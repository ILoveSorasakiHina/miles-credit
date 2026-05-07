import gc
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.fft
import torch.nn.functional as F
import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset
from credit.scheduler import update_on_batch
from credit.trainers.utils import cycle, accum_log
from credit.trainers.base_trainer import BaseTrainer
from credit.data import concat_and_reshape, reshape_only
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer

# ====================================================================== #
# 修改 #1：import CrossFormer，用來讀取 class-level _feat_hidden_storage
# 原因：crossformer.py forward 把 feat_hidden 寫到 CrossFormer._feat_hidden_storage（class-level dict）
#       因為 FSDP 會包裝 model 導致 instance attribute (self._last_feat_hidden) 在不同 step 抓不到
#       (上次 debug 確認：FORWARD 寫入的 self_id 跟 trainer 讀到的 base_model_id 不同)
#       Class-level dict 不受 instance wrap 影響，所有地方看到的都是同一份
# ====================================================================== #
from credit.models.crossformer import CrossFormer

import optuna
import torch

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int):
        super().__init__(model, rank)
        logger.info("Loading a multi-step trainer class")
        self._criterion_optimizer = None

    def train_one_epoch(
        self, epoch, conf, trainloader, optimizer, criterion, scaler, scheduler, metrics
    ):
        # ==== 為 criterion 建立獨立 optimizer ====
        if self._criterion_optimizer is None:
            base_crit = criterion.module if hasattr(criterion, 'module') else criterion
            if hasattr(base_crit, 'parameters'):
                crit_params = list(filter(lambda p: p.requires_grad, base_crit.parameters()))
                if len(crit_params) > 0:
                    self._criterion_optimizer = torch.optim.AdamW(
                        crit_params,
                        lr=float(conf["trainer"]["learning_rate"]),
                        weight_decay=float(conf["trainer"]["weight_decay"]),
                        betas=(0.9, 0.95),
                    )
                    if self.rank == 0:
                        logger.info(
                            f"Created separate criterion optimizer with "
                            f"{len(crit_params)} parameter tensors "
                            f"({sum(p.numel() for p in crit_params):,} total params)"
                        )

        batches_per_epoch = conf["trainer"]["batches_per_epoch"]
        grad_max_norm = conf["trainer"].get("grad_max_norm", 0.0)
        amp = conf["trainer"]["amp"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False
        forecast_length = conf["data"]["forecast_len"]
        ensemble_size = conf["trainer"].get("ensemble_size", 1)
        if ensemble_size > 1:
            logger.info(f"ensemble training with ensemble_size {ensemble_size}")
        logger.info(f"Using grad-max-norm value: {grad_max_norm}")

        varnum_diag = len(conf["data"]["diagnostic_variables"])

        static_dim_size = (
            len(conf["data"]["dynamic_forcing_variables"])
            + len(conf["data"]["forcing_variables"])
            + len(conf["data"]["static_variables"])
        )

        ensemble_size = conf["trainer"].get("ensemble_size", 1)

        retain_graph = conf["data"].get("retain_graph", False)

        if "backprop_on_timestep" in conf["data"]:
            backprop_on_timestep = conf["data"]["backprop_on_timestep"]
        else:
            backprop_on_timestep = list(range(0, conf["data"]["forecast_len"] + 1 + 1))

        assert (
            forecast_length <= backprop_on_timestep[-1]
        ), f"forecast_length ({forecast_length + 1}) must not exceed the max value in backprop_on_timestep {backprop_on_timestep}"

        if (
            conf["trainer"]["use_scheduler"]
            and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda"
        ):
            scheduler.step()

        if conf["data"]["data_clamp"] is None:
            flag_clamp = False
        else:
            flag_clamp = True
            clamp_min = float(conf["data"]["data_clamp"][0])
            clamp_max = float(conf["data"]["data_clamp"][1])

        post_conf = conf["model"]["post_conf"]
        flag_mass_conserve = False
        flag_water_conserve = False
        flag_energy_conserve = False

        if post_conf["activate"]:
            if post_conf["global_mass_fixer"]["activate"]:
                if post_conf["global_mass_fixer"]["activate_outside_model"]:
                    logger.info("Activate GlobalMassFixer outside of model")
                    flag_mass_conserve = True
                    opt_mass = GlobalMassFixer(post_conf)

            if post_conf["global_water_fixer"]["activate"]:
                if post_conf["global_water_fixer"]["activate_outside_model"]:
                    logger.info("Activate GlobalWaterFixer outside of model")
                    flag_water_conserve = True
                    opt_water = GlobalWaterFixer(post_conf)

            if post_conf["global_energy_fixer"]["activate"]:
                if post_conf["global_energy_fixer"]["activate_outside_model"]:
                    logger.info("Activate GlobalEnergyFixer outside of model")
                    flag_energy_conserve = True
                    opt_energy = GlobalEnergyFixer(post_conf)

        if not isinstance(trainloader.dataset, IterableDataset):
            if hasattr(trainloader.dataset, "batches_per_epoch"):
                dataset_batches_per_epoch = trainloader.dataset.batches_per_epoch()
            elif hasattr(trainloader.sampler, "batches_per_epoch"):
                dataset_batches_per_epoch = trainloader.sampler.batches_per_epoch()
            else:
                dataset_batches_per_epoch = len(trainloader)
            batches_per_epoch = (
                batches_per_epoch
                if 0 < batches_per_epoch < dataset_batches_per_epoch
                else dataset_batches_per_epoch
            )

        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch), total=batches_per_epoch, leave=True
        )

        self.model.train()

        dl = cycle(trainloader)
        results_dict = defaultdict(list)
        for steps in range(batches_per_epoch):
            logs = {}
            loss = 0
            stop_forecast = False
            y_pred = None
            while not stop_forecast:
                batch = next(dl)
                forecast_step = batch["forecast_step"].item()
                if forecast_step == 1:
                    if "x_surf" in batch:
                        x = concat_and_reshape(batch["x"], batch["x_surf"]).to(
                            self.device
                        )
                    else:
                        x = reshape_only(batch["x"]).to(self.device)

                    if ensemble_size > 1:
                        x = torch.repeat_interleave(x, ensemble_size, 0)

                if "x_forcing_static" in batch:
                    x_forcing_batch = (
                        batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4)
                    )
                    if ensemble_size > 1:
                        x_forcing_batch = torch.repeat_interleave(
                            x_forcing_batch, ensemble_size, 0
                        )
                    x = torch.cat((x, x_forcing_batch), dim=1)

                if flag_clamp:
                    x = torch.clamp(x, min=clamp_min, max=clamp_max)

                x = x.float()
                with torch.autocast(device_type="cuda", enabled=amp):
                    y_pred = self.model(x)

                # ============================================================== #
                # 修改 #2：從 CrossFormer class-level storage 抓 feat_hidden
                # 原因：上次 debug 確認 self.model.module._last_feat_hidden 在 FSDP 下
                #       第二個 batch 開始就拿不到（id 不一致 + FSDP reshard 行為）
                #       改用 class-level dict 不受 wrap 影響，永遠拿得到
                # ============================================================== #
                storage = getattr(CrossFormer, '_feat_hidden_storage', {})
                feat_hidden = storage.pop('last', None)
                # ============================================================== #

                # ==== 判斷是 diffusion 還是 MSE 模式 ====
                base_criterion = criterion.module if hasattr(criterion, 'module') else criterion
                is_diffusion = (
                    hasattr(base_criterion, 'deterministic_predict') 
                    and feat_hidden is not None
                )

                # ==== Rollout 用 y_pred（兩種模式都一樣）====
                y_physical = y_pred

                if flag_mass_conserve:
                    if forecast_step == 1:
                        x_init = x.clone()

                if flag_mass_conserve:
                    input_dict = {"y_pred": y_physical, "x": x_init}
                    input_dict = opt_mass(input_dict)
                    y_physical = input_dict["y_pred"]

                if flag_water_conserve:
                    input_dict = {"y_pred": y_physical, "x": x}
                    input_dict = opt_water(input_dict)
                    y_physical = input_dict["y_pred"]

                if flag_energy_conserve:
                    input_dict = {"y_pred": y_physical, "x": x}
                    input_dict = opt_energy(input_dict)
                    y_physical = input_dict["y_pred"]

                if forecast_step in backprop_on_timestep:
                    if "y_surf" in batch:
                        y = concat_and_reshape(batch["y"], batch["y_surf"]).to(
                            self.device
                        )
                    else:
                        y = reshape_only(batch["y"]).to(self.device)

                    if "y_diag" in batch:
                        y_diag_batch = (
                            batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4)
                        )
                        y = torch.cat((y, y_diag_batch), dim=1)

                    if flag_clamp:
                        y = torch.clamp(y, min=clamp_min, max=clamp_max)

                    with torch.autocast(enabled=amp, device_type="cuda"):
                        if is_diffusion:
                            # Diffusion B2 模式：MSE + λ * diffusion
                            loss_mse = F.mse_loss(y_pred, y.to(y_pred.dtype))
                            loss_diffusion = criterion(y.to(y_pred.dtype), feat_hidden).mean()
                            loss = loss_mse + 0.5 * loss_diffusion
                        else:
                            # MSE 模式：原本邏輯
                            loss = criterion(y.to(y_pred.dtype), y_pred).mean()

                    accum_log(logs, {"loss": loss.item()})
                    scaler.scale(loss).backward(retain_graph=retain_graph)

                if distributed:
                    torch.distributed.barrier()

                stop_forecast = batch["stop_forecast"].item()
                if stop_forecast:
                    break

                if not retain_graph:
                    y_pred = y_pred.detach()
                    y_physical = y_physical.detach()
                
                if x.shape[2] == 1:
                    if "y_diag" in batch:
                        x = y_physical[:, :-varnum_diag, ...]
                    else:
                        x = y_physical
                else:
                    if static_dim_size == 0:
                        x_detach = x[:, :, 1:, ...].detach()
                    else:
                        x_detach = x[:, :-static_dim_size, 1:, ...].detach()

                    if "y_diag" in batch:
                        x = torch.cat(
                            [x_detach, y_physical[:, :-varnum_diag, ...]],
                            dim=2,
                        )
                    else:
                        x = torch.cat([x_detach, y_physical], dim=2)

            if distributed:
                torch.distributed.barrier()

            scaler.unscale_(optimizer)
            if grad_max_norm == "dynamic":
                local_norm = torch.norm(
                    torch.stack(
                        [
                            p.grad.detach().norm(2)
                            for p in self.model.parameters()
                            if p.grad is not None
                        ]
                    )
                )

                if distributed:
                    dist.all_reduce(local_norm, op=dist.ReduceOp.SUM)
                global_norm = local_norm.sqrt()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=global_norm
                )
            elif grad_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=grad_max_norm
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # step criterion optimizer（FSDP 路徑手動處理）
            if self._criterion_optimizer is not None:
                if distributed:
                    for group in self._criterion_optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                self._criterion_optimizer.step()
                self._criterion_optimizer.zero_grad()

            metrics_dict = metrics(y_pred, y)
            for name, value in metrics_dict.items():
                value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                if distributed:
                    dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                results_dict[f"train_{name}"].append(value[0].item())

            batch_loss = torch.Tensor([logs["loss"]]).cuda(self.device)
            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())
            results_dict["train_forecast_len"].append(forecast_length + 1)

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                print(
                    results_dict["train_loss"],
                    batch["x"].shape,
                    batch["y"].shape,
                    batch["index"],
                )

                if distributed and dist.is_initialized():
                    logger.error("Loss is non-finite. Destroying process group before pruning.")
                    dist.destroy_process_group()

                try:
                    raise optuna.TrialPruned()
                except Exception as E:
                    raise E

            to_print = "Epoch: {} train_loss: {:.6f} train_acc: {:.6f} train_mae: {:.6f} forecast_len: {:.6f}".format(
                epoch,
                np.mean(results_dict["train_loss"]),
                np.mean(results_dict["train_acc"]),
                np.mean(results_dict["train_mae"]),
                forecast_length + 1,
            )
            if ensemble_size > 1:
                to_print += f" std: {np.mean(results_dict['train_std']):.6f}"
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            if self.rank == 0:
                batch_group_generator.update(1)
                batch_group_generator.set_description(to_print)

            if (
                conf["trainer"]["use_scheduler"]
                and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_batch
            ):
                scheduler.step()

        batch_group_generator.close()

        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def validate(self, epoch, conf, valid_loader, criterion, metrics):
        self.model.eval()

        varnum_diag = len(conf["data"]["diagnostic_variables"])

        static_dim_size = (
            len(conf["data"]["dynamic_forcing_variables"])
            + len(conf["data"]["forcing_variables"])
            + len(conf["data"]["static_variables"])
        )

        valid_batches_per_epoch = conf["trainer"]["valid_batches_per_epoch"]
        history_len = (
            conf["data"]["valid_history_len"]
            if "valid_history_len" in conf["data"]
            else conf["history_len"]
        )
        forecast_len = (
            conf["data"]["valid_forecast_len"]
            if "valid_forecast_len" in conf["data"]
            else conf["forecast_len"]
        )
        ensemble_size = conf["trainer"].get("ensemble_size", 1)

        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        results_dict = defaultdict(list)

        if not isinstance(valid_loader.dataset, IterableDataset):
            if hasattr(valid_loader.dataset, "batches_per_epoch"):
                dataset_batches_per_epoch = valid_loader.dataset.batches_per_epoch()
            elif hasattr(valid_loader.sampler, "batches_per_epoch"):
                dataset_batches_per_epoch = valid_loader.sampler.batches_per_epoch()
            else:
                dataset_batches_per_epoch = len(valid_loader)
            valid_batches_per_epoch = (
                valid_batches_per_epoch
                if 0 < valid_batches_per_epoch < dataset_batches_per_epoch
                else dataset_batches_per_epoch
            )

        if conf["data"]["data_clamp"] is None:
            flag_clamp = False
        else:
            flag_clamp = True
            clamp_min = float(conf["data"]["data_clamp"][0])
            clamp_max = float(conf["data"]["data_clamp"][1])

        post_conf = conf["model"]["post_conf"]
        flag_mass_conserve = False
        flag_water_conserve = False
        flag_energy_conserve = False

        if post_conf["activate"]:
            if post_conf["global_mass_fixer"]["activate"]:
                if post_conf["global_mass_fixer"]["activate_outside_model"]:
                    logger.info("Activate GlobalMassFixer outside of model")
                    flag_mass_conserve = True
                    opt_mass = GlobalMassFixer(post_conf)

            if post_conf["global_water_fixer"]["activate"]:
                if post_conf["global_water_fixer"]["activate_outside_model"]:
                    logger.info("Activate GlobalWaterFixer outside of model")
                    flag_water_conserve = True
                    opt_water = GlobalWaterFixer(post_conf)

            if post_conf["global_energy_fixer"]["activate"]:
                if post_conf["global_energy_fixer"]["activate_outside_model"]:
                    logger.info("Activate GlobalEnergyFixer outside of model")
                    flag_energy_conserve = True
                    opt_energy = GlobalEnergyFixer(post_conf)

        batch_group_generator = tqdm.tqdm(
            range(valid_batches_per_epoch), total=valid_batches_per_epoch, leave=True
        )

        stop_forecast = False
        dl = cycle(valid_loader)
        with torch.no_grad():
            for steps in range(valid_batches_per_epoch):
                loss = 0
                stop_forecast = False
                y_pred = None
                while not stop_forecast:
                    batch = next(dl)
                    forecast_step = batch["forecast_step"].item()
                    stop_forecast = batch["stop_forecast"].item()
                    
                    if forecast_step == 1:
                        if "x_surf" in batch:
                            x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device)
                        else:
                            x = reshape_only(batch["x"]).to(self.device)
                        if ensemble_size > 1:
                            x = torch.repeat_interleave(x, ensemble_size, 0)

                    if "x_forcing_static" in batch:
                        x_forcing_batch = (
                            batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4)
                        )
                        if ensemble_size > 1:
                            x_forcing_batch = torch.repeat_interleave(x_forcing_batch, ensemble_size, 0)
                        x = torch.cat((x, x_forcing_batch), dim=1)

                    if flag_clamp:
                        x = torch.clamp(x, min=clamp_min, max=clamp_max)

                    y_pred = self.model(x.float())

                    # ============================================================== #
                    # 修改 #3：validate 也改用 class-level storage（同樣理由）
                    # ============================================================== #
                    storage = getattr(CrossFormer, '_feat_hidden_storage', {})
                    feat_hidden = storage.pop('last', None)
                    # ============================================================== #

                    base_criterion = criterion.module if hasattr(criterion, 'module') else criterion
                    is_diffusion = (
                        hasattr(base_criterion, 'deterministic_predict') 
                        and feat_hidden is not None
                    )

                    # 兩種 inference 路徑
                    y_physical_mse = y_pred

                    if is_diffusion:
                        y_physical_diff = base_criterion.deterministic_predict(feat_hidden)
                        if y_physical_diff.shape[-2:] != y_pred.shape[-2:]:
                            if y_physical_diff.dim() == 5:
                                B, C, T, H, W = y_physical_diff.shape
                                y_physical_diff = y_physical_diff.reshape(B, C, H, W)
                                y_physical_diff = F.interpolate(
                                    y_physical_diff, size=y_pred.shape[-2:],
                                    mode='bilinear', align_corners=False
                                )
                                y_physical_diff = y_physical_diff.unsqueeze(2)
                            else:
                                y_physical_diff = F.interpolate(
                                    y_physical_diff, size=y_pred.shape[-2:],
                                    mode='bilinear', align_corners=False
                                )
                    else:
                        y_physical_diff = None

                    y_physical = y_physical_mse

                    if flag_mass_conserve:
                        if forecast_step == 1:
                            x_init = x.clone()
                    if flag_mass_conserve:
                        input_dict = {"y_pred": y_physical, "x": x_init}
                        input_dict = opt_mass(input_dict)
                        y_physical = input_dict["y_pred"]
                    if flag_water_conserve:
                        input_dict = {"y_pred": y_physical, "x": x}
                        input_dict = opt_water(input_dict)
                        y_physical = input_dict["y_pred"]
                    if flag_energy_conserve:
                        input_dict = {"y_pred": y_physical, "x": x}
                        input_dict = opt_energy(input_dict)
                        y_physical = input_dict["y_pred"]

                    if forecast_step == (forecast_len + 1):
                        if "y_surf" in batch:
                            y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device)
                        else:
                            y = reshape_only(batch["y"]).to(self.device)
                        if "y_diag" in batch:
                            y_diag_batch = batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4)
                            y = torch.cat((y, y_diag_batch), dim=1)
                        if flag_clamp:
                            y = torch.clamp(y, min=clamp_min, max=clamp_max)

                        # Loss 計算（兩種模式分流）
                        if is_diffusion:
                            loss = criterion(y.to(y_pred.dtype), feat_hidden).mean()
                        else:
                            loss = criterion(y.to(y_pred.dtype), y_pred).mean()

                        # MSE baseline 的 metrics
                        metrics_dict_mse = metrics(y_physical_mse.float(), y.float())
                        for name, value in metrics_dict_mse.items():
                            value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                            if distributed:
                                dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                            results_dict[f"valid_{name}"].append(value[0].item())

                        # Diffusion 路徑的 metrics
                        if is_diffusion and y_physical_diff is not None:
                            metrics_dict_diff = metrics(y_physical_diff.float(), y.float())
                            for name, value in metrics_dict_diff.items():
                                value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                                if distributed:
                                    dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                                results_dict[f"valid_{name}_diff"].append(value[0].item())

                        assert stop_forecast
                        break

                    elif history_len == 1:
                        if "y_diag" in batch:
                            x = y_physical[:, :-varnum_diag, ...].detach()
                        else:
                            x = y_physical.detach()
                    else:
                        if static_dim_size == 0:
                            x_detach = x[:, :, 1:, ...].detach()
                        else:
                            x_detach = x[:, :-static_dim_size, 1:, ...].detach()
                        if "y_diag" in batch:
                            x = torch.cat([x_detach, y_physical[:, :-varnum_diag, ...].detach()], dim=2)
                        else:
                            x = torch.cat([x_detach, y_physical.detach()], dim=2)

                batch_loss = torch.Tensor([loss.item()]).cuda(self.device)
                if distributed:
                    torch.distributed.barrier()
                results_dict["valid_loss"].append(batch_loss[0].item())
                results_dict["valid_forecast_len"].append(forecast_len + 1)
                stop_forecast = False

                to_print = "Epoch: {} valid_loss: {:.6f} valid_acc: {:.6f} valid_mae: {:.6f}".format(
                    epoch,
                    np.mean(results_dict["valid_loss"]),
                    np.mean(results_dict["valid_acc"]),
                    np.mean(results_dict["valid_mae"]),
                )
                if ensemble_size > 1:
                    to_print += f" std: {np.mean(results_dict['valid_std']):.6f}"
                if self.rank == 0:
                    batch_group_generator.update(1)
                    batch_group_generator.set_description(to_print)

        batch_group_generator.close()

        if distributed:
            torch.distributed.barrier()

        torch.cuda.empty_cache()
        gc.collect()

        return results_dict