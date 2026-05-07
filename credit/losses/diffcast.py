"""
Segmentation Diffusion Loss (SegDiffLoss)
=========================================
Inspired by MMPD (Multi-Mode Patch Diffusion) loss for time series forecasting,
adapted for 2D dense prediction tasks.

Supports two modes:
- 'regression':      target is continuous (B, C, H, W), e.g. weather prediction
- 'classification':  target is integer labels (B, H, W), e.g. semantic segmentation

Usage:
------
    backbone = YourBackbone()

    # === Regression mode (continuous output) ===
    loss_module = SegDiffusionLoss(
        output_channels=3,             # number of output channels
        feature_channels=256,          # channels of backbone feature map
        mode='regression',             # continuous output
    )
    
    features = backbone(images)        # (B, C_feat, H, W)
    loss = loss_module(features, target)  # target: (B, 3, H, W) float tensor
    loss.backward()
    
    pred = loss_module.deterministic_predict(features)  # (B, 3, H, W) continuous values
    
    # === Classification mode (segmentation) ===
    loss_module = SegDiffusionLoss(
        output_channels=21,            # number of classes
        feature_channels=256,
        mode='classification',         # integer labels
    )
    
    features = backbone(images)        # (B, C_feat, H, W)
    loss = loss_module(features, masks)  # masks: (B, H, W) long tensor
    loss.backward()
    
    logits = loss_module.deterministic_predict(features)  # (B, 21, H, W)
    seg_map = logits.argmax(dim=1)     # (B, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 1. Diffusion Schedule
# =============================================================================

class DiffusionSchedule:
    """
    Linear noise schedule for diffusion.
    Precomputes alpha_bar and related constants.
    """
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_steps = num_steps
        
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
    
    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self
    
    def add_noise(self, x0, noise, k):
        """
        Forward process: q(x_k | x_0) = N(sqrt(alpha_bar_k) * x_0, (1-alpha_bar_k) * I)
        
        Args:
            x0: clean sample, (B, C, H, W)
            noise: Gaussian noise, same shape as x0
            k: diffusion step indices, (B,)
        Returns:
            xk: noisy sample at step k
        """
        sqrt_ab = self.sqrt_alpha_bar[k].view(-1, 1, 1, 1)
        sqrt_1_ab = self.sqrt_one_minus_alpha_bar[k].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_1_ab * noise

    def find_anchor_step(self):
        """Find k* such that alpha_bar_k* is close to 0.5"""
        return (self.alpha_bar - 0.5).abs().argmin().item()


# =============================================================================
# 2. Timestep Embedding
# =============================================================================

class TimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, k):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=k.device, dtype=torch.float32) * -emb)
        emb = k.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.mlp(emb)


# =============================================================================
# 3. AdaLN (Adaptive Layer Normalization)
# =============================================================================

class AdaLN(nn.Module):
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, channels * 3),
        )
        # 修改：只 zero-init bias，讓 gamma/beta/gate 從 0 開始
        # 但 weight 用預設初始化，確保梯度能流
        # 實際上 DiT 的做法：bias zero-init，weight 也 zero-init，但輸出用 gate 機制
        # 這裡改成：weight 正常初始化，但改 forward 的起始點
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
    
    def forward(self, x, condition):
        gamma, beta, gate = self.proj(condition).chunk(3, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        x = self.norm(x)
        # 修改：gate 從 1 開始（不是從 0），這樣訊號能流過
        return (1 + gate) * ((1 + gamma) * x + beta)



# =============================================================================
# 4. ConvNeXt-style Denoiser Block
# =============================================================================

class ConvNeXtDenoiserBlock(nn.Module):
    """
    Depthwise large-kernel Conv -> AdaLN -> 1x1 Conv -> GELU -> 1x1 Conv -> Residual
    
    2D analog of MMPD's Patch Consistent MLP:
    - Large kernel = looking at adjacent patches for consistency
    - AdaLN = injecting backbone tokens as condition
    """
    def __init__(self, channels, condition_dim, kernel_size=7, expansion=4):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels)
        self.adaln = AdaLN(channels, condition_dim)
        hidden = channels * expansion
        self.pw1 = nn.Conv2d(channels, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, channels, 1)
        # nn.init.zeros_(self.pw2.weight)
        # nn.init.zeros_(self.pw2.bias)
    
    def forward(self, x, condition):
        residual = x
        x = self.dwconv(x)
        x = self.adaln(x, condition)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return residual + x


# =============================================================================
# 5. Full Denoiser Network
# =============================================================================

class ConvNeXtDenoiser(nn.Module):
    """
    Lightweight ConvNeXt-style denoiser.

    Condition injection:
    - Global:  backbone features (avg pooled) + timestep -> AdaLN condition vector
    - Spatial: backbone features (projected) concatenated with noisy input, then fused by 1x1 conv
    """
    def __init__(self, output_channels, feature_channels, hidden_dim=128,
                 num_blocks=4, kernel_size=7, expansion=4):
        super().__init__()
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        
        # Timestep embedding
        self.time_emb = TimestepEmbedding(hidden_dim)
        
        # Global condition from backbone
        self.feat_pool = nn.AdaptiveAvgPool2d(1)
        self.feat_proj = nn.Linear(feature_channels, hidden_dim)
        
        # Spatial condition from backbone
        self.input_proj = nn.Conv2d(output_channels, hidden_dim, 1)
        self.feat_spatial_proj = nn.Conv2d(feature_channels, hidden_dim, 1)
        self.fuse_proj = nn.Conv2d(hidden_dim * 2, hidden_dim, 1)


        self.blocks = nn.ModuleList([
            ConvNeXtDenoiserBlock(hidden_dim, hidden_dim, kernel_size, expansion)
            for _ in range(num_blocks)
        ])
        self.output_norm = nn.GroupNorm(1, hidden_dim)
        self.output_proj = nn.Conv2d(hidden_dim, output_channels, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, noisy_input, backbone_features, k):
        """
        Args:
            noisy_input: (B, output_channels, H, W)
            backbone_features: (B, C_feat, H, W)
            k: (B,) diffusion timestep
        Returns:
            predicted_noise: (B, output_channels, H, W)
        """
        # Global condition
        t_emb = self.time_emb(k)
        feat_global = self.feat_pool(backbone_features).flatten(1)
        feat_global = self.feat_proj(feat_global)
        condition = t_emb + feat_global
        
        # Spatial condition
        feat_spatial = self.feat_spatial_proj(backbone_features)
        if feat_spatial.shape[-2:] != noisy_input.shape[-2:]:
            feat_spatial = F.interpolate(
                feat_spatial, size=noisy_input.shape[-2:], mode='bilinear', align_corners=False
            )
        
        # Forward
        x = self.fuse_proj(torch.cat([self.input_proj(noisy_input), feat_spatial], dim=1))
        for block in self.blocks:
            x = block(x, condition)
        x = self.output_norm(x)
        return self.output_proj(x)


# =============================================================================
# 6. Main Module: Diffusion Loss for Dense Prediction
# =============================================================================


class SegDiffusionLoss(nn.Module):
    """
    DiffCast-style Residual Diffusion Loss for dense prediction.
    
    對齊 DiffCast (Yu et al., CVPR 2024) 的設計：
    - Backbone 輸出 y_pred，但作為 condition 時不 detach
    - Residual r = y - y_pred 也不 detach（讓 diffusion 梯度回傳 backbone）
    - 用 ConditionEncoder 把 y_pred 編碼到 hidden 空間（對應 GlobalNet 簡化版）
    - 訓練 loss = α * diffusion + (1-α) * MSE(y_pred, y)，α=0.5
    - 端到端聯合訓練（DiffCast Table 3 證實 end-to-end 比 frozen 好）
    """
    def __init__(
        self,
        output_channels,
        feature_channels,           # 目前用不到，保留只為 config 相容
        mode='regression',
        hidden_dim=128,
        num_blocks=4,
        kernel_size=7,
        expansion=4,
        diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        lambda_weight=0.5,          # 預設 0.5，對齊 DiffCast α
        residual_scale=1.0,
        cond_encoder_layers=3,      # ConditionEncoder 層數
        cond_encoder_dim=128,       # ConditionEncoder 輸出維度
    ):
        super().__init__()
        assert mode == 'regression', "Residual diffusion only supports regression mode"
        
        self.output_channels = output_channels
        self.mode = mode
        self.lambda_weight = lambda_weight
        self.diffusion_steps = diffusion_steps
        self.residual_scale = residual_scale
        
        self.schedule = DiffusionSchedule(diffusion_steps, beta_start, beta_end)
        
        # ==== 新增：Condition Encoder（對應 DiffCast 的 GlobalNet 簡化版）====
        # 把 y_pred (output_channels) 編碼到更抽象的 hidden space
        # 避免「condition 和 target 在同一空間造成 identity mapping」的問題
        cond_layers = []
        c_in = output_channels
        for i in range(cond_encoder_layers):
            cond_layers += [
                nn.Conv2d(c_in, cond_encoder_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, cond_encoder_dim),
                nn.SiLU(),
            ]
            c_in = cond_encoder_dim
        self.condition_encoder = nn.Sequential(*cond_layers)
        # ====================================================================
        
        # Denoiser 的 feature_channels 改用 cond_encoder_dim（不是原本的 71）
        self.denoiser = ConvNeXtDenoiser(
            output_channels=output_channels,
            feature_channels=cond_encoder_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            expansion=expansion,
        )
        self._anchor_step = None
    
    @property
    def anchor_step(self):
        if self._anchor_step is None:
            self._anchor_step = self.schedule.find_anchor_step()
        return self._anchor_step
    
    def _ensure_schedule_device(self, device):
        if self.schedule.betas.device != device:
            self.schedule = self.schedule.to(device)
    
    @staticmethod
    def _squeeze_if_5d(t):
        if t.dim() == 5 and t.shape[2] == 1:
            return t.squeeze(2), True
        return t, False
    
    @staticmethod
    def _unsqueeze_if_needed(t, was_5d):
        if was_5d:
            return t.unsqueeze(2)
        return t
    
    def forward(self, target, backbone_prediction):
        """
        Args:
            target: (B, C, H, W) or (B, C, 1, H, W) — ground truth y
            backbone_prediction: same shape — backbone 的粗預測 y_pred
        Returns:
            total_loss: scalar
        """
        device = backbone_prediction.device
        self._ensure_schedule_device(device)
        if next(self.denoiser.parameters()).device != device:
            self.to(device)
        
        # 統一成 4D
        y_pred, _ = self._squeeze_if_5d(backbone_prediction)
        y_true, _ = self._squeeze_if_5d(target)
        y_true = y_true.float()
        y_pred = y_pred.float()
        
        if y_true.shape[-2:] != y_pred.shape[-2:]:
            y_true = F.interpolate(y_true, size=y_pred.shape[-2:], mode='nearest')
        
        # ==== 關鍵改動 1：residual 計算不 detach y_pred ====
        # 之前: residual = (y_true - y_pred.detach()) * scale
        # DiffCast 風格：完全不 detach，讓梯度可以從 residual 流回 backbone
        residual = (y_true - y_pred) * self.residual_scale
        # ==================================================
        
        # ==== 關鍵改動 2：condition 過 ConditionEncoder，且不 detach ====
        # 之前: y_pred_cond_for_denoiser = y_pred.detach()  # 直接用、且 detach
        # DiffCast 風格：先 encode 到 hidden space，且不 detach
        condition = self.condition_encoder(y_pred)
        # ================================================================
        
        B = residual.shape[0]
        
        # === Diffusion Loss ===
        k = torch.randint(0, self.diffusion_steps, (B,), device=device)
        noise = torch.randn_like(residual)
        rk = self.schedule.add_noise(residual, noise, k)
        predicted_noise = self.denoiser(rk, condition, k)
        loss_diffusion = F.mse_loss(predicted_noise, noise)
        
        # === Deterministic Loss (MMPD Eq.8 / DiffCast inner) ===
        k_star = self.anchor_step
        alpha_bar_star = self.schedule.alpha_bar[k_star]
        scale = torch.sqrt(alpha_bar_star / (1 - alpha_bar_star))
        
        k_star_tensor = torch.full((B,), k_star, device=device, dtype=torch.long)
        zero_input = torch.zeros_like(residual)
        predicted_noise_det = self.denoiser(zero_input, condition, k_star_tensor)
        target_det = scale * residual
        loss_deterministic = F.mse_loss(predicted_noise_det, -target_det)
        
        # === Backbone MSE Loss ===
        # 對應 DiffCast 的 deterministic loss L_P
        loss_backbone = F.mse_loss(y_pred, y_true)
        
        # ==== 診斷 print ====
        if not hasattr(self, '_diag_step'):
            self._diag_step = 0
        self._diag_step += 1
        
        if self._diag_step % 50 == 0:
            try:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
            except Exception:
                rank = 0
            
            if rank == 0:
                with torch.no_grad():
                    res_std = residual.std().item()
                    res_abs_mean = residual.abs().mean().item()
                    
                    y_pred_std = y_pred.std().item()
                    y_true_std = y_true.std().item()
                    cond_std = condition.std().item()
                    
                    inv_scale_ = torch.sqrt((1 - alpha_bar_star) / alpha_bar_star)
                    pred_residual = (-inv_scale_ * predicted_noise_det) / self.residual_scale
                    true_residual_unscaled = residual / self.residual_scale
                    
                    pred_res_std = pred_residual.std().item()
                    residual_mse = F.mse_loss(pred_residual, true_residual_unscaled).item()
                    residual_mse_if_zero = (true_residual_unscaled ** 2).mean().item()
                    
                    op_w_norm = self.denoiser.output_proj.weight.norm().item()
                    
                    print(
                        f"\n[DIAG step={self._diag_step}] "
                        f"y_pred_std={y_pred_std:.3f} y_true_std={y_true_std:.3f} cond_std={cond_std:.3f} | "
                        f"residual std={res_std:.3f} abs_mean={res_abs_mean:.3f} | "
                        f"pred_residual std={pred_res_std:.3f} | "
                        f"residual_MSE={residual_mse:.4f} (zero_baseline={residual_mse_if_zero:.4f}) | "
                        f"L_diff={loss_diffusion.item():.4f} L_det={loss_deterministic.item():.4f} L_back={loss_backbone.item():.4f} | "
                        f"output_proj.norm={op_w_norm:.4f}",
                        flush=True
                    )
        
        # ==== 關鍵改動 3：DiffCast 風格的 loss 組合 ====
        # 對應 DiffCast Eq. 12: L = α * L_ε + (1-α) * L_P
        # L_ε 包含 diffusion 內部的 L_diff + L_det（沿用 MMPD Eq.8）
        # L_P 是 backbone 對齊 y 的 MSE
        # α = self.lambda_weight，預設 0.5（DiffCast 也是 0.5）
        loss_diff_inner = 0.99 * loss_diffusion + 0.01 * loss_deterministic
        # 內層用論文預設 0.99（diffusion 主導，deterministic 只是輕微正規化）
        
        total_loss = (
            self.lambda_weight * loss_diff_inner       # diffusion 部分
            + (1 - self.lambda_weight) * loss_backbone # backbone MSE 部分
        )
        # ===============================================
        
        return total_loss
    
    @torch.no_grad()
    def deterministic_predict(self, backbone_prediction):
        """推論: y_physical = y_pred + predicted_residual"""
        device = backbone_prediction.device
        self._ensure_schedule_device(device)
        if next(self.denoiser.parameters()).device != device:
            self.to(device)
        
        y_pred, was_5d = self._squeeze_if_5d(backbone_prediction)
        B = y_pred.shape[0]
        
        # 編碼 condition
        condition = self.condition_encoder(y_pred)
        
        k_star = self.anchor_step
        alpha_bar_star = self.schedule.alpha_bar[k_star]
        inv_scale = torch.sqrt((1 - alpha_bar_star) / alpha_bar_star)
        
        k_star_tensor = torch.full((B,), k_star, device=device, dtype=torch.long)
        zero_input = torch.zeros(
            B, self.output_channels,
            y_pred.shape[2], y_pred.shape[3],
            device=device
        )
        
        predicted_noise = self.denoiser(zero_input, condition, k_star_tensor)
        predicted_residual = (-inv_scale * predicted_noise) / self.residual_scale
        
        y_physical = y_pred + predicted_residual
        
        return self._unsqueeze_if_needed(y_physical, was_5d)
    
    @torch.no_grad()
    def probabilistic_predict(self, backbone_prediction, num_samples=10, infer_steps=20):
        """機率預測（也要過 ConditionEncoder）"""
        device = backbone_prediction.device
        self._ensure_schedule_device(device)
        
        y_pred, was_5d = self._squeeze_if_5d(backbone_prediction)
        B, _, H, W = y_pred.shape
        
        condition = self.condition_encoder(y_pred)
        
        step_indices = torch.linspace(
            self.diffusion_steps - 1, 0, infer_steps, device=device
        ).long()
        
        all_samples = []
        for _ in range(num_samples):
            x = torch.randn(B, self.output_channels, H, W, device=device)
            for i in range(len(step_indices)):
                k = step_indices[i]
                k_tensor = k.unsqueeze(0).expand(B)
                predicted_noise = self.denoiser(x, condition, k_tensor)
                
                alpha = self.schedule.alphas[k]
                alpha_bar = self.schedule.alpha_bar[k]
                beta = self.schedule.betas[k]
                x0_pred = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
                
                if i < len(step_indices) - 1:
                    k_prev = step_indices[i + 1]
                    alpha_bar_prev = self.schedule.alpha_bar[k_prev]
                    coef1 = torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar)
                    coef2 = torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
                    mean = coef1 * x0_pred + coef2 * x
                    var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                    x = mean + torch.sqrt(var) * torch.randn_like(x)
                else:
                    x = x0_pred
            
            sample_physical = y_pred + x / self.residual_scale
            sample_physical = self._unsqueeze_if_needed(sample_physical, was_5d)
            all_samples.append(sample_physical)
        
        return torch.stack(all_samples)    


# =============================================================================
# 7. Quick Test / Example
# =============================================================================

if __name__ == "__main__":
    B, C_feat, H, W = 2, 256, 64, 64

    backbone_features = torch.randn(B, C_feat, H, W)

    # =====================
    # Regression mode test
    # =====================
    print("=" * 50)
    print("Regression Mode")
    print("=" * 50)
    
    output_channels = 3  # e.g., predicting 3 continuous variables
    target_reg = torch.randn(B, output_channels, H, W)
    
    loss_reg = SegDiffusionLoss(
        output_channels=output_channels,
        feature_channels=C_feat,
        mode='regression',
    )
    
    # Training
    loss = loss_reg(backbone_features, target_reg)
    print(f"Training loss: {loss.item():.4f}")
    
    # Deterministic inference
    pred = loss_reg.deterministic_predict(backbone_features)
    print(f"Prediction shape: {pred.shape}")  # (B, 3, H, W) continuous values
    
    # ========================
    # Classification mode test
    # ========================
    print()
    print("=" * 50)
    print("Classification Mode")
    print("=" * 50)
    
    num_classes = 21
    target_cls = torch.randint(0, num_classes, (B, H, W))
    
    loss_cls = SegDiffusionLoss(
        output_channels=num_classes,
        feature_channels=C_feat,
        mode='classification',
    )
    
    # Training
    loss = loss_cls(backbone_features, target_cls)
    print(f"Training loss: {loss.item():.4f}")
    
    # Deterministic inference
    logits = loss_cls.deterministic_predict(backbone_features)
    seg_map = logits.argmax(dim=1)
    print(f"Logits shape: {logits.shape}")    # (B, 21, H, W)
    print(f"Seg map shape: {seg_map.shape}")  # (B, H, W)
    
    # ========================
    # Parameter count
    # ========================
    print()
    print("=" * 50)
    print("Parameter Count")
    print("=" * 50)
    total_params = sum(p.numel() for p in loss_reg.parameters())
    denoiser_params = sum(p.numel() for p in loss_reg.denoiser.parameters())
    print(f"Total:    {total_params:,} (~{total_params / 1e6:.2f}M)")
    print(f"Denoiser: {denoiser_params:,} (~{denoiser_params / 1e6:.2f}M)")