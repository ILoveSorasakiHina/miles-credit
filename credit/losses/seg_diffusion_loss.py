"""
Segmentation Diffusion Loss (SegDiffLoss)
=========================================
Inspired by MMPD (Multi-Mode Patch Diffusion) loss for time series forecasting,
adapted for 2D dense prediction tasks.

Supports two modes:
- 'regression':      target is continuous (B, C, H, W), e.g. weather prediction
- 'classification':  target is integer labels (B, H, W), e.g. semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 1. Diffusion Schedule
# =============================================================================

class DiffusionSchedule:
    """Linear noise schedule for diffusion."""
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
        sqrt_ab = self.sqrt_alpha_bar[k].view(-1, 1, 1, 1)
        sqrt_1_ab = self.sqrt_one_minus_alpha_bar[k].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_1_ab * noise

    def find_anchor_step(self):
        return (self.alpha_bar - 0.5).abs().argmin().item()


# =============================================================================
# 2. Timestep Embedding
# =============================================================================

class TimestepEmbedding(nn.Module):
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
# 3. AdaLN
# =============================================================================

class AdaLN(nn.Module):
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, channels * 3),
        )
        # AdaLN 的 zero-init 是合理的：(1+gamma)*(x)+beta，gamma=beta=0 時退化為恆等
        # gate 從 0 開始也 OK，因為下面 forward 用 (1+gate) 確保訊號流過
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
    
    def forward(self, x, condition):
        gamma, beta, gate = self.proj(condition).chunk(3, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        x = self.norm(x)
        return (1 + gate) * ((1 + gamma) * x + beta)


# =============================================================================
# 4. ConvNeXt-style Denoiser Block
# =============================================================================

class ConvNeXtDenoiserBlock(nn.Module):
    def __init__(self, channels, condition_dim, kernel_size=7, expansion=4):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels)
        self.adaln = AdaLN(channels, condition_dim)
        hidden = channels * expansion
        self.pw1 = nn.Conv2d(channels, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, channels, 1)
    
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
        
        # ===================================================================== #
        # 修改：拿掉 output_proj 的 zero-init，改用小幅度 normal init
        # 原因：原本 zero-init 讓 predicted_noise 一開始全 0，
        #       loss = ||noise - 0||² ≈ 1 (常數)，梯度極小，訓練半天 output_proj.norm
        #       才從 0 → 0.038，denoiser 完全沒學起來。
        # 改用 std=0.02 normal init（DiT 等 diffusion 工作的標準做法），
        # 初始 predicted_noise 量級小但非零，梯度從第一步就有意義。
        # ===================================================================== #
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, noisy_input, backbone_features, k):
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
    MMPD-style Latent Condition Diffusion Loss for dense prediction.
    
    Condition: backbone 中間層 feat_hidden (256ch latent representation)
    Target: y 本身（物理量 ground truth），不是殘差
    
    訓練時: 
      Loss_diffusion = λ * L_diff + (1-λ) * L_det  (MMPD Eq.8)
      註：L_backbone (MSE) 由 trainer 另外計算，不在這裡加
    
    推論時: y_physical = denoiser.deterministic_predict(feat_hidden)
    """
    def __init__(
        self,
        output_channels,
        feature_channels,
        mode='regression',
        hidden_dim=128,
        num_blocks=4,
        kernel_size=7,
        expansion=4,
        diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        lambda_weight=0.99,
        **kwargs,
    ):
        super().__init__()
        assert mode == 'regression', "Only regression mode supported"
        
        self.output_channels = output_channels
        self.mode = mode
        self.lambda_weight = lambda_weight
        self.diffusion_steps = diffusion_steps
        
        self.schedule = DiffusionSchedule(diffusion_steps, beta_start, beta_end)
        self.denoiser = ConvNeXtDenoiser(
            output_channels=output_channels,
            feature_channels=feature_channels,
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
    
    def forward(self, target, feat_hidden):
        """
        Args:
            target: (B, C, H, W) or (B, C, 1, H, W) — ground truth y
            feat_hidden: (B, C_feat, H', W') — backbone 中間層特徵
        Returns:
            total_loss: scalar (僅 diffusion 部分)
        """
        device = feat_hidden.device
        self._ensure_schedule_device(device)
        if next(self.denoiser.parameters()).device != device:
            self.to(device)
        
        # 統一 target 成 4D
        y_true, _ = self._squeeze_if_5d(target)
        y_true = y_true.float()
        
        if feat_hidden.dim() == 5:
            B, C, _, H, W = feat_hidden.shape
            feat_hidden = feat_hidden.reshape(B, C, H, W)
        
        B = y_true.shape[0]
        
        # === Diffusion Loss ===
        y0 = y_true
        k = torch.randint(0, self.diffusion_steps, (B,), device=device)
        noise = torch.randn_like(y0)
        yk = self.schedule.add_noise(y0, noise, k)
        
        predicted_noise = self.denoiser(yk, feat_hidden, k)
        loss_diffusion = F.mse_loss(predicted_noise, noise)
        
        # === Deterministic Loss (MMPD Eq.8) ===
        k_star = self.anchor_step
        alpha_bar_star = self.schedule.alpha_bar[k_star]
        scale = torch.sqrt(alpha_bar_star / (1 - alpha_bar_star))
        
        k_star_tensor = torch.full((B,), k_star, device=device, dtype=torch.long)
        zero_input = torch.zeros_like(y0)
        predicted_noise_det = self.denoiser(zero_input, feat_hidden, k_star_tensor)
        target_det = scale * y0
        loss_deterministic = F.mse_loss(predicted_noise_det, -target_det)
        
        # === 診斷 print ===
        # 修改：拿掉 forward 裡的 grad 檢查（時機不對，永遠印 0）
        #       grad 檢查移到 trainer 的 backward 之後做
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
                    y_true_std = y_true.std().item()
                    feat_std = feat_hidden.std().item()
                    feat_max = feat_hidden.abs().max().item()
                    
                    inv_scale_ = torch.sqrt((1 - alpha_bar_star) / alpha_bar_star)
                    pred_y_from_denoiser = -inv_scale_ * predicted_noise_det
                    pred_y_std = pred_y_from_denoiser.std().item()
                    
                    y_mse = F.mse_loss(pred_y_from_denoiser, y_true).item()
                    y_zero_baseline = (y_true ** 2).mean().item()
                    
                    op_w_norm = self.denoiser.output_proj.weight.norm().item()
                    
                    print(
                        f"\n[DIAG step={self._diag_step}] "
                        f"y_true_std={y_true_std:.3f} feat_std={feat_std:.3f} feat_max={feat_max:.2f} | "
                        f"pred_y_std={pred_y_std:.3f} | "
                        f"y_MSE={y_mse:.4f} (zero_baseline={y_zero_baseline:.4f}) | "
                        f"L_diff={loss_diffusion.item():.4f} L_det={loss_deterministic.item():.4f} | "
                        f"output_proj.norm={op_w_norm:.4f}",
                        flush=True
                    )
        
        # === MMPD Eq.8 風格組合 ===
        total_loss = (
            self.lambda_weight * loss_diffusion
            + (1 - self.lambda_weight) * loss_deterministic
        )
        return total_loss
    
    @torch.no_grad()
    def deterministic_predict(self, feat_hidden):
        device = feat_hidden.device
        self._ensure_schedule_device(device)
        if next(self.denoiser.parameters()).device != device:
            self.to(device)
        
        is_5d = False
        if feat_hidden.dim() == 5:
            is_5d = True
            B, C, _, H, W = feat_hidden.shape
            feat_hidden = feat_hidden.reshape(B, C, H, W)
        
        B = feat_hidden.shape[0]
        H = feat_hidden.shape[2]
        W = feat_hidden.shape[3]
        
        k_star = self.anchor_step
        alpha_bar_star = self.schedule.alpha_bar[k_star]
        inv_scale = torch.sqrt((1 - alpha_bar_star) / alpha_bar_star)
        
        k_star_tensor = torch.full((B,), k_star, device=device, dtype=torch.long)
        zero_input = torch.zeros(
            B, self.output_channels, H, W, device=device
        )
        
        predicted_noise = self.denoiser(zero_input, feat_hidden, k_star_tensor)
        y_physical = -inv_scale * predicted_noise
        
        if is_5d:
            y_physical = y_physical.unsqueeze(2)
        
        return y_physical
    
    @torch.no_grad()
    def probabilistic_predict(self, feat_hidden, num_samples=10, infer_steps=20):
        device = feat_hidden.device
        self._ensure_schedule_device(device)
        
        is_5d = False
        if feat_hidden.dim() == 5:
            is_5d = True
            B, C, _, H, W = feat_hidden.shape
            feat_hidden = feat_hidden.reshape(B, C, H, W)
        
        B, _, H, W = feat_hidden.shape
        
        step_indices = torch.linspace(
            self.diffusion_steps - 1, 0, infer_steps, device=device
        ).long()
        
        all_samples = []
        for _ in range(num_samples):
            x = torch.randn(B, self.output_channels, H, W, device=device)
            for i in range(len(step_indices)):
                k = step_indices[i]
                k_tensor = k.unsqueeze(0).expand(B)
                predicted_noise = self.denoiser(x, feat_hidden, k_tensor)
                
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
            
            if is_5d:
                x = x.unsqueeze(2)
            all_samples.append(x)
        
        return torch.stack(all_samples)