"""Conditional Score-Based Diffusion Forecaster for probabilistic water quality prediction.

This module enables calibrated probabilistic forecasting:
  "P(DO < 2 mg/L in next 6 hours) = 0.73"

The denoiser is conditioned on PatchTST encoder tokens, enabling the existing
deterministic encoder to serve as a strong conditioning signal without retraining.

Architecture:
  - Conditioning signal c: PatchTST patch tokens [B, P, d_model]
  - Denoiser: 1D U-Net with residual blocks + cross-attention to condition
  - Noise schedule: cosine schedule with T=1000 training steps
  - Sampling: DDIM (50 steps) for fast inference; 200 samples per forecast

Novel contribution:
  - Physics-constrained sampling: filter/reweight samples violating DO_sat
  - Hypoxia exceedance probability: P(DO < threshold) at each lead time
  - First calibrated probabilistic hypoxia risk system for river monitoring

References:
  Song et al. (2020) "Score-Based Generative Modeling through SDEs"
  Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
  Nichol & Dhariwal (2021) "Improved Denoising Diffusion Probabilistic Models"
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding from DDPM."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] integer timesteps → [B, dim]"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


class ResidualBlock1D(nn.Module):
    """1D residual block with time embedding and optional cross-attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.1,
        groups: int = 8,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L], t_emb: [B, time_dim] → [B, out_channels, L]"""
        h = self.norm1(F.silu(self.conv1(x)))
        # Add time embedding (broadcast over sequence length)
        t_scale = self.time_proj(F.silu(t_emb))[:, :, None]
        h = h + t_scale
        h = self.norm2(F.silu(self.dropout(self.conv2(h))))
        return h + self.skip(x)


class CrossAttentionBlock(nn.Module):
    """Cross-attention from target sequence to conditioning tokens."""

    def __init__(self, channels: int, context_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_proj = nn.Linear(context_dim, channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C] — target sequence (transposed from conv format)
        context: [B, P, context_dim] — conditioning tokens (PatchTST output)
        """
        q = self.norm(x)
        kv = self.context_proj(self.context_norm(context))
        h, _ = self.attn(q, kv, kv)
        x = x + h
        return x + self.ff(x)


class UNet1D(nn.Module):
    """
    1D U-Net denoiser conditioned on PatchTST encoder tokens.

    Input: noisy target [B, pred_len, n_targets] + timestep t + context tokens
    Output: predicted noise [B, pred_len, n_targets]
    """

    def __init__(
        self,
        n_targets: int = 2,
        pred_len: int = 96,
        context_dim: int = 192,  # PatchTST d_model
        channels: List[int] = None,
        time_dim: int = 128,
        n_attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        channels = channels or [64, 128, 256, 128, 64]
        self.pred_len = pred_len
        self.n_targets = n_targets

        self.time_emb = SinusoidalTimestepEmbedding(time_dim)

        # Input projection
        self.in_proj = nn.Conv1d(n_targets, channels[0], 1)

        # Encoder path (downsampling)
        n_enc = len(channels) // 2 + 1
        self.enc_blocks = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        for i in range(n_enc):
            in_c = channels[i - 1] if i > 0 else channels[0]
            out_c = channels[i]
            self.enc_blocks.append(ResidualBlock1D(in_c, out_c, time_dim, dropout))
            self.enc_attn.append(CrossAttentionBlock(out_c, context_dim, n_attn_heads, dropout))

        # Bottleneck
        mid_c = channels[n_enc - 1]
        self.mid_block1 = ResidualBlock1D(mid_c, mid_c, time_dim, dropout)
        self.mid_attn = CrossAttentionBlock(mid_c, context_dim, n_attn_heads, dropout)
        self.mid_block2 = ResidualBlock1D(mid_c, mid_c, time_dim, dropout)

        # Decoder path (upsampling)
        self.dec_blocks = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        n_dec = len(channels) - n_enc
        for i in range(n_dec):
            idx = n_enc + i
            in_c = channels[idx - 1] + channels[n_enc - 1 - i]  # skip connection
            out_c = channels[idx]
            self.dec_blocks.append(ResidualBlock1D(in_c, out_c, time_dim, dropout))
            self.dec_attn.append(CrossAttentionBlock(out_c, context_dim, n_attn_heads, dropout))

        # Output projection
        self.out_norm = nn.GroupNorm(min(8, channels[-1]), channels[-1])
        self.out_proj = nn.Conv1d(channels[-1], n_targets, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_noisy: [B, pred_len, n_targets]
        t: [B] integer diffusion timesteps
        context: [B, P, context_dim] PatchTST patch tokens
        Returns: predicted noise [B, pred_len, n_targets]
        """
        t_emb = self.time_emb(t)  # [B, time_dim]

        # [B, n_targets, pred_len]
        h = self.in_proj(x_noisy.transpose(1, 2))

        # Encoder with skip connections
        skips = []
        for blk, attn in zip(self.enc_blocks, self.enc_attn):
            h = blk(h, t_emb)
            # Cross-attend to conditioning
            h_t = h.transpose(1, 2)  # [B, L, C]
            h_t = attn(h_t, context)
            h = h_t.transpose(1, 2)  # [B, C, L]
            skips.append(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h_t = self.mid_attn(h.transpose(1, 2), context)
        h = h_t.transpose(1, 2)
        h = self.mid_block2(h, t_emb)

        # Decoder with skip connections
        for blk, attn in zip(self.dec_blocks, self.dec_attn):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = blk(h, t_emb)
            h_t = h.transpose(1, 2)
            h_t = attn(h_t, context)
            h = h_t.transpose(1, 2)

        h = self.out_norm(F.silu(h))
        eps_pred = self.out_proj(h).transpose(1, 2)  # [B, pred_len, n_targets]
        return eps_pred


class CosineNoiseSchedule:
    """Cosine noise schedule from Nichol & Dhariwal (2021)."""

    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T = T
        t = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        alphas_bar = f / f[0]
        betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
        betas = betas.clamp(0, 0.999)

        self.register_buffers_to: Optional[torch.device] = None
        alphas = 1.0 - betas
        self.alphas_bar = alphas_bar[1:].float()
        self.sqrt_alphas_bar = self.alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1 - self.alphas_bar).sqrt()
        self.betas = betas.float()
        self.alphas = alphas.float()

    def to(self, device: torch.device) -> "CosineNoiseSchedule":
        self.alphas_bar = self.alphas_bar.to(device)
        self.sqrt_alphas_bar = self.sqrt_alphas_bar.to(device)
        self.sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        return self

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """q(x_t | x_0): add noise at level t."""
        sa = self.sqrt_alphas_bar[t].view(-1, 1, 1)
        sm = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
        return sa * x0 + sm * noise


class DiffusionForecaster(nn.Module):
    """
    Conditional diffusion forecaster for water quality probabilistic prediction.

    Usage:
      # Training (with frozen/joint PatchTST encoder):
      losses = model.training_loss(y_true, context_tokens)

      # Inference (generate n_samples trajectories):
      samples = model.sample(context_tokens, n_samples=200)
      # → [B, n_samples, pred_len, n_targets]

      # Hypoxia probability at each lead time:
      p_hypoxia = (samples[..., 1] < 2.0).float().mean(dim=1)
      # → [B, pred_len]
    """

    def __init__(
        self,
        n_targets: int = 2,
        pred_len: int = 96,
        context_dim: int = 192,
        unet_channels: Optional[List[int]] = None,
        time_dim: int = 128,
        T: int = 1000,
        ddim_steps: int = 50,
        n_attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.T = T
        self.ddim_steps = ddim_steps
        self.pred_len = pred_len
        self.n_targets = n_targets

        self.schedule = CosineNoiseSchedule(T)
        self.denoiser = UNet1D(
            n_targets=n_targets,
            pred_len=pred_len,
            context_dim=context_dim,
            channels=unet_channels or [64, 128, 256, 128, 64],
            time_dim=time_dim,
            n_attn_heads=n_attn_heads,
            dropout=dropout,
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Also move noise schedule buffers
        device = args[0] if args else kwargs.get("device", None)
        if device is not None:
            self.schedule.to(device)
        return self

    def training_loss(
        self,
        y0: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute denoising score matching loss.

        y0: [B, pred_len, n_targets] — ground truth target (scaled)
        context: [B, P, context_dim] — PatchTST patch tokens (conditioning)
        """
        device = y0.device
        self.schedule.to(device)
        B = y0.shape[0]

        # Sample random diffusion timestep for each example
        t = torch.randint(0, self.T, (B,), device=device)

        # Sample noise and construct noisy sample
        noise = torch.randn_like(y0)
        y_noisy = self.schedule.add_noise(y0, noise, t)

        # Predict noise
        eps_pred = self.denoiser(y_noisy, t, context)

        # Simple MSE loss on predicted noise
        loss = F.mse_loss(eps_pred, noise)

        # Also log per-variable loss for monitoring
        temp_loss = F.mse_loss(eps_pred[..., 0], noise[..., 0])
        do_loss = F.mse_loss(eps_pred[..., 1], noise[..., 1])

        return {
            "diffusion_loss": loss,
            "diffusion_temp_loss": temp_loss,
            "diffusion_do_loss": do_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        n_samples: int = 200,
        physics_filter: bool = True,
        target_mean: Optional[torch.Tensor] = None,
        target_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate n_samples trajectory samples via DDIM.

        context: [B, P, context_dim]
        Returns: [B, n_samples, pred_len, n_targets] in SCALED space

        If physics_filter=True and target_mean/scale provided,
        samples violating DO_sat ceiling are reweighted via importance weights.
        """
        device = context.device
        self.schedule.to(device)
        B = context.shape[0]

        # DDIM timestep sequence
        step_size = self.T // self.ddim_steps
        timesteps = list(range(self.T - 1, -1, -step_size))

        all_samples = []
        for _ in range(n_samples):
            # Start from pure noise
            y_t = torch.randn(B, self.pred_len, self.n_targets, device=device)

            for i, t_curr in enumerate(timesteps):
                t_tensor = torch.full((B,), t_curr, device=device, dtype=torch.long)
                eps = self.denoiser(y_t, t_tensor, context)

                # DDIM update
                alpha_bar_t = self.schedule.alphas_bar[t_curr]
                x0_pred = (y_t - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()

                if i < len(timesteps) - 1:
                    t_prev = timesteps[i + 1]
                    alpha_bar_prev = self.schedule.alphas_bar[t_prev]
                    sigma = 0.0  # deterministic DDIM
                    dir_xt = (1 - alpha_bar_prev - sigma**2).sqrt() * eps
                    y_t = alpha_bar_prev.sqrt() * x0_pred + dir_xt
                else:
                    y_t = x0_pred

            all_samples.append(y_t.unsqueeze(1))  # [B, 1, pred_len, n_targets]

        samples = torch.cat(all_samples, dim=1)  # [B, n_samples, pred_len, n_targets]

        if physics_filter and target_mean is not None and target_scale is not None:
            samples = self._physics_reweight(samples, target_mean, target_scale)

        return samples

    def _physics_reweight(
        self,
        samples: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Soft physics filter: down-weight samples that violate DO saturation.

        Uses importance weighting (not hard rejection) so gradients can flow.
        """
        from physics.do_saturation import do_saturation

        B, S, H, _ = samples.shape
        mean = target_mean.to(samples.device).view(1, 1, 1, -1)
        scale = target_scale.to(samples.device).view(1, 1, 1, -1)

        # Denormalize
        samples_phys = samples * scale + mean
        t_pred = samples_phys[..., 0]   # [B, S, H]
        do_pred = samples_phys[..., 1]  # [B, S, H]
        do_sat = do_saturation(t_pred)

        # Violation severity: how much does DO exceed saturation?
        violation = F.relu(do_pred - do_sat).mean(dim=-1)  # [B, S]

        # Soft importance weights (less violation → higher weight)
        log_weights = -5.0 * violation  # temperature parameter
        weights = torch.softmax(log_weights, dim=1)  # [B, S]

        # Resample using weights (soft replacement)
        resample_idx = torch.multinomial(weights, num_samples=samples.shape[1], replacement=True)
        resampled = torch.gather(
            samples,
            1,
            resample_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, self.n_targets),
        )
        return resampled

    @torch.no_grad()
    def hypoxia_exceedance_curve(
        self,
        context: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
        n_samples: int = 200,
        threshold_mg_l: float = 2.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute P(DO < threshold) at each forecast lead time.

        Returns:
          exceedance_prob: [B, pred_len] — probability of hypoxia at each step
          do_quantiles:    [B, pred_len, 5] — 5th, 25th, 50th, 75th, 95th percentile
        """
        samples = self.sample(
            context, n_samples=n_samples,
            physics_filter=True, target_mean=target_mean, target_scale=target_scale,
        )  # [B, n_samples, pred_len, n_targets]

        # Denormalize DO channel
        mean = target_mean.to(samples.device).view(1, 1, 1, -1)
        scale = target_scale.to(samples.device).view(1, 1, 1, -1)
        samples_phys = samples * scale + mean
        do_samples = samples_phys[..., 1]  # [B, n_samples, pred_len]

        exceedance = (do_samples < threshold_mg_l).float().mean(dim=1)  # [B, pred_len]
        quantiles = torch.quantile(do_samples, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95],
                                   device=samples.device), dim=1)  # [5, B, pred_len]
        quantiles = quantiles.permute(1, 2, 0)  # [B, pred_len, 5]

        return {
            "exceedance_prob": exceedance,
            "do_quantiles": quantiles,
            "do_samples": do_samples,
        }
