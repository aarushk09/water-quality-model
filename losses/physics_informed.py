"""Composite physics-informed loss in physical units with horizon weighting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from physics.do_saturation import do_saturation, reaeration_residual


@dataclass
class PhysicsLossConfig:
    huber_delta: float = 1.0
    horizon_tau: float = 48.0
    lambda_supersat: float = 0.15
    lambda_nonneg: float = 0.05
    lambda_solubility: float = 0.02
    lambda_reaeration: float = 0.03
    lambda_derivative: float = 0.05
    lambda_amplitude: float = 0.10       # NEW: diurnal amplitude matching
    derivative_horizon: int = 24
    physics_warmup_epochs: int = 0
    physics_ramp_epochs: int = 20
    physics_max_scale: float = 1.0
    physics_horizon_steps: int = 24
    physics_mode: str = "full"
    short_horizon_steps: int = 12
    short_horizon_weight: float = 1.5
    short_horizon_epoch_fraction: float = 0.35
    short_horizon_tail_weight: float = 0.5
    k_reaer: float = 0.5
    dt_hours: float = 0.25
    # Per-variable loss weights: upweight DO which is harder to predict
    temp_loss_weight: float = 1.0
    do_loss_weight: float = 2.0


@dataclass
class PhysicsInformedLoss:
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    cfg: PhysicsLossConfig = field(default_factory=PhysicsLossConfig)
    current_epoch: int = 1
    max_epochs: int = 300

    def _physics_scale(self) -> float:
        """Legacy curriculum scale (kept for API compat; physics always active)."""
        w = self.cfg.physics_warmup_epochs
        r = self.cfg.physics_ramp_epochs
        e = self.current_epoch
        cap = self.cfg.physics_max_scale
        if self.cfg.physics_mode == "off":
            return 0.0
        if e <= w:
            return 0.0
        if e >= w + r:
            return cap
        progress = (e - w) / max(r, 1)
        return cap * 0.5 * (1.0 - math.cos(math.pi * progress))

    def _reaeration_weight(self) -> float:
        """Ramp reaeration penalty from lambda_reaeration to 3× over ramp_epochs."""
        base = self.cfg.lambda_reaeration
        ramp = self.cfg.physics_ramp_epochs
        e = self.current_epoch
        if ramp <= 0:
            return base * 3.0
        progress = min(1.0, max(0.0, (e - 1) / ramp))
        return base + (base * 2.0) * progress

    def _horizon_weights(self, h: int, device: torch.device) -> torch.Tensor:
        k = torch.arange(h, device=device, dtype=torch.float32)
        return torch.exp(-k / self.cfg.horizon_tau)

    def _to_physical(self, y: torch.Tensor) -> torch.Tensor:
        mean = self.target_mean.to(y.device).view(1, 1, 1, -1)
        scale = self.target_scale.to(y.device).view(1, 1, 1, -1)
        return y * scale + mean

    def _masked_weighted_mae(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        yp = self._to_physical(y_pred)
        yt = self._to_physical(y_true)
        loss_elem = F.l1_loss(yp, yt, reduction="none")
        w = weights.view(1, 1, -1, 1)
        loss_elem = loss_elem * w
        if mask is not None:
            m = mask.view(-1, y_pred.shape[1], 1, 1)
            denom = (m.sum() * y_pred.shape[-1] * y_pred.shape[-2] * weights.sum())
            denom = denom.clamp(min=1.0)
            return (loss_elem * m).sum() / denom
        return loss_elem.mean()

    def forecast_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, h, _ = y_pred.shape
        hw = self._horizon_weights(h, y_pred.device)

        # Per-variable weighted MAE (upweight DO)
        var_w = torch.tensor(
            [self.cfg.temp_loss_weight, self.cfg.do_loss_weight],
            device=y_pred.device, dtype=torch.float32
        ).view(1, 1, 1, 2)
        yp_phys = self._to_physical(y_pred)
        yt_phys = self._to_physical(y_true)
        loss_elem = F.l1_loss(yp_phys, yt_phys, reduction="none") * var_w
        hw_view = hw.view(1, 1, -1, 1)
        loss_elem = loss_elem * hw_view
        if mask is not None:
            m = mask.view(-1, n, 1, 1)
            denom = (m.sum() * 2 * h * hw.sum()).clamp(min=1.0)
            loss = (loss_elem * m).sum() / denom
        else:
            loss = loss_elem.mean()

        # Short-horizon emphasis
        sh = min(self.cfg.short_horizon_steps, h)
        sh_elem = F.l1_loss(yp_phys[:, :, :sh], yt_phys[:, :, :sh], reduction="none") * var_w
        sh_elem = sh_elem * hw_view[:, :, :sh]
        if mask is not None:
            m = mask.view(-1, n, 1, 1)
            sh_denom = (m.sum() * 2 * sh * hw[:sh].sum()).clamp(min=1.0)
            short = (sh_elem * m).sum() / sh_denom
        else:
            short = sh_elem.mean()

        frac = self.cfg.short_horizon_epoch_fraction
        if self.current_epoch <= int(self.max_epochs * frac):
            w = self.cfg.short_horizon_weight
        else:
            w = self.cfg.short_horizon_tail_weight
        loss = loss + w * short

        # Amplitude penalty: penalize under-prediction of diurnal range
        # This directly targets the peak-smoothing problem
        if self.cfg.lambda_amplitude > 0:
            pred_range = yp_phys.max(dim=2).values - yp_phys.min(dim=2).values  # [B,N,2]
            true_range = yt_phys.max(dim=2).values - yt_phys.min(dim=2).values  # [B,N,2]
            # Penalize under-prediction of amplitude (asymmetric: smoothing is worse)
            amp_err = F.relu(true_range - pred_range)  # only penalize under-prediction
            amplitude_loss = (amp_err * var_w.squeeze(2).squeeze(2)).mean()
            loss = loss + self.cfg.lambda_amplitude * amplitude_loss

        return loss

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        y_pred, y_true: [B, N, H, 2] scaled (prefer physics-projected predictions).
        """
        device = y_pred.device
        forecast_loss = self.forecast_loss(y_pred, y_true, mask)
        h = y_pred.shape[2]

        y_phys = self._to_physical(y_pred)
        t_pred = y_phys[..., 0]
        do_pred = y_phys[..., 1]
        do_sat = do_saturation(t_pred).clamp(min=0.5)

        rel_super = F.relu(do_pred - do_sat) / do_sat
        supersat = rel_super.pow(2).mean()
        nonneg = F.relu(-do_pred).pow(2).mean()
        solubility = ((do_pred - do_sat).abs() / do_sat).mean()

        ph_steps = min(self.cfg.physics_horizon_steps, h)
        reaer = torch.tensor(0.0, device=device)
        deriv = torch.tensor(0.0, device=device)
        if ph_steps > 1:
            do_s = do_pred[:, :, :ph_steps]
            t_s = t_pred[:, :, :ph_steps]
            reaer = reaeration_residual(
                do_s, t_s, dt_hours=self.cfg.dt_hours, k_reaer=self.cfg.k_reaer
            )
            sh_deriv = min(self.cfg.derivative_horizon, ph_steps - 1)
            if sh_deriv > 0:
                dt = self.cfg.dt_hours
                t_true = self._to_physical(y_true)[..., 0]
                dt_pred = (t_s[:, :, 1 : sh_deriv + 1] - t_s[:, :, :sh_deriv]) / dt
                dt_true = (t_true[:, :, 1 : sh_deriv + 1] - t_true[:, :, :sh_deriv]) / dt
                deriv = (dt_pred - dt_true).abs().mean()

        c = self.cfg
        reaer_w = self._reaeration_weight()
        physics = (
            c.lambda_supersat * supersat
            + c.lambda_nonneg * nonneg
            + c.lambda_solubility * solubility
            + reaer_w * reaer
            + c.lambda_derivative * deriv
        )

        loss = forecast_loss + physics
        return {
            "loss": loss,
            "mse": forecast_loss,
            "physics_violation": physics,
            "physics_supersat": supersat,
            "physics_derivative": deriv,
            "short_horizon_loss": torch.tensor(0.0, device=device),
            "physics_nonneg": nonneg,
            "physics_reaeration": reaer,
        }
