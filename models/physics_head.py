"""Differentiable physics projection for temperature and dissolved oxygen."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from physics.do_saturation import do_saturation


class PhysicsProjectionHead(nn.Module):
    """
    Apply DO saturation ceiling and non-negativity in physical units, then
    map back to scaled target space for loss compatibility.
    """

    def __init__(
        self,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
        do_floor: float = 0.0,
        do_eps: float = 0.05,
    ):
        super().__init__()
        self.register_buffer("target_mean", target_mean.view(1, 1, 1, 2))
        self.register_buffer("target_scale", target_scale.view(1, 1, 1, 2))
        self.do_floor = do_floor
        self.do_eps = do_eps

    def to_physical(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return y_scaled * self.target_scale + self.target_mean

    def to_scaled(self, y_phys: torch.Tensor) -> torch.Tensor:
        return (y_phys - self.target_mean) / self.target_scale.clamp(min=1e-6)

    def forward(
        self, y_scaled: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        y_scaled: [B, N, H, 2]
        Returns (y_scaled_projected, y_physical).
        """
        y_phys = self.to_physical(y_scaled)
        t = y_phys[..., 0]
        do_raw = y_phys[..., 1]
        do_sat = do_saturation(t).clamp(min=self.do_eps)
        do_proj = torch.clamp(do_raw, min=self.do_floor)
        do_proj = torch.minimum(do_proj, do_sat)
        y_phys_proj = torch.stack([t, do_proj], dim=-1)
        y_scaled_proj = self.to_scaled(y_phys_proj)
        return y_scaled_proj, y_phys_proj
