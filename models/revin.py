"""Reversible Instance Normalization (RevIN) for stable time-series forecasting."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Per-sample normalization over the time dimension.

    See: Kim et al., Reversible Instance Normalization for Accurate Time-Series
    Forecasting against Distribution Shift, ICLR 2022.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        self._mean: Optional[torch.Tensor] = None
        self._stdev: Optional[torch.Tensor] = None

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._stdev = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        x = (x - self._mean) / self._stdev
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._stdev is None:
            return x
        if self.affine:
            x = (x - self.beta) / (self.gamma + self.eps)
        return x * self._stdev + self._mean
