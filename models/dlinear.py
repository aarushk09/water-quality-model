"""DLinear baseline: trend/seasonal decomposition + per-variable linear forecast."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvg(nn.Module):
  def __init__(self, kernel_size: int):
    super().__init__()
    self.kernel_size = kernel_size

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, C]
    if self.kernel_size <= 1:
      return x
    pad = (self.kernel_size - 1) // 2
    x_t = x.transpose(1, 2)
    x_pad = F.pad(x_t, (pad, pad), mode="replicate")
    trend = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1)
    return trend.transpose(1, 2)


class DLinear(nn.Module):
  """
  Decompose input into trend + seasonal, forecast each with Linear(seq_len, pred_len).
  Input: [B, T, F]  Output: [B, pred_len, 2] (temp, DO)
  """

  def __init__(self, seq_len: int, pred_len: int, n_vars: int = 2, kernel_size: int = 25):
    super().__init__()
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.n_vars = n_vars
    self.moving_avg = MovingAvg(kernel_size)
    self.linear_trend = nn.Linear(seq_len, pred_len)
    self.linear_seasonal = nn.Linear(seq_len, pred_len)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, F] — use first n_vars channels (temp, DO)
    x_tgt = x[..., : self.n_vars]
    trend = self.moving_avg(x_tgt)
    seasonal = x_tgt - trend
    # [B, C, T] for linear layers
    trend = trend.transpose(1, 2)
    seasonal = seasonal.transpose(1, 2)
    trend_out = self.linear_trend(trend)
    seasonal_out = self.linear_seasonal(seasonal)
    out = (trend_out + seasonal_out).transpose(1, 2)
    return out
