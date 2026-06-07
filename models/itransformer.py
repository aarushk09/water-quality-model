"""iTransformer: inverted attention across variables (ICLR 2024)."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.revin import RevIN


class iTransformerBlock(nn.Module):
  def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
    super().__init__()
    self.norm1 = nn.LayerNorm(d_model)
    self.attn = nn.MultiheadAttention(
      d_model, n_heads, dropout=dropout, batch_first=True
    )
    self.norm2 = nn.LayerNorm(d_model)
    self.ffn = nn.Sequential(
      nn.Linear(d_model, d_ff),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(d_ff, d_model),
      nn.Dropout(dropout),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, F, d_model]
    h = self.norm1(x)
    attn_out, _ = self.attn(h, h, h, need_weights=False)
    x = x + attn_out
    x = x + self.ffn(self.norm2(x))
    return x


class iTransformerEncoder(nn.Module):
  """
  Input x: [B, T, F] → output [B, pred_len, n_targets]
  Each variable is a token; attention is across F not T.
  """

  def __init__(
    self,
    seq_len: int,
    pred_len: int,
    n_vars: int,
    n_targets: int = 2,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    dropout: float = 0.1,
    use_revin: bool = True,
  ):
    super().__init__()
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.n_vars = n_vars
    self.n_targets = n_targets
    self.use_revin = use_revin
    if use_revin:
      self.revin = RevIN(n_vars)
    self.var_embed = nn.Linear(seq_len, d_model)
    d_ff = d_model * 4
    self.blocks = nn.ModuleList(
      [iTransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
    )
    self.head = nn.Linear(d_model, pred_len)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, F]
    if self.use_revin:
      x = self.revin.norm(x)
    x = x.transpose(1, 2)  # [B, F, T]
    h = self.var_embed(x)  # [B, F, d_model]
    for blk in self.blocks:
      h = blk(h)
    # Project target variables (first n_targets) to horizon
    h_tgt = h[:, : self.n_targets, :]
    out = self.head(h_tgt)  # [B, n_targets, pred_len]
    out = out.transpose(1, 2)  # [B, pred_len, n_targets]
    if self.use_revin:
      b, pl, _ = out.shape
      full = torch.zeros(b, pl, self.n_vars, device=out.device, dtype=out.dtype)
      full[..., : self.n_targets] = out
      full = self.revin.denorm(full)
      out = full[..., : self.n_targets]
    return out
