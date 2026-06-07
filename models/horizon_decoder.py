"""Cross-attention horizon decoder with coupled temperature / DO heads."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class HorizonCrossAttentionDecoder(nn.Module):
    """
    Learned horizon queries attend to patch tokens; separate temp and DO heads.
    Optional GRU refinement over horizon steps.
    """

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_ar_refinement: bool = True,
        residual_output: bool = True,
        n_fut_cov: int = 0,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.residual_output = residual_output
        self.d_model = d_model
        self.n_fut_cov = n_fut_cov
        if n_fut_cov > 0:
            self.fut_proj = nn.Sequential(
                nn.Linear(n_fut_cov, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.fut_proj = None
        self.horizon_queries = nn.Parameter(torch.randn(pred_len, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)

        self.head_temp = nn.Linear(d_model, 1)
        self.head_do = nn.Linear(d_model + 1, 1)

        self.use_ar_refinement = use_ar_refinement
        if use_ar_refinement:
            ar_in_dim = d_model + 2 + n_fut_cov
            self.ar_gru = nn.GRU(ar_in_dim, d_model, batch_first=True)
            self.ar_proj = nn.Linear(d_model, d_model)

        self._init_heads()

    def _init_heads(self) -> None:
        gain = 0.02 if self.residual_output else 0.1
        nn.init.xavier_uniform_(self.head_temp.weight, gain=gain)
        nn.init.zeros_(self.head_temp.bias)
        nn.init.xavier_uniform_(self.head_do.weight, gain=gain)
        nn.init.zeros_(self.head_do.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        return_attention: bool = False,
        fut_cov: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        tokens: [B, P, d_model]
        fut_cov: optional [B, pred_len, n_fut_cov] known future meteorology (scaled)
        returns: [B, pred_len, 2] in scaled target space
        """
        b, _, _ = tokens.shape
        kv = self.norm_kv(tokens)
        q = self.norm_q(self.horizon_queries.unsqueeze(0).expand(b, -1, -1))
        if fut_cov is not None and self.fut_proj is not None:
            q = q + self.fut_proj(fut_cov)
        h, attn_weights = self.cross_attn(q, kv, kv, need_weights=return_attention)
        h = self.norm_ff(h + self.ff(h))

        if self.use_ar_refinement:
            temp0 = self.head_temp(h).squeeze(-1)
            do0 = self.head_do(torch.cat([h, temp0.unsqueeze(-1)], dim=-1)).squeeze(-1)
            ar_parts = [h, temp0.unsqueeze(-1), do0.unsqueeze(-1)]
            if fut_cov is not None and self.n_fut_cov > 0:
                ar_parts.append(fut_cov)
            elif self.n_fut_cov > 0:
                ar_parts.append(
                    torch.zeros(b, self.pred_len, self.n_fut_cov, device=h.device)
                )
            ar_in = torch.cat(ar_parts, dim=-1)
            h_ref, _ = self.ar_gru(ar_in)
            h = h + self.ar_proj(h_ref)

        temp = self.head_temp(h).squeeze(-1)
        do_in = torch.cat([h, temp.unsqueeze(-1)], dim=-1)
        do = self.head_do(do_in).squeeze(-1)
        y = torch.stack([temp, do], dim=-1)
        attn = attn_weights if return_attention else None
        return y, attn
