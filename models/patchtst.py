"""PatchTST encoder + horizon cross-attention decoder."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.horizon_decoder import HorizonCrossAttentionDecoder
from models.revin import RevIN


class PatchEmbedding(nn.Module):
    def __init__(self, n_vars: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len * n_vars, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, t, c = x.shape
        patches = []
        for start in range(0, t - self.patch_len + 1, self.stride):
            patch = x[:, start : start + self.patch_len, :].reshape(b, -1)
            patches.append(patch)
        if not patches:
            pad = self.patch_len - t
            x_pad = torch.nn.functional.pad(x, (0, 0, pad, 0))
            patches = [x_pad.reshape(b, -1)]
        p = torch.stack(patches, dim=1)
        return self.norm(self.proj(p)), p.shape[1]


class PatchTSTEncoder(nn.Module):
    def __init__(
        self,
        n_vars: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        pred_len: int = 96,
        n_targets: int = 2,
        use_revin: bool = True,
        use_horizon_decoder: bool = True,
        decoder_heads: Optional[int] = None,
        use_ar_refinement: bool = True,
        use_local_conv: bool = True,
        residual_output: bool = True,
        n_fut_cov: int = 0,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.n_targets = n_targets
        self.n_vars = n_vars
        self.use_revin = use_revin
        self.use_horizon_decoder = use_horizon_decoder
        self.d_model = d_model
        self.use_local_conv = use_local_conv

        if use_revin:
            self.revin = RevIN(n_vars)

        if use_local_conv:
            self.local_conv = nn.Sequential(
                nn.Conv1d(n_vars, n_vars, kernel_size=5, padding=2, groups=n_vars),
                nn.GELU(),
                nn.Conv1d(n_vars, n_vars, kernel_size=3, padding=1, groups=n_vars),
            )

        self.patch_embed = PatchEmbedding(n_vars, patch_len, stride, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        dec_heads = decoder_heads or n_heads
        if use_horizon_decoder:
            self.horizon_decoder = HorizonCrossAttentionDecoder(
                d_model=d_model,
                pred_len=pred_len,
                n_heads=dec_heads,
                dropout=dropout,
                use_ar_refinement=use_ar_refinement,
                residual_output=residual_output,
                n_fut_cov=n_fut_cov,
            )
        else:
            self.head_norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, pred_len * n_targets)
            self._init_legacy_head()

        self._last_attn: Optional[torch.Tensor] = None

    def _init_legacy_head(self) -> None:
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        fut_cov: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [B, T, C] -> y: [B, pred_len, n_targets]
        """
        if self.use_revin:
            x = self.revin.norm(x)

        if self.use_local_conv:
            x = self.local_conv(x.transpose(1, 2)).transpose(1, 2)

        emb, _ = self.patch_embed(x)
        encoded = self.encoder(emb)

        if self.use_horizon_decoder:
            y, attn = self.horizon_decoder(
                encoded, return_attention=return_attention, fut_cov=fut_cov
            )
            self._last_attn = attn
            return y, attn

        if return_attention:
            norm = encoded / (encoded.norm(dim=-1, keepdim=True) + 1e-8)
            self._last_attn = torch.bmm(norm, norm.transpose(1, 2))

        pooled = self.head_norm(encoded.mean(dim=1))
        out = self.head(pooled).view(-1, self.pred_len, self.n_targets)
        attn = self._last_attn if return_attention else None
        return out, attn

    @property
    def last_attention(self) -> Optional[torch.Tensor]:
        return self._last_attn
