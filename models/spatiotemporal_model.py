"""GAT spatial front-end + temporal backbone + physics projection."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from models.dlinear import DLinear
from models.gat_layer import SpatialGAT, apply_gat_over_time
from models.itransformer import iTransformerEncoder
from models.patchtst import PatchTSTEncoder
from models.physics_head import PhysicsProjectionHead
from models.tcn import TCNBackbone


class SpatioTemporalWaterModel(nn.Module):
    """
    Optional GAT -> temporal backbone -> physics projection.

    Input x: [B, N, T, F]
    Output y_hat: [B, N, H, 2] (scaled, physics-projected)
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        pred_len: int,
        edge_index: torch.Tensor,
        backbone: Literal["patchtst", "tcn", "dlinear", "itransformer"] = "patchtst",
        gat_hidden: int = 64,
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        tcn_channels: Optional[list] = None,
        tcn_kernel_size: int = 3,
        use_revin: bool = True,
        use_horizon_decoder: bool = True,
        use_physics_head: bool = True,
        target_mean: Optional[torch.Tensor] = None,
        target_scale: Optional[torch.Tensor] = None,
        use_ar_refinement: bool = True,
        use_residual_baseline: bool = True,
        use_local_conv: bool = True,
        n_fut_cov: int = 0,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.use_physics_head = use_physics_head
        self.use_residual_baseline = use_residual_baseline
        self.backbone_name = backbone
        self.skip_gat = backbone in ("dlinear", "itransformer")
        self.register_buffer("edge_index", edge_index.long())
        if edge_attr is not None:
            self.register_buffer("edge_attr", edge_attr.float())
        else:
            self.edge_attr = None

        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 1
        self.gat = None
        if not self.skip_gat:
            self.gat = SpatialGAT(
                in_channels=n_features,
                hidden_channels=gat_hidden,
                num_nodes=num_nodes,
                heads=gat_heads,
                dropout=gat_dropout,
            )
            gat_out = self.gat.out_dim
        else:
            gat_out = n_features

        if backbone == "dlinear":
            self.backbone = DLinear(seq_len, pred_len, n_vars=2)
            self.use_horizon_decoder = False
        elif backbone == "itransformer":
            self.backbone = iTransformerEncoder(
                seq_len=seq_len,
                pred_len=pred_len,
                n_vars=n_features,
                n_targets=2,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                use_revin=use_revin,
            )
            self.use_horizon_decoder = False
        elif backbone == "tcn":
            channels = tcn_channels or [64, 64, 128]
            self.backbone = TCNBackbone(
                n_vars=gat_out,
                channels=channels,
                kernel_size=tcn_kernel_size,
                dropout=dropout,
                pred_len=pred_len,
                n_targets=2,
            )
            self.use_horizon_decoder = False
        else:
            self.backbone = PatchTSTEncoder(
                n_vars=gat_out,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                patch_len=patch_len,
                stride=patch_stride,
                dropout=dropout,
                pred_len=pred_len,
                n_targets=2,
                use_revin=use_revin,
                use_horizon_decoder=use_horizon_decoder,
                decoder_heads=n_heads,
                use_ar_refinement=use_ar_refinement,
                use_local_conv=use_local_conv,
                residual_output=use_residual_baseline,
                n_fut_cov=n_fut_cov,
            )
            self.use_horizon_decoder = use_horizon_decoder

        if use_physics_head and target_mean is not None and target_scale is not None:
            self.physics_head = PhysicsProjectionHead(target_mean, target_scale)
        else:
            self.physics_head = None

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        apply_physics: bool = True,
        fut_cov: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        b, n, t, f = x.shape
        attn = None

        if self.skip_gat:
            h_flat = x.reshape(b * n, t, f)
            y_flat = self.backbone(h_flat)
        else:
            h, _ = apply_gat_over_time(
                self.gat, x, self.edge_index, return_attention, edge_attr=self.edge_attr
            )
            h_flat = h.reshape(b * n, t, h.shape[-1])
            fut_flat = None
            if fut_cov is not None:
                fut_flat = fut_cov.reshape(b * n, fut_cov.shape[2], fut_cov.shape[3])
            y_flat, attn = self.backbone(
                h_flat, return_attention=return_attention, fut_cov=fut_flat
            )

        y_hat = y_flat.reshape(b, n, self.pred_len, 2)

        if self.use_residual_baseline:
            last = x[:, :, -1, :2]
            baseline = last.unsqueeze(2).expand(-1, -1, self.pred_len, -1)
            y_hat = baseline + y_hat

        if apply_physics and self.physics_head is not None:
            y_hat, _ = self.physics_head(y_hat)
        return y_hat, attn
