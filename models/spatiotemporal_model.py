"""GAT spatial front-end + temporal backbone + physics projection."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

from models.dlinear import DLinear
from models.gat_layer import SpatialGAT, apply_gat_over_time
from models.itransformer import iTransformerEncoder
from models.koopman_encoder import KoopmanEncoder
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
        backbone: Literal["patchtst", "tcn", "dlinear", "itransformer", "koopman_patchtst"] = "patchtst",
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
        # Koopman-specific parameters
        koopman_latent_dim: int = 32,
        koopman_lambda_recon: float = 1.0,
        koopman_lambda_pred: float = 1.0,
        koopman_lambda_multi: float = 0.5,
        koopman_lambda_spectral: float = 0.01,
        koopman_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.use_physics_head = use_physics_head
        self.use_residual_baseline = use_residual_baseline
        self.backbone_name = backbone
        self.use_koopman = backbone == "koopman_patchtst"
        self.koopman_loss_weight = koopman_loss_weight
        # Normalize backbone name for downstream logic
        _effective_backbone = "patchtst" if self.use_koopman else backbone
        self.skip_gat = _effective_backbone in ("dlinear", "itransformer")
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

        # Koopman encoder (parallel branch for dynamics discovery)
        self.koopman: Optional[KoopmanEncoder] = None
        if self.use_koopman:
            self.koopman = KoopmanEncoder(
                n_features=gat_out,
                latent_dim=koopman_latent_dim,
                lambda_recon=koopman_lambda_recon,
                lambda_pred=koopman_lambda_pred,
                lambda_multi=koopman_lambda_multi,
                lambda_spectral=koopman_lambda_spectral,
            )
            self.koopman.set_proj_dim(d_model)
            # Koopman context gate: learns how much to blend Koopman vs. PatchTST.
            # Initialize deep-negative so sigmoid ≈ 0.007 (near-zero Koopman influence
            # at start). Gate grows naturally as the encoder learns good dynamics.
            self.koopman_gate = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5)≈0.007

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
        koopman_losses = None

        if self.skip_gat:
            h_flat = x.reshape(b * n, t, f)
            y_flat = self.backbone(h_flat)
        else:
            h, _ = apply_gat_over_time(
                self.gat, x, self.edge_index, return_attention, edge_attr=self.edge_attr
            )
            h_flat = h.reshape(b * n, t, h.shape[-1])

            # Koopman branch: encode entire sequence, get latent at last step
            if self.koopman is not None:
                z_last, koopman_losses = self.koopman(h_flat, return_losses=self.training)
                # Project Koopman latent into PatchTST query space
                koopman_proj = self.koopman.project_to_model(z_last)  # [B*N, d_model]

            fut_flat = None
            if fut_cov is not None:
                fut_flat = fut_cov.reshape(b * n, fut_cov.shape[2], fut_cov.shape[3])

            if self.koopman is not None and koopman_proj is not None:
                # Inject Koopman context as an extra token prepended to patch sequence
                # This lets cross-attention in horizon decoder attend to Koopman dynamics
                y_flat, attn = self.backbone(
                    h_flat,
                    return_attention=return_attention,
                    fut_cov=fut_flat,
                    koopman_ctx=koopman_proj,
                    koopman_gate=torch.sigmoid(self.koopman_gate),
                )
            else:
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

    def get_koopman_losses(self) -> Optional[Dict[str, torch.Tensor]]:
        """Returns Koopman auxiliary losses (call after forward in training loop)."""
        return None  # losses returned directly from forward; stored here for legacy API

    def get_patch_tokens(
        self, x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Extract PatchTST patch tokens for use as conditioning signal in diffusion.
        Returns [B, P, d_model] patch token sequence.
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        b, n, t, f = x.shape
        if self.skip_gat:
            h_flat = x.reshape(b * n, t, f)
        else:
            h, _ = apply_gat_over_time(self.gat, x, self.edge_index, False)
            h_flat = h.reshape(b * n, t, h.shape[-1])

        if not hasattr(self.backbone, "patch_embed") or not hasattr(self.backbone, "encoder"):
            return None

        # Run through PatchTST up to encoder (before horizon decoder)
        bb = self.backbone
        xr = h_flat
        if bb.use_revin:
            xr = bb.revin.norm(xr)
        if bb.use_local_conv:
            xr = bb.local_conv(xr.transpose(1, 2)).transpose(1, 2)
        emb, _ = bb.patch_embed(xr)
        tokens = bb.encoder(emb)  # [B*N, P, d_model]
        # Use forecast node only
        tokens = tokens.view(b, n, tokens.shape[1], tokens.shape[2])[:, 0]  # [B, P, d_model]
        return tokens
