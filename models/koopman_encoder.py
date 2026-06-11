"""Koopman Operator Encoder for river dynamics discovery.

This module learns a neural lifting of physical observables (T, DO, Q, time)
into a latent space where dynamics are LINEAR — the Koopman representation.

The transition matrix K is a learned 32×32 matrix. After training:
  - SVD(K) → eigenvalue spectrum reveals dominant oscillation modes
  - Diurnal cycle at 1/24h⁻¹, dam release at higher frequencies
  - Multi-step consistency guarantees K captures true system structure

References:
  Brunton et al. (2021) "Modern Koopman Theory for Dynamical Systems"
  Lusch et al. (2018) "Deep learning for universal linear embeddings of nonlinear dynamics"
  Morton et al. (2019) "Deep dynamical modeling and control of unsteady fluid flows"
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KoopmanEncoder(nn.Module):
    """
    Neural Koopman encoder-decoder with learned linear transition.

    Maps a physical state x_t ∈ ℝ^F  →  latent z_t ∈ ℝ^{latent_dim}
    such that z_{t+1} ≈ K · z_t  (K learned, approximately linear dynamics).

    Triple loss:
      1. Reconstruction: ||D(E(x)) - x||²
      2. One-step prediction: ||E(x_{t+1}) - K·E(x_t)||²
      3. Multi-step consistency: ||K^n·E(x_t) - E(x_{t+n})||² for n in multi_steps
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 32,
        encoder_hidden: int = 256,
        n_encoder_layers: int = 3,
        multi_steps: Tuple[int, ...] = (1, 4, 24),
        spectral_radius_target: float = 0.98,
        lambda_recon: float = 1.0,
        lambda_pred: float = 1.0,
        lambda_multi: float = 0.5,
        lambda_spectral: float = 0.01,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.multi_steps = multi_steps
        self.spectral_radius_target = spectral_radius_target
        self.lambda_recon = lambda_recon
        self.lambda_pred = lambda_pred
        self.lambda_multi = lambda_multi
        self.lambda_spectral = lambda_spectral

        # Encoder: physical state → Koopman latent
        enc_layers = []
        in_dim = n_features
        for i in range(n_encoder_layers):
            out_dim = encoder_hidden if i < n_encoder_layers - 1 else latent_dim
            enc_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout) if i < n_encoder_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: Koopman latent → physical state reconstruction
        dec_layers = []
        in_dim = latent_dim
        for i in range(n_encoder_layers):
            out_dim = encoder_hidden if i < n_encoder_layers - 1 else n_features
            dec_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < n_encoder_layers - 1 else nn.Identity(),
                nn.GELU() if i < n_encoder_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < n_encoder_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        self.decoder = nn.Sequential(*dec_layers)

        # Koopman operator K: the key learned transition matrix
        # Initialize close to identity (small perturbation from stable dynamics)
        self.K = nn.Parameter(torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim))

        # Projection head: projects Koopman latent to PatchTST d_model space
        # Used to inject Koopman dynamics into the Transformer as an auxiliary stream
        self.latent_proj: Optional[nn.Linear] = None  # initialized lazily via set_proj_dim

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.K:
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_proj_dim(self, d_model: int) -> None:
        """Set the projection head dimension for PatchTST integration."""
        self.latent_proj = nn.Linear(self.latent_dim, d_model)
        nn.init.xavier_uniform_(self.latent_proj.weight, gain=0.1)
        nn.init.zeros_(self.latent_proj.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., F] → z: [..., latent_dim]"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., latent_dim] → x_hat: [..., F]"""
        return self.decoder(z)

    def advance(self, z: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Apply K^steps to latent state z: [..., latent_dim] → [..., latent_dim]"""
        out = z
        for _ in range(steps):
            out = out @ self.K.T  # [..., latent_dim] @ [latent_dim, latent_dim]
        return out

    def project_to_model(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Project Koopman latent to PatchTST d_model dim, if proj head is set."""
        if self.latent_proj is not None:
            return self.latent_proj(z)
        return None

    def koopman_loss(
        self,
        x_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all Koopman training losses.

        x_seq: [B, T, F] — sequence of physical states

        Returns dict with scalar losses:
          - recon_loss: reconstruction fidelity E(x) → D → x
          - pred_loss:  one-step linear prediction E(x_{t+1}) ≈ K·E(x_t)
          - multi_loss: multi-step consistency
          - spectral_loss: encourages K eigenvalues near unit circle (stability)
          - total: weighted sum
        """
        B, T, n_feat = x_seq.shape

        # Encode all timesteps
        z_seq = self.encode(x_seq)  # [B, T, latent_dim]

        # 1. Reconstruction loss
        x_recon = self.decode(z_seq)
        recon_loss = F.mse_loss(x_recon, x_seq)

        # 2. One-step prediction loss: K·z_t ≈ z_{t+1}
        if T > 1:
            z_pred = z_seq[:, :-1, :] @ self.K.T  # [B, T-1, latent_dim]
            pred_loss = F.mse_loss(z_pred, z_seq[:, 1:, :])
        else:
            pred_loss = torch.tensor(0.0, device=x_seq.device)

        # 3. Multi-step consistency: K^n·z_t ≈ z_{t+n}
        multi_loss = torch.tensor(0.0, device=x_seq.device)
        count = 0
        for n in self.multi_steps:
            if n >= T:
                continue
            z_start = z_seq[:, :T - n, :]   # [B, T-n, latent_dim]
            z_target = z_seq[:, n:, :]       # [B, T-n, latent_dim]
            z_advanced = self.advance(z_start, steps=n)
            multi_loss = multi_loss + F.mse_loss(z_advanced, z_target)
            count += 1
        if count > 0:
            multi_loss = multi_loss / count

        # 4. Spectral regularization: keep spectral radius near target
        # Use Frobenius norm surrogate (exact eigenvalues costly in training)
        # ||K||_F ≈ sqrt(sum of squared singular values) ≥ spectral radius
        k_norm = torch.linalg.norm(self.K, ord="fro")
        target_norm = self.spectral_radius_target * math.sqrt(self.latent_dim)
        spectral_loss = F.mse_loss(k_norm, torch.tensor(target_norm, device=x_seq.device))

        total = (
            self.lambda_recon * recon_loss
            + self.lambda_pred * pred_loss
            + self.lambda_multi * multi_loss
            + self.lambda_spectral * spectral_loss
        )

        return {
            "koopman_total": total,
            "koopman_recon": recon_loss,
            "koopman_pred": pred_loss,
            "koopman_multi": multi_loss,
            "koopman_spectral": spectral_loss,
        }

    @torch.no_grad()
    def get_eigenspectrum(self) -> Dict[str, torch.Tensor]:
        """
        Compute eigendecomposition of K for physics discovery analysis.

        Returns eigenvalues (complex), their magnitudes, and frequencies.
        Dominant eigenvalues map to dominant oscillation modes.
        """
        # Use CPU for eigendecomposition (more stable numerics)
        K_cpu = self.K.detach().cpu().float()
        eigenvalues = torch.linalg.eigvals(K_cpu)  # complex tensor

        magnitudes = eigenvalues.abs()
        # Frequency in cycles per 15-min timestep → convert to cycles/hour
        angles = eigenvalues.angle()  # radians per timestep
        freq_per_hour = angles.abs() / (2 * math.pi) * 4  # 4 timesteps per hour
        period_hours = torch.where(
            freq_per_hour > 1e-6,
            1.0 / freq_per_hour,
            torch.full_like(freq_per_hour, float("inf"))
        )

        # Sort by magnitude (dominant modes first)
        sort_idx = magnitudes.argsort(descending=True)
        return {
            "eigenvalues_real": eigenvalues.real[sort_idx],
            "eigenvalues_imag": eigenvalues.imag[sort_idx],
            "magnitudes": magnitudes[sort_idx],
            "freq_per_hour": freq_per_hour[sort_idx],
            "period_hours": period_hours[sort_idx],
        }

    def forward(
        self,
        x: torch.Tensor,
        return_losses: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        x: [B, T, F] → (z_last, losses_dict)

        Returns the Koopman latent at the last timestep (for decoder input)
        and optionally the training losses.
        """
        z = self.encode(x)  # [B, T, latent_dim]
        losses = self.koopman_loss(x) if return_losses else None
        return z[:, -1, :], losses  # return last-step latent + losses
