#!/usr/bin/env python3
"""
Post-hoc Koopman Analysis on Frozen Frontier-FT Representations.

Architecture motivation:
  - The Frontier-FT model's GAT encoder has learned rich water-quality
    spatiotemporal representations after 40+ epochs of physics-constrained training.
  - Jointly training a Koopman encoder alongside the forecast head causes
    gradient interference — the Koopman auxiliary losses reshape h_flat in ways
    that degrade physical predictions.
  - Solution: freeze the entire Frontier-FT backbone, extract h_flat features
    for the full dataset, then train a standalone KoopmanEncoder on those
    fixed representations. This gives a clean Koopman eigenspectrum of the
    LEARNED latent dynamics without any feedback into forecast quality.

Scientific contribution:
  - The eigenvalues of the Koopman operator (K) reveal which dynamical modes
    govern the water-quality latent space (oscillatory vs. dissipative).
  - Modes with |λ| ≈ 1 represent persistent cycles (diel temperature oscillation,
    seasonal DO, storm pulses). Modes with |λ| < 1 decay — fast transients.
  - This analysis cannot be performed on raw sensor data (nonlinear mixing);
    it requires the linearising representation h_flat learned by the GAT.
"""

from __future__ import annotations
import argparse, sys, json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.spatiotemporal_model import SpatioTemporalWaterModel as SpatiotemporalModel
from models.koopman_encoder import KoopmanEncoder
from models.gat_layer import apply_gat_over_time
from training.checkpointing import load_checkpoint
from data.dataset import build_dataloaders_from_config


# ─────────────────────────────────────────────────────────────────────────────
def load_frontier_model(ckpt_path: Path, device: torch.device, cfg: dict, bundle) -> SpatiotemporalModel:
    """Load Frontier-FT backbone weights only (no optimizer/scheduler needed)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    from training.evaluate import build_model_from_bundle, load_state_dict_compatible

    # Temporarily disable physics head for feature extraction
    use_physics_head_orig = cfg["model"].get("use_physics_head", True)
    cfg["model"]["use_physics_head"] = False

    model = build_model_from_bundle(cfg, bundle)

    # Restore physics head setting
    cfg["model"]["use_physics_head"] = use_physics_head_orig

    load_state_dict_compatible(model, ckpt["model"])
    model.to(device)
    model.eval()

    # Freeze ALL parameters — we never backprop into the Frontier model
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"Loaded Frontier-FT backbone from epoch {ckpt.get('epoch','?')} "
          f"(best_val={ckpt.get('best_val', float('nan')):.4f})")
    return model, ckpt


@torch.no_grad()
def extract_features(
    model: SpatiotemporalModel,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Run the frozen GAT encoder over all batches, collect h_flat tensors."""
    all_h: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="Extracting GAT features"):
        x = batch["x"].to(device)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        b, n, t, f = x.shape

        if model.skip_gat:
            h_flat = x.reshape(b * n, t, f)
        else:
            h, _ = apply_gat_over_time(
                model.gat, x, model.edge_index, False,
                edge_attr=model.edge_attr
            )
            h_flat = h.reshape(b * n, t, h.shape[-1])  # [B*N, T, D_gat]

        all_h.append(h_flat.cpu())

    return torch.cat(all_h, dim=0)  # [N_total, T, D_gat]


def train_koopman_standalone(
    h_all: torch.Tensor,
    gat_dim: int,
    latent_dim: int = 32,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    save_dir: Path = Path("checkpoints/koopman_posthoc"),
    log_path: Path = Path("logs/koopman_posthoc/train_log.csv"),
) -> KoopmanEncoder:
    """Train KoopmanEncoder on fixed h_flat representations."""
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = KoopmanEncoder(
        n_features  = gat_dim,
        latent_dim  = latent_dim,
        lambda_recon = 1.0,
        lambda_pred  = 1.0,
        lambda_multi = 0.5,
        lambda_spectral = 0.01,
    ).to(device)
    encoder.set_proj_dim(gat_dim)   # project_to_model not used but set for API

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    dataset = TensorDataset(h_all)  # [N, T, D]
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    log_path.write_text("epoch,train_loss,recon,pred,multi\n")
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        encoder.train()
        totals = {"total": 0.0, "recon": 0.0, "pred": 0.0, "multi": 0.0}
        n_batches = 0

        for (h_batch,) in loader:
            h_batch = h_batch.to(device)
            optimizer.zero_grad()
            _, losses = encoder(h_batch, return_losses=True)
            loss = losses["koopman_total"]
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            totals["total"] += losses["koopman_total"].item()
            totals["recon"] += losses["koopman_recon"].item()
            totals["pred"]  += losses["koopman_pred"].item()
            totals["multi"] += losses.get("koopman_multi", torch.tensor(0.0)).item()
            n_batches += 1

        scheduler.step()

        if n_batches == 0:
            print(f"Epoch {epoch}: all batches non-finite — skipping")
            continue

        avg = {k: v / n_batches for k, v in totals.items()}
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Ep {epoch:3d}/{epochs}: "
              f"loss={avg['total']:.4f} "
              f"recon={avg['recon']:.4f} "
              f"pred={avg['pred']:.4f} "
              f"multi={avg['multi']:.4f} "
              f"lr={lr_now:.2e}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg['total']:.6f},{avg['recon']:.6f},"
                    f"{avg['pred']:.6f},{avg['multi']:.6f}\n")

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "best_loss": best_loss,
                "latent_dim": latent_dim,
                "gat_dim": gat_dim,
            }, save_dir / "best.pt")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "loss": avg["total"],
            }, save_dir / f"epoch_{epoch:03d}.pt")

    print(f"\nBest loss: {best_loss:.4f}. Checkpoints in {save_dir}")
    return encoder


@torch.no_grad()
def compute_koopman_spectrum(encoder: KoopmanEncoder, device: torch.device) -> dict:
    """
    Extract the Koopman operator K and compute its eigenspectrum.
    Handles different attribute names for the K matrix across encoder versions.
    """
    # Find the Koopman dynamics matrix — it's an nn.Parameter named K
    K_tensor = None
    for attr in ["K", "koopman_matrix", "linear_dynamics"]:
        if hasattr(encoder, attr):
            candidate = getattr(encoder, attr)
            if isinstance(candidate, nn.Parameter) or isinstance(candidate, torch.Tensor):
                K_tensor = candidate.data.cpu().float().numpy()
                print(f"  Using encoder.{attr} as K matrix (shape {K_tensor.shape})")
                break
            elif isinstance(candidate, nn.Linear):
                K_tensor = candidate.weight.data.cpu().float().numpy()
                print(f"  Using encoder.{attr}.weight as K matrix (shape {K_tensor.shape})")
                break

    if K_tensor is None:
        print("WARNING: Could not find Koopman dynamics matrix. Returning empty spectrum.")
        return {"eigenvalues_real": [], "eigenvalues_imag": [], "magnitudes": [], "period_hours": []}

    eigenvalues = np.linalg.eigvals(K_tensor)

    magnitudes = np.abs(eigenvalues)
    phases     = np.angle(eigenvalues)
    # Convert phase to period in time-steps (1 step = 15 min → hourly scale)
    # |λ| ≈ 1: persistent; |λ| < 1: decaying; frequency = phase / (2π)
    freq_hz   = np.abs(phases) / (2 * np.pi)              # cycles per time-step
    period_ts = np.where(freq_hz > 0, 1.0 / freq_hz, np.inf)  # time-steps per cycle
    period_h  = period_ts * 0.25  # 15-min steps → hours

    # Sort by magnitude descending (most persistent modes first)
    idx = np.argsort(magnitudes)[::-1]
    spectrum = {
        "eigenvalues_real": eigenvalues.real[idx].tolist(),
        "eigenvalues_imag": eigenvalues.imag[idx].tolist(),
        "magnitudes":       magnitudes[idx].tolist(),
        "period_hours":     period_h[idx].tolist(),
        "K_matrix":         K_tensor.tolist(),
    }
    return spectrum


def main():
    parser = argparse.ArgumentParser(description="Post-hoc Koopman analysis on frozen Frontier-FT features")
    parser.add_argument("--frontier-ckpt", type=str,
                        default="checkpoints/frontier_v1_finetune/best.pt",
                        help="Path to Frontier-FT best checkpoint")
    parser.add_argument("--config", type=str,
                        default="configs/exp_frontier_v1_finetune.yaml",
                        help="Config used for Frontier-FT (for data loading)")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--spectrum-only", action="store_true",
                        help="Skip training, just load best checkpoint and compute spectrum")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    ckpt_path = ROOT / args.frontier_ckpt

    if not args.spectrum_only:
        # ── 1. Build data loaders (use same config as Frontier-FT) ──────────
        import yaml
        with open(ROOT / args.config) as f:
            cfg = yaml.safe_load(f)

        bundle = build_dataloaders_from_config(cfg, project_root=ROOT)
        train_loader = bundle.train_loader

        # ── 2. Load frozen Frontier-FT backbone ─────────────────────────────
        model, ckpt = load_frontier_model(ckpt_path, device, cfg, bundle)

        # ── 3. Extract frozen GAT features ──────────────────────────────────
        print("\nExtracting frozen GAT features from training set...")
        gat_dim = cfg["model"].get("gat_hidden", 64) if not model.skip_gat else cfg["model"]["n_features"]
        h_all = extract_features(model, train_loader, device)
        print(f"Feature tensor shape: {tuple(h_all.shape)}  "
              f"({h_all.shape[0]} sequences × {h_all.shape[1]} timesteps × {gat_dim} dims)")

        # Free GPU memory before Koopman training
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

        # ── 4. Train standalone KoopmanEncoder ──────────────────────────────
        print("\nTraining standalone KoopmanEncoder on frozen features...")
        encoder = train_koopman_standalone(
            h_all,
            gat_dim     = gat_dim,
            latent_dim  = args.latent_dim,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            device      = device,
            save_dir    = ROOT / "checkpoints/koopman_posthoc",
            log_path    = ROOT / "logs/koopman_posthoc/train_log.csv",
        )
    else:
        # Load best encoder checkpoint
        enc_ckpt = ROOT / "checkpoints/koopman_posthoc/best.pt"
        enc_data = torch.load(enc_ckpt, map_location=device, weights_only=False)
        gat_dim    = enc_data["gat_dim"]
        latent_dim = enc_data["latent_dim"]
        encoder = KoopmanEncoder(
            n_features=gat_dim, latent_dim=latent_dim,
            lambda_recon=1.0, lambda_pred=1.0, lambda_multi=0.5, lambda_spectral=0.01,
        ).to(device)
        encoder.set_proj_dim(gat_dim)
        encoder.load_state_dict(enc_data["encoder"])
        print(f"Loaded encoder from epoch {enc_data['epoch']} (loss={enc_data.get('best_loss','?')})")

    # ── 5. Compute and save Koopman eigenspectrum ────────────────────────────
    print("\nComputing Koopman eigenspectrum...")
    spectrum = compute_koopman_spectrum(encoder, device)

    out_path = ROOT / "logs/koopman_posthoc/spectrum.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(spectrum, f, indent=2)
    print(f"Spectrum saved to {out_path}")

    # Print top modes
    print("\nTop 10 Koopman modes (by persistence |λ|):")
    print(f"  {'Mode':>4}  {'|λ|':>7}  {'Period (h)':>12}  {'Re(λ)':>9}  {'Im(λ)':>9}")
    for i in range(min(10, len(spectrum["magnitudes"]))):
        mag   = spectrum["magnitudes"][i]
        per   = spectrum["period_hours"][i]
        re    = spectrum["eigenvalues_real"][i]
        im    = spectrum["eigenvalues_imag"][i]
        per_s = f"{per:.1f} h" if per < 1e6 else "∞ (DC)"
        print(f"  {i+1:>4}  {mag:>7.4f}  {per_s:>12}  {re:>9.4f}  {im:>9.4f}")


if __name__ == "__main__":
    main()
