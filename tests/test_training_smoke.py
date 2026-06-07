"""Fast checks that training stack builds and loss decreases on a few steps."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloaders_from_config
from losses.physics_informed import PhysicsInformedLoss, PhysicsLossConfig
from training.evaluate import build_model_from_bundle


def _load_cfg() -> dict:
    with open(ROOT / "configs" / "high_accuracy.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["project_root"] = str(ROOT)
    cfg["training"]["epochs"] = 3
    cfg["training"]["batch_size"] = 8
    cfg["physics"]["physics_warmup_epochs"] = 2
    cfg["physics"]["physics_ramp_epochs"] = 1
    return cfg


def test_dataset_builds():
    cfg = _load_cfg()
    bundle = build_dataloaders_from_config(cfg, project_root=ROOT)
    assert bundle.n_features >= 21
    batch = next(iter(bundle.train_loader))
    assert batch["x"].shape[-1] == bundle.n_features
    assert batch["y"].shape[-1] == 2


def test_physics_scale_curriculum():
    cfg = PhysicsLossConfig(physics_warmup_epochs=5, physics_ramp_epochs=10, physics_max_scale=0.3)
    loss_fn = PhysicsInformedLoss(
        target_mean=torch.zeros(2),
        target_scale=torch.ones(2),
        cfg=cfg,
        current_epoch=1,
    )
    assert loss_fn._physics_scale() == 0.0
    loss_fn.current_epoch = 5
    assert loss_fn._physics_scale() == 0.0
    loss_fn.current_epoch = 10
    assert 0.0 < loss_fn._physics_scale() < 0.3
    loss_fn.current_epoch = 20
    assert loss_fn._physics_scale() == 0.3


def test_forward_and_loss_finite():
    cfg = _load_cfg()
    bundle = build_dataloaders_from_config(cfg, project_root=ROOT)
    model = build_model_from_bundle(cfg, bundle)
    model.eval()
    batch = next(iter(bundle.train_loader))
    with torch.no_grad():
        y_hat, _ = model(batch["x"])
    ts = bundle.feature_engineer.target_scaler
    crit = PhysicsInformedLoss(
        target_mean=torch.tensor(ts.mean_, dtype=torch.float32),
        target_scale=torch.tensor(ts.scale_, dtype=torch.float32),
        cfg=PhysicsLossConfig(physics_mode="soft"),
    )
    out = crit(y_hat, batch["y"])
    assert torch.isfinite(out["loss"])


def test_residual_baseline_near_persistence():
    cfg = _load_cfg()
    bundle = build_dataloaders_from_config(cfg, project_root=ROOT)
    model = build_model_from_bundle(cfg, bundle)
    model.eval()
    batch = next(iter(bundle.train_loader))
    with torch.no_grad():
        y_hat, _ = model(batch["x"])
    last = batch["x"][:, :, -1, :2]
    baseline = last.unsqueeze(2).expand_as(y_hat)
    delta = (y_hat - baseline).abs().mean()
    assert delta.item() < 5.0
