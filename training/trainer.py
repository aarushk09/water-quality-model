"""PyTorch training loop with checkpointing, resume, and long-run safeguards."""

from __future__ import annotations

import csv
import json
import math
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
)
from tqdm import tqdm

from data.dataset import DatasetBundle
from losses.physics_informed import PhysicsInformedLoss, PhysicsLossConfig
from models.spatiotemporal_model import SpatioTemporalWaterModel
from training.checkpointing import load_checkpoint, save_checkpoint
from training.device import resolve_device
from training.evaluate import build_model_from_bundle, evaluate_loader, print_metrics
from training.metrics import MetricsConfig, evaluate_batch_physical, mean_physical_rmse

# Fixed CSV columns so physical metrics rows do not change the schema mid-run.
LOG_FIELDNAMES = [
    "epoch",
    "train_loss",
    "train_mse",
    "train_physics_violation",
    "val_loss",
    "val_mse",
    "val_physics_violation",
    "val_temperature_rmse",
    "val_temperature_mae",
    "val_temperature_r2",
    "val_dissolved_oxygen_rmse",
    "val_dissolved_oxygen_mae",
    "val_dissolved_oxygen_r2",
    "val_hypoxia_f1",
    "val_hypoxia_prevalence",
    "lr",
    "best_val_mse",
    "wait",
    "val_phys_temperature_rmse",
    "val_phys_temperature_mae",
    "val_phys_temperature_r2",
    "val_phys_dissolved_oxygen_rmse",
    "val_phys_dissolved_oxygen_mae",
    "val_phys_dissolved_oxygen_r2",
    "val_phys_hypoxia_f1",
    "val_phys_hypoxia_prevalence",
    "val_phys_mean_rmse",
    "best_val_phys_mean_rmse",
]


class Trainer:
    def __init__(
        self,
        model: SpatioTemporalWaterModel,
        bundle: DatasetBundle,
        cfg: dict,
        device: Optional[torch.device] = None,
        resume_path: Optional[Path] = None,
    ):
        self.model = model
        self.bundle = bundle
        self.cfg = cfg
        train_cfg = cfg["training"]
        pref = train_cfg.get("device", "auto")
        self.device = device or resolve_device(pref)
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
            betas=(0.9, 0.95),
        )
        self.epochs = int(train_cfg["epochs"])
        sched_name = train_cfg.get("lr_scheduler", "plateau")
        self._use_onecycle = sched_name == "onecycle"
        self._use_cosine_warmup = sched_name == "cosine_warmup"
        if self._use_onecycle:
            steps_per_epoch = max(len(bundle.train_loader), 1)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=train_cfg.get("max_lr", train_cfg["learning_rate"] * 3),
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=train_cfg.get("onecycle_pct_start", 0.1),
                div_factor=train_cfg.get("onecycle_div_factor", 25.0),
                final_div_factor=train_cfg.get("onecycle_final_div_factor", 1e4),
            )
            self._plateau_scheduler = None
        elif self._use_cosine_warmup:
            warmup_epochs = int(train_cfg.get("warmup_epochs", 15))
            warmup_epochs = max(1, min(warmup_epochs, self.epochs - 1))
            warmup = LinearLR(
                self.optimizer,
                start_factor=train_cfg.get("warmup_start_factor", 0.1),
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=max(self.epochs - warmup_epochs, 1),
                eta_min=train_cfg.get("min_lr", 1e-6),
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
            self._plateau_scheduler = None
        else:
            self._plateau_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=train_cfg.get("lr_factor", 0.5),
                patience=train_cfg.get("lr_patience", 4),
                min_lr=train_cfg.get("min_lr", 1e-7),
            )
            self.scheduler = self._plateau_scheduler

        ts = bundle.feature_engineer.target_scaler
        phys_cfg = cfg.get("physics", {})
        loss_cfg = PhysicsLossConfig(
            huber_delta=phys_cfg.get("huber_delta", 1.0),
            horizon_tau=phys_cfg.get("horizon_tau", 48.0),
            lambda_supersat=phys_cfg.get("lambda_supersat", 0.5),
            lambda_nonneg=phys_cfg.get("lambda_nonneg", 0.1),
            lambda_solubility=phys_cfg.get("lambda_solubility", 0.05),
            lambda_reaeration=phys_cfg.get("lambda_reaeration", 0.1),
            lambda_derivative=phys_cfg.get("lambda_derivative", 0.2),
            derivative_horizon=phys_cfg.get("derivative_horizon", 24),
            physics_warmup_epochs=phys_cfg.get("physics_warmup_epochs", 40),
            physics_ramp_epochs=phys_cfg.get("physics_ramp_epochs", 80),
            physics_max_scale=phys_cfg.get("physics_max_scale", 0.35),
            physics_horizon_steps=phys_cfg.get("physics_horizon_steps", 24),
            physics_mode=phys_cfg.get("physics_mode", "soft"),
            short_horizon_steps=phys_cfg.get("short_horizon_steps", 12),
            short_horizon_weight=phys_cfg.get("short_horizon_weight", 2.0),
            short_horizon_epoch_fraction=phys_cfg.get("short_horizon_epoch_fraction", 0.2),
            short_horizon_tail_weight=phys_cfg.get("short_horizon_tail_weight", 0.5),
            k_reaer=phys_cfg.get("k_reaer", 0.5),
            dt_hours=phys_cfg.get("dt_hours", 0.25),
        )
        self.criterion = PhysicsInformedLoss(
            target_mean=torch.tensor(ts.mean_, dtype=torch.float32),
            target_scale=torch.tensor(ts.scale_, dtype=torch.float32),
            cfg=loss_cfg,
            current_epoch=1,
            max_epochs=self.epochs,
        )
        self.grad_clip = train_cfg.get("grad_clip", 0.5)
        self.patience = int(train_cfg.get("early_stopping_patience", 15))
        self.min_epochs = int(train_cfg.get("min_epochs", 0))
        self.enable_early_stopping = train_cfg.get("enable_early_stopping", True)
        self.monitor_key = train_cfg.get("monitor_key", "val_phys_mean_rmse")
        self.save_last_every_epoch = train_cfg.get("save_last_every_epoch", True)
        self.checkpoint_every = int(train_cfg.get("checkpoint_every_epochs", 10))
        self.keep_last_n = int(train_cfg.get("keep_last_n_epoch_checkpoints", 3))
        self.eval_physical_every = int(train_cfg.get("eval_physical_every_epochs", 5))
        self.skip_nan_batches = train_cfg.get("skip_nan_batches", True)
        self.max_nan_epochs = int(train_cfg.get("max_consecutive_nan_epochs", 3))

        self.ckpt_dir = Path(train_cfg["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(train_cfg.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = self.log_dir / "train_log.csv"
        self.log_jsonl = self.log_dir / "train_log.jsonl"

        self.metrics_cfg = MetricsConfig(
            hypoxia_threshold_mg_l=cfg["metrics"]["hypoxia_threshold_mg_l"],
            hypoxia_sensitivity_mg_l=cfg["metrics"]["hypoxia_sensitivity_mg_l"],
            sudden_drop_mg_l=cfg["metrics"]["sudden_drop_mg_l"],
            sudden_drop_steps=cfg["metrics"]["sudden_drop_steps"],
        )

        self.best_val = float("inf")
        self.ema_val = float("inf")
        self.ema_decay = float(train_cfg.get("ema_decay", 0.92))
        self.use_ema_for_best = train_cfg.get("use_ema_for_best", True)
        self.wait = 0
        self.start_epoch = 1
        self.nan_epoch_streak = 0
        self._interrupted = False
        self._epoch_ckpts: List[Path] = []

        # Persist config snapshot once per run
        import yaml

        with open(self.ckpt_dir / "run_config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)

        if resume_path is not None:
            self._resume_from(resume_path)
        elif train_cfg.get("auto_resume_last", False) and (self.ckpt_dir / "last.pt").exists():
            print(f"Auto-resuming from {self.ckpt_dir / 'last.pt'}")
            self._resume_from(self.ckpt_dir / "last.pt")

        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame) -> None:
        print("\nInterrupt received — saving last checkpoint before exit...")
        self._interrupted = True

    def _resume_from(self, path: Path) -> None:
        ckpt = load_checkpoint(
            path,
            self.device,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.best_val = float(ckpt.get("best_val", ckpt.get("val_mse", float("inf"))))
        self.wait = int(ckpt.get("wait", 0))
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(
            f"Resumed from epoch {ckpt.get('epoch')}: "
            f"best_val={self.best_val:.4f}, next_epoch={self.start_epoch}"
        )

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        non_blocking = self.device.type in ("cuda", "mps")
        return tensor.to(self.device, non_blocking=non_blocking)

    def _train_step_batch(self, batch: Dict[str, torch.Tensor]) -> Optional[Dict[str, float]]:
        x = self._to_device(batch["x"])
        y = self._to_device(batch["y"])
        mask = batch.get("mask")
        if mask is not None:
            mask = self._to_device(mask)
        fut_cov = batch.get("fut_cov")
        if fut_cov is not None:
            fut_cov = self._to_device(fut_cov)
        y_hat, _ = self.model(x, fut_cov=fut_cov)
        out = self.criterion(y_hat, y, mask)
        loss = out["loss"]
        if not torch.isfinite(loss):
            if self.skip_nan_batches:
                return None
            raise RuntimeError("Non-finite loss encountered during training.")
        return out

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "mse": 0.0, "physics_violation": 0.0}
        n = 0
        skipped = 0
        for batch in tqdm(self.bundle.train_loader, desc="train", leave=False):
            if self._interrupted:
                break
            self.optimizer.zero_grad(set_to_none=True)
            out = self._train_step_batch(batch)
            if out is None:
                skipped += 1
                continue
            out["loss"].backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self._use_onecycle:
                self.scheduler.step()
            for k in totals:
                totals[k] += out[k].item()
            n += 1
        if skipped:
            print(f"  (skipped {skipped} batches with non-finite loss)")
        if n == 0:
            return {k: float("nan") for k in totals}
        return {k: v / n for k, v in totals.items()}

    @torch.no_grad()
    def eval_epoch(self, loader, prefix: str = "val") -> Dict[str, float]:
        self.model.eval()
        totals = {"loss": 0.0, "mse": 0.0, "physics_violation": 0.0}
        metric_accum: Dict[str, float] = {}
        n = 0
        for batch in loader:
            x = self._to_device(batch["x"])
            y = self._to_device(batch["y"])
            mask = batch.get("mask")
            if mask is not None:
                mask = self._to_device(mask)
            fut_cov = batch.get("fut_cov")
            if fut_cov is not None:
                fut_cov = self._to_device(fut_cov)
            y_hat, _ = self.model(x, fut_cov=fut_cov)
            out = self.criterion(y_hat, y, mask)
            for k in totals:
                totals[k] += out[k].item()
            mb = evaluate_batch_physical(
                y,
                y_hat,
                self.bundle.feature_engineer,
                self.metrics_cfg,
                forecast_node=self.bundle.forecast_node,
            )
            for k, v in mb.items():
                if k == "mean_rmse":
                    continue
                metric_accum[k] = metric_accum.get(k, 0.0) + (v if v == v else 0.0)
            metric_accum["mean_rmse"] = metric_accum.get("mean_rmse", 0.0) + mb["mean_rmse"]
            n += 1
        result = {f"{prefix}_{k}": v / max(n, 1) for k, v in totals.items()}
        for k, v in metric_accum.items():
            key = f"{prefix}_phys_{k}" if k != "mean_rmse" else f"{prefix}_phys_mean_rmse"
            result[key] = v / max(n, 1)
        return result

    def _row_for_csv(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Pad to fixed columns; use empty string when physical eval not run this epoch."""
        out: Dict[str, Any] = {}
        for key in LOG_FIELDNAMES:
            val = row.get(key, "")
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                val = ""
            out[key] = val
        return out

    def _row_for_jsonl(self, row: Dict[str, Any]) -> Dict[str, Any]:
        safe = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                safe[k] = None
            elif isinstance(v, (int, float)):
                safe[k] = float(v)
            else:
                safe[k] = v
        return safe

    def _append_log(self, row: Dict[str, Any]) -> None:
        csv_row = self._row_for_csv(row)
        write_header = not self.log_csv.exists() or self.log_csv.stat().st_size == 0
        with open(self.log_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(csv_row)
            f.flush()
        with open(self.log_jsonl, "a") as f:
            f.write(json.dumps(self._row_for_jsonl(row)) + "\n")
            f.flush()

    def _save_all_checkpoints(
        self,
        epoch: int,
        val_m: Dict[str, float],
        is_best: bool,
    ) -> None:
        val_mse = val_m[self.monitor_key]
        extra = {"history_note": "long_run"}
        if self.save_last_every_epoch:
            save_checkpoint(
                self.ckpt_dir / "last.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                cfg=self.cfg,
                epoch=epoch,
                best_val=self.best_val,
                wait=self.wait,
                val_mse=val_mse,
                feature_scaler=self.bundle.feature_engineer.feature_scaler,
                target_scaler=self.bundle.feature_engineer.target_scaler,
                extra=extra,
            )
        if is_best:
            save_checkpoint(
                self.ckpt_dir / "best.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                cfg=self.cfg,
                epoch=epoch,
                best_val=self.best_val,
                wait=self.wait,
                val_mse=val_mse,
                feature_scaler=self.bundle.feature_engineer.feature_scaler,
                target_scaler=self.bundle.feature_engineer.target_scaler,
                extra=extra,
            )
        if self.checkpoint_every > 0 and epoch % self.checkpoint_every == 0:
            ep_path = self.ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                ep_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                cfg=self.cfg,
                epoch=epoch,
                best_val=self.best_val,
                wait=self.wait,
                val_mse=val_mse,
                feature_scaler=self.bundle.feature_engineer.feature_scaler,
                target_scaler=self.bundle.feature_engineer.target_scaler,
            )
            self._epoch_ckpts.append(ep_path)
            while len(self._epoch_ckpts) > self.keep_last_n:
                old = self._epoch_ckpts.pop(0)
                if old.exists():
                    old.unlink()

    def _prepare_log_files(self) -> None:
        """On a fresh run, rotate partial logs so CSV header matches LOG_FIELDNAMES."""
        if self.start_epoch > 1:
            return
        for path in (self.log_csv, self.log_jsonl):
            if path.exists() and path.stat().st_size > 0:
                backup = path.with_name(path.stem + "_partial" + path.suffix)
                path.rename(backup)
                print(f"Rotated previous log to {backup}")

    def fit(self) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        self._prepare_log_files()

        print(
            f"Training epochs {self.start_epoch}–{self.epochs} "
            f"(early_stop={'on' if self.enable_early_stopping else 'off'}, "
            f"patience={self.patience}, min_epochs={self.min_epochs})"
        )

        for epoch in range(self.start_epoch, self.epochs + 1):
            if self._interrupted:
                break

            self.criterion.current_epoch = epoch
            train_m = self.train_epoch()
            val_m = self.eval_epoch(self.bundle.val_loader, prefix="val")

            if not all(math.isfinite(train_m.get(k, float("nan"))) for k in ("loss", "mse")):
                self.nan_epoch_streak += 1
                print(f"Epoch {epoch}: non-finite train metrics — skipping scheduler/update")
                if self.nan_epoch_streak >= self.max_nan_epochs:
                    print("Too many consecutive NaN epochs; stopping.")
                    break
                continue
            self.nan_epoch_streak = 0

            monitor_val = val_m.get(
                self.monitor_key,
                val_m.get("val_phys_mean_rmse", val_m.get("val_mse", float("inf"))),
            )
            if self._use_cosine_warmup:
                self.scheduler.step()
            elif not self._use_onecycle and self._plateau_scheduler is not None:
                self._plateau_scheduler.step(monitor_val)
            lr = self.optimizer.param_groups[0]["lr"]

            row: Dict[str, Any] = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_m.items()},
                **val_m,
                "lr": lr,
                "best_val_mse": self.best_val,
                "best_val_phys_mean_rmse": self.best_val,
                "wait": self.wait,
            }

            if self.eval_physical_every > 0 and epoch % self.eval_physical_every == 0:
                phys = evaluate_loader(
                    self.model,
                    self.bundle.val_loader,
                    self.bundle.feature_engineer,
                    self.device,
                    self.metrics_cfg,
                    forecast_node=self.bundle.forecast_node,
                )
                for k, v in phys.items():
                    row[f"val_phys_{k}"] = v
                row["val_phys_mean_rmse"] = mean_physical_rmse(phys)
                print(
                    f"  val physical — temp RMSE {phys['temperature_rmse']:.3f} °C, "
                    f"DO RMSE {phys['dissolved_oxygen_rmse']:.3f} mg/L"
                )
            elif "val_phys_mean_rmse" in val_m:
                print(
                    f"  val physical — mean RMSE {val_m['val_phys_mean_rmse']:.3f} "
                    f"(temp R² {val_m.get('val_phys_temperature_r2', float('nan')):.3f})"
                )

            history.append(row)
            self._append_log(row)

            if self.use_ema_for_best and math.isfinite(monitor_val):
                if math.isinf(self.ema_val):
                    self.ema_val = monitor_val
                else:
                    self.ema_val = (
                        self.ema_decay * self.ema_val
                        + (1.0 - self.ema_decay) * monitor_val
                    )
                score = self.ema_val
            else:
                score = monitor_val

            is_best = score < self.best_val
            if is_best:
                self.best_val = score
                self.wait = 0
            else:
                self.wait += 1

            self._save_all_checkpoints(epoch, val_m, is_best)

            print(
                f"Epoch {epoch}/{self.epochs}: train_loss={train_m['loss']:.4f} "
                f"val_mse={val_m.get('val_mse', float('nan')):.4f} "
                f"val_phys_rmse={val_m.get('val_phys_mean_rmse', float('nan')):.4f} "
                f"phys_loss={val_m.get('val_physics_violation', 0):.4f} "
                f"lr={lr:.2e} "
                f"{'[best]' if is_best else f'wait={self.wait}'}"
            )

            if (
                self.enable_early_stopping
                and epoch >= self.min_epochs
                and self.wait >= self.patience
            ):
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no {self.monitor_key} improvement for {self.patience} epochs)."
                )
                break

        if self._interrupted:
            print("Training interrupted — last.pt contains the latest state for --resume.")

        ckpt_path = self.ckpt_dir / "best.pt"
        if ckpt_path.exists():
            load_checkpoint(ckpt_path, self.device, model=self.model)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            print(
                f"Loaded best checkpoint (epoch {ckpt.get('epoch', '?')}, "
                f"val_mse={ckpt.get('val_mse', float('nan')):.4f})"
            )

        test_phys = evaluate_loader(
            self.model,
            self.bundle.test_loader,
            self.bundle.feature_engineer,
            self.device,
            self.metrics_cfg,
            forecast_node=self.bundle.forecast_node,
        )
        print_metrics("Test", test_phys)

        return {
            "history": history,
            "best_val_mse": self.best_val,
            "test_metrics": test_phys,
            "last_epoch": history[-1]["epoch"] if history else 0,
            "interrupted": self._interrupted,
        }


# build_model_from_bundle is imported from training.evaluate
