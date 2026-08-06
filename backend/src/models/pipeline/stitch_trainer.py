"""
backend/src/models/stitch_net/trainer.py
==========================================
StitchTrainer — complete training orchestrator for AnimeStitchNet.

Features
--------
* Mixed-precision (AMP) via torch.cuda.amp.GradScaler.
* Cosine LR schedule with linear warmup.
* Checkpoint save / resume (model + optimiser + scheduler state).
* Per-epoch validation: pixel-space alignment error metric.
* Thread-safe GUI callbacks (on_log, on_metrics, on_epoch_end) so
  StitchTrainTab can show live progress without blocking the Qt event loop.
* Optional LoFTR knowledge distillation: replace synthetic GT with LoFTR
  affine predictions as soft labels (blended at distill_weight).
* TorchScript export on training completion.

Usage (standalone)
------------------
    trainer = StitchTrainer(config)
    trainer.train()

Usage (from GUI, runs in a daemon thread)
-----------------------------------------
    trainer = StitchTrainer(config,
        on_log=log_signal.emit,
        on_metrics=metric_signal.emit,
        on_epoch_end=epoch_signal.emit)
    threading.Thread(target=trainer.train, daemon=True).start()
"""

from __future__ import annotations

# --------------------------------
import gc
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

# --- Relocated Nested Imports ---
from animation.models.stitch_net import AnimeStitchNet
from animation.models.data.stitch_dataset import (
    SyntheticStitchDataset,
    stitch_collate_fn,
)
from backend.src.models.wrappers.loftr_wrapper import LoFTRWrapper
from animation.models.pipeline.stitch_losses import StitchNetLoss
from backend.src.constants.pipeline import DEFAULT_CONFIG

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# StitchTrainer
# ---------------------------------------------------------------------------


class StitchTrainer:
    """Full training orchestrator for AnimeStitchNet."""

    is_cancelled: bool = False  # class-level flag (set by StitchTrainTab.cancel)

    def __init__(
        self,
        config: Dict,
        on_log: Optional[Callable[[str], None]] = None,
        on_metrics: Optional[Callable[[Dict], None]] = None,
        on_epoch_end: Optional[Callable[[int, Dict], None]] = None,
    ):
        self.cfg = {**DEFAULT_CONFIG, **config}
        self.on_log = on_log or print
        self.on_metrics = on_metrics or (lambda _: None)
        self.on_epoch_end = on_epoch_end or (lambda *_: None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        StitchTrainer.is_cancelled = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self):
        try:
            self._setup()
            self._run_loop()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    def cancel(cls):
        cls.is_cancelled = True

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self):
        # relocated: from animation.models.stitch_net import AnimeStitchNet
        # relocated: from animation.models.data.stitch_dataset import (
            # relocated: SyntheticStitchDataset,
            # relocated: stitch_collate_fn,
        # relocated: )
        # relocated: from animation.models.pipeline.stitch_losses import StitchNetLoss

        cfg = self.cfg

        # ── Model ────────────────────────────────────────────────────────
        self._log("Building AnimeStitchNet…")
        self.model = AnimeStitchNet(
            enc_channels=cfg["enc_channels"],
            num_heads=cfg["num_heads"],
            num_ca_layers=cfg["num_ca_layers"],
            pretrained=cfg["pretrained"],
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        self._log(f"  Parameters: {n_params:,}")

        # ── Dataset ───────────────────────────────────────────────────────
        self._log(f"Loading dataset from '{cfg['image_dir']}'…")
        full_ds = SyntheticStitchDataset(
            image_dir=cfg["image_dir"],
            patch_hw=tuple(cfg["patch_hw"]),
            max_dx=cfg["max_dx"],
            max_dy=cfg["max_dy"],
            max_angle=cfg["max_angle"],
            max_log_s=cfg["max_log_s"],
            mpeg_noise_prob=cfg["mpeg_noise_prob"],
            dimming_prob=cfg["dimming_prob"],
            neg_pair_prob=cfg["neg_pair_prob"],
            dataset_size=cfg["dataset_size"],
            augment=cfg["augment"],
        )
        n_val = max(1, int(len(full_ds) * cfg["val_split"]))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        nw = min(cfg["num_workers"], os.cpu_count() or 1)
        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=nw,
            collate_fn=stitch_collate_fn,
            pin_memory=pin,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=nw,
            collate_fn=stitch_collate_fn,
            pin_memory=pin,
        )
        self._log(f"  Train: {n_train}  Val: {n_val}  Batch: {cfg['batch_size']}")

        # ── Loss ──────────────────────────────────────────────────────────
        self.criterion = StitchNetLoss(
            lambda_param=cfg["lambda_param"],
            lambda_photo=cfg["lambda_photo"],
            lambda_sym=cfg["lambda_sym"],
            huber_delta=cfg["huber_delta"],
            warmup_steps=cfg["warmup_steps"],
        )

        # ── Optimiser and scheduler ─────────────────────────────────────────
        self.opt = AdamW(
            self.model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        total_steps = cfg["epochs"] * len(self.train_loader)
        warmup_steps = cfg["warmup_epochs"] * len(self.train_loader)
        self.sched = _WarmupCosineScheduler(self.opt, warmup_steps, total_steps)

        # ── AMP ───────────────────────────────────────────────────────────
        use_amp = cfg["amp"] and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=use_amp)
        self.use_amp = use_amp

        # ── Output dir ────────────────────────────────────────────────────
        self.out_dir = Path(cfg["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ── Optional LoFTR teacher ────────────────────────────────────────
        self.loftr = None
        if cfg.get("loftr_distill"):
            try:
                # relocated: from backend.src.models.wrappers.loftr_wrapper import LoFTRWrapper

                self.loftr = LoFTRWrapper()
                self._log("  LoFTR distillation enabled.")
            except Exception as e:
                self._log(f"  LoFTR unavailable ({e}) — distillation disabled.")

        # ── Resume ────────────────────────────────────────────────────────
        self.start_epoch = 0
        latest = self.out_dir / "latest.pt"
        if latest.exists():
            self._load_checkpoint(latest)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def _run_loop(self):
        cfg = self.cfg
        epochs = cfg["epochs"]
        self._log(f"Training on {self.device}  ({epochs} epochs).")

        for epoch in range(self.start_epoch, epochs):
            if StitchTrainer.is_cancelled:
                self._log("Cancelled.")
                break

            t0 = time.time()
            tr_metrics = self._train_epoch(epoch, epochs)
            val_metrics = self._val_epoch()
            elapsed = time.time() - t0

            epoch_m = {
                "epoch": epoch + 1,
                "lr": self.opt.param_groups[0]["lr"],
                "elapsed": elapsed,
                **{f"train_{k}": v for k, v in tr_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self._log(
                f"Epoch {epoch + 1}/{epochs}  "
                f"loss={tr_metrics['total']:.4f}  "
                f"val_loss={val_metrics['total']:.4f}  "
                f"val_px={val_metrics['px_error']:.2f}px  "
                f"lr={epoch_m['lr']:.2e}  "
                f"t={elapsed:.1f}s"
            )
            self.on_epoch_end(epoch + 1, epoch_m)

            if (epoch + 1) % cfg["save_every"] == 0 or (epoch + 1) == epochs:
                self._save_checkpoint(epoch + 1)

        self._log("Training complete.")
        self._export_torchscript()

    # ------------------------------------------------------------------
    # Single training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, total_epochs: int) -> Dict:
        self.model.train()
        accum = {k: 0.0 for k in ["total", "param", "photo", "sym"]}
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if StitchTrainer.is_cancelled:
                break

            fi = batch["frame_i"].to(self.device, non_blocking=True)
            fj = batch["frame_j"].to(self.device, non_blocking=True)
            target = batch["params"].to(self.device, non_blocking=True)
            is_neg = batch["is_neg"].to(self.device, non_blocking=True)

            if self.loftr is not None and not is_neg.all():
                target = self._loftr_distill(fi, fj, target, is_neg)

            self.opt.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                pred_ij = self.model(fi, fj)
                pred_ji = self.model(fj, fi)
                losses = self.criterion(pred_ij, pred_ji, target, fi, fj, is_neg)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            self.criterion.step()

            for k in accum:
                accum[k] += losses[k].item()
            n_batches += 1

            if (batch_idx + 1) % self.cfg["log_every"] == 0:
                m = {k: accum[k] / n_batches for k in accum}
                m.update(
                    {
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                        "lr": self.opt.param_groups[0]["lr"],
                    }
                )
                self.on_metrics(m)
                self._log(
                    f"  [{epoch + 1}/{total_epochs}][{batch_idx + 1}/"
                    f"{len(self.train_loader)}]  "
                    f"loss={m['total']:.4f}  param={m['param']:.4f}  "
                    f"photo={m['photo']:.4f}"
                )

        return {k: accum[k] / max(n_batches, 1) for k in accum}

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------

    def _val_epoch(self) -> Dict:
        self.model.eval()
        accum = {k: 0.0 for k in ["total", "param", "photo", "sym"]}
        px_sum = 0.0
        n_batches = 0
        pH, pW = self.cfg["patch_hw"]

        with torch.no_grad():
            for batch in self.val_loader:
                fi = batch["frame_i"].to(self.device)
                fj = batch["frame_j"].to(self.device)
                target = batch["params"].to(self.device)
                is_neg = batch["is_neg"].to(self.device)

                pred_ij = self.model(fi, fj)
                pred_ji = self.model(fj, fi)
                losses = self.criterion(pred_ij, pred_ji, target, fi, fj, is_neg)

                for k in accum:
                    accum[k] += losses[k].item()

                pos = ~is_neg
                if pos.any():
                    err_dx = (pred_ij[pos, 0] - target[pos, 0]).abs() * pW
                    err_dy = (pred_ij[pos, 1] - target[pos, 1]).abs() * pH
                    px_sum += (err_dx + err_dy).mean().item()

                n_batches += 1

        n = max(n_batches, 1)
        return {**{k: accum[k] / n for k in accum}, "px_error": px_sum / n}

    # ------------------------------------------------------------------
    # LoFTR knowledge distillation
    # ------------------------------------------------------------------

    def _loftr_distill(
        self,
        fi: torch.Tensor,
        fj: torch.Tensor,
        target: torch.Tensor,
        is_neg: torch.Tensor,
    ) -> torch.Tensor:
        # relocated: from animation.models.stitch_net.model import AnimeStitchNet

        w = self.cfg["distill_weight"]
        pH, pW = self.cfg["patch_hw"]
        new_target = target.clone()

        fi_np = (fi.cpu().numpy() * 255).astype(np.uint8)
        fj_np = (fj.cpu().numpy() * 255).astype(np.uint8)

        for b in range(fi.shape[0]):
            if is_neg[b]:
                continue
            bgr_i = cv2.cvtColor(fi_np[b, 0], cv2.COLOR_GRAY2BGR)
            bgr_j = cv2.cvtColor(fj_np[b, 0], cv2.COLOR_GRAY2BGR)
            try:
                M, conf = self.loftr.get_affine_partial(bgr_i, bgr_j)
                if M is not None and conf > 0.3:
                    M_t = torch.from_numpy(M).float().to(self.device)
                    lp = AnimeStitchNet.affine_to_params(M_t, pH, pW).squeeze(0)
                    new_target[b] = (1.0 - w) * target[b] + w * lp
            except Exception:
                pass

        return new_target

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimiser": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
            "config": self.cfg,
        }
        path = self.out_dir / f"ckpt_epoch{epoch:03d}.pt"
        torch.save(ckpt, path)
        torch.save(ckpt, self.out_dir / "latest.pt")
        self._log(f"Checkpoint saved → {path.name}")

    def _load_checkpoint(self, path: Path):
        self._log(f"Resuming from {path.name} …")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimiser"])
        self.sched.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"]
        self._log(f"  Resumed at epoch {self.start_epoch}.")

    # ------------------------------------------------------------------
    # TorchScript export
    # ------------------------------------------------------------------

    def _export_torchscript(self):
        try:
            self.model.eval()
            pH, pW = self.cfg["patch_hw"]
            d_i = torch.zeros(1, 1, pH, pW, device=self.device)
            d_j = torch.zeros(1, 1, pH, pW, device=self.device)
            traced = torch.jit.trace(self.model, (d_i, d_j), strict=False)
            ts_path = self.out_dir / "anime_stitch_net.torchscript"
            traced.save(str(ts_path))
            self._log(f"TorchScript exported → {ts_path.name}")
        except Exception as e:
            self._log(f"TorchScript export skipped ({e}).")

    # ------------------------------------------------------------------
    def _log(self, msg: str):
        self.on_log(msg)


# ---------------------------------------------------------------------------
# load_stitch_net — convenience loader for inference
# ---------------------------------------------------------------------------


def load_stitch_net(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> "AnimeStitchNet":
    """
    Load a trained AnimeStitchNet from a .pt checkpoint or .torchscript file.
    Returns the model in eval mode on the requested device.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if checkpoint_path.endswith(".torchscript"):
        model = torch.jit.load(checkpoint_path, map_location=dev)
    else:
        ckpt = torch.load(checkpoint_path, map_location=dev)
        cfg = ckpt.get("config", {})
        model = AnimeStitchNet(
            enc_channels=cfg.get("enc_channels", 256),
            num_heads=cfg.get("num_heads", 8),
            num_ca_layers=cfg.get("num_ca_layers", 2),
            pretrained=False,
        )
        model.load_state_dict(ckpt["model"])
        model.to(dev)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Warmup-Cosine LR scheduler
# ---------------------------------------------------------------------------


class _WarmupCosineScheduler:
    """Linear warmup then cosine annealing — operates per step (not per epoch)."""

    def __init__(self, opt, warmup_steps: int, total_steps: int):
        self.opt = opt
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._step = 0
        self._base_lrs = [g["lr"] for g in opt.param_groups]

    def step(self):
        self._step += 1
        s, w, T = self._step, self.warmup_steps, self.total_steps
        for g, base_lr in zip(self.opt.param_groups, self._base_lrs, strict=False):
            if s < w:
                g["lr"] = base_lr * (s / max(w, 1))
            else:
                progress = (s - w) / max(T - w, 1)
                g["lr"] = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, d: dict):
        self._step = d.get("step", 0)
