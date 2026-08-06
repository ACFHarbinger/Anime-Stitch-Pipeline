"""
gui/src/tabs/models/train/stitch_train_tab.py
===============================================
Training tab for AnimeStitchNet.

Follows the same pattern as GANTrainTab / LoRATrainTab:
  • Uses collect() / set_config() for UnifiedTrainTab integration.
  • Runs StitchTrainer in a daemon thread, communicates via Qt Signals.
  • Shows a live rolling mini-chart and per-epoch progress bar.
"""

from __future__ import annotations

import math
import threading

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from backend.src.pipeline.stitch_trainer import DEFAULT_CONFIG, StitchTrainer

    _TRAINER_OK = True
except ImportError:
    _TRAINER_OK = False
    DEFAULT_CONFIG = {}


# ---------------------------------------------------------------------------
# Mini loss chart (unicode block characters)
# ---------------------------------------------------------------------------


class _MiniChart(QLabel):
    MAX_POINTS = 64

    def __init__(self):
        super().__init__()
        self._values: list = []
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(20)
        self.setStyleSheet("font-family: monospace; font-size: 11px;")

    def push(self, v: float):
        self._values.append(v)
        if len(self._values) > self.MAX_POINTS:
            self._values.pop(0)
        self._redraw()

    def _redraw(self):
        if not self._values:
            return
        lo, hi = min(self._values), max(self._values)
        span = hi - lo or 1e-9
        blocks = " ▁▂▃▄▅▆▇█"
        bar = "".join(blocks[min(8, int(((v - lo) / span) * 8))] for v in self._values)
        self.setText(f"loss  {bar}  {self._values[-1]:.4f}")


# ---------------------------------------------------------------------------
# StitchTrainTab
# ---------------------------------------------------------------------------


class StitchTrainTab(QWidget):
    """Training tab for AnimeStitchNet — integrates into UnifiedTrainTab."""

    sig_log = Signal(str)
    sig_metrics = Signal(dict)
    sig_epoch = Signal(int, dict)
    sig_done = Signal(str, str)  # (status: 'ok'|'error'|'cancel', message)

    def __init__(self):
        super().__init__()
        self._thread: threading.Thread | None = None
        self._init_ui()
        self.sig_log.connect(self._on_log)
        self.sig_metrics.connect(self._on_metrics)
        self.sig_epoch.connect(self._on_epoch)
        self.sig_done.connect(self._on_done)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Dataset ───────────────────────────────────────────────────────
        dg = QGroupBox("Dataset")
        dgl = QFormLayout(dg)

        self._img_dir = QLineEdit()
        self._img_dir.setPlaceholderText("Folder of anime PNG/JPG frames")
        btn_img = QPushButton("Browse…")
        btn_img.setFixedWidth(80)
        btn_img.clicked.connect(lambda: self._browse_dir(self._img_dir))
        row_img = QHBoxLayout()
        row_img.addWidget(self._img_dir)
        row_img.addWidget(btn_img)
        dgl.addRow("Image dir:", row_img)

        self._ckpt_dir = QLineEdit("stitch_checkpoints")
        btn_ckpt = QPushButton("Browse…")
        btn_ckpt.setFixedWidth(80)
        btn_ckpt.clicked.connect(lambda: self._browse_dir(self._ckpt_dir))
        row_ckpt = QHBoxLayout()
        row_ckpt.addWidget(self._ckpt_dir)
        row_ckpt.addWidget(btn_ckpt)
        dgl.addRow("Output dir:", row_ckpt)

        self._dataset_size = QSpinBox()
        self._dataset_size.setRange(1_000, 1_000_000)
        self._dataset_size.setValue(DEFAULT_CONFIG.get("dataset_size", 50_000))
        self._dataset_size.setSingleStep(5_000)
        dgl.addRow("Dataset size:", self._dataset_size)

        self._val_split = QDoubleSpinBox()
        self._val_split.setRange(0.01, 0.30)
        self._val_split.setValue(DEFAULT_CONFIG.get("val_split", 0.10))
        self._val_split.setDecimals(2)
        self._val_split.setSingleStep(0.05)
        dgl.addRow("Val split:", self._val_split)
        root.addWidget(dg)

        # ── Augmentation ──────────────────────────────────────────────────
        ag = QGroupBox("Augmentation")
        agl = QFormLayout(ag)

        self._patch_h = QSpinBox()
        self._patch_h.setRange(64, 512)
        self._patch_h.setValue(256)
        self._patch_h.setSingleStep(32)
        self._patch_w = QSpinBox()
        self._patch_w.setRange(64, 512)
        self._patch_w.setValue(256)
        self._patch_w.setSingleStep(32)
        pw_row = QHBoxLayout()
        pw_row.addWidget(self._patch_h)
        pw_row.addWidget(QLabel("×"))
        pw_row.addWidget(self._patch_w)
        agl.addRow("Patch H × W:", pw_row)

        self._max_dx = QDoubleSpinBox()
        self._max_dx.setRange(0.05, 1.0)
        self._max_dx.setValue(0.50)
        self._max_dx.setDecimals(2)
        agl.addRow("Max |dx| (frac):", self._max_dx)

        self._max_dy = QDoubleSpinBox()
        self._max_dy.setRange(0.05, 1.0)
        self._max_dy.setValue(0.50)
        self._max_dy.setDecimals(2)
        agl.addRow("Max |dy| (frac):", self._max_dy)

        self._max_angle_deg = QSpinBox()
        self._max_angle_deg.setRange(1, 45)
        self._max_angle_deg.setValue(30)
        agl.addRow("Max angle (°):", self._max_angle_deg)

        self._mpeg_prob = QDoubleSpinBox()
        self._mpeg_prob.setRange(0.0, 1.0)
        self._mpeg_prob.setValue(0.30)
        self._mpeg_prob.setDecimals(2)
        agl.addRow("MPEG noise prob:", self._mpeg_prob)

        self._dimming_prob = QDoubleSpinBox()
        self._dimming_prob.setRange(0.0, 1.0)
        self._dimming_prob.setValue(0.40)
        self._dimming_prob.setDecimals(2)
        agl.addRow("Dimming prob:", self._dimming_prob)

        self._neg_prob = QDoubleSpinBox()
        self._neg_prob.setRange(0.0, 0.50)
        self._neg_prob.setValue(0.10)
        self._neg_prob.setDecimals(2)
        agl.addRow("Neg pair prob:", self._neg_prob)
        root.addWidget(ag)

        # ── Model ─────────────────────────────────────────────────────────
        mg = QGroupBox("Model")
        mgl = QFormLayout(mg)

        self._enc_channels = QSpinBox()
        self._enc_channels.setRange(64, 512)
        self._enc_channels.setValue(256)
        self._enc_channels.setSingleStep(64)
        mgl.addRow("Encoder channels:", self._enc_channels)

        self._num_heads = QComboBox()
        for h in [4, 8, 16]:
            self._num_heads.addItem(str(h), h)
        self._num_heads.setCurrentIndex(1)  # 8
        mgl.addRow("Attn heads:", self._num_heads)

        self._num_ca_layers = QSpinBox()
        self._num_ca_layers.setRange(1, 6)
        self._num_ca_layers.setValue(2)
        mgl.addRow("Cross-attn layers:", self._num_ca_layers)

        self._pretrained = QCheckBox("Use ImageNet pretrained backbone")
        self._pretrained.setChecked(True)
        mgl.addRow("", self._pretrained)
        root.addWidget(mg)

        # ── Training ──────────────────────────────────────────────────────
        tg = QGroupBox("Training")
        tgl = QFormLayout(tg)

        self._epochs = QSpinBox()
        self._epochs.setRange(1, 500)
        self._epochs.setValue(30)
        tgl.addRow("Epochs:", self._epochs)

        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 256)
        self._batch_size.setValue(32)
        tgl.addRow("Batch size:", self._batch_size)

        self._lr = QDoubleSpinBox()
        self._lr.setRange(1e-6, 1e-2)
        self._lr.setValue(3e-4)
        self._lr.setDecimals(6)
        tgl.addRow("Learning rate:", self._lr)

        self._warmup_epochs = QSpinBox()
        self._warmup_epochs.setRange(0, 20)
        self._warmup_epochs.setValue(2)
        tgl.addRow("Warmup epochs:", self._warmup_epochs)

        self._num_workers = QSpinBox()
        self._num_workers.setRange(0, 16)
        self._num_workers.setValue(4)
        tgl.addRow("DataLoader workers:", self._num_workers)

        self._amp_cb = QCheckBox("Mixed precision (AMP)")
        self._amp_cb.setChecked(True)
        tgl.addRow("", self._amp_cb)

        self._loftr_distill_cb = QCheckBox("LoFTR knowledge distillation")
        self._loftr_distill_cb.setChecked(False)
        tgl.addRow("", self._loftr_distill_cb)
        root.addWidget(tg)

        # ── Loss weights ──────────────────────────────────────────────────
        lg = QGroupBox("Loss Weights")
        lgl = QFormLayout(lg)

        self._l_param = QDoubleSpinBox()
        self._l_param.setRange(0.0, 10.0)
        self._l_param.setValue(1.0)
        self._l_param.setDecimals(2)
        lgl.addRow("λ param (affine regression):", self._l_param)

        self._l_photo = QDoubleSpinBox()
        self._l_photo.setRange(0.0, 10.0)
        self._l_photo.setValue(0.5)
        self._l_photo.setDecimals(2)
        lgl.addRow("λ photo (ZNCC photometric):", self._l_photo)

        self._l_sym = QDoubleSpinBox()
        self._l_sym.setRange(0.0, 5.0)
        self._l_sym.setValue(0.2)
        self._l_sym.setDecimals(2)
        lgl.addRow("λ sym (forward/backward):", self._l_sym)
        root.addWidget(lg)

        # ── Control buttons ───────────────────────────────────────────────
        ctrl = QHBoxLayout()
        self._btn_start = QPushButton("▶  Start Training")
        self._btn_cancel = QPushButton("■  Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_start.setStyleSheet(
            "background:#4CAF50; color:white; font-weight:bold; padding:8px 16px;"
        )
        self._btn_cancel.setStyleSheet(
            "background:#f44336; color:white; font-weight:bold; padding:8px 16px;"
        )
        self._btn_start.clicked.connect(self._start)
        self._btn_cancel.clicked.connect(self._cancel)
        ctrl.addWidget(self._btn_start)
        ctrl.addWidget(self._btn_cancel)
        root.addLayout(ctrl)

        # ── Progress ──────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, self._epochs.value())
        self._progress.setValue(0)
        root.addWidget(self._progress)

        self._chart = _MiniChart()
        root.addWidget(self._chart)

        # ── Log ───────────────────────────────────────────────────────────
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setFixedHeight(180)
        self._log_box.setStyleSheet("font-family: monospace; font-size: 11px;")
        root.addWidget(self._log_box)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(str)
    def _on_log(self, msg: str):
        self._log_box.append(msg)
        self._log_box.verticalScrollBar().setValue(
            self._log_box.verticalScrollBar().maximum()
        )

    @Slot(dict)
    def _on_metrics(self, m: dict):
        if "total" in m:
            self._chart.push(float(m["total"]))

    @Slot(int, dict)
    def _on_epoch(self, epoch: int, m: dict):
        self._progress.setMaximum(self._epochs.value())
        self._progress.setValue(epoch)

    @Slot(str, str)
    def _on_done(self, status: str, msg: str):
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        if _TRAINER_OK:
            StitchTrainer.is_cancelled = False
        if status == "ok":
            QMessageBox.information(self, "Training complete", msg)
        elif status == "cancel":
            QMessageBox.warning(self, "Cancelled", msg)
        else:
            QMessageBox.critical(self, "Error", msg)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse_dir(self, line_edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select directory")
        if d:
            line_edit.setText(d)

    def _start(self):
        if not _TRAINER_OK:
            QMessageBox.critical(
                self,
                "Missing dependencies",
                "Could not import StitchTrainer.  "
                "Ensure PyTorch and torchvision are installed.",
            )
            return

        img_dir = self._img_dir.text().strip()
        if not img_dir:
            QMessageBox.warning(
                self, "No directory", "Please select an image directory."
            )
            return

        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setValue(0)
        self._progress.setMaximum(self._epochs.value())
        self._log_box.clear()

        config = self.collect()
        trainer = StitchTrainer(
            config=config,
            on_log=lambda m: self.sig_log.emit(m),
            on_metrics=lambda m: self.sig_metrics.emit(m),
            on_epoch_end=lambda e, m: self.sig_epoch.emit(e, m),
        )

        def _run():
            try:
                trainer.train()
                self.sig_done.emit("ok", "Training finished successfully.")
            except Exception as exc:
                if StitchTrainer.is_cancelled:
                    self.sig_done.emit("cancel", "Training cancelled by user.")
                else:
                    self.sig_done.emit("error", str(exc))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _cancel(self):
        if _TRAINER_OK:
            StitchTrainer.cancel()
        self._btn_cancel.setEnabled(False)
        self._on_log("Cancellation requested…")

    # ------------------------------------------------------------------
    # UnifiedTrainTab integration
    # ------------------------------------------------------------------

    def collect(self) -> dict:
        return {
            "image_dir": self._img_dir.text().strip(),
            "output_dir": self._ckpt_dir.text().strip(),
            "dataset_size": self._dataset_size.value(),
            "val_split": self._val_split.value(),
            "patch_hw": [self._patch_h.value(), self._patch_w.value()],
            "max_dx": self._max_dx.value(),
            "max_dy": self._max_dy.value(),
            "max_angle": math.radians(self._max_angle_deg.value()),
            "mpeg_noise_prob": self._mpeg_prob.value(),
            "dimming_prob": self._dimming_prob.value(),
            "neg_pair_prob": self._neg_prob.value(),
            "enc_channels": self._enc_channels.value(),
            "num_heads": self._num_heads.currentData(),
            "num_ca_layers": self._num_ca_layers.value(),
            "pretrained": self._pretrained.isChecked(),
            "epochs": self._epochs.value(),
            "batch_size": self._batch_size.value(),
            "lr": self._lr.value(),
            "warmup_epochs": self._warmup_epochs.value(),
            "num_workers": self._num_workers.value(),
            "amp": self._amp_cb.isChecked(),
            "loftr_distill": self._loftr_distill_cb.isChecked(),
            "lambda_param": self._l_param.value(),
            "lambda_photo": self._l_photo.value(),
            "lambda_sym": self._l_sym.value(),
        }

    def set_config(self, cfg: dict):
        if "image_dir" in cfg:
            self._img_dir.setText(cfg["image_dir"])
        if "output_dir" in cfg:
            self._ckpt_dir.setText(cfg["output_dir"])
        if "dataset_size" in cfg:
            self._dataset_size.setValue(cfg["dataset_size"])
        if "val_split" in cfg:
            self._val_split.setValue(cfg["val_split"])
        if "epochs" in cfg:
            self._epochs.setValue(cfg["epochs"])
        if "batch_size" in cfg:
            self._batch_size.setValue(cfg["batch_size"])
        if "lr" in cfg:
            self._lr.setValue(cfg["lr"])
        if "lambda_param" in cfg:
            self._l_param.setValue(cfg["lambda_param"])
        if "lambda_photo" in cfg:
            self._l_photo.setValue(cfg["lambda_photo"])
        if "lambda_sym" in cfg:
            self._l_sym.setValue(cfg["lambda_sym"])

    # For BaseGenerativeTab compatibility
    def get_default_config(self) -> dict:
        return self.collect()
