from typing import Dict

import cv2
import numpy as np
from PySide6.QtCore import (
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QImage,
    QPixmap,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ....styles import apply_shadow_effect


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


def _apply_color_correction(bgr: np.ndarray, cc: dict) -> np.ndarray:
    """Apply per-frame color correction dict to a BGR uint8 image."""
    if not cc:
        return bgr
    img = bgr.astype(np.float32)
    # Brightness [-100 .. +100]
    img += float(cc.get("brightness", 0))
    # Contrast  [0.1 .. 3.0, 1.0 = no change]
    img = (img - 127.5) * float(cc.get("contrast", 1.0)) + 127.5
    # Saturation [0.0 .. 3.0, 1.0 = no change]
    sat = float(cc.get("saturation", 1.0))
    if sat != 1.0:
        gray = img.mean(axis=2, keepdims=True)
        img = gray + sat * (img - gray)
    # Gamma [0.2 .. 5.0, 1.0 = no change]
    gamma = float(cc.get("gamma", 1.0))
    if gamma != 1.0:
        img = np.sign(img) * (np.abs(img / 255.0) ** (1.0 / gamma)) * 255.0
    # White balance — temperature offset in blue/red channels [-50 .. +50]
    temp = float(cc.get("temperature", 0))
    img[:, :, 0] -= temp  # blue
    img[:, :, 2] += temp  # red
    return np.clip(img, 0, 255).astype(np.uint8)


class _CCSlider(QWidget):
    """Label + QSlider row with live value display."""

    changed = Signal(str, float)  # key, value

    def __init__(
        self,
        label: str,
        key: str,
        lo: float,
        hi: float,
        default: float,
        step: float = 1.0,
    ):
        super().__init__()
        self._key = key
        self._lo = lo
        self._range = hi - lo
        self._step = step
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(f"{label}:")
        lbl.setFixedWidth(90)
        lbl.setStyleSheet("color:#ccc;")
        row.addWidget(lbl)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 200)
        self._slider.setValue(int((default - lo) / (hi - lo) * 200))
        self._slider.valueChanged.connect(self._emit)
        row.addWidget(self._slider, 1)
        self._val_lbl = QLabel(f"{default:.2f}")
        self._val_lbl.setFixedWidth(44)
        self._val_lbl.setStyleSheet("color:#aaa; font-size:10px;")
        row.addWidget(self._val_lbl)

        # Reset button
        self._default = default
        btn = QPushButton("↺")
        btn.setFixedWidth(24)
        btn.setToolTip("Reset to default")
        btn.clicked.connect(self._reset)
        row.addWidget(btn)

    def value(self) -> float:
        return self._lo + self._slider.value() / 200.0 * self._range

    def set_value(self, v: float):
        self._slider.setValue(int((v - self._lo) / self._range * 200))

    def _emit(self):
        v = self.value()
        self._val_lbl.setText(f"{v:.2f}")
        self.changed.emit(self._key, v)

    def _reset(self):
        self.set_value(self._default)


class ColorCorrectionWidget(QWidget):
    """Per-frame color/exposure corrections."""

    corrections_changed = Signal(dict)  # full dict for current frame

    def __init__(self, parent=None):
        super().__init__(parent)
        self._corrections: dict = {}
        self._path: str = ""
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        # Preview
        prev_row = QHBoxLayout()
        self._prev_lbl = QLabel("No image loaded")
        self._prev_lbl.setFixedHeight(160)
        self._prev_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._prev_lbl.setStyleSheet("background:#111; color:#555;")
        self._prev_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        prev_row.addWidget(self._prev_lbl, 1)
        root.addLayout(prev_row)

        # Sliders
        self._sliders: Dict[str, _CCSlider] = {}
        _defs = [
            ("Brightness", "brightness", -100.0, 100.0, 0.0),
            ("Contrast", "contrast", 0.1, 3.0, 1.0),
            ("Saturation", "saturation", 0.0, 3.0, 1.0),
            ("Gamma", "gamma", 0.2, 5.0, 1.0),
            ("Temperature", "temperature", -50.0, 50.0, 0.0),
        ]
        for name, key, lo, hi, default in _defs:
            s = _CCSlider(name, key, lo, hi, default)
            s.changed.connect(self._on_changed)
            self._sliders[key] = s
            root.addWidget(s)

        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset All")
        btn_reset.clicked.connect(self._reset_all)
        apply_shadow_effect(btn_reset, radius=4, y_offset=2)
        btn_row.addWidget(btn_reset)

        btn_match = QPushButton("Match Adjacent →")
        btn_match.setToolTip(
            "Auto-match this frame's histogram to the next frame "
            "(requires both frames to be loaded)."
        )
        btn_match.clicked.connect(self._match_adjacent)
        apply_shadow_effect(btn_match, radius=4, y_offset=2)
        btn_row.addWidget(btn_match)
        btn_row.addStretch()
        root.addLayout(btn_row)
        root.addStretch()

        # Debounce for preview refresh
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._refresh_preview)

    def load_frame(self, path: str, existing_cc: dict | None = None):
        self._path = path
        self._corrections = dict(existing_cc or {})
        for key, slider in self._sliders.items():
            slider.set_value(self._corrections.get(key, slider._default))
        self._refresh_preview()

    def get_corrections(self) -> dict:
        return dict(self._corrections)

    def set_adjacent_path(self, path: str):
        self._adj_path = path

    def _on_changed(self, key: str, value: float):
        self._corrections[key] = value
        self._debounce.start()
        self.corrections_changed.emit(dict(self._corrections))

    def _reset_all(self):
        for slider in self._sliders.values():
            slider._reset()
        self._corrections.clear()
        self._refresh_preview()
        self.corrections_changed.emit({})

    def _match_adjacent(self):
        adj = getattr(self, "_adj_path", "")
        if not self._path or not adj:
            return
        ref = cv2.imread(adj)
        src = cv2.imread(self._path)
        if ref is None or src is None:
            return

        ref_l = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32)
        src_l = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32)

        mean_diff = float(ref_l.mean() - src_l.mean())
        ratio = float(ref_l.std()) / max(float(src_l.std()), 0.1)

        self._sliders["brightness"].set_value(mean_diff)
        ratio = max(0.1, min(3.0, ratio))
        self._sliders["contrast"].set_value(ratio)

    def _refresh_preview(self):
        if not self._path:
            return
        bgr = cv2.imread(self._path)
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        scale = min(1.0, 400 / max(w, h, 1))
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        bgr = _apply_color_correction(bgr, self._corrections)
        qi = _bgr_to_qimage(bgr)
        pm = QPixmap.fromImage(qi)
        self._prev_lbl.setPixmap(
            pm.scaled(
                self._prev_lbl.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
