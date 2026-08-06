"""Off-thread thumbnail / metrics QRunnable workers shared across every panel.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, Signal
from PySide6.QtGui import QImage


class _ThumbHub(QObject):
    loaded = Signal(str, int, object)  # path, generation, QImage


class _MetricsSignals(QObject):
    ready = Signal(str)  # formatted metrics string


class _MetricsTask(QRunnable):
    """Off-thread Laplacian sharpness + file-size metrics for the result preview overlay."""

    def __init__(self, path: str, signals: _MetricsSignals):
        super().__init__()
        self._path = path
        self._signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            stat_size = os.stat(self._path).st_size / (1024 * 1024)
            img_gray = cv2.imread(self._path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                self._signals.ready.emit(f"Size: {stat_size:.1f} MB")
                return
            h, w = img_gray.shape
            lap_var = float(np.var(cv2.Laplacian(img_gray, cv2.CV_64F)))
            self._signals.ready.emit(
                f"{w}×{h}  |  {stat_size:.1f} MB  |  Sharpness: {lap_var:.0f}"
            )
        except Exception:
            self._signals.ready.emit("")


class _ThumbTask(QRunnable):
    def __init__(self, path: str, size: int, generation: int, hub: "_ThumbHub"):
        super().__init__()
        self._path = path
        self._size = size
        self._gen = generation
        self._hub = hub
        self.setAutoDelete(True)

    def run(self):
        img = QImage(self._path)
        if not img.isNull():
            img = img.scaled(
                self._size,
                self._size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._hub.loaded.emit(self._path, self._gen, img)


__all__ = ["_ThumbHub", "_MetricsSignals", "_MetricsTask", "_ThumbTask"]
