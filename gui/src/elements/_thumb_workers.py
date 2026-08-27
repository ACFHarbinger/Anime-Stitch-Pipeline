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


class _ScansComparisonSignals(QObject):
    """Deliver the generated SCANS image and its quality readouts to the UI."""

    ready = Signal(str, object)  # output path, {"ASP": str, "SCANS": str}
    failed = Signal(str)


class _ScansComparisonTask(QRunnable):
    """Generate a SCANS baseline off the GUI thread after an ASP stitch."""

    def __init__(
        self,
        frame_paths: list[str],
        asp_output_path: str,
        scans_output_path: str,
        signals: _ScansComparisonSignals,
    ) -> None:
        super().__init__()
        self._frame_paths = frame_paths
        self._asp_output_path = asp_output_path
        self._scans_output_path = scans_output_path
        self._signals = signals
        self.setAutoDelete(True)

    @staticmethod
    def _quality_metrics(path: str) -> str:
        from asp_backend.core.pipeline.safety_metrics import (
            ghosting_score_v2,
            seam_visibility_score,
        )

        image = cv2.imread(path)
        if image is None:
            return "Metrics unavailable"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        ghosting = ghosting_score_v2(image)
        seam_gradient = seam_visibility_score(image)
        return (
            f"Sharpness: {sharpness:.0f}  |  Ghosting: {ghosting:.1f}  |  "
            f"Seam gradient: {seam_gradient:.1f}"
        )

    def run(self) -> None:
        try:
            from asp_backend.alignment.canvas import _scan_stitch_fallback

            frames = [cv2.imread(path) for path in self._frame_paths]
            usable_frames = [frame for frame in frames if frame is not None]
            if len(usable_frames) < 2:
                raise RuntimeError("At least two readable source frames are required.")
            _scan_stitch_fallback(usable_frames, self._scans_output_path)
            self._signals.ready.emit(
                self._scans_output_path,
                {
                    "ASP": self._quality_metrics(self._asp_output_path),
                    "SCANS": self._quality_metrics(self._scans_output_path),
                },
            )
        except Exception as exc:
            self._signals.failed.emit(str(exc))


class _ThumbTask(QRunnable):
    def __init__(self, path: str, size: int, generation: int, hub: _ThumbHub):
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


__all__ = [
    "_ThumbHub",
    "_MetricsSignals",
    "_MetricsTask",
    "_ScansComparisonSignals",
    "_ScansComparisonTask",
    "_ThumbTask",
]
