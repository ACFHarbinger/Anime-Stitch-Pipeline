"""BGR/QImage/QPixmap conversion helpers shared by the match editor.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap

from gui.src.constants import CONF_HIGH, CONF_LOW, CONF_MED


def _conf_color(c: float) -> QColor:
    if c >= 0.7:
        return CONF_HIGH
    if c >= 0.5:
        return CONF_MED
    return CONF_LOW


def _bgr_to_qpixmap(bgr: np.ndarray, max_dim: int = 600) -> QPixmap:
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        bgr = cv2.resize(
            bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h2, w2 = rgb.shape[:2]
    qi = QImage(rgb.data, w2, h2, 3 * w2, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi.copy())


def _mask_to_qpixmap(bgr: np.ndarray, mask: np.ndarray, max_dim: int = 600) -> QPixmap:
    overlay = bgr.copy()
    fg = mask < 128
    overlay[fg] = (
        (overlay[fg] * 0.3 + np.array([180, 60, 60]) * 0.7)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return _bgr_to_qpixmap(overlay, max_dim)


def _qimage_to_qpixmap(qi: QImage, max_dim: int = 0) -> QPixmap:
    px = QPixmap.fromImage(qi)
    if max_dim > 0 and max(px.width(), px.height()) > max_dim:
        px = px.scaled(
            max_dim,
            max_dim,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return px


__all__ = ["_conf_color", "_bgr_to_qpixmap", "_mask_to_qpixmap", "_qimage_to_qpixmap"]
