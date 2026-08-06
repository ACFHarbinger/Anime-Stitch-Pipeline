"""HITL Checkpoint 3.5 — Interactive seam boundary editor.

Shows a downsampled canvas preview with N-1 draggable horizontal lines
representing the ownership boundaries between adjacent frames.  The user
drags lines to place each seam where the character overlap looks best, then
clicks "Resume Pipeline" to pass the adjusted boundaries back to the worker.

The dialog is purely for vertical-scroll stitches (boundaries are horizontal
y-coordinates).  For horizontal scrolls `_composite_foreground` short-circuits
before calling `_find_optimal_boundaries`, so this checkpoint is never reached.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)
from gui.src.constants.elements import DIALOG_BOUNDARY_EDITOR_DIALOG__MAX_PREVIEW_H, DIALOG__MAX_PREVIEW_W, _LABEL_COLOR, _LINE_COLOR


def _bgr_to_pixmap(bgr: np.ndarray, max_w: int, max_h: int) -> tuple[QPixmap, float]:
    """Resize a BGR frame to fit within max_w × max_h, return (pixmap, scale)."""
    h, w = bgr.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h2, w2 = rgb.shape[:2]
    qimg = QImage(rgb.data, w2, h2, 3 * w2, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg), scale


class _DraggableLine(QGraphicsLineItem):
    """Horizontal line that can be dragged vertically within canvas bounds."""

    def __init__(self, y: float, width: float, index: int, canvas_h: float):
        super().__init__(0, y, width, y)
        self._index = index
        self._canvas_h = canvas_h
        pen = QPen(_LINE_COLOR)
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.CursorShape.SizeVerCursor)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            new_pos = QPointF(0.0, value.y())
            new_pos.setY(max(0.0, min(new_pos.y(), self._canvas_h - 1)))
            return new_pos
        return super().itemChange(change, value)

    def current_y(self) -> float:
        return self.y() + self.line().y1()


class BoundaryEditorDialog(QDialog):
    """Show a canvas preview with draggable seam-boundary lines."""

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HITL: Seam Boundary Editor — Checkpoint 3.5")
        self.resize(560, 780)

        canvas_bgr: np.ndarray = data["canvas_preview"]
        self._boundaries_full: List[float] = list(data["boundaries"])
        self._canvas_h_full: int = int(data["canvas_h"])
        self._canvas_w_full: int = int(data["canvas_w"])
        frame_count: int = int(data.get("frame_count", len(self._boundaries_full) + 1))

        # Downsample canvas for display
        pixmap, self._scale = _bgr_to_pixmap(canvas_bgr, DIALOG__MAX_PREVIEW_W, DIALOG_BOUNDARY_EDITOR_DIALOG__MAX_PREVIEW_H)
        self._preview_w = pixmap.width()
        self._preview_h = pixmap.height()

        # Scale boundary y-coordinates to preview space
        boundaries_px = [b * self._scale for b in self._boundaries_full]

        self._scene = QGraphicsScene(self)
        bg_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(bg_item)

        # Add draggable lines
        self._lines: List[_DraggableLine] = []
        for idx, by in enumerate(boundaries_px):
            line = _DraggableLine(by, self._preview_w, idx, self._preview_h)
            self._scene.addItem(line)
            self._lines.append(line)

            # Label: "← frame i | frame i+1 →"
            label = self._scene.addText(f"▸ seam {idx + 1}", QFont("monospace", 8))
            label.setDefaultTextColor(_LABEL_COLOR)
            label.setPos(4, by - 16)

        # Frame-range labels along the left edge
        # tys_sorted = sorted(data.get("frame_tys_scaled", []))
        for fi in range(frame_count):
            lbl_y = (
                boundaries_px[fi - 1] if fi > 0 else 0
            ) + (
                (boundaries_px[fi] if fi < len(boundaries_px) else self._preview_h)
                - (boundaries_px[fi - 1] if fi > 0 else 0)
            ) / 2
            txt = self._scene.addText(f"F{fi}", QFont("monospace", 7))
            txt.setDefaultTextColor(QColor(180, 220, 255))
            txt.setPos(self._preview_w - 30, lbl_y - 8)

        self._view = QGraphicsView(self._scene)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._view.setFixedWidth(self._preview_w + 4)
        self._view.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._view.setSceneRect(QRectF(0, 0, self._preview_w, self._preview_h))

        self._status = QLabel(
            f"{len(self._lines)} seam boundaries — drag red lines to adjust"
        )
        self._status.setStyleSheet("color: #aaa; font-size: 10px;")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_reset = QPushButton("Reset to Auto")
        btn_reset.clicked.connect(self._reset)
        btn_resume = QPushButton("Resume Pipeline")
        btn_resume.setDefault(True)
        btn_resume.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        btn_row.addWidget(btn_resume)
        btn_row.addWidget(btn_cancel)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(self._status)
        root.addWidget(self._view, stretch=1)
        root.addLayout(btn_row)

    def _reset(self):
        boundaries_px = [b * self._scale for b in self._boundaries_full]
        for line, by in zip(self._lines, boundaries_px, strict=False):
            line.setY(0)
            line.setLine(0, by, self._preview_w, by)

    def adjusted_boundaries(self) -> List[float]:
        """Return boundary y-coordinates scaled back to full canvas space."""
        result = []
        for line in self._lines:
            y_preview = line.current_y()
            result.append(y_preview / max(self._scale, 1e-6))
        return result
