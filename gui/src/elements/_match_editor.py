"""Draggable-anchor LoFTR match preview scene/view for the Stitch sub-tab.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QSizePolicy,
)

from gui.src.constants import ANCHOR_RADIUS, MAX_DISPLAYED_MATCHES
from ._pixmap_utils import _bgr_to_qpixmap, _conf_color, _mask_to_qpixmap


class _AnchorHandle(QGraphicsEllipseItem):
    def __init__(
        self,
        cx: float,
        cy: float,
        color: QColor,
        moved_cb,
        radius: int = ANCHOR_RADIUS,
    ):
        r = radius
        super().__init__(-r, -r, r * 2, r * 2)
        self._moved_cb = moved_cb
        self.setPos(cx, cy)
        self.setBrush(QBrush(color))
        pen = QPen(Qt.GlobalColor.white)
        pen.setWidth(1)
        self.setPen(pen)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges
            | QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        )
        self.setZValue(10)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged:
            self._moved_cb(self, value)
        return super().itemChange(change, value)


class _MatchScene(QGraphicsScene):
    affine_updated = Signal(object)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._pix_a: Optional[QGraphicsPixmapItem] = None
        self._pix_b: Optional[QGraphicsPixmapItem] = None
        self._offset_x: float = 0.0
        self._scale_a: float = 1.0
        self._scale_b: float = 1.0
        self._orig_h_a = self._orig_w_a = self._orig_h_b = self._orig_w_b = 0
        self._match_lines: list = []
        self._anchors_a: List[_AnchorHandle] = []
        self._anchors_b: List[_AnchorHandle] = []
        self._affine_overlay: Optional[QGraphicsRectItem] = None
        self._hint_label: Optional[QGraphicsTextItem] = None

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(80)
        self._debounce.timeout.connect(self._recompute_affine)

    def load_pair(self, img_a, img_b, orig_h_a, orig_w_a, orig_h_b, orig_w_b):
        self.clear()
        self._match_lines = []
        self._anchors_a = []
        self._anchors_b = []
        self._affine_overlay = None

        pix_a = _bgr_to_qpixmap(img_a)
        pix_b = _bgr_to_qpixmap(img_b)
        self._pix_a = self.addPixmap(pix_a)
        self._pix_a.setPos(0, 0)

        gap = 12
        self._offset_x = float(pix_a.width() + gap)
        self._pix_b = self.addPixmap(pix_b)
        self._pix_b.setPos(self._offset_x, 0)

        self._scale_a = pix_a.width() / orig_w_a if orig_w_a else 1.0
        self._scale_b = pix_b.width() / orig_w_b if orig_w_b else 1.0
        self._orig_h_a, self._orig_w_a = orig_h_a, orig_w_a
        self._orig_h_b, self._orig_w_b = orig_h_b, orig_w_b

        self.setSceneRect(
            QRectF(
                0,
                0,
                self._offset_x + pix_b.width(),
                max(pix_a.height(), pix_b.height()),
            )
        )
        sep_pen = QPen(QColor(120, 120, 120, 200))
        sep_pen.setWidth(2)
        sep = self.addLine(
            self._offset_x - gap / 2,
            0,
            self._offset_x - gap / 2,
            max(pix_a.height(), pix_b.height()),
            sep_pen,
        )
        sep.setZValue(1)

        lbl = self.addText("Click 'Compute Matches' to show correspondences.")
        lbl.setDefaultTextColor(QColor(180, 180, 180))
        lbl.setFont(QFont("monospace", 10))
        lbl.setPos(10, max(pix_a.height(), pix_b.height()) / 2 - 10)
        lbl.setZValue(5)
        self._hint_label = lbl

    def show_matches(self, pts1, pts2, conf):
        if self._hint_label is not None:
            self.removeItem(self._hint_label)
            self._hint_label = None

        for item in self._match_lines + self._anchors_a + self._anchors_b:
            self.removeItem(item)
        self._match_lines = []
        self._anchors_a = []
        self._anchors_b = []

        if len(pts1) > MAX_DISPLAYED_MATCHES:
            idx = np.argsort(conf)[::-1][:MAX_DISPLAYED_MATCHES]
            pts1, pts2, conf = pts1[idx], pts2[idx], conf[idx]

        for p1, p2, c in zip(pts1, pts2, conf, strict=False):
            color = _conf_color(float(c))
            sx_a = float(p1[0]) * self._scale_a
            sy_a = float(p1[1]) * self._scale_a
            sx_b = float(p2[0]) * self._scale_b + self._offset_x
            sy_b = float(p2[1]) * self._scale_b

            line_pen = QPen(color)
            line_pen.setWidthF(1.2)
            line = self.addLine(sx_a, sy_a, sx_b, sy_b, line_pen)
            line.setZValue(3)
            self._match_lines.append(line)

            h_a = _AnchorHandle(sx_a, sy_a, color, self._on_anchor_moved)
            h_b = _AnchorHandle(sx_b, sy_b, color, self._on_anchor_moved)
            self.addItem(h_a)
            self.addItem(h_b)
            self._anchors_a.append(h_a)
            self._anchors_b.append(h_b)

    def show_mask(self, img_a, mask):
        if self._pix_a is None:
            return
        self._pix_a.setPixmap(_mask_to_qpixmap(img_a, mask))

    def _on_anchor_moved(self, handle: _AnchorHandle, _pos):
        idx_a = self._anchors_a.index(handle) if handle in self._anchors_a else -1
        idx_b = self._anchors_b.index(handle) if handle in self._anchors_b else -1
        idx = idx_a if idx_a >= 0 else idx_b
        if 0 <= idx < len(self._match_lines):
            line_item: QGraphicsLineItem = self._match_lines[idx]
            ha = self._anchors_a[idx]
            hb = self._anchors_b[idx]
            line_item.setLine(
                ha.scenePos().x(),
                ha.scenePos().y(),
                hb.scenePos().x(),
                hb.scenePos().y(),
            )
        self._debounce.start()

    def _recompute_affine(self):
        if len(self._anchors_a) < 3:
            self.affine_updated.emit(None)
            return
        pts_a = np.array(
            [
                [h.scenePos().x() / self._scale_a, h.scenePos().y() / self._scale_a]
                for h in self._anchors_a
            ],
            dtype=np.float32,
        )
        pts_b = np.array(
            [
                (
                    (h.scenePos().x() - self._offset_x) / self._scale_b,
                    h.scenePos().y() / self._scale_b,
                )
                for h in self._anchors_b
            ],
            dtype=np.float32,
        )
        M, _ = cv2.estimateAffinePartial2D(
            pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=4.0
        )
        self.affine_updated.emit(M)

    def get_affine_from_anchors(self) -> Optional[np.ndarray]:
        if len(self._anchors_a) < 3:
            return None
        pts_a = np.array(
            [
                [h.scenePos().x() / self._scale_a, h.scenePos().y() / self._scale_a]
                for h in self._anchors_a
            ],
            dtype=np.float32,
        )
        pts_b = np.array(
            [
                (
                    (h.scenePos().x() - self._offset_x) / self._scale_b,
                    h.scenePos().y() / self._scale_b,
                )
                for h in self._anchors_b
            ],
            dtype=np.float32,
        )
        M, _ = cv2.estimateAffinePartial2D(
            pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=4.0
        )
        return M


class _MatchView(QGraphicsView):
    def __init__(self, scene: _MatchScene):
        super().__init__(scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)

    def fit(self):
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


__all__ = ["_AnchorHandle", "_MatchScene", "_MatchView"]
