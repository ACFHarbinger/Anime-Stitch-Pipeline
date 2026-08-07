from __future__ import annotations

import copy
import math
import os
from collections.abc import Callable

import cv2
import numpy as np
from gui.src.constants.elements import _FRAME_COLORS, _HIGHLIGHT_PEN
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


def _bgr_thumb_to_qpixmap(bgr: np.ndarray, w: int, h: int) -> QPixmap:
    thumb = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class _DraggableFrameItem(QGraphicsRectItem):
    """A draggable canvas frame rectangle that keeps _nudges in sync on drag."""

    def __init__(
        self,
        idx: int,
        frame_w: float,
        frame_h: float,
        base_tx: float,
        base_ty: float,
        nudge_list: list[list[float]],
        on_select: Callable[[int], None],
    ):
        super().__init__(0.0, 0.0, frame_w, frame_h)
        self._idx = idx
        self._base_tx = base_tx
        self._base_ty = base_ty
        self._nudge_list = nudge_list
        self._on_select = on_select

        self.setPos(base_tx + nudge_list[idx][0], base_ty + nudge_list[idx][1])
        self.setTransformOriginPoint(frame_w / 2.0, frame_h / 2.0)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(1)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            new_x: float = value.x()
            new_y: float = value.y()
            self._nudge_list[self._idx][0] = new_x - self._base_tx
            self._nudge_list[self._idx][1] = new_y - self._base_ty
            return value
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange and value:
            self._on_select(self._idx)
        return super().itemChange(change, value)


class CanvasInspectorDialog(QDialog):
    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Canvas Layout Inspector — Stage 8 (HITL)")
        self.resize(1040, 680)

        self._canvas_h: int = int(data["canvas_h"])
        self._canvas_w: int = int(data["canvas_w"])
        self._frame_h: int = int(data["frame_h"])
        self._frame_w: int = int(data["frame_w"])
        self._affines: list[list[list[float]]] = copy.deepcopy(data["affines"])
        self._image_paths: list[str] = list(data.get("image_paths", []))
        self._thumbnails: list[np.ndarray | None] = list(data.get("thumbnails", []))
        n = len(self._affines)
        self._nudges: list[list[float]] = [[0.0, 0.0] for _ in range(n)]
        self._rot_angles: list[float] = [0.0] * n
        self._scale_factors: list[float] = [1.0] * n
        self._drag_items: list[_DraggableFrameItem] = []
        self._selected_idx: int | None = None
        self._suppress_list_signal: bool = False

        self._build_ui()
        self._populate_scene()
        self._populate_list()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QColor(18, 18, 18))
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        splitter.addWidget(self._view)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        self._list = QListWidget()
        self._list.setMaximumWidth(220)
        self._list.currentRowChanged.connect(self._on_list_row_changed)
        right_layout.addWidget(self._list, stretch=1)

        self._tx_label = QLabel("tx: —  ty: —")
        self._tx_label.setStyleSheet("font-size: 10px; color: #ccc;")
        right_layout.addWidget(self._tx_label)

        # Nudge step control
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step (px):"))
        self._step_spin = QSpinBox()
        self._step_spin.setRange(1, 200)
        self._step_spin.setValue(10)
        self._step_spin.setMaximumWidth(70)
        step_row.addWidget(self._step_spin)
        step_row.addStretch()
        right_layout.addLayout(step_row)

        nudge_grid = QWidget()
        ng = QVBoxLayout(nudge_grid)
        ng.setSpacing(2)

        row_up = QHBoxLayout()
        btn_up = QPushButton("Nudge Up")
        btn_up.clicked.connect(lambda: self._nudge(0, -self._step_spin.value()))
        row_up.addStretch()
        row_up.addWidget(btn_up)
        row_up.addStretch()
        ng.addLayout(row_up)

        row_lr = QHBoxLayout()
        btn_left = QPushButton("Nudge Left")
        btn_left.clicked.connect(lambda: self._nudge(-self._step_spin.value(), 0))
        btn_right = QPushButton("Nudge Right")
        btn_right.clicked.connect(lambda: self._nudge(self._step_spin.value(), 0))
        row_lr.addWidget(btn_left)
        row_lr.addWidget(btn_right)
        ng.addLayout(row_lr)

        row_dn = QHBoxLayout()
        btn_dn = QPushButton("Nudge Down")
        btn_dn.clicked.connect(lambda: self._nudge(0, self._step_spin.value()))
        row_dn.addStretch()
        row_dn.addWidget(btn_dn)
        row_dn.addStretch()
        ng.addLayout(row_dn)

        right_layout.addWidget(nudge_grid)

        # Rotation / scale controls
        rot_scale_box = QWidget()
        rs_layout = QVBoxLayout(rot_scale_box)
        rs_layout.setContentsMargins(0, 4, 0, 0)
        rs_layout.setSpacing(3)

        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Rotation (°):"))
        self._rot_spin = QDoubleSpinBox()
        self._rot_spin.setRange(-180.0, 180.0)
        self._rot_spin.setSingleStep(0.5)
        self._rot_spin.setDecimals(1)
        self._rot_spin.setValue(0.0)
        self._rot_spin.setEnabled(False)
        self._rot_spin.setMaximumWidth(80)
        rot_row.addWidget(self._rot_spin)
        rs_layout.addLayout(rot_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale:"))
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.1, 3.0)
        self._scale_spin.setSingleStep(0.01)
        self._scale_spin.setDecimals(3)
        self._scale_spin.setValue(1.0)
        self._scale_spin.setEnabled(False)
        self._scale_spin.setMaximumWidth(80)
        scale_row.addWidget(self._scale_spin)
        rs_layout.addLayout(scale_row)

        right_layout.addWidget(rot_scale_box)

        self._rot_spin.valueChanged.connect(self._on_rot_changed)
        self._scale_spin.valueChanged.connect(self._on_scale_changed)

        btn_reset = QPushButton("Reset Frame")
        btn_reset.clicked.connect(self._reset_frame)
        right_layout.addWidget(btn_reset)

        splitter.addWidget(right)
        splitter.setSizes([780, 220])

        root.addWidget(splitter, stretch=1)

        bottom = QHBoxLayout()
        btn_resume = QPushButton("Resume Pipeline")
        btn_resume.setDefault(True)
        btn_resume.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        bottom.addStretch()
        bottom.addWidget(btn_resume)
        bottom.addWidget(btn_cancel)
        root.addLayout(bottom)

    def _populate_scene(self):
        self._scene.clear()
        self._drag_items = []

        border_pen = QPen(QColor(80, 80, 80, 200))
        border_pen.setWidth(2)
        self._scene.addRect(0, 0, self._canvas_w, self._canvas_h, border_pen, QColor(0, 0, 0, 0))

        fh, fw = float(self._frame_h), float(self._frame_w)
        for idx, aff in enumerate(self._affines):
            base_tx = aff[0][2]
            base_ty = aff[1][2]

            color = _FRAME_COLORS[idx % len(_FRAME_COLORS)]
            pen = QPen(color.darker(150))
            pen.setWidth(2)

            drag_item = _DraggableFrameItem(
                idx=idx,
                frame_w=fw,
                frame_h=fh,
                base_tx=base_tx,
                base_ty=base_ty,
                nudge_list=self._nudges,
                on_select=self._on_scene_frame_clicked,
            )
            drag_item.setPen(pen)
            drag_item.setBrush(QBrush(color))

            # Apply stored rotation/scale visually
            drag_item.setRotation(self._rot_angles[idx])
            drag_item.setScale(self._scale_factors[idx])

            self._scene.addItem(drag_item)

            # Thumbnail as child item (moves/rotates/scales with the frame rect)
            if idx < len(self._thumbnails) and self._thumbnails[idx] is not None:
                try:
                    pix = _bgr_thumb_to_qpixmap(self._thumbnails[idx], int(fw), int(fh)) # pyrefly: ignore[bad-argument-type]
                    pix_item = QGraphicsPixmapItem(pix, drag_item)
                    pix_item.setOpacity(0.6)
                    pix_item.setPos(0.0, 0.0)
                    drag_item.setBrush(QBrush(QColor(0, 0, 0, 0)))
                except Exception:
                    pass

            lbl = self._scene.addSimpleText(str(idx))
            lbl.setBrush(QBrush(QColor(255, 255, 255, 210)))
            cur_tx = base_tx + self._nudges[idx][0]
            cur_ty = base_ty + self._nudges[idx][1]
            lbl.setPos(
                cur_tx + fw / 2 - lbl.boundingRect().width() / 2,
                cur_ty + fh / 2 - lbl.boundingRect().height() / 2,
            )
            lbl.setZValue(2)

            self._drag_items.append(drag_item)

        self._view.fitInView(
            self._scene.itemsBoundingRect().adjusted(-10, -10, 10, 10),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._apply_highlight()

    def _populate_list(self):
        self._list.blockSignals(True)
        self._list.clear()
        for idx in range(len(self._affines)):
            name = ""
            if idx < len(self._image_paths):
                name = os.path.basename(self._image_paths[idx])
            item = QListWidgetItem(f"{idx}: {name}")
            self._list.addItem(item)
        self._list.blockSignals(False)

    def _on_list_row_changed(self, row: int):
        self._selected_idx = row if 0 <= row < len(self._affines) else None
        self._sync_scene_selection()
        self._apply_highlight()
        self._update_tx_label()
        self._update_transform_controls()

    def _on_scene_frame_clicked(self, idx: int):
        """Called by _DraggableFrameItem when it becomes selected in the scene."""
        self._selected_idx = idx
        self._update_tx_label()
        self._update_transform_controls()
        self._suppress_list_signal = True
        try:
            self._list.setCurrentRow(idx)
        finally:
            self._suppress_list_signal = False
        self._apply_highlight()

    def _on_scene_selection_changed(self):
        """Keep tx_label updated when a drag ends or selection changes in scene."""
        selected = self._scene.selectedItems()
        if selected:
            item = selected[0]
            if isinstance(item, _DraggableFrameItem):
                self._update_tx_label()

    def _sync_scene_selection(self):
        """Select the matching drag item in the scene to match the list selection."""
        for i, item in enumerate(self._drag_items):
            item.setSelected(i == self._selected_idx)

    def _on_frame_selected(self, row: int):
        self._on_list_row_changed(row)

    def _apply_highlight(self):
        for idx, item in enumerate(self._drag_items):
            if idx == self._selected_idx:
                item.setPen(_HIGHLIGHT_PEN)
            else:
                color = _FRAME_COLORS[idx % len(_FRAME_COLORS)]
                pen = QPen(color.darker(150))
                pen.setWidth(2)
                item.setPen(pen)

    def _update_tx_label(self):
        idx = self._selected_idx
        if idx is None or idx >= len(self._affines):
            self._tx_label.setText("tx: —  ty: —")
            return
        aff = self._affines[idx]
        tx = aff[0][2] + self._nudges[idx][0]
        ty = aff[1][2] + self._nudges[idx][1]
        ndx, ndy = self._nudges[idx]
        self._tx_label.setText(f"tx: {tx:.1f}  ty: {ty:.1f}  (nudge {ndx:+.0f}, {ndy:+.0f})")

    def _update_transform_controls(self):
        idx = self._selected_idx
        enabled = idx is not None and idx < len(self._affines)
        self._rot_spin.setEnabled(enabled)
        self._scale_spin.setEnabled(enabled)
        if not enabled:
            return
        self._rot_spin.blockSignals(True)
        self._scale_spin.blockSignals(True)
        self._rot_spin.setValue(self._rot_angles[idx])
        self._scale_spin.setValue(self._scale_factors[idx])
        self._rot_spin.blockSignals(False)
        self._scale_spin.blockSignals(False)

    def _on_rot_changed(self, val: float):
        idx = self._selected_idx
        if idx is None or idx >= len(self._drag_items):
            return
        self._rot_angles[idx] = val
        self._drag_items[idx].setRotation(val)

    def _on_scale_changed(self, val: float):
        idx = self._selected_idx
        if idx is None or idx >= len(self._drag_items):
            return
        self._scale_factors[idx] = val
        self._drag_items[idx].setScale(val)

    def _nudge(self, dx: float, dy: float):
        idx = self._selected_idx
        if idx is None or idx >= len(self._drag_items):
            return
        self._nudges[idx][0] += dx
        self._nudges[idx][1] += dy
        aff = self._affines[idx]
        new_pos = QPointF(aff[0][2] + self._nudges[idx][0], aff[1][2] + self._nudges[idx][1])
        self._drag_items[idx].setPos(new_pos)
        self._update_tx_label()

    def _reset_frame(self):
        idx = self._selected_idx
        if idx is None or idx >= len(self._drag_items):
            return
        self._nudges[idx] = [0.0, 0.0]
        self._rot_angles[idx] = 0.0
        self._scale_factors[idx] = 1.0
        aff = self._affines[idx]
        item = self._drag_items[idx]
        item.setPos(QPointF(aff[0][2], aff[1][2]))
        item.setRotation(0.0)
        item.setScale(1.0)
        self._update_tx_label()
        self._update_transform_controls()

    def adjusted_affines(self) -> list[list[list[float]]]:
        """Return affines with nudge (tx/ty) and rotation/scale modifications applied."""
        result = copy.deepcopy(self._affines)
        for idx in range(len(result)):
            # Apply additional rotation+scale on top of the BA 2x2 block
            theta = math.radians(self._rot_angles[idx])
            s = self._scale_factors[idx]
            c = math.cos(theta) * s
            st = math.sin(theta) * s
            R = np.array([[c, -st], [st, c]], dtype=np.float64)
            orig_2x2 = np.array(
                [[result[idx][0][0], result[idx][0][1]],
                 [result[idx][1][0], result[idx][1][1]]],
                dtype=np.float64,
            )
            new_2x2 = R @ orig_2x2
            result[idx][0][0] = float(new_2x2[0, 0])
            result[idx][0][1] = float(new_2x2[0, 1])
            result[idx][1][0] = float(new_2x2[1, 0])
            result[idx][1][1] = float(new_2x2[1, 1])
            # Apply translation nudge
            result[idx][0][2] += self._nudges[idx][0]
            result[idx][1][2] += self._nudges[idx][1]
        return result
