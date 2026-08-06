from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import (
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.interpolate import RBFInterpolator

from ....styles import apply_shadow_effect


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


class _MeshPin(QGraphicsEllipseItem):
    R = 6

    def __init__(self, canvas: "_MeshCanvas", gi: int, gj: int):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R)
        self._canvas = canvas
        self._gi = gi
        self._gj = gj
        self.setBrush(QBrush(QColor("#FF9800")))
        self.setPen(QPen(QColor("white"), 1.5))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(20)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._canvas._on_pin_moved(self._gi, self._gj)
        return super().itemChange(change, value)


class _MeshCanvas(QGraphicsView):
    """Deformable-grid canvas.  Drag grid intersections to warp the image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._pins: Dict[Tuple[int, int], _MeshPin] = {}
        self._grid_lines: List[QGraphicsLineItem] = []
        self._img_w = 1
        self._img_h = 1
        self._rows = 5
        self._cols = 8
        self._orig_positions: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._displaced: Optional[Any] = None

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor("#111")))
        self.setMinimumSize(300, 250)

    def load_image(self, bgr: np.ndarray, rows: int = 5, cols: int = 8):
        self._img_h, self._img_w = bgr.shape[:2]
        self._rows, self._cols = rows, cols
        qi = _bgr_to_qimage(bgr)
        pm = QPixmap.fromImage(qi)
        scene = self._scene
        for line in self._grid_lines:
            scene.removeItem(line)
        self._grid_lines.clear()
        for pin in self._pins.values():
            scene.removeItem(pin)
        self._pins.clear()
        self._orig_positions.clear()

        if self._pix_item:
            scene.removeItem(self._pix_item)
        self._pix_item = QGraphicsPixmapItem(pm)
        scene.addItem(self._pix_item)
        scene.setSceneRect(0, 0, self._img_w, self._img_h)

        dx = self._img_w / cols
        dy = self._img_h / rows
        for gi in range(rows + 1):
            for gj in range(cols + 1):
                x = gj * dx
                y = gi * dy
                pin = _MeshPin(self, gi, gj)
                pin.setPos(x, y)
                scene.addItem(pin)
                self._pins[(gi, gj)] = pin
                self._orig_positions[(gi, gj)] = (x, y)

        self._draw_grid_lines()
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        QTimer.singleShot(0, self._fit_image)

    def reset_pins(self):
        for (gi, gj), pin in self._pins.items():
            ox, oy = self._orig_positions[(gi, gj)]
            pin.setPos(ox, oy)
        self._draw_grid_lines()
        self._displaced = None

    def _on_pin_moved(self, gi, gj):
        self._draw_grid_lines()

    def _draw_grid_lines(self):
        scene = self._scene
        for line in self._grid_lines:
            scene.removeItem(line)
        self._grid_lines.clear()
        pen = QPen(QColor("#FF980088"), 1)
        for gi in range(self._rows + 1):
            for gj in range(self._cols):
                p1 = self._pins[(gi, gj)].pos()
                p2 = self._pins[(gi, gj + 1)].pos()
                item = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
                item.setPen(pen)
                item.setZValue(5)
                scene.addItem(item)
                self._grid_lines.append(item)
        for gi in range(self._rows):
            for gj in range(self._cols + 1):
                p1 = self._pins[(gi, gj)].pos()
                p2 = self._pins[(gi + 1, gj)].pos()
                item = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
                item.setPen(pen)
                item.setZValue(5)
                scene.addItem(item)
                self._grid_lines.append(item)

    def compute_warp(self) -> Optional[Tuple[Any, Any]]:
        """Return warped image via thin-plate spline from pin displacements."""
        src_pts = []
        dst_pts = []
        for (gi, gj), pin in self._pins.items():
            ox, oy = self._orig_positions[(gi, gj)]
            nx, ny = pin.pos().x(), pin.pos().y()
            if abs(nx - ox) > 0.5 or abs(ny - oy) > 0.5:
                src_pts.append([ox, oy])
                dst_pts.append([nx, ny])

        if not src_pts:
            return None

        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        try:
            xs = np.arange(self._img_w, dtype=np.float32)
            ys = np.arange(self._img_h, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

            interp_x = RBFInterpolator(
                src, dst[:, 0] - src[:, 0], kernel="thin_plate_spline"
            )
            interp_y = RBFInterpolator(
                src, dst[:, 1] - src[:, 1], kernel="thin_plate_spline"
            )

            dx_field = interp_x(grid_pts).reshape(self._img_h, self._img_w)
            dy_field = interp_y(grid_pts).reshape(self._img_h, self._img_w)

            map_x = (gx + dx_field).astype(np.float32)
            map_y = (gy + dy_field).astype(np.float32)
            self._displaced = (map_x, map_y)
            return map_x, map_y

        except ImportError:
            tps = cv2.createThinPlateSplineShapeTransformer()  # pyrefly: ignore[missing-attribute]
            tps.estimateTransformation(
                dst.reshape(1, -1, 2),
                src.reshape(1, -1, 2),
                [cv2.DMatch(i, i, 0) for i in range(len(src))],
            )
            xs = np.arange(self._img_w, dtype=np.float32)
            ys = np.arange(self._img_h, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            pts_in = (
                np.column_stack([gx.ravel(), gy.ravel()])
                .reshape(1, -1, 2)
                .astype(np.float32)
            )
            _, pts_out = tps.applyTransformation(pts_in)
            pts_out = pts_out.reshape(self._img_h, self._img_w, 2)
            return pts_out[:, :, 0], pts_out[:, :, 1]

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def _fit_image(self):
        if self._pix_item is not None:
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_image()


class MeshWarpWidget(QWidget):
    """Pin-based mesh warp panel."""

    warp_applied = Signal(np.ndarray)  # warped BGR image

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bgr: Optional[np.ndarray] = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        tb = QHBoxLayout()
        tb.addWidget(QLabel("Grid:"))

        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(2, 20)
        self._rows_spin.setValue(5)
        self._rows_spin.setPrefix("rows: ")
        tb.addWidget(self._rows_spin)

        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(2, 30)
        self._cols_spin.setValue(8)
        self._cols_spin.setPrefix("cols: ")
        tb.addWidget(self._cols_spin)

        btn_reload = QPushButton("Rebuild Grid")
        btn_reload.clicked.connect(self._rebuild_grid)
        apply_shadow_effect(btn_reload, radius=4, y_offset=2)
        tb.addWidget(btn_reload)

        btn_reset = QPushButton("Reset Pins")
        btn_reset.clicked.connect(lambda: self._canvas.reset_pins())
        apply_shadow_effect(btn_reset, radius=4, y_offset=2)
        tb.addWidget(btn_reset)

        tb.addStretch()

        btn_apply = QPushButton("⚡ Apply Warp")
        btn_apply.setStyleSheet(
            "background:#1976D2;color:white;font-weight:bold;padding:4px 8px;"
        )
        btn_apply.clicked.connect(self._apply_warp)
        apply_shadow_effect(btn_apply, radius=4, y_offset=2)
        tb.addWidget(btn_apply)

        root.addLayout(tb)

        self._canvas = _MeshCanvas()
        root.addWidget(self._canvas, 1)

        self._status = QLabel("Load an image, drag grid pins to warp, then Apply Warp.")
        self._status.setStyleSheet("color:#aaa; font-size:10px;")
        root.addWidget(self._status)

    def load_image(self, bgr: np.ndarray):
        self._bgr = bgr
        self._canvas.load_image(bgr, self._rows_spin.value(), self._cols_spin.value())
        self._status.setText("Drag orange pins to warp.  Apply Warp when done.")

    def _rebuild_grid(self):
        if self._bgr is not None:
            self._canvas.load_image(
                self._bgr, self._rows_spin.value(), self._cols_spin.value()
            )

    def _apply_warp(self):
        if self._bgr is None:
            return
        result = self._canvas.compute_warp()
        if result is None:
            self._status.setText("No displacement — drag pins first.")
            return
        map_x, map_y = result
        warped = cv2.remap(
            self._bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        self._bgr = warped
        self._canvas.load_image(
            warped, self._rows_spin.value(), self._cols_spin.value()
        )
        self.warp_applied.emit(warped)
        self._status.setText("Warp applied.  You can continue refining or export.")
