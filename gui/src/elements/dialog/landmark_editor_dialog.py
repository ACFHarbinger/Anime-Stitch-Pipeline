"""§2.9A — BigWarp Landmark Editor Dialog.

User clicks corresponding points on side-by-side frame thumbnails to define
landmark pairs used to override a failed edge in bundle adjustment.  Supports
pure translation (1 pair), partial affine (2 pairs), or full affine (3+ pairs).

Usage
-----
    dlg = LandmarkEditorDialog(frame_i_bgr, frame_j_bgr, i=2, j=3, parent=self)
    if dlg.exec() == QDialog.Accepted:
        pairs = dlg.landmark_pairs()   # List[((xi,yi),(xj,yj))]
        edge = _build_landmark_affine(i, j, pairs)
"""

from __future__ import annotations

import cv2
import numpy as np
from gui.src.constants.elements import _COLORS, _MARKER_R, _THUMB_H, _THUMB_W
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _bgr_to_pixmap(arr: np.ndarray, max_w: int = _THUMB_W, max_h: int = _THUMB_H) -> QPixmap:
    h, w = arr.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if scale < 1.0:
        arr = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))), cv2.INTER_AREA) # pyrefly: ignore [no-matching-overload]
    h, w = arr.shape[:2]
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class _ClickableScene(QGraphicsScene):
    """QGraphicsScene that emits click position when user clicks on the frame pixmap."""

    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self._callback = callback

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            self._callback(pos)
        super().mousePressEvent(event)


class _FrameView(QWidget):
    """One panel: a frame thumbnail with clickable landmark markers."""

    def __init__(self, frame_bgr: np.ndarray, label: str, parent=None):
        super().__init__(parent)
        self._pixmap = _bgr_to_pixmap(frame_bgr)
        self._scale_x = frame_bgr.shape[1] / max(1, self._pixmap.width())
        self._scale_y = frame_bgr.shape[0] / max(1, self._pixmap.height())
        self._markers: list[tuple[float, float]] = []  # image-space coords
        self._click_cb = None  # set by parent after construction

        self._scene = _ClickableScene(self._on_click)
        self._pix_item = QGraphicsPixmapItem(self._pixmap)
        self._scene.addItem(self._pix_item)

        self._view = QGraphicsView(self._scene)
        self._view.setFixedSize(self._pixmap.width() + 4, self._pixmap.height() + 4)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        lbl.setFont(font)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(lbl)
        lay.addWidget(self._view)

    def _on_click(self, scene_pos: QPointF):
        ix = scene_pos.x() * self._scale_x
        iy = scene_pos.y() * self._scale_y
        if self._click_cb:
            self._click_cb(ix, iy)

    def add_marker(self, ix: float, iy: float, color: QColor, index: int):
        self._markers.append((ix, iy))
        sx = ix / self._scale_x
        sy = iy / self._scale_y
        ellipse = QGraphicsEllipseItem(sx - _MARKER_R, sy - _MARKER_R, _MARKER_R * 2, _MARKER_R * 2)
        pen = QPen(color, 2)
        ellipse.setPen(pen)
        ellipse.setBrush(color)
        ellipse.setOpacity(0.8)
        self._scene.addItem(ellipse)
        text = self._scene.addText(str(index + 1))
        text.setDefaultTextColor(Qt.GlobalColor.white)
        text.setPos(sx - _MARKER_R + 1, sy - _MARKER_R - 1)

    def clear_markers(self):
        self._markers.clear()
        self._scene.clear()
        self._pix_item = QGraphicsPixmapItem(self._pixmap)
        self._scene.addItem(self._pix_item)


class LandmarkEditorDialog(QDialog):
    """§2.9A BigWarp-style Landmark Editor.

    Side-by-side thumbnails of two frames.  Click alternately on the LEFT frame
    then the RIGHT frame to add a matching landmark pair.  A connecting line is
    drawn between pairs with the same index colour.

    Parameters
    ----------
    frame_i_bgr : (H, W, 3) uint8 BGR image for the source frame (frame i).
    frame_j_bgr : (H, W, 3) uint8 BGR image for the target frame (frame j).
    i : pipeline frame index for frame_i (display only).
    j : pipeline frame index for frame_j (display only).
    parent : parent QWidget.

    Methods
    -------
    landmark_pairs() → List[((xi,yi),(xj,yj))]
        Returns the list of user-placed landmark correspondences in image space.
    """

    def __init__(
        self,
        frame_i_bgr: np.ndarray,
        frame_j_bgr: np.ndarray,
        i: int = 0,
        j: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"§2.9A  Landmark Editor — Frames {i} → {j}")
        self._i = i
        self._j = j
        self._pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._pending_left: tuple[float, float] | None = None  # waiting for right click

        self._view_i = _FrameView(frame_i_bgr, f"Frame {i}  (click first)")
        self._view_j = _FrameView(frame_j_bgr, f"Frame {j}  (click second)")
        self._view_i._click_cb = self._on_left_click
        self._view_j._click_cb = self._on_right_click

        self._status = QLabel(
            "Click a point on the LEFT frame, then the matching point on the RIGHT frame."
        )
        self._status.setWordWrap(True)
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._undo_btn = QPushButton("Undo Last Pair")
        self._undo_btn.clicked.connect(self._undo_last)
        self._undo_btn.setEnabled(False)

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._clear_all)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        self._ok_btn = btn_box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_btn.setEnabled(False)
        self._ok_btn.setText("Inject Edge")

        frames_row = QHBoxLayout()
        frames_row.addWidget(self._view_i)
        frames_row.addSpacing(8)
        frames_row.addWidget(self._view_j)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(self._undo_btn)
        ctrl_row.addWidget(self._clear_btn)
        ctrl_row.addStretch()

        self._pair_label = QLabel("Landmark pairs: 0  (min 1 for translation, 3 for full affine)")
        self._pair_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay = QVBoxLayout(self)
        lay.addLayout(frames_row)
        lay.addWidget(self._status)
        lay.addWidget(self._pair_label)
        lay.addLayout(ctrl_row)
        lay.addWidget(btn_box)
        self.adjustSize()

    # ------------------------------------------------------------------
    # Click handlers
    # ------------------------------------------------------------------

    def _on_left_click(self, ix: float, iy: float):
        self._pending_left = (ix, iy)
        self._status.setText(
            f"Left point set at ({ix:.0f}, {iy:.0f}).  "
            "Now click the matching point on the RIGHT frame."
        )
        color = _COLORS[len(self._pairs) % len(_COLORS)]
        self._view_i.add_marker(ix, iy, color, len(self._pairs))

    def _on_right_click(self, jx: float, jy: float):
        if self._pending_left is None:
            self._status.setText("Click a point on the LEFT frame first.")
            return
        ix, iy = self._pending_left
        self._pending_left = None
        color = _COLORS[len(self._pairs) % len(_COLORS)]
        self._view_j.add_marker(jx, jy, color, len(self._pairs))
        self._pairs.append(((ix, iy), (jx, jy)))
        self._refresh_ui()

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _undo_last(self):
        if not self._pairs:
            return
        self._pairs.pop()
        self._pending_left = None
        self._redraw_all_markers()
        self._refresh_ui()

    def _clear_all(self):
        self._pairs.clear()
        self._pending_left = None
        self._view_i.clear_markers()
        self._view_j.clear_markers()
        self._refresh_ui()

    def _redraw_all_markers(self):
        self._view_i.clear_markers()
        self._view_j.clear_markers()
        for k, ((ix, iy), (jx, jy)) in enumerate(self._pairs):
            color = _COLORS[k % len(_COLORS)]
            self._view_i.add_marker(ix, iy, color, k)
            self._view_j.add_marker(jx, jy, color, k)

    def _refresh_ui(self):
        n = len(self._pairs)
        self._pair_label.setText(
            f"Landmark pairs: {n}  —  "
            + ("translation (1 pt)" if n == 1 else
               "partial affine (4-DOF)" if n == 2 else
               f"full affine (6-DOF, {n} pts)")
        )
        self._ok_btn.setEnabled(n >= 1)
        self._undo_btn.setEnabled(n > 0)
        if n == 0:
            self._status.setText(
                "Click a point on the LEFT frame, then the matching point on the RIGHT frame."
            )

    # ------------------------------------------------------------------
    # Accept / result
    # ------------------------------------------------------------------

    def _on_accept(self):
        if self._pairs:
            self.accept()

    def landmark_pairs(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Return user-placed landmark correspondences in image-pixel space."""
        return list(self._pairs)
