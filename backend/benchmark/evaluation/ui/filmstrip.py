"""FilmstripWidget: a scrollable strip of clickable thumbnails for a sequence
of related images, with a horizontal/vertical layout toggle.

Built for the Artifacts tab's "Pipeline stages" groups (e.g.
``stage02_normalised (16)``), which used to be browsed one frame at a time via
a slider — a real filmstrip showing every frame at once is a much better match
for "compare these 16 renders" than scrubbing through them one at a time, and
it's the same "multiple sequential images in one view" pattern the benchmark's
own ``animation_phases.png`` report plot uses, just interactive.

Thumbnails are decoded and scaled once per ``set_frames()`` call and cached as
QPixmaps; toggling orientation only re-flows the existing pixmaps into a
row or a column, so it stays instant regardless of how many frames are loaded.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..constants.user_interface import COL_ACCENT
from .image_panel import bgr_to_qimage
from .theme import subtle

ORIENTATION_HORIZONTAL = "horizontal"
ORIENTATION_VERTICAL = "vertical"

_THUMB_LONG_EDGE = 96


class _Thumbnail(QLabel):
    """A single clickable thumbnail — QLabel has no native click signal."""

    clicked = Signal()

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._selected = False

    def mousePressEvent(self, event) -> None:  # noqa: D102 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        border = f"2px solid {COL_ACCENT}" if selected else "2px solid transparent"
        self.setStyleSheet(f"border: {border};")


def _make_thumbnail_pixmap(img_bgr: np.ndarray, long_edge: int) -> QPixmap:
    h, w = img_bgr.shape[:2]
    scale = long_edge / float(max(h, w))
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    small = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    return QPixmap.fromImage(bgr_to_qimage(small))


class FilmstripWidget(QWidget):
    """Thumbnails of every frame in a sequence, plus an orientation toggle."""

    frameSelected = Signal(int)
    orientationChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._orientation = ORIENTATION_HORIZONTAL
        self._paths: list[str] = []
        self._thumbnails: list[_Thumbnail] = []
        self._current_index = -1

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(subtle("Layout"))
        self._orientation_group = QButtonGroup(self)
        # Short labels ("H"/"V" + a tooltip) rather than "Horizontal"/"Vertical":
        # in vertical mode the whole filmstrip (this header included) is
        # constrained to a narrow column, where full words clip.
        self._h_btn = QPushButton("H")
        self._h_btn.setToolTip("Horizontal filmstrip, below the preview")
        self._v_btn = QPushButton("V")
        self._v_btn.setToolTip("Vertical filmstrip, beside the preview")
        for btn, orientation in ((self._h_btn, ORIENTATION_HORIZONTAL), (self._v_btn, ORIENTATION_VERTICAL)):
            btn.setCheckable(True)
            btn.setFixedWidth(28)
            btn.clicked.connect(lambda _c, o=orientation: self.set_orientation(o))
            self._orientation_group.addButton(btn)
            header.addWidget(btn)
        self._h_btn.setChecked(True)
        header.addStretch(1)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._strip_host = QWidget()
        self._scroll.setWidget(self._strip_host)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addLayout(header)
        outer.addWidget(self._scroll)
        self._apply_scroll_policy()

    def _apply_scroll_policy(self) -> None:
        if self._orientation == ORIENTATION_HORIZONTAL:
            self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setMaximumHeight(_THUMB_LONG_EDGE + 40)
            self.setMaximumWidth(16777215)
        else:
            self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.setMaximumWidth(_THUMB_LONG_EDGE + 40)
            self.setMaximumHeight(16777215)

    def set_orientation(self, orientation: str) -> None:
        self._orientation = orientation
        self._h_btn.setChecked(orientation == ORIENTATION_HORIZONTAL)
        self._v_btn.setChecked(orientation == ORIENTATION_VERTICAL)
        self._apply_scroll_policy()
        self._relayout()  # thumbnails are already decoded; just re-flow them
        self.orientationChanged.emit(orientation)

    def orientation(self) -> str:
        return self._orientation

    def set_frames(self, paths: list[str]) -> None:
        """Decode and thumbnail every path — the one potentially slow step,
        done once per selection rather than on every orientation toggle."""
        self._paths = list(paths)
        self._thumbnails = []
        for path in self._paths:
            img = cv2.imread(path)
            pixmap = (
                _make_thumbnail_pixmap(img, _THUMB_LONG_EDGE)
                if img is not None
                else QPixmap()
            )
            thumb = _Thumbnail(pixmap)
            thumb.setToolTip(os.path.basename(path))
            index = len(self._thumbnails)
            thumb.clicked.connect(lambda i=index: self._on_thumb_clicked(i))
            self._thumbnails.append(thumb)
        self._current_index = -1
        self._relayout()

    def _on_thumb_clicked(self, index: int) -> None:
        self.set_current(index)
        self.frameSelected.emit(index)

    def set_current(self, index: int) -> None:
        if self._current_index == index:
            return
        if 0 <= self._current_index < len(self._thumbnails):
            self._thumbnails[self._current_index].set_selected(False)
        self._current_index = index
        if 0 <= index < len(self._thumbnails):
            self._thumbnails[index].set_selected(True)

    def _relayout(self) -> None:
        self._strip_host = QWidget()
        layout = (
            QHBoxLayout(self._strip_host)
            if self._orientation == ORIENTATION_HORIZONTAL
            else QVBoxLayout(self._strip_host)
        )
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        for thumb in self._thumbnails:
            thumb.setParent(self._strip_host)
            layout.addWidget(thumb)
        layout.addStretch(1)
        self._strip_host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        # QScrollArea.setWidget() deletes the previously-set widget itself once
        # replaced — an extra deleteLater() here double-frees it.
        self._scroll.setWidget(self._strip_host)
