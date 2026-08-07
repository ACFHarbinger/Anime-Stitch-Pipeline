from __future__ import annotations

import os

import cv2
import numpy as np
from gui.src.constants.elements import _CARD_H, _CARD_W, _DIFF_BAR_H, _DIFF_HIGH
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _diff_color(diff: float) -> QColor:
    t = min(1.0, diff / _DIFF_HIGH)
    r = int(t * 220 + (1 - t) * 60)
    g = int((1 - t) * 200 + t * 60)
    return QColor(r, g, 60)


def _bgr_to_qimage(bgr: np.ndarray, max_w: int, max_h: int) -> QImage:
    h, w = bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h2, w2 = rgb.shape[:2]
    return QImage(rgb.data, w2, h2, 3 * w2, QImage.Format.Format_RGB888).copy()


class _DiffBar(QWidget):
    def __init__(self, diff: float, parent=None):
        super().__init__(parent)
        self._color = _diff_color(diff)
        self.setFixedHeight(_DIFF_BAR_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QBrush(self._color))


class _ThumbnailCard(QWidget):
    def __init__(self, path: str, thumb: np.ndarray, diff: float, parent=None):
        super().__init__(parent)
        self._path = path
        self.setFixedWidth(_CARD_W)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._check = QCheckBox()
        self._check.setChecked(True)
        layout.addWidget(self._check, alignment=Qt.AlignmentFlag.AlignHCenter)

        thumb_label = QLabel()
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        qimg = _bgr_to_qimage(thumb, _CARD_W - 8, _CARD_H)
        thumb_label.setPixmap(QPixmap.fromImage(qimg))
        thumb_label.setFixedHeight(_CARD_H)
        layout.addWidget(thumb_label)

        layout.addWidget(_DiffBar(diff))

        name = os.path.basename(path)
        name_label = QLabel(name[:18] + "…" if len(name) > 19 else name)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet("font-size: 9px; color: #ccc;")
        name_label.setWordWrap(False)
        layout.addWidget(name_label)

    @property
    def path(self) -> str:
        return self._path

    @property
    def included(self) -> bool:
        return self._check.isChecked()

    def set_included(self, v: bool):
        self._check.setChecked(v)


class SelectionReviewDialog(QDialog):
    def __init__(self, data: dict, title: str = "Frame Selection Review — Stage 4", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(980, 340)
        self._paths: list[str] = list(data["paths"])
        thumbnails: list[np.ndarray] = list(data["thumbnails"])
        diffs: list[float] = list(data.get("frame_diffs", [0.0] * len(self._paths)))

        while len(diffs) < len(self._paths):
            diffs.append(0.0)

        self._cards: list[_ThumbnailCard] = [
            _ThumbnailCard(p, t, d)
            for p, t, d in zip(self._paths, thumbnails, diffs, strict=False)
        ]
        self._build_ui()
        self._update_status()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        ctrl = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(self._select_all)
        btn_none = QPushButton("Deselect All")
        btn_none.clicked.connect(self._deselect_all)
        btn_up = QPushButton("Move Up")
        btn_up.clicked.connect(self._move_up)
        btn_dn = QPushButton("Move Down")
        btn_dn.clicked.connect(self._move_down)
        self._status_label = QLabel()
        self._status_label.setStyleSheet("color: #aaa; font-size: 10px;")
        for b in (btn_all, btn_none, btn_up, btn_dn):
            ctrl.addWidget(b)
        ctrl.addSpacing(10)
        ctrl.addWidget(self._status_label, stretch=1)
        root.addLayout(ctrl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._card_container = QWidget()
        self._card_layout = QHBoxLayout(self._card_container)
        self._card_layout.setContentsMargins(4, 4, 4, 4)
        self._card_layout.setSpacing(6)
        self._card_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        for card in self._cards:
            self._card_layout.addWidget(card)
        self._card_layout.addStretch()

        scroll.setWidget(self._card_container)
        scroll.setFixedHeight(_CARD_H + _DIFF_BAR_H + 80)
        root.addWidget(scroll, stretch=1)

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

    def _select_all(self):
        for c in self._cards:
            c.set_included(True)
        self._update_status()

    def _deselect_all(self):
        for c in self._cards:
            c.set_included(False)
        self._update_status()

    def _move_up(self):
        for i, c in enumerate(self._cards):
            if c.included and i > 0:
                self._swap_cards(i - 1, i)
                break
        self._update_status()

    def _move_down(self):
        for i in range(len(self._cards) - 1, -1, -1):
            if self._cards[i].included and i < len(self._cards) - 1:
                self._swap_cards(i, i + 1)
                break
        self._update_status()

    def _swap_cards(self, a: int, b: int):
        self._cards[a], self._cards[b] = self._cards[b], self._cards[a]
        layout = self._card_layout
        wa = layout.itemAt(a).widget()
        wb = layout.itemAt(b).widget()
        assert wa is not None and wb is not None
        layout.removeWidget(wa)
        layout.removeWidget(wb)
        layout.insertWidget(a, wb)
        layout.insertWidget(b, wa)

    def _update_status(self):
        n_sel = sum(1 for c in self._cards if c.included)
        self._status_label.setText(f"{n_sel} of {len(self._cards)} frames selected")

    def selected_paths(self) -> list[str]:
        return [c.path for c in self._cards if c.included]
