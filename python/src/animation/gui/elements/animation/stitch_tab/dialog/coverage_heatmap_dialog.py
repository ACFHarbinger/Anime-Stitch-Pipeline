from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from gui.src.constants.elements import DIALOG__MAX_PREVIEW_H, _BAR_W


def _coverage_color(count: int) -> QColor:
    if count == 0:
        return QColor(200, 50, 50)
    if count == 1:
        return QColor(220, 140, 40)
    return QColor(60, 190, 80)


class _CanvasPreview(QWidget):
    def __init__(self, bgr: np.ndarray, parent=None):
        super().__init__(parent)
        h, w = bgr.shape[:2]
        scale = min(1.0, DIALOG__MAX_PREVIEW_H / h)
        if scale < 1.0:
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rh, rw = rgb.shape[:2]
        qimg = QImage(rgb.data, rw, rh, 3 * rw, QImage.Format.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self.setFixedSize(self._pix.width(), self._pix.height())

    def paintEvent(self, _event):
        p = QPainter(self)
        p.drawPixmap(0, 0, self._pix)


class _CoverageBarChart(QWidget):
    def __init__(self, coverage: np.ndarray, parent=None):
        super().__init__(parent)
        self._coverage = coverage
        self.setFixedWidth(_BAR_W)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QBrush(QColor(20, 20, 20)))
        cov = self._coverage
        n = len(cov)
        if n == 0:
            return
        h = self.height()
        w = self.width()
        max_val = max(int(cov.max()), 1)
        row_h = max(1, h / n)
        for i, cnt in enumerate(cov):
            y = int(i * row_h)
            bar_w = int((cnt / max_val) * (w - 2))
            color = _coverage_color(int(cnt))
            p.fillRect(QRect(0, y, bar_w, max(1, int(row_h))), QBrush(color))


class CoverageHeatmapDialog(QDialog):
    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Coverage Heatmap — Stage 9")
        self.resize(860, 500)

        canvas_preview: np.ndarray = data["canvas_preview"]
        coverage: np.ndarray = np.asarray(data["frame_count_per_row"], dtype=np.int32)

        self._build_ui(canvas_preview, coverage)

    def _build_ui(self, canvas_preview: np.ndarray, coverage: np.ndarray):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        body = QHBoxLayout()
        body.setSpacing(10)

        preview_widget = _CanvasPreview(canvas_preview)
        body.addWidget(preview_widget, alignment=Qt.AlignmentFlag.AlignTop)

        chart = _CoverageBarChart(coverage)
        body.addWidget(chart, stretch=1)

        root.addLayout(body, stretch=1)

        n = len(coverage)
        if n > 0:
            mn = int(coverage.min())
            mx = int(coverage.max())
            single_pct = 100.0 * int((coverage == 1).sum()) / n
            stats_text = f"Min coverage: {mn}  Max: {mx}  Single-frame rows: {single_pct:.1f}%"
        else:
            stats_text = "No coverage data."
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("font-size: 10px; color: #aaa;")
        root.addWidget(stats_label)

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
