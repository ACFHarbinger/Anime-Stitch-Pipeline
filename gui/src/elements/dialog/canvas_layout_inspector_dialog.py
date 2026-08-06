import json
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


def parse_canvas_json(path: str) -> dict:
    with open(path, "r") as fh:
        raw = json.load(fh)
    return {
        "canvas_h": int(raw.get("canvas_h", 0)),
        "canvas_w": int(raw.get("canvas_w", 0)),
        "frame_h": int(raw.get("frame_h", 0)),
        "frame_w": int(raw.get("frame_w", 0)),
        "T_global": [float(v) for v in raw.get("T_global", [0.0, 0.0])],
        "affines_final": [
            [[float(v) for v in row] for row in m] for m in raw.get("affines_final", [])
        ],
    }


def _canvas_frame_corners(
    affine_2x3: List[List[float]], frame_h: int, frame_w: int
) -> List[Tuple[float, float]]:
    a, b, tx = affine_2x3[0]
    c, d, ty = affine_2x3[1]
    pts = [(0, 0), (frame_w, 0), (frame_w, frame_h), (0, frame_h)]
    return [(a * x + b * y + tx, c * x + d * y + ty) for (x, y) in pts]


class CanvasLayoutInspectorDialog(QDialog):
    _FRAME_COLORS = [
        QColor(100, 149, 237, 110),
        QColor(100, 220, 130, 110),
        QColor(255, 165, 0, 110),
        QColor(210, 100, 210, 110),
        QColor(255, 215, 0, 110),
        QColor(32, 178, 170, 110),
        QColor(255, 99, 71, 110),
        QColor(173, 216, 230, 110),
    ]

    def __init__(self, canvas_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Canvas Layout Inspector")
        self.setModal(False)
        self._data: Optional[dict] = canvas_data
        self._build_ui()
        if self._data is not None:
            self._populate()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        toolbar = QHBoxLayout()
        btn_load = QPushButton("Load JSON…")
        btn_load.clicked.connect(self._load_file)
        self._stats_label = QLabel("No data loaded.")
        self._stats_label.setStyleSheet("color: #aaa; font-size: 11px;")
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        toolbar.addWidget(btn_load)
        toolbar.addSpacing(10)
        toolbar.addWidget(self._stats_label, 1)
        toolbar.addWidget(btn_close)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._scene = QGraphicsScene()
        self._scene.setBackgroundBrush(QColor(18, 18, 18))
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setMinimumWidth(560)
        self._view.setMinimumHeight(420)
        splitter.addWidget(self._view)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Frame", "tx", "ty"])
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMinimumWidth(200)
        splitter.addWidget(self._table)

        splitter.setSizes([720, 240])
        root.addWidget(splitter)
        self.resize(980, 620)

    def _populate(self):
        if self._data is None:
            return
        self._scene.clear()

        canvas_h = self._data["canvas_h"]
        canvas_w = self._data["canvas_w"]
        frame_h = self._data["frame_h"]
        frame_w = self._data["frame_w"]
        affines = self._data["affines_final"]
        N = len(affines)

        border_pen = QPen(QColor(80, 80, 80, 200))
        border_pen.setWidth(2)
        self._scene.addRect(0, 0, canvas_w, canvas_h, border_pen, QColor(0, 0, 0, 0))

        self._table.setRowCount(0)

        for idx, aff in enumerate(affines):
            if frame_h <= 0 or frame_w <= 0:
                continue
            corners = _canvas_frame_corners(aff, frame_h, frame_w)
            color = self._FRAME_COLORS[idx % len(self._FRAME_COLORS)]
            edge_pen = QPen(color.darker(160))
            edge_pen.setWidth(2)

            path = QPainterPath()
            path.moveTo(corners[0][0], corners[0][1])
            for x, y in corners[1:]:
                path.lineTo(x, y)
            path.closeSubpath()
            self._scene.addPath(path, edge_pen, QBrush(color))

            cx = sum(x for (x, y) in corners) / len(corners)
            cy = sum(y for (x, y) in corners) / len(corners)
            lbl = self._scene.addSimpleText(str(idx))
            lbl.setBrush(QBrush(QColor(255, 255, 255, 230)))
            br = lbl.boundingRect()
            lbl.setPos(cx - br.width() / 2, cy - br.height() / 2)

            row = self._table.rowCount()
            self._table.insertRow(row)
            tx = float(aff[0][2])
            ty = float(aff[1][2])
            self._table.setItem(row, 0, QTableWidgetItem(str(idx)))
            self._table.setItem(row, 1, QTableWidgetItem(f"{tx:.1f}"))
            self._table.setItem(row, 2, QTableWidgetItem(f"{ty:.1f}"))

        self._view.fitInView(
            self._scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio
        )
        self._stats_label.setText(f"{N} frames · {canvas_w}×{canvas_h} canvas")

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Canvas Info JSON",
            "",
            "JSON (*.json)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return
        try:
            self._data = parse_canvas_json(path)
            self._populate()
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
