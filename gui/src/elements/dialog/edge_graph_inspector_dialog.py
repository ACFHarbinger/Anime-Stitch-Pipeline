import json
import os

import numpy as np
from gui.src.constants import (
    CONF_HIGH,
    CONF_LOW,
    CONF_MED,
)
from gui.src.styles import apply_shadow_effect
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


def _conf_color(c: float) -> QColor:
    if c >= 0.7:
        return CONF_HIGH
    if c >= 0.5:
        return CONF_MED
    return CONF_LOW


def parse_edge_json(path: str) -> list[dict]:
    """Load and normalise an ASP stage05_edges.json file."""
    with open(path) as fh:
        raw = json.load(fh)
    result = []
    for rec in raw:
        if not isinstance(rec, dict) or "i" not in rec or "j" not in rec:
            continue
        result.append(
            {
                "i": int(rec["i"]),
                "j": int(rec["j"]),
                "dx": float(rec.get("dx", 0.0)),
                "dy": float(rec.get("dy", 0.0)),
                "conf": float(rec.get("conf", 0.0)),
                "method": str(rec.get("method", "?")),
            }
        )
    return result


def _edge_graph_node_positions(
    n: int, radius: float = 150.0
) -> list[tuple[float, float]]:
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]
    return [
        (
            float(radius * np.cos(2 * np.pi * k / n - np.pi / 2)),
            float(radius * np.sin(2 * np.pi * k / n - np.pi / 2)),
        )
        for k in range(n)
    ]


class EdgeGraphInspectorDialog(QDialog):
    """Read-only viewer for the ASP stage-5 LoFTR edge graph."""

    _NODE_R = 18

    def __init__(
        self,
        edges: list[dict] | None = None,
        frame_paths: list[str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edge Graph Inspector — Stage 5")
        self.resize(920, 580)
        self._edges: list[dict] = edges if edges is not None else []
        self._frame_paths: list[str] = frame_paths or []
        self._build_ui()
        if edges is not None:
            self._populate()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        toolbar = QHBoxLayout()
        btn_load = QPushButton("Load JSON…")
        btn_load.clicked.connect(self._load_file)
        apply_shadow_effect(btn_load, radius=4, y_offset=2)
        self._stats_label = QLabel("No data loaded.")
        self._stats_label.setStyleSheet("color: #999; font-size: 10px;")
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        toolbar.addWidget(btn_load)
        toolbar.addSpacing(8)
        toolbar.addWidget(self._stats_label, stretch=1)
        toolbar.addWidget(btn_close)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setBackgroundBrush(QBrush(QColor(24, 24, 32)))
        self._view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["From", "To", "Conf", "Method", "dx", "dy"]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setMinimumWidth(240)
        self._table.setMaximumWidth(340)

        splitter.addWidget(self._view)
        splitter.addWidget(self._table)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

    def _populate(self):
        self._scene.clear()
        edges = self._edges
        if not edges:
            self._stats_label.setText("No edges.")
            self._table.setRowCount(0)
            return

        n_nodes = max(max(e["i"], e["j"]) for e in edges) + 1
        n_low = sum(1 for e in edges if e["conf"] < 0.5)
        self._stats_label.setText(
            f"{n_nodes} frames · {len(edges)} edges · {n_low} low-conf"
        )

        positions = _edge_graph_node_positions(n_nodes)

        for e in edges:
            xi, yi = positions[e["i"]]
            xj, yj = positions[e["j"]]
            color = _conf_color(e["conf"])
            pen = QPen(color, max(1, int(1 + e["conf"] * 4)))
            item = self._scene.addLine(xi, yi, xj, yj, pen)
            item.setToolTip(
                f"Edge {e['i']}→{e['j']}  conf={e['conf']:.3f}\n"
                f"dx={e['dx']:+.1f}  dy={e['dy']:+.1f}  method={e['method']}"
            )

        r = self._NODE_R
        for k, (x, y) in enumerate(positions):
            pen = QPen(QColor(180, 210, 255, 200), 1)
            brush = QBrush(QColor(50, 100, 190, 200))
            self._scene.addEllipse(x - r, y - r, 2 * r, 2 * r, pen, brush)
            label_text = str(k)
            if 0 <= k < len(self._frame_paths):
                name = os.path.basename(self._frame_paths[k])
                label_text += f"\n{name[:9] + '…' if len(name) > 10 else name}"
            text = self._scene.addText(label_text)
            font = QFont()
            font.setPointSize(7)
            text.setFont(font)
            text.setDefaultTextColor(QColor(230, 230, 240))
            br = text.boundingRect()
            text.setPos(x - br.width() / 2, y - br.height() / 2)

        self._view.fitInView(
            self._scene.itemsBoundingRect().adjusted(-20, -20, 20, 20),
            Qt.AspectRatioMode.KeepAspectRatio,
        )

        sorted_edges = sorted(edges, key=lambda e: e["conf"])
        self._table.setRowCount(len(sorted_edges))
        for row, e in enumerate(sorted_edges):
            color = _conf_color(e["conf"])
            for col, val in enumerate(
                [
                    str(e["i"]),
                    str(e["j"]),
                    f"{e['conf']:.3f}",
                    e["method"],
                    f"{e['dx']:+.1f}",
                    f"{e['dy']:+.1f}",
                ]
            ):
                cell = QTableWidgetItem(val)
                cell.setForeground(QBrush(color))
                self._table.setItem(row, col, cell)

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Edge Graph JSON",
            "",
            "JSON (*.json)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return
        try:
            self._edges = parse_edge_json(path)
            self._populate()
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
