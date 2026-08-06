from __future__ import annotations

import math
from typing import List, Tuple

import cv2
from asp_backend.core.pipeline import _build_landmark_affine
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
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .landmark_editor_dialog import LandmarkEditorDialog
from gui.src.constants.elements import _CENTRE, _CONF_DIS, _CONF_HIGH, _CONF_LOW, _CONF_MANUAL, _CONF_MED, _NODE_R, _RADIUS


class _ManualEdgeDialog(QDialog):
    """Small dialog to capture a user-defined edge (i, j, dx, dy, weight)."""

    def __init__(self, n_frames: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Manual Edge")
        layout = QFormLayout(self)

        self._i = QSpinBox()
        self._i.setRange(0, max(0, n_frames - 1))
        layout.addRow("From frame (i):", self._i)

        self._j = QSpinBox()
        self._j.setRange(0, max(0, n_frames - 1))
        self._j.setValue(min(1, n_frames - 1))
        layout.addRow("To frame (j):", self._j)

        self._dx = QDoubleSpinBox()
        self._dx.setRange(-9999.0, 9999.0)
        self._dx.setSingleStep(1.0)
        self._dx.setDecimals(1)
        layout.addRow("dx (horizontal shift):", self._dx)

        self._dy = QDoubleSpinBox()
        self._dy.setRange(-9999.0, 9999.0)
        self._dy.setSingleStep(1.0)
        self._dy.setDecimals(1)
        layout.addRow("dy (vertical shift):", self._dy)

        self._weight = QDoubleSpinBox()
        self._weight.setRange(0.0, 1.0)
        self._weight.setSingleStep(0.05)
        self._weight.setDecimals(2)
        self._weight.setValue(0.90)
        layout.addRow("Confidence weight:", self._weight)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def edge_dict(self) -> dict:
        return {
            "i": self._i.value(),
            "j": self._j.value(),
            "dx": self._dx.value(),
            "dy": self._dy.value(),
            "conf": self._weight.value(),
            "method": "manual",
        }


def _conf_color(conf: float) -> QColor:
    if conf >= 0.7:
        return _CONF_HIGH
    if conf >= 0.5:
        return _CONF_MED
    return _CONF_LOW


def _node_positions(n: int) -> List[Tuple[float, float]]:
    if n <= 0:
        return []
    if n == 1:
        return [(_CENTRE, _CENTRE)]
    return [
        (
            _CENTRE + _RADIUS * math.cos(2 * math.pi * k / n - math.pi / 2),
            _CENTRE + _RADIUS * math.sin(2 * math.pi * k / n - math.pi / 2),
        )
        for k in range(n)
    ]


def _mst_edge_set(edges: List[dict]) -> set:
    sorted_by_conf = sorted(edges, key=lambda e: e["conf"], reverse=True)
    parent: dict[int, int] = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), x)
            x = parent.get(x, x)
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    kept: set = set()
    for e in sorted_by_conf:
        if union(e["i"], e["j"]):
            kept.add(id(e))
    return kept


class EdgeReviewDialog(QDialog):
    _NODE_R = _NODE_R

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edge Graph Review — Stage 5")
        self.resize(960, 600)
        self._edges: List[dict] = list(data.get("edges", []))
        self._image_paths: List[str] = list(data.get("image_paths", []))
        self._enabled: List[bool] = [True] * len(self._edges)
        self._manual_edges: List[dict] = []  # S89: user-added edges
        self._n_frames: int = data.get("n_frames", 0) or (
            (max((max(e["i"], e["j"]) for e in self._edges), default=-1) + 1)
            if self._edges
            else 0
        )
        self._build_ui()
        self._populate()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        toolbar = QHBoxLayout()
        btn_mst = QPushButton("Keep MST Only")
        btn_mst.clicked.connect(self._apply_mst)
        btn_all = QPushButton("Enable All")
        btn_all.clicked.connect(self._enable_all)
        btn_add = QPushButton("Add Edge…")  # S89
        btn_add.setToolTip(
            "Manually specify a connection between two frames.\n"
            "Use this when LoFTR failed to match a pair you know should connect."
        )
        btn_add.clicked.connect(self._on_add_edge)
        btn_landmark = QPushButton("Landmark Editor…")  # §2.9A
        btn_landmark.setToolTip(
            "§2.9A  Select two frames in the table, then click here to place\n"
            "corresponding landmark points on the frame thumbnails.\n"
            "Builds a precise affine transform from your point pairs."
        )
        btn_landmark.clicked.connect(self._on_landmark_edit)
        self._status_label = QLabel()
        self._status_label.setStyleSheet("color: #999; font-size: 10px;")
        btn_resume = QPushButton("Resume Pipeline")
        btn_resume.setDefault(True)
        btn_resume.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        for w in (btn_mst, btn_all, btn_add, btn_landmark):
            toolbar.addWidget(w)
        toolbar.addSpacing(8)
        toolbar.addWidget(self._status_label, stretch=1)
        toolbar.addWidget(btn_resume)
        toolbar.addWidget(btn_cancel)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setBackgroundBrush(QBrush(QColor(24, 24, 32)))
        self._view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        splitter.addWidget(self._view)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["On", "From", "To", "Conf", "Method", "dx", "dy"]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setMinimumWidth(260)
        self._table.setMaximumWidth(380)
        self._table.cellChanged.connect(self._on_cell_changed)
        splitter.addWidget(self._table)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

    def _populate(self):
        self._scene.clear()
        edges = self._edges
        if not edges:
            self._status_label.setText("No edges.")
            self._table.setRowCount(0)
            return

        n_nodes = max(max(e["i"], e["j"]) for e in edges) + 1
        positions = _node_positions(n_nodes)

        for idx, e in enumerate(edges):
            xi, yi = positions[e["i"]]
            xj, yj = positions[e["j"]]
            enabled = self._enabled[idx]
            color = _conf_color(e["conf"]) if enabled else _CONF_DIS
            pen = QPen(color, max(1, int(1 + e["conf"] * 4)) if enabled else 1)
            if not enabled:
                pen.setStyle(Qt.PenStyle.DashLine)
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
            if 0 <= k < len(self._image_paths):
                import os

                name = os.path.basename(self._image_paths[k])
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

        self._table.blockSignals(True)
        self._table.setRowCount(len(edges))
        for row, (e, enabled) in enumerate(zip(edges, self._enabled, strict=False)):
            color = _conf_color(e["conf"]) if enabled else _CONF_DIS
            chk = QTableWidgetItem()
            chk.setCheckState(
                Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked
            )
            self._table.setItem(row, 0, chk)
            for col, val in enumerate(
                [
                    str(e["i"]),
                    str(e["j"]),
                    f"{e['conf']:.3f}",
                    e["method"],
                    f"{e['dx']:+.1f}",
                    f"{e['dy']:+.1f}",
                ],
                start=1,
            ):
                cell = QTableWidgetItem(val)
                cell.setForeground(QBrush(color))
                self._table.setItem(row, col, cell)
        self._table.blockSignals(False)

        # S89: render manual edges in purple above the existing graph
        for me in self._manual_edges:
            if me["i"] < len(positions) and me["j"] < len(positions):
                xi, yi = positions[me["i"]]
                xj, yj = positions[me["j"]]
                pen = QPen(_CONF_MANUAL, 3)
                pen.setStyle(Qt.PenStyle.DotLine)
                item = self._scene.addLine(xi, yi, xj, yj, pen)
                item.setToolTip(
                    f"Manual edge {me['i']}→{me['j']}\n"
                    f"dx={me['dx']:+.1f}  dy={me['dy']:+.1f}  conf={me['conf']:.2f}"
                )

        # Append manual edges to the table
        n_base = len(edges)
        total_rows = n_base + len(self._manual_edges)
        self._table.blockSignals(True)
        self._table.setRowCount(total_rows)
        for row, me in enumerate(self._manual_edges, start=n_base):
            chk = QTableWidgetItem()
            chk.setCheckState(Qt.CheckState.Checked)
            chk.setFlags(chk.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)  # always on
            self._table.setItem(row, 0, chk)
            for col, val in enumerate(
                [
                    str(me["i"]),
                    str(me["j"]),
                    f"{me['conf']:.3f}",
                    "manual",
                    f"{me['dx']:+.1f}",
                    f"{me['dy']:+.1f}",
                ],
                start=1,
            ):
                cell = QTableWidgetItem(val)
                cell.setForeground(QBrush(_CONF_MANUAL))
                self._table.setItem(row, col, cell)
        self._table.blockSignals(False)

        n_on = sum(self._enabled)
        n_low = sum(1 for e, en in zip(edges, self._enabled, strict=False) if en and e["conf"] < 0.5)
        self._status_label.setText(
            f"{n_nodes} frames · {len(edges)} edges · {n_on} enabled"
            f" · {n_low} low-conf · {len(self._manual_edges)} manual"
        )

    def _on_add_edge(self):
        """S89: Open ManualEdgeDialog and append the new edge."""
        dlg = _ManualEdgeDialog(n_frames=self._n_frames, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._manual_edges.append(dlg.edge_dict())
            self._populate()

    def _on_landmark_edit(self):
        """§2.9A: Open LandmarkEditorDialog for the selected table row's frame pair."""
        rows = {idx.row() for idx in self._table.selectedIndexes()}
        if not rows:
            QMessageBox.information(
                self,
                "Landmark Editor",
                "Select a row in the edge table first, then click Landmark Editor.",
            )
            return
        row = sorted(rows)[0]
        # Resolve edge at this row (base or manual)
        n_base = len(self._edges)
        edge = self._edges[row] if row < n_base else self._manual_edges[row - n_base]
        fi, fj = edge["i"], edge["j"]

        if not self._image_paths:
            QMessageBox.warning(
                self,
                "Landmark Editor",
                "Frame image paths are not available — cannot open landmark editor.",
            )
            return
        if fi >= len(self._image_paths) or fj >= len(self._image_paths):
            QMessageBox.warning(
                self,
                "Landmark Editor",
                f"Frame indices {fi}/{fj} exceed available image paths ({len(self._image_paths)}).",
            )
            return

        frame_i = cv2.imread(self._image_paths[fi])
        frame_j = cv2.imread(self._image_paths[fj])
        if frame_i is None or frame_j is None:
            QMessageBox.warning(
                self,
                "Landmark Editor",
                "Could not load frame images from disk.",
            )
            return

        dlg = LandmarkEditorDialog(frame_i, frame_j, i=fi, j=fj, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            pairs = dlg.landmark_pairs()
            if pairs:
                new_edge = _build_landmark_affine(fi, fj, pairs)
                # Represent as a unified edge dict compatible with the table
                M = new_edge["M"]
                new_edge_display = {
                    "i": fi,
                    "j": fj,
                    "M": M,
                    "pts_i": new_edge["pts_i"],
                    "pts_j": new_edge["pts_j"],
                    "dx": float(M[0, 2]),
                    "dy": float(M[1, 2]),
                    "conf": new_edge["weight"],
                    "method": f"landmark({len(pairs)}pts)",
                }
                self._manual_edges.append(new_edge_display)
                self._populate()

    def _on_cell_changed(self, row: int, col: int):
        if col != 0:
            return
        item = self._table.item(row, 0)
        if item is None:
            return
        self._enabled[row] = item.checkState() == Qt.CheckState.Checked
        self._populate()

    def _apply_mst(self):
        mst_ids = _mst_edge_set(self._edges)
        for idx, e in enumerate(self._edges):
            self._enabled[idx] = id(e) in mst_ids
        self._populate()

    def _enable_all(self):
        self._enabled = [True] * len(self._edges)
        self._populate()

    def accepted_edges(self) -> List[dict]:
        """Return enabled original edges + all manual edges (S89)."""
        return [e for e, en in zip(self._edges, self._enabled, strict=False) if en] + list(
            self._manual_edges
        )
