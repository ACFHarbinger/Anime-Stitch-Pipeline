"""Graph sub-tab: node-graph stitch planner UI and execution.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ....helpers.animation import GraphStitchWorker
from ....styles import apply_shadow_effect
from ....windows.settings.splitter_persistence import persist_splitter
from ._node_graph_items import _StitchOpNode
from ._node_graph_scene import _NodeScene, _NodeView
from ._thumbnail_file_picker import _ThumbnailFilePicker


class _GraphPanelMixin:
    def _build_graph_panel(self) -> QWidget:
        """
        Graph-based stitch planner.

        Left sidebar  — toolbar buttons + selected-node property editor.
        Centre        — node graph canvas (_NodeView / _NodeScene).
        Bottom        — shared pipeline config + progress + run/cancel.

        Workflow
        --------
        1. Click "Add Image" to drop _SourceNode items onto the canvas.
        2. Click "Add Stitch Op" to create a _StitchOpNode.
        3. Drag from an output port (right side, blue) to an input port
           (left side, green) to connect nodes.
        4. Select a _StitchOpNode and set its output path in the sidebar.
        5. Hit "Run Graph" — operations execute in topological order.
        """
        from gui.src.tabs.animation.stencil import GraphPanel

        panel = GraphPanel(self)
        vbox_lay = QVBoxLayout(panel)
        vbox_lay.setContentsMargins(0, 0, 0, 0)
        v_split = QSplitter(Qt.Orientation.Vertical)
        vbox_lay.addWidget(v_split)

        # ── scene ────────────────────────────────────────────────────────
        self._node_scene = _NodeScene(self)
        self._node_scene.plan_changed.connect(self._graph_refresh_plan)
        self._node_view = _NodeView(self._node_scene)

        # ── main splitter ────────────────────────────────────────────────
        split = QSplitter(Qt.Orientation.Horizontal)

        # ── LEFT: toolbar + properties ───────────────────────────────────
        left_w = QWidget()
        left_w.setFixedWidth(190)
        left_lay = QVBoxLayout(left_w)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(6)

        # Toolbar buttons
        btn_add_img = QPushButton("+ Add Image(s)")
        btn_add_img.setToolTip("Add image source nodes.")
        btn_add_img.clicked.connect(self._graph_add_sources)
        apply_shadow_effect(btn_add_img, radius=4, y_offset=2)

        btn_add_op = QPushButton("+ Add Stitch Op")
        btn_add_op.setToolTip("Add a stitch operation node.")
        btn_add_op.clicked.connect(self._graph_add_op)
        apply_shadow_effect(btn_add_op, radius=4, y_offset=2)

        btn_grow = QPushButton("+ Input Port")
        btn_grow.setToolTip("Add an extra input port to the selected stitch-op node.")
        btn_grow.clicked.connect(self._graph_grow_input)
        apply_shadow_effect(btn_grow, radius=4, y_offset=2)

        btn_del = QPushButton("Delete Selected")
        btn_del.setToolTip("Remove selected nodes / edges (Del).")
        btn_del.clicked.connect(self._node_scene.remove_selected)
        apply_shadow_effect(btn_del, radius=4, y_offset=2)

        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._node_scene.clear_graph)
        apply_shadow_effect(btn_clear, radius=4, y_offset=2)

        for b in (btn_add_img, btn_add_op, btn_grow, btn_del, btn_clear):
            left_lay.addWidget(b)

        left_lay.addSpacing(8)

        # Selected-node properties
        props_group = QGroupBox("Operation Properties")
        props_form = QVBoxLayout(props_group)
        props_form.setSpacing(4)

        props_form.addWidget(QLabel("Step name:"))
        self._graph_name_edit = QLineEdit()
        self._graph_name_edit.setPlaceholderText("e.g. Pair A")
        self._graph_name_edit.textChanged.connect(self._graph_apply_props)
        props_form.addWidget(self._graph_name_edit)

        btn_grow_inline = QPushButton("+ Input Port")
        btn_grow_inline.clicked.connect(self._graph_grow_input)
        props_form.addWidget(btn_grow_inline)

        left_lay.addWidget(props_group)

        # Plan summary
        plan_group = QGroupBox("Execution Plan")
        plan_lay = QVBoxLayout(plan_group)
        self._graph_plan_label = QLabel("(no ops)")
        self._graph_plan_label.setWordWrap(True)
        self._graph_plan_label.setStyleSheet("color:#999; font-size:10px;")
        plan_lay.addWidget(self._graph_plan_label)
        left_lay.addWidget(plan_group)

        left_lay.addStretch()
        split.addWidget(left_w)

        # ── CENTRE: node canvas ───────────────────────────────────────────
        split.addWidget(self._node_view)
        split.setStretchFactor(1, 1)
        v_split.addWidget(split)

        # ── BOTTOM: pipeline options + progress + run ─────────────────────
        bottom_group = QGroupBox("Pipeline and Execution")
        bottom_lay = QVBoxLayout(bottom_group)
        bottom_lay.setSpacing(4)

        # Reuse same pipeline toggles as Stitch tab (separate widget instances)
        pipe_row = QHBoxLayout()
        self._gph_chk_basic = QCheckBox("BaSiC")
        self._gph_chk_birefnet = QCheckBox("BiRefNet")
        self._gph_chk_loftr = QCheckBox("LoFTR")
        self._gph_chk_ecc = QCheckBox("ECC")
        self._gph_chk_fg = QCheckBox("Composite FG")
        for chk, default in (
            (self._gph_chk_basic, True),
            (self._gph_chk_birefnet, True),
            (self._gph_chk_loftr, True),
            (self._gph_chk_ecc, True),
            (self._gph_chk_fg, True),
        ):
            chk.setChecked(default)
            pipe_row.addWidget(chk)
        pipe_row.addStretch()

        # Combined Renderer and Output Row
        config_row = QHBoxLayout()
        config_row.addWidget(QLabel("Renderer:"))
        self._gph_renderer = QComboBox()
        self._gph_renderer.addItems(["median", "first", "blend"])
        config_row.addWidget(self._gph_renderer)

        config_row.addSpacing(20)

        config_row.addWidget(QLabel("Output dir:"))
        self._gph_out_dir_edit = QLineEdit()
        self._gph_out_dir_edit.setText("images")
        self._gph_out_dir_edit.setToolTip("Directory where stitch outputs are saved.")
        config_row.addWidget(self._gph_out_dir_edit)
        btn_out_dir = QPushButton("Browse…")
        btn_out_dir.setFixedWidth(110)
        btn_out_dir.clicked.connect(self._graph_browse_output_dir)
        config_row.addWidget(btn_out_dir)
        config_row.addStretch()

        self._gph_progress = QProgressBar()
        self._gph_progress.setRange(0, 100)
        self._gph_progress.setValue(0)
        self._gph_stage_lbl = QLabel("Idle")
        self._gph_stage_lbl.setStyleSheet("color:#aaa; font-size:10px;")

        btn_row = QHBoxLayout()
        self._gph_btn_run = QPushButton("▶ Run Graph")
        self._gph_btn_run.setStyleSheet(
            "QPushButton{background:#2e7d32;color:white;font-weight:bold;padding:6px;}"
            "QPushButton:hover{background:#388e3c;}"
        )
        self._gph_btn_run.clicked.connect(self._graph_run)
        apply_shadow_effect(self._gph_btn_run, radius=6, y_offset=3)

        self._gph_btn_cancel = QPushButton("Cancel")
        self._gph_btn_cancel.setEnabled(False)
        self._gph_btn_cancel.clicked.connect(self._graph_cancel)
        apply_shadow_effect(self._gph_btn_cancel, radius=4, y_offset=2)

        btn_row.addWidget(self._gph_btn_run)
        btn_row.addWidget(self._gph_btn_cancel)
        btn_row.addStretch()

        self._gph_log = QTextEdit()
        self._gph_log.setReadOnly(True)
        self._gph_log.setFixedHeight(75)
        self._gph_log.setFont(QFont("Monospace", 8))

        bottom_lay.addLayout(pipe_row)
        bottom_lay.addLayout(config_row)
        bottom_lay.addLayout(btn_row)
        bottom_lay.addWidget(self._gph_stage_lbl)
        bottom_lay.addWidget(self._gph_progress)
        bottom_lay.addWidget(self._gph_log)

        v_split.addWidget(bottom_group)
        v_split.setStretchFactor(0, 1)
        v_split.setStretchFactor(1, 0)
        v_split.setSizes([1000, 300])

        persist_splitter(v_split, "GraphPanel/vertical")
        persist_splitter(split, "GraphPanel/horizontal")
        # Connect scene selection changes to property editor
        self._node_scene.selectionChanged.connect(self._graph_on_selection_changed)

        return panel

    def _graph_add_sources(self):
        dlg = _ThumbnailFilePicker(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        for p in dlg.selected_paths():
            self._node_scene.add_source(p)

    def _graph_add_op(self):
        op = self._node_scene.add_stitch_op()
        # Select the new op so its properties appear immediately
        self._node_scene.clearSelection()
        op.setSelected(True)

    def _graph_grow_input(self):
        # Prefer currently selected op; fall back to last selected so sidebar
        # buttons work even if clicking them briefly moved focus away.
        target: Optional[_StitchOpNode] = None
        for item in self._node_scene.selectedItems():
            if isinstance(item, _StitchOpNode):
                target = item
                break
        if target is None:
            target = self._last_selected_op
        if target is not None:
            target.grow_input()
            self._node_scene.plan_changed.emit()
            self._node_scene.update()

    def _graph_on_selection_changed(self):
        try:
            items = self._node_scene.selectedItems()
        except RuntimeError:
            return

        for item in items:
            if isinstance(item, _StitchOpNode):
                self._last_selected_op = item
                self._graph_name_edit.blockSignals(True)
                self._graph_name_edit.setText(item.step_name)
                self._graph_name_edit.blockSignals(False)
                return

    def _graph_apply_props(self):
        try:
            items = self._node_scene.selectedItems()
        except RuntimeError:
            return
        for item in items:
            if isinstance(item, _StitchOpNode):
                item.step_name = self._graph_name_edit.text()
                item._title = f"⊞ {item.step_name}"
                item.update()
                self._node_scene.plan_changed.emit()
                break

    def _graph_browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self._gph_out_dir_edit.text() or "images",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if d:
            self._gph_out_dir_edit.setText(d)

    def _graph_refresh_plan(self):
        try:
            plan = self._node_scene.get_plan()
        except RuntimeError:
            return
        if not plan:
            self._graph_plan_label.setText("(no ops)")
            return
        lines = []
        for step in plan:
            ins = ", ".join(
                os.path.basename(i) if not i.startswith("op_") else i
                for i in step["inputs"]
            )
            out = os.path.basename(step["output"]) or "(not set)"
            lines.append(f"{step['name']}: [{ins}] → {out}")
        self._graph_plan_label.setText("\n".join(lines))

    def _graph_run(self):
        plan = self._node_scene.get_plan()
        if not plan:
            QMessageBox.warning(self, "Graph", "No stitch operations defined.")
            return
        for step in plan:
            if not step["inputs"]:
                QMessageBox.warning(
                    self, "Graph", f"Step '{step['name']}' has no connected inputs."
                )
                return

        # Auto-assign output paths under the chosen output directory.
        out_dir = self._gph_out_dir_edit.text().strip() or "images"
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        for i, step in enumerate(plan, start=1):
            if not step["output"]:
                slug = (
                    re.sub(r"[^\w]+", "_", step["name"]).strip("_").lower() or f"op{i}"
                )
                step["output"] = os.path.join(out_dir, f"{slug}.png")
                # keep the node's stored path in sync so get_plan() stays consistent
                for item in self._node_scene.items():
                    if (
                        isinstance(item, _StitchOpNode)
                        and item.step_name == step["name"]
                    ):
                        item.output_path = step["output"]
                        break

        cfg = {
            "use_basic": self._gph_chk_basic.isChecked(),
            "use_birefnet": self._gph_chk_birefnet.isChecked(),
            "use_loftr": self._gph_chk_loftr.isChecked(),
            "use_ecc": self._gph_chk_ecc.isChecked(),
            "composite_fg": self._gph_chk_fg.isChecked(),
            "renderer": self._gph_renderer.currentText(),
            "laplacian_bands": 5,
            "motion_model": "translation",
            "edge_crop": 30,
        }

        self._graph_worker = GraphStitchWorker(plan, cfg)
        self._graph_thread = self._graph_worker
        self._graph_worker.sig_step.connect(self._graph_on_step)
        self._graph_worker.sig_stage.connect(self._graph_on_stage)
        self._graph_worker.sig_log.connect(self._graph_log_append)
        self._graph_worker.sig_finished.connect(self._graph_on_finished)
        self._graph_worker.sig_error.connect(self._graph_on_error)
        self._graph_worker.finished.connect(self._graph_on_thread_done)
        self._graph_worker.finished.connect(self._graph_worker.deleteLater)

        self._gph_btn_run.setEnabled(False)
        self._gph_btn_cancel.setEnabled(True)
        self._gph_log.clear()
        self._graph_worker.start()

    def _graph_cancel(self):
        if self._graph_worker:
            self._graph_worker.cancel()

    @Slot(int, int, str)
    def _graph_on_step(self, current: int, total: int, name: str):
        pct = int((current - 1) / total * 100)
        self._gph_progress.setValue(pct)
        self._gph_stage_lbl.setText(f"Step {current}/{total}: {name}")

    @Slot(int, int, str)
    def _graph_on_stage(self, stage: int, total: int, label: str):
        self._gph_stage_lbl.setText(f"  ↳ [{stage}/{total}] {label}")

    @Slot(list)
    def _graph_on_finished(self, paths: list):
        self._gph_progress.setValue(100)
        self._gph_stage_lbl.setText("Done.")
        self._graph_log_append(f"\n✓ Graph complete. Outputs: {paths}")

    @Slot(str)
    def _graph_on_error(self, msg: str):
        self._gph_stage_lbl.setText(f"Error: {msg}")
        self._graph_log_append(f"[ERROR] {msg}")

    def _graph_on_thread_done(self):
        self._gph_btn_run.setEnabled(True)
        self._gph_btn_cancel.setEnabled(False)

    def _graph_log_append(self, msg: str):
        self._gph_log.append(msg)
        sb = self._gph_log.verticalScrollBar()
        sb.setValue(sb.maximum())


__all__ = ["_GraphPanelMixin"]
