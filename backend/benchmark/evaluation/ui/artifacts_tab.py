"""Artifacts tab: browse the static PNGs the benchmark already wrote for this
test — ``output/plots/`` (8 diagnostic plots) and ``output/panorama_stages/``
(100+ per-stage renders).

``discovery`` has always resolved ``plots_dir`` and ``stage_dir`` and the old
dashboard never surfaced either (issue #123 defect 7), so the only way to see a
stage render was a file manager. Stage renders are grouped by their
``stageNN_<label>`` prefix, because a flat list of 100+ ``..._frameNN.png`` names
is unusable; picking a group shows every member as a clickable filmstrip
(``filmstrip.py``), togglable between horizontal and vertical layout — the
same "multiple sequential images in one view" pattern the benchmark's own
``animation_phases.png`` report plot uses, just interactive and click-to-enlarge
instead of a single flat raster.

Images load into a zoomable panel, which matters most here: a
``stage11_fg_composite`` render is exactly where a torn-anatomy defect becomes
attributable to a stage.
"""

from __future__ import annotations

import os

import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QBoxLayout,
    QHBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..constants.user_interface import DISPLAY_RAW, MODE_NAVIGATE
from ..other import discovery
from ..other.discovery import TestAssets
from .filmstrip import FilmstripWidget
from .image_panel import ImagePanel
from .theme import subtle


class ArtifactsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._groups: dict[str, list[str]] = {}
        self._plots: list[str] = []
        self._current: list[str] = []

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setMaximumWidth(280)
        self._tree.currentItemChanged.connect(self._on_selected)

        self._panel = ImagePanel("artifact", "Artifact")
        self._panel.set_mode(MODE_NAVIGATE)
        self._panel.set_display_mode(DISPLAY_RAW)

        self._filmstrip = FilmstripWidget()
        self._filmstrip.frameSelected.connect(self._show)
        self._filmstrip.orientationChanged.connect(lambda _o: self._relayout_right())
        self._filmstrip.setVisible(False)

        self._caption = subtle("Select an artifact.")
        self._caption.setWordWrap(True)

        # The main preview (caption + zoomable panel) is a stable child widget;
        # only the OUTER arrangement around [preview, filmstrip] toggles between
        # a QVBoxLayout and a QHBoxLayout, matching the filmstrip's own
        # orientation — a horizontal filmstrip (thumbnails flowing left-right)
        # belongs below the preview, a vertical one belongs beside it, the way
        # most photo browsers place a vertical filmstrip column.
        preview = QVBoxLayout()
        preview.setContentsMargins(0, 0, 0, 0)
        preview.setSpacing(4)
        preview.addWidget(self._caption)
        preview.addWidget(self._panel, stretch=1)
        self._preview_host = QWidget()
        self._preview_host.setLayout(preview)

        self._right_host = QWidget()
        self._right_layout: QBoxLayout | None = None
        self._relayout_right()

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        outer.addWidget(self._tree)
        outer.addWidget(self._right_host, stretch=1)

    def _relayout_right(self) -> None:
        from .filmstrip import ORIENTATION_VERTICAL

        if self._right_layout is not None:
            self._right_layout.removeWidget(self._preview_host)
            self._right_layout.removeWidget(self._filmstrip)
            QWidget().setLayout(self._right_layout)  # detach the emptied layout

        vertical_strip = self._filmstrip.orientation() == ORIENTATION_VERTICAL
        layout = QHBoxLayout() if vertical_strip else QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._preview_host, stretch=1)
        layout.addWidget(self._filmstrip)
        self._right_layout = layout
        self._right_host.setLayout(layout)

    def set_assets(self, assets: TestAssets | None) -> None:
        self._tree.clear()
        self._panel.set_image(None)
        self._filmstrip.setVisible(False)
        self._current = []
        if assets is None:
            self._caption.setText("No test loaded.")
            return

        self._plots = discovery.list_plot_images(assets.plots_dir)
        self._groups = discovery.stage_groups(assets.stage_dir)

        if self._plots:
            root = QTreeWidgetItem([f"Plots ({len(self._plots)})"])
            self._tree.addTopLevelItem(root)
            for path in self._plots:
                child = QTreeWidgetItem([os.path.splitext(os.path.basename(path))[0]])
                child.setData(0, Qt.ItemDataRole.UserRole, [path])
                root.addChild(child)
            root.setExpanded(True)
        if self._groups:
            root = QTreeWidgetItem([f"Pipeline stages ({len(self._groups)})"])
            self._tree.addTopLevelItem(root)
            for prefix, paths in self._groups.items():
                child = QTreeWidgetItem([f"{prefix}  ({len(paths)})"])
                child.setData(0, Qt.ItemDataRole.UserRole, paths)
                root.addChild(child)
            root.setExpanded(True)
        if not self._plots and not self._groups:
            self._caption.setText(
                "This test has no plots or stage renders on disk.\n"
                "They are written per-run under {dataset}/output/{plots,panorama_stages}/."
            )
        else:
            self._caption.setText("Select an artifact.")

    def _on_selected(self, current: QTreeWidgetItem | None, _previous) -> None:
        if current is None:
            return
        paths = current.data(0, Qt.ItemDataRole.UserRole)
        if not paths:
            return
        self._current = paths
        multi = len(paths) > 1
        self._filmstrip.setVisible(multi)
        if multi:
            self._filmstrip.set_frames(paths)
            self._filmstrip.set_current(0)
        self._show(0)

    def _show(self, index: int) -> None:
        if not (0 <= index < len(self._current)):
            return
        path = self._current[index]
        img = cv2.imread(path)
        if img is None:
            self._caption.setText(f"Could not decode {os.path.basename(path)}")
            self._panel.set_image(None)
            return
        self._panel.set_image(img)
        self._filmstrip.set_current(index)
        count_note = f"  ({index + 1}/{len(self._current)})" if len(self._current) > 1 else ""
        self._caption.setText(f"{os.path.basename(path)}   {img.shape[1]}x{img.shape[0]}{count_note}")
