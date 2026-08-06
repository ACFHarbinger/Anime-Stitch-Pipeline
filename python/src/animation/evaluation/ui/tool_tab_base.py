"""Shared base for "pick a tool, run it against the current test, show the
result" — the pattern the visualisations, comparison and diagnostics tabs all
follow, so the control-row / result-area plumbing exists once.

Two changes from the old version. Image results now go into a zoomable
``ImagePanel`` instead of a ``QLabel`` scaled to a hard-coded 900x700 — a
diff map of a 1700px panorama at 900px wide hides exactly the fine
misalignment it was computed to reveal. And results are rendered through a
small worker-free cache keyed on (tool, inputs) so re-selecting a tool a user
already ran is instant, which matters at a ~28 s/test pace.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..constants.logic import FIGURE_CACHE_SIZE
from ..constants.schema import COMPARATOR_TITLES
from ..constants.user_interface import DISPLAY_RAW, MODE_NAVIGATE
from .image_panel import ImagePanel
from .mpl_canvas import PlotPane
from .theme import subtle


class ToolTabBase(QWidget):
    """A tool list on the left, the rendered result on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._images: Dict[str, np.ndarray] = {}
        self._metrics: Dict = {}
        self._tools: List[Tuple[str, Callable[[], object]]] = []
        self._cache: "OrderedDict[str, QWidget]" = OrderedDict()
        self._cache_token = ""
        self._current_widget: Optional[QWidget] = None

        self._tool_list = QListWidget()
        self._tool_list.setMaximumWidth(230)
        self._tool_list.currentRowChanged.connect(self._on_tool_selected)

        self._controls = QVBoxLayout()
        self._controls.setContentsMargins(0, 0, 0, 0)
        self._controls.setSpacing(4)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self._tool_list, stretch=1)
        left_layout.addLayout(self._controls)

        self._result_host = QWidget()
        self._result_layout = QVBoxLayout(self._result_host)
        self._result_layout.setContentsMargins(0, 0, 0, 0)
        self._result_layout.setSpacing(4)
        self._status = subtle("Select a tool.")
        self._status.setWordWrap(True)
        self._result_layout.addWidget(self._status)
        self._result_slot = QVBoxLayout()
        self._result_layout.addLayout(self._result_slot, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._result_host)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

    # -- registration --------------------------------------------------------

    def _add_tool(self, name: str, handler: Callable[[], object]) -> None:
        self._tools.append((name, handler))
        self._tool_list.addItem(name)

    def _add_control(self, widget: QWidget) -> None:
        self._controls.addWidget(widget)

    def _add_control_row(self, label: str, widget: QWidget) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        text = QLabel(label)
        text.setMinimumWidth(72)
        row.addWidget(text)
        row.addWidget(widget, stretch=1)
        container = QWidget()
        container.setLayout(row)
        self._controls.addWidget(container)

    def _comparator_combo(self, keys: Optional[List[str]] = None) -> QComboBox:
        combo = QComboBox()
        self._populate_combo(combo, keys or [])
        return combo

    @staticmethod
    def _populate_combo(combo: QComboBox, keys: List[str]) -> None:
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for key in keys:
            combo.addItem(COMPARATOR_TITLES.get(key, key), key)
        if previous in keys:
            combo.setCurrentIndex(keys.index(previous))
        combo.blockSignals(False)

    # -- content -------------------------------------------------------------

    def set_context(self, images: Dict[str, np.ndarray], metrics: Dict, token: str) -> None:
        """New test loaded. ``token`` identifies it, so the result cache is
        dropped when the underlying images change."""
        self._images = images
        self._metrics = metrics
        if token != self._cache_token:
            self._cache.clear()
            self._cache_token = token
        self._on_context_changed()

    def _on_context_changed(self) -> None:
        """Hook for subclasses to refresh their comparator selectors."""

    def available(self) -> List[str]:
        return list(self._images)

    def image(self, key: Optional[str]) -> Optional[np.ndarray]:
        return self._images.get(key) if key else None

    def refresh(self) -> None:
        """Re-render the selected tool, e.g. after a slider moved."""
        row = self._tool_list.currentRow()
        if row >= 0:
            self._run_tool(row, use_cache=False)

    # -- execution -----------------------------------------------------------

    def _on_tool_selected(self, row: int) -> None:
        if row >= 0:
            self._run_tool(row, use_cache=True)

    def _cache_key(self, name: str) -> str:
        """Subclasses fold their control state in, so moving a slider produces
        a different key rather than serving a stale render."""
        return name

    def _run_tool(self, row: int, use_cache: bool) -> None:
        if not (0 <= row < len(self._tools)):
            return
        name, handler = self._tools[row]
        key = self._cache_key(name)
        if use_cache and key in self._cache:
            self._cache.move_to_end(key)
            self._mount(self._cache[key])
            return
        self._status.setText(f"Computing {name}…")
        self._status.repaint()
        try:
            widget = self._build_result(handler())
        except Exception as exc:  # a bad frame must not take the tool down
            self._status.setText(f"{name} failed: {type(exc).__name__}: {exc}")
            self._clear_slot()
            return
        if widget is None:
            self._status.setText(f"{name}: not available for this test.")
            self._clear_slot()
            return
        self._cache[key] = widget
        while len(self._cache) > FIGURE_CACHE_SIZE:
            _old_key, old_widget = self._cache.popitem(last=False)
            if old_widget is not self._current_widget:
                old_widget.setParent(None)
        self._status.setText(name)
        self._mount(widget)

    def _build_result(self, result) -> Optional[QWidget]:
        """Wrap a handler's return value in a display widget."""
        if result is None:
            return None
        if isinstance(result, QWidget):
            return result
        if isinstance(result, Figure):
            return PlotPane(result)
        if isinstance(result, np.ndarray):
            return self._image_widget(result)
        if isinstance(result, str):
            return self._text_widget(result)
        raise TypeError(f"Unsupported tool result type: {type(result).__name__}")

    @staticmethod
    def _image_widget(img_bgr: np.ndarray) -> QWidget:
        panel = ImagePanel("result", "Result")
        panel.set_mode(MODE_NAVIGATE)
        panel.set_display_mode(DISPLAY_RAW)
        panel.set_image(img_bgr)
        return panel

    @staticmethod
    def _text_widget(text: str) -> QWidget:
        from PySide6.QtWidgets import QPlainTextEdit

        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        view.setStyleSheet("font-family: monospace;")
        view.setPlainText(text)
        return view

    def _clear_slot(self) -> None:
        if self._current_widget is not None:
            self._result_slot.removeWidget(self._current_widget)
            self._current_widget.hide()
            self._current_widget = None

    def _mount(self, widget: QWidget) -> None:
        self._clear_slot()
        self._result_slot.addWidget(widget)
        widget.show()
        self._current_widget = widget

    def set_status(self, text: str) -> None:
        self._status.setText(text)
