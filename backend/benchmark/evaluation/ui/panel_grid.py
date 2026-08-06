"""PanelGrid: the N-way comparator display — which comparators are visible,
how they're laid out, and cross-panel view locking.

The old dashboard hard-wired exactly three panels in a fixed row, which left
the Overmix comparator invisible even though all 97 tests have one on disk
(issue #123 defect 8). Here the visible set and the layout are both runtime
choices over whatever comparators the test actually has.

View locking mirrors the *relative* zoom factor and a normalized viewport
centre rather than an absolute pixel scale, because comparators legitimately
produce different canvas sizes — see the note in ``image_panel.py``. It also
mirrors pan, which the old "Synchronized zoom" checkbox never did.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..constants.schema import COMPARATOR_KEYS, COMPARATOR_TITLES
from ..constants.user_interface import (
    LAYOUT_COLUMN,
    LAYOUT_GRID,
    LAYOUT_ROW,
    LAYOUT_STACK,
)
from ..other.schema import BoundingBox
from .image_panel import ImagePanel
from .theme import current_palette

# Manhattan distance (px) a press has to travel before it counts as a
# reorder-drag rather than a click — matches the small-drag tolerance used
# elsewhere (e.g. the defect-region rubber band).
_DRAG_THRESHOLD_PX = 8


class _PanelCell(QWidget):
    """One titled panel — the title bar doubles as the focus indicator, since
    at a keyboard-driven pace the user needs to see which panel a 0-4 keypress
    will score without moving the mouse.

    The title bar is also the drag handle for reordering panels: pressing and
    dragging it onto another panel's cell moves this one into that panel's
    slot. Implemented as plain mouse-event tracking (an event filter on the
    title label) rather than Qt's native QDrag, since this is a same-window
    reorder that doesn't need the OS drag-and-drop subsystem — and QDrag's
    nested event loop isn't exercisable under the offscreen QPA this package
    is tested under.
    """

    def __init__(self, key: str, panel: ImagePanel, on_reorder: Callable[[str, str], None], parent=None):
        super().__init__(parent)
        self.key = key
        self.panel = panel
        self._on_reorder = on_reorder
        self._press_pos = None
        self._dragging = False
        self._focused = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.title_label = QLabel(panel.title)
        self.title_label.setProperty("role", "panelTitle")
        self.title_label.setCursor(Qt.CursorShape.OpenHandCursor)
        self.title_label.setToolTip(f"{panel.title} — drag onto another panel to reorder")
        self.title_label.installEventFilter(self)
        self.info_label = QLabel("")
        self.info_label.setProperty("role", "subtle")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header = QGridLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.title_label, 0, 0)
        header.addWidget(self.info_label, 0, 1)
        header.setColumnStretch(0, 1)
        layout.addLayout(header)
        layout.addWidget(panel, stretch=1)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

    def set_focused(self, focused: bool) -> None:
        self._focused = focused
        palette = current_palette()
        color = palette["accent"] if focused else palette["border"]
        weight = "600" if focused else "500"
        self.title_label.setStyleSheet(
            f"padding: 3px 6px; border: 1px solid {color}; border-radius: 4px;"
            f" font-weight: {weight}; color: {palette['bg'] if focused else 'inherit'};"
            f" background: {palette['accent'] if focused else palette['surface_hi']};"
        )

    def refresh_theme(self) -> None:
        """Re-derive this cell's inline focus-chip style from the now-current
        palette. ``set_focused`` builds an inline stylesheet (not cascaded QSS,
        since the focus colour needs to differ from every other button), so a
        theme change leaves it stale unless re-applied explicitly."""
        self.set_focused(self._focused)

    def set_info(self, text: str) -> None:
        self.info_label.setText(text)

    # -- drag-to-reorder -------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        if watched is not self.title_label:
            return super().eventFilter(watched, event)
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self._dragging = False
        elif kind == QEvent.Type.MouseMove and self._press_pos is not None:
            moved = (event.position().toPoint() - self._press_pos).manhattanLength()
            if moved > _DRAG_THRESHOLD_PX and not self._dragging:
                self._dragging = True
                self.title_label.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif kind == QEvent.Type.MouseButtonRelease:
            if self._dragging:
                target = self._cell_under_global_pos(event.globalPosition().toPoint())
                if target is not None and target is not self:
                    self._on_reorder(self.key, target.key)
            self._press_pos = None
            self._dragging = False
            self.title_label.setCursor(Qt.CursorShape.OpenHandCursor)
        return False  # never consume — the title label has nothing else to do with these

    def _cell_under_global_pos(self, global_pos) -> Optional["_PanelCell"]:
        grid = self.parentWidget()
        while grid is not None and not isinstance(grid, PanelGrid):
            grid = grid.parentWidget()
        if grid is None:
            return None
        for cell in grid.cells.values():
            if cell.isVisible() and cell.rect().contains(cell.mapFromGlobal(global_pos)):
                return cell
        return None


class PanelGrid(QWidget):
    """Owns one ``ImagePanel`` per comparator key and arranges the visible
    subset."""

    focusChanged = Signal(str)
    bboxDrawn = Signal(str, object)
    pointPicked = Signal(str, float, float)
    regionPicked = Signal(str, float, float, float, float)
    pixelHovered = Signal(str, int, int, object)
    pixelPinned = Signal(str, int, int, object)

    orderChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.panels: Dict[str, ImagePanel] = {}
        self.cells: Dict[str, _PanelCell] = {}
        self._visible: List[str] = []
        self._available: List[str] = []
        # A user preference, not a per-test fact — persists across the whole
        # session as the user navigates the 97-test queue, only ever changed
        # by an explicit drag. Independent of COMPARATOR_KEYS' fixed order so
        # e.g. Ground Truth can be dragged in between two stitcher outputs.
        self._order: List[str] = list(COMPARATOR_KEYS)
        self._layout_mode = LAYOUT_ROW
        self._locked = True
        self._focus_key: Optional[str] = None
        self._images: Dict[str, np.ndarray] = {}

        for key in COMPARATOR_KEYS:
            panel = ImagePanel(key, COMPARATOR_TITLES[key])
            panel.viewChanged.connect(lambda z, cx, cy, k=key: self._on_view_changed(k, z, cx, cy))
            panel.bboxDrawn.connect(lambda data, k=key: self.bboxDrawn.emit(k, data))
            panel.pointPicked.connect(lambda x, y, k=key: self.pointPicked.emit(k, x, y))
            panel.regionPicked.connect(lambda x, y, w, h, k=key: self.regionPicked.emit(k, x, y, w, h))
            panel.pixelHovered.connect(lambda x, y, bgr, k=key: self.pixelHovered.emit(k, x, y, bgr))
            panel.pixelPinned.connect(lambda x, y, bgr, k=key: self.pixelPinned.emit(k, x, y, bgr))
            panel.focusRequested.connect(lambda k=key: self.set_focus(k))
            self.panels[key] = panel
            self.cells[key] = _PanelCell(key, panel, self.reorder)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(6)
        self._stack = QStackedWidget()
        # Prevent the internal stacked widget from reporting a sizeHint based
        # on the max of its children (large image panels at native resolution)
        # — that hint would propagate through the QVBoxLayout inside PanelGrid
        # even though PanelGrid itself has Ignored horizontal policy, because
        # QLayout.minimumSize() still honours children's minimumSizeHint.
        self._stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._host_stack = QStackedWidget()
        self._host_stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self._host_stack.addWidget(self._grid_host)
        self._host_stack.addWidget(self._stack)
        outer.addWidget(self._host_stack)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

    # -- content -------------------------------------------------------------

    def set_images(self, images: Dict[str, np.ndarray]) -> None:
        """Load a test's comparators. Panels for absent comparators are cleared
        and dropped from the visible set, so a test without ground truth simply
        shows fewer panels rather than an empty box. The user's custom panel
        order (self._order) is a session-wide preference and is untouched
        here — only which comparators are available changes per test."""
        self._images = images
        available_set = {key for key in COMPARATOR_KEYS if key in images}
        self._available = [key for key in self._order if key in available_set]
        for key, panel in self.panels.items():
            panel.set_image(images.get(key))
            size = panel.image_size()
            self.cells[key].set_info(f"{size[0]}x{size[1]}" if size else "not available")
        desired = [key for key in (self._visible or self._order) if key in available_set]
        self.set_visible(desired or self._available)
        if self._focus_key not in self._visible:
            self.set_focus(self._visible[0] if self._visible else None)

    def available(self) -> List[str]:
        return list(self._available)

    def visible(self) -> List[str]:
        return list(self._visible)

    def image(self, key: str) -> Optional[np.ndarray]:
        return self._images.get(key)

    def images(self) -> Dict[str, np.ndarray]:
        return dict(self._images)

    def restore_bboxes(self, bboxes: Iterable[BoundingBox]) -> None:
        boxes = list(bboxes)
        for panel in self.panels.values():
            panel.clear_overlays()
            panel.restore_bboxes(boxes)

    # -- layout --------------------------------------------------------------

    def set_visible(self, keys: Iterable[str]) -> None:
        available_set = set(self._available)
        self._visible = [k for k in self._order if k in set(keys) and k in available_set]
        self._relayout()

    def order(self) -> List[str]:
        return list(self._order)

    def set_order(self, order: Iterable[str]) -> None:
        """Replace the whole display order at once (e.g. from a restored
        session preference). Unknown keys are dropped; any comparator missing
        from ``order`` keeps its relative position at the end."""
        given = [k for k in order if k in COMPARATOR_KEYS]
        missing = [k for k in self._order if k not in given]
        self._order = given + missing
        self._available = [k for k in self._order if k in set(self._available)]
        self._visible = [k for k in self._order if k in set(self._visible)]
        self._relayout()

    def reorder(self, source_key: str, target_key: str) -> None:
        """Move ``source_key`` into ``target_key``'s current slot, shifting
        target and everything after it back by one — the usual "drag item A
        onto item B" semantics. A no-op for two unknown or identical keys."""
        if source_key == target_key or source_key not in self._order or target_key not in self._order:
            return
        order = [k for k in self._order if k != source_key]
        order.insert(order.index(target_key), source_key)
        self._order = order
        self._available = [k for k in order if k in set(self._available)]
        visible_set = set(self._visible)
        self._visible = [k for k in order if k in visible_set]
        self._relayout()
        self.orderChanged.emit(list(order))

    def set_layout_mode(self, mode: str) -> None:
        self._layout_mode = mode
        self._relayout()

    def layout_mode(self) -> str:
        return self._layout_mode

    def _detach_all(self) -> None:
        for cell in self.cells.values():
            cell.setParent(None)
        while self._grid.count():
            self._grid.takeAt(0)
        while self._stack.count():
            self._stack.removeWidget(self._stack.widget(0))
        # QGridLayout remembers a stretch factor per column/row index for the
        # life of the layout object — takeAt() does not clear it. Left alone,
        # unchecking comparators down to fewer columns leaves the old columns'
        # stretch=1 in place with nothing in them, so Qt still reserves their
        # share of width and the visible panels don't reflow to fill the space
        # (reported: "large chunk empty when several stitcher engines are
        # unselected", and the same reason unchecking Ground Truth looked like
        # it did nothing — the freed column stayed reserved as blank space).
        # COMPARATOR_KEYS is a safe upper bound on columns/rows ever used.
        max_span = len(COMPARATOR_KEYS)
        for i in range(max_span):
            self._grid.setColumnStretch(i, 0)
            self._grid.setRowStretch(i, 0)

    def _relayout(self) -> None:
        self._detach_all()
        keys = self._visible
        if self._layout_mode == LAYOUT_STACK:
            self._host_stack.setCurrentIndex(1)
            for key in keys:
                self._stack.addWidget(self.cells[key])
                self.cells[key].show()
            if keys:
                target = self._focus_key if self._focus_key in keys else keys[0]
                self._stack.setCurrentWidget(self.cells[target])
            return

        self._host_stack.setCurrentIndex(0)
        if self._layout_mode == LAYOUT_COLUMN:
            columns = 1
        elif self._layout_mode == LAYOUT_GRID:
            columns = 2 if len(keys) > 2 else max(1, len(keys))
        else:  # LAYOUT_ROW
            columns = max(1, len(keys))
        for index, key in enumerate(keys):
            row, col = divmod(index, columns)
            self._grid.addWidget(self.cells[key], row, col)
            self.cells[key].show()
        for col in range(columns):
            self._grid.setColumnStretch(col, 1)
        rows = -(-len(keys) // columns) if keys else 1
        for row in range(rows):
            self._grid.setRowStretch(row, 1)
        # Fitting is deferred to the event loop: the cells have no final
        # geometry yet, so computing a fit scale now would use stale viewport
        # sizes and every panel would come up at the wrong zoom.
        for key in keys:
            self.panels[key].fit_to_view(emit=False)

    # -- focus ---------------------------------------------------------------

    def focus_key(self) -> Optional[str]:
        return self._focus_key

    def set_focus(self, key: Optional[str]) -> None:
        if key is not None and key not in self.panels:
            return
        self._focus_key = key
        for panel_key, cell in self.cells.items():
            cell.set_focused(panel_key == key)
        if key is not None and self._layout_mode == LAYOUT_STACK and key in self._visible:
            self._stack.setCurrentWidget(self.cells[key])
        self.focusChanged.emit(key or "")

    def cycle_focus(self, step: int = 1) -> None:
        if not self._visible:
            return
        index = self._visible.index(self._focus_key) if self._focus_key in self._visible else -1 if step > 0 else 0
        self.set_focus(self._visible[(index + step) % len(self._visible)])

    def focused_panel(self) -> Optional[ImagePanel]:
        return self.panels.get(self._focus_key) if self._focus_key else None

    def refresh_theme(self) -> None:
        """Re-derive every cell's inline focus-chip style after a theme
        change (see ``_PanelCell.refresh_theme``)."""
        for cell in self.cells.values():
            cell.refresh_theme()

    # -- view control --------------------------------------------------------

    def set_locked(self, locked: bool) -> None:
        self._locked = locked
        if locked:
            panel = self.focused_panel()
            if panel is not None and panel.has_image():
                cx, cy = panel.center_norm()
                self._on_view_changed(panel.key, panel.zoom(), cx, cy)

    def locked(self) -> bool:
        return self._locked

    def _on_view_changed(self, source_key: str, zoom: float, cx: float, cy: float) -> None:
        if not self._locked:
            return
        for key in self._visible:
            if key != source_key:
                self.panels[key].apply_external_view(zoom, cx, cy)

    def fit_all(self) -> None:
        for key in self._visible:
            self.panels[key].fit_to_view(emit=False)

    def set_mode(self, mode: str) -> None:
        for panel in self.panels.values():
            panel.set_mode(mode)

    def set_display_mode(self, mode: str) -> None:
        for panel in self.panels.values():
            panel.set_display_mode(mode)

    def zoom_focused(self, step: float) -> None:
        panel = self.focused_panel()
        if panel is not None:
            panel.set_zoom(panel.zoom() * step)
