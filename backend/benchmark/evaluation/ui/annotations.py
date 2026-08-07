"""Cross-panel edge annotations, the defect-tagging dialog, and the annotation
list.

An Edge is a *chain* of 2 or more endpoints across any of the visible panels
(e.g. "this seam in ASP corresponds to this clean region in Overmix and this
point in ground truth") — each endpoint independently a point or a region.
Panels each own their own ``QGraphicsScene``, so an edge line can't be a
``QGraphicsItem`` in any of them — it's painted on a transparent overlay
stacked over the whole panel grid, and ``sync_geometry()`` must be called by
the container on resize.

The annotation list stores its indices in ``Qt.UserRole`` data rather than
re-parsing its own display strings to recover them, which is what the old
``_remove_selected`` did (``text[1:].split("]", 1)`` — one label containing a
``]`` away from breaking).
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..constants.schema import (
    COMPARATOR_TITLES,
    DEFECT_TITLES,
    DEFECTS,
    SEVERITY_LABELS,
)
from ..constants.user_interface import COL_EDGE
from ..other.schema import Edge, EdgePoint
from .image_panel import ImagePanel


class DefectDialog(QDialog):
    """Captures a drawn region's defect class, severity and description.

    Replaces the old flow's bare ``QInputDialog.getText``, which could only
    collect a free-text label — so nothing a user drew was ever machine-
    classifiable, and §0.2's metric-calibration item had no structured
    failure-mode data to correlate against.
    """

    def __init__(self, image_title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tag defect region")
        self.setMinimumWidth(400)
        form = QFormLayout()
        form.addRow(QLabel(f"Region drawn on <b>{image_title}</b>"))

        self.defect_combo = QComboBox()
        for key, title, hint in DEFECTS:
            self.defect_combo.addItem(f"{title} — {hint}", key)
        form.addRow("Defect class", self.defect_combo)

        self.severity_combo = QComboBox()
        for value, label in sorted(SEVERITY_LABELS.items()):
            self.severity_combo.addItem(f"{value} — {label}", value)
        self.severity_combo.setCurrentIndex(1)
        form.addRow("Severity", self.severity_combo)

        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("optional — what exactly is wrong here")
        form.addRow("Note", self.label_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.label_edit.setFocus()

    def result_values(self) -> tuple[str, int, str]:
        return (
            self.defect_combo.currentData(),
            int(self.severity_combo.currentData()),
            self.label_edit.text().strip(),
        )


class EdgeOverlay(QWidget):
    """Transparent overlay painting edge chains across panel boundaries."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._panels: dict[str, ImagePanel] = {}
        self._edges: list[Edge] = []
        self._pending: list[EdgePoint] = []

    def register_panel(self, key: str, panel: ImagePanel) -> None:
        self._panels[key] = panel

    def register_panels(self, panels: dict[str, ImagePanel]) -> None:
        self._panels.update(panels)

    def sync_geometry(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())
            self.raise_()

    def set_edges(self, edges: list[Edge]) -> None:
        self._edges = list(edges)
        self.update()

    def set_pending(self, points: list[EdgePoint]) -> None:
        """The in-progress chain's endpoints so far, drawn dashed and
        unlabelled — visual feedback for a link that hasn't been finished yet."""
        self._pending = list(points)
        self.update()

    def _to_overlay_point(self, ep: EdgePoint) -> QPoint | None:
        panel = self._panels.get(ep.image)
        container = self.parentWidget()
        if panel is None or container is None or not panel.isVisible():
            return None
        view_pt = panel.scene_point_to_view(ep.x, ep.y)
        if view_pt is None:
            return None
        return panel.mapTo(container, view_pt.toPoint())

    def _to_overlay_rect(self, ep: EdgePoint) -> tuple[QPoint, QPoint] | None:
        panel = self._panels.get(ep.image)
        container = self.parentWidget()
        if panel is None or container is None or not panel.isVisible():
            return None
        top_left = panel.scene_point_to_view(ep.x, ep.y)
        bottom_right = panel.scene_point_to_view(ep.x + ep.w, ep.y + ep.h)
        if top_left is None or bottom_right is None:
            return None
        return panel.mapTo(container, top_left.toPoint()), panel.mapTo(container, bottom_right.toPoint())

    def _draw_chain(self, painter: QPainter, points: list[EdgePoint], label: str) -> None:
        """One connected chain: a line through consecutive endpoints, a
        marker (ellipse for a point, rect for a region) at each, and the
        label once at the middle segment's midpoint."""
        anchors: list[QPoint] = []
        for ep in points:
            if ep.is_region:
                rect = self._to_overlay_rect(ep)
                if rect is None:
                    anchors.append(None)
                    continue
                top_left, bottom_right = rect
                painter.drawRect(top_left.x(), top_left.y(),
                                  bottom_right.x() - top_left.x(), bottom_right.y() - top_left.y())
                anchors.append(QPoint((top_left.x() + bottom_right.x()) // 2,
                                       (top_left.y() + bottom_right.y()) // 2))
            else:
                point = self._to_overlay_point(ep)
                anchors.append(point)
                if point is not None:
                    painter.drawEllipse(point, 4, 4)
        segments = [(anchors[i], anchors[i + 1]) for i in range(len(anchors) - 1)]
        for a, b in segments:
            if a is not None and b is not None:
                painter.drawLine(a, b)
        if label:
            valid = [p for p in anchors if p is not None]
            if valid:
                mid_index = len(valid) // 2
                painter.drawText(valid[mid_index], label)

    def paintEvent(self, event) -> None:  # noqa: D102 - Qt override
        if not self._edges and not self._pending:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont("Sans", 9))
        painter.setPen(QPen(QColor(COL_EDGE), 2, Qt.PenStyle.DashLine))
        for edge in self._edges:
            self._draw_chain(painter, edge.points, edge.label)
        if len(self._pending) >= 1:
            painter.setPen(QPen(QColor(COL_EDGE), 2, Qt.PenStyle.DotLine))
            self._draw_chain(painter, self._pending, "")
        painter.end()


class EdgeBuilder:
    """Accumulates an open-ended chain of endpoints for an in-progress link,
    across any number of ``pointPicked``/``regionPicked`` signals from
    (possibly different) panels, until the caller explicitly finishes or
    cancels it.

    Deliberately does *not* prompt for a label itself, and does not
    auto-finish at any particular count — the old two-point version raised a
    modal dialog from inside ``add_point()`` and closed the link the instant a
    second point arrived, which put a Qt dependency in the middle of the state
    machine and made "link 3+ images" impossible to express: there was no way
    to say "one more point, please" instead of "done."
    """

    def __init__(self):
        self._points: list[EdgePoint] = []

    def pending(self) -> list[EdgePoint]:
        return list(self._points)

    def count(self) -> int:
        return len(self._points)

    def can_finish(self) -> bool:
        return len(self._points) >= 2

    def reset(self) -> None:
        self._points = []

    def add(self, image_key: str, x: float, y: float, w: float = 0.0, h: float = 0.0) -> None:
        self._points.append(EdgePoint(image=image_key, x=x, y=y, w=w, h=h))

    def finish(self, label: str) -> Edge | None:
        """Returns the completed ``Edge`` and clears the pending chain, or
        ``None`` (and leaves the chain untouched) if fewer than 2 endpoints
        have been added yet."""
        if not self.can_finish():
            return None
        edge = Edge(points=self._points, label=label)
        self._points = []
        return edge


class AnnotationListWidget(QWidget):
    """The current test's regions and links, with per-row removal."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(False)
        layout.addWidget(self.list_widget)
        row = QHBoxLayout()
        self.remove_btn = QPushButton("Remove selected")
        self.remove_btn.clicked.connect(self._remove_selected)
        row.addWidget(self.remove_btn)
        row.addStretch(1)
        layout.addLayout(row)
        self._on_remove: Callable[[str, int], None] | None = None

    def set_remove_callback(self, callback: Callable[[str, int], None]) -> None:
        self._on_remove = callback

    def refresh(self, bboxes: list, edges: list) -> None:
        self.list_widget.clear()
        for i, b in enumerate(bboxes):
            title = COMPARATOR_TITLES.get(b.image, b.image)
            defect = DEFECT_TITLES.get(b.defect, b.defect) or "untagged"
            severity = f" · {SEVERITY_LABELS.get(b.severity, '')}" if b.severity else ""
            detail = f" — {b.label}" if b.label else ""
            item = QListWidgetItem(f"▢ {title}: {defect}{severity}{detail}")
            item.setData(Qt.ItemDataRole.UserRole, ("bbox", i))
            item.setToolTip(f"x={b.x:.3f} y={b.y:.3f} w={b.w:.3f} h={b.h:.3f}")
            self.list_widget.addItem(item)
        for i, e in enumerate(edges):
            chain = " ⟷ ".join(COMPARATOR_TITLES.get(p.image, p.image) for p in e.points)
            item = QListWidgetItem(f"↔ {chain} — {e.label or '(no description)'}")
            item.setData(Qt.ItemDataRole.UserRole, ("edge", i))
            self.list_widget.addItem(item)
        if not bboxes and not edges:
            placeholder = QListWidgetItem("No regions or links yet.")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list_widget.addItem(placeholder)
        self.remove_btn.setEnabled(bool(bboxes or edges))

    def _remove_selected(self) -> None:
        item = self.list_widget.currentItem()
        if item is None or self._on_remove is None:
            return
        payload = item.data(Qt.ItemDataRole.UserRole)
        if not payload:
            return
        kind, index = payload
        self._on_remove(kind, index)
