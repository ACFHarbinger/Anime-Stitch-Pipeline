"""Node-graph item classes (ports, edges, nodes) for the Graph sub-tab.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import os

from gui.src.constants import (
    EDGE_COLOR,
    NODE_BODY_HEIGHT,
    NODE_HDR_HEIGHT,
    NODE_THUMB_HEIGHT,
    NODE_WIDTH,
    PORT_RADIUS,
)
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImageReader,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
)


class _Port(QGraphicsEllipseItem):
    """Input (left) or output (right) connection port on a graph node."""

    def __init__(self, node, is_input: bool, index: int = 0):
        r = PORT_RADIUS
        super().__init__(-r, -r, r * 2, r * 2)
        self.node = node
        self.is_input = is_input
        self.index = index
        self.edges: list[_GraphEdge] = []
        self.setZValue(15)
        self.setAcceptHoverEvents(True)
        color = QColor(100, 180, 100) if is_input else QColor(80, 160, 230)
        self.setBrush(QBrush(color))
        pen = QPen(Qt.GlobalColor.white)
        pen.setWidth(1)
        self.setPen(pen)

    def scene_center(self):
        return self.mapToScene(QPointF(0, 0))

    def hoverEnterEvent(self, event):
        self.setScale(1.35)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def update_edges(self):
        for e in self.edges:
            e.update_path()


class _GraphEdge(QGraphicsPathItem):
    """Cubic-Bezier edge from an output port to an input port."""

    def __init__(self, src: _Port, dst: _Port | None = None):
        super().__init__()
        self.src = src
        self.dst = dst
        pen = QPen(EDGE_COLOR, 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)
        self.setZValue(5)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.update_path()

    def update_path(self, end_pos=None):
        p1 = self.src.scene_center()
        p2 = (
            end_pos
            if end_pos is not None
            else (self.dst.scene_center() if self.dst else p1)
        )
        dx = max(abs(p2.x() - p1.x()) * 0.5, 60)
        path = QPainterPath(p1)
        path.cubicTo(p1 + QPointF(dx, 0), p2 - QPointF(dx, 0), p2)
        self.setPath(path)

    def remove_self(self):
        if self in self.src.edges:
            self.src.edges.remove(self)
        if self.dst and self in self.dst.edges:
            self.dst.edges.remove(self)
        if self.scene():
            self.scene().removeItem(self)


class _BaseNode(QGraphicsRectItem):
    """Draggable rounded-rect node with title bar and ports."""

    def __init__(self, title: str, hdr_color: QColor, x: float = 0, y: float = 0):
        h = NODE_HDR_HEIGHT + NODE_BODY_HEIGHT
        super().__init__(0, 0, NODE_WIDTH, h)
        self.setPos(x, y)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        self.setZValue(10)
        self._title = title
        self._hdr_color = hdr_color
        self._input_ports: list[_Port] = []
        self._output_port: _Port | None = None
        self.setPen(QPen(QColor(70, 70, 75), 1))
        self.setBrush(QBrush(QColor(42, 42, 48)))

    # ── port helpers ─────────────────────────────────────────────────────

    def _place_input_ports(self):
        n = len(self._input_ports)
        h = self.rect().height()
        for i, p in enumerate(self._input_ports):
            p.setPos(0, NODE_HDR_HEIGHT + (i + 1) * (h - NODE_HDR_HEIGHT) / (n + 1))

    def add_input_port(self) -> _Port:
        p = _Port(self, is_input=True, index=len(self._input_ports))
        p.setParentItem(self)
        self._input_ports.append(p)
        self._place_input_ports()
        return p

    def set_output_port(self) -> _Port:
        p = _Port(self, is_input=False)
        p.setParentItem(self)
        p.setPos(NODE_WIDTH, self.rect().height() / 2)
        self._output_port = p
        return p

    @property
    def output_port(self):
        return self._output_port

    @property
    def input_ports(self):
        return self._input_ports

    # ── drawing ──────────────────────────────────────────────────────────

    def paint(self, painter: QPainter, option, widget=None):
        r = self.rect()
        # body
        painter.setBrush(QBrush(QColor(42, 42, 48)))
        sel_pen = (
            QPen(QColor(0, 188, 212), 2)
            if self.isSelected()
            else QPen(QColor(70, 70, 75), 1)
        )
        painter.setPen(sel_pen)
        painter.drawRoundedRect(r, 7, 7)
        # header fill clipped to top rounded corners
        painter.save()
        painter.setClipRect(QRectF(0, 0, NODE_WIDTH, NODE_HDR_HEIGHT))
        painter.setBrush(QBrush(self._hdr_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(r, 7, 7)
        painter.restore()
        # title
        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont("sans-serif", 8, QFont.Weight.Bold))
        painter.drawText(
            QRectF(10, 0, NODE_WIDTH - 20, NODE_HDR_HEIGHT),
            Qt.AlignmentFlag.AlignVCenter,
            self._title,
        )
        # body text
        painter.setFont(QFont("sans-serif", 7))
        painter.setPen(QColor(190, 190, 190))
        painter.drawText(
            QRectF(
                10,
                NODE_HDR_HEIGHT + 4,
                NODE_WIDTH - 20,
                self.rect().height() - NODE_HDR_HEIGHT - 8,
            ),
            Qt.AlignmentFlag.AlignTop
            | Qt.AlignmentFlag.AlignLeft
            | Qt.TextFlag.TextWordWrap,
            self._body_text(),
        )

    def _body_text(self) -> str:
        return ""

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged:
            for p in self._input_ports:
                p.update_edges()
            if self._output_port:
                self._output_port.update_edges()
        return super().itemChange(change, value)


class _SourceNode(_BaseNode):
    """Image-source node — no inputs, one output. Shows an inline thumbnail."""

    _HDR = QColor(30, 90, 140)

    def __init__(self, path: str, x: float = 0, y: float = 0):
        # Taller rect to accommodate thumbnail
        super().__init__(os.path.basename(path), self._HDR, x, y)
        self.path = path
        self.setRect(0, 0, NODE_WIDTH, NODE_HDR_HEIGHT + NODE_THUMB_HEIGHT)
        self._thumb: QPixmap | None = self._load_thumb(path)
        self.set_output_port()  # repositioned after setRect

    @staticmethod
    def _load_thumb(path: str) -> QPixmap | None:
        """Load a downscaled thumbnail efficiently via QImageReader."""
        reader = QImageReader(path)
        if not reader.canRead():
            return None
        orig = reader.size()
        if orig.isValid() and orig.width() > 0:
            tw = NODE_WIDTH - 4
            th = NODE_THUMB_HEIGHT - 4
            scale = min(tw / orig.width(), th / orig.height())
            reader.setScaledSize(
                QSize(
                    max(1, int(orig.width() * scale)),
                    max(1, int(orig.height() * scale)),
                )
            )
        qi = reader.read()
        return QPixmap.fromImage(qi) if not qi.isNull() else None

    def paint(self, painter: QPainter, option, widget=None):
        # Draw base (header + border)
        super().paint(painter, option, widget)
        # Draw thumbnail centred in body area
        if self._thumb:
            px = self._thumb
            x = int((NODE_WIDTH - px.width()) / 2)
            y = NODE_HDR_HEIGHT + int((NODE_THUMB_HEIGHT - px.height()) / 2)
            painter.drawPixmap(x, y, px)
        else:
            # Fallback: dim placeholder
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("sans-serif", 8))
            painter.drawText(
                QRectF(0, NODE_HDR_HEIGHT, NODE_WIDTH, NODE_THUMB_HEIGHT),
                Qt.AlignmentFlag.AlignCenter,
                "(no preview)",
            )

    def _body_text(self) -> str:
        return ""  # thumbnail replaces text


class _StitchOpNode(_BaseNode):
    """Stitch-operation node — N inputs, one output."""

    _HDR = QColor(90, 50, 130)

    def __init__(self, name: str, output_path: str = "", x: float = 0, y: float = 0):
        super().__init__(f"⊞ {name}", self._HDR, x, y)
        self.step_name = name
        self.output_path = output_path
        self.add_input_port()
        self.add_input_port()
        self.set_output_port()

    def grow_input(self):
        self.add_input_port()
        new_h = NODE_HDR_HEIGHT + NODE_BODY_HEIGHT + (len(self._input_ports) - 2) * 22
        self.prepareGeometryChange()
        self.setRect(0, 0, NODE_WIDTH, new_h)
        self._place_input_ports()
        if self._output_port:
            self._output_port.setPos(NODE_WIDTH, new_h / 2)
        self.update()

    def _body_text(self) -> str:
        n_conn = sum(1 for p in self._input_ports if p.edges)
        return f"Inputs connected: {n_conn}/{len(self._input_ports)}"


__all__ = ["_Port", "_GraphEdge", "_BaseNode", "_SourceNode", "_StitchOpNode"]
