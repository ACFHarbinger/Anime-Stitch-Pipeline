"""Node-graph scene and view for the Graph sub-tab.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import os

from gui.src.constants import NODE_BODY_HEIGHT, NODE_HDR_HEIGHT
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QPainter, QWheelEvent
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QSizePolicy

from ._node_graph_items import _BaseNode, _GraphEdge, _Port, _SourceNode, _StitchOpNode


class _NodeScene(QGraphicsScene):
    """Manages source nodes, stitch-op nodes, and their connections."""

    plan_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_edge: _GraphEdge | None = None
        self._drag_src: _Port | None = None

    # ── node factory ─────────────────────────────────────────────────────

    def add_source(self, path: str, pos: QPointF | None = None) -> _SourceNode:
        if pos is None:
            x, y = self._next_pos(col=0)
        else:
            x, y = pos.x(), pos.y()
        node = _SourceNode(path, x, y)
        self.addItem(node)
        self.plan_changed.emit()
        return node

    def add_stitch_op(self, name: str = "", output: str = "") -> _StitchOpNode:
        x, y = self._next_pos(col=1)
        n = sum(1 for i in self.items() if isinstance(i, _StitchOpNode))
        node = _StitchOpNode(name or f"Op {n + 1}", output, x, y)
        self.addItem(node)
        self.plan_changed.emit()
        return node

    def remove_selected(self):
        for item in list(self.selectedItems()):
            if isinstance(item, _GraphEdge):
                item.remove_self()
            elif isinstance(item, _BaseNode):
                for p in item.input_ports + (
                    [item.output_port] if item.output_port else []
                ):
                    for e in list(p.edges):
                        e.remove_self()
                self.removeItem(item)
        self.plan_changed.emit()

    def clear_graph(self):
        self.clear()
        self._drag_edge = None
        self._drag_src = None
        self.plan_changed.emit()

    def _next_pos(self, col: int = 0) -> tuple[float, float]:
        nodes = [i for i in self.items() if isinstance(i, _BaseNode)]
        col_nodes = [n for n in nodes if (n.scenePos().x() > 260) == (col > 0)]
        y = (
            max((n.scenePos().y() for n in col_nodes), default=30)
            + NODE_HDR_HEIGHT
            + NODE_BODY_HEIGHT
            + 20
        )
        return (30.0 if col == 0 else 300.0, y if col_nodes else 30.0)

    # ── port-drag connection ──────────────────────────────────────────────

    def _port_at(self, pos) -> _Port | None:
        for item in self.items(pos):
            if isinstance(item, _Port):
                return item
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            port = self._port_at(event.scenePos())
            if port:
                if not port.is_input:
                    # Start new connection
                    self._drag_src = port
                    self._drag_edge = _GraphEdge(port)
                    self.addItem(self._drag_edge)
                    return
                elif port.is_input and port.edges:
                    # Detach existing connection
                    edge = port.edges.pop()
                    self._drag_src = edge.src
                    self._drag_edge = edge
                    edge.dst = None
                    edge.update_path(event.scenePos())
                    self.plan_changed.emit()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_edge:
            self._drag_edge.update_path(event.scenePos())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_edge and self._drag_src:
            dst = self._port_at(event.scenePos())
            if (
                dst
                and dst.is_input
                and dst.node is not self._drag_src.node
                and not dst.edges
            ):
                self._drag_edge.dst = dst
                self._drag_src.edges.append(self._drag_edge)
                dst.edges.append(self._drag_edge)
                self._drag_edge.update_path()
                self.plan_changed.emit()
            else:
                # Remove from source port list if it was already registered
                if self._drag_edge in self._drag_src.edges:
                    self._drag_src.edges.remove(self._drag_edge)
                self.removeItem(self._drag_edge)
                self.plan_changed.emit()
            self._drag_edge = None
            self._drag_src = None
            return
        super().mouseReleaseEvent(event)

    # ── plan extraction ───────────────────────────────────────────────────

    def get_plan(self) -> list[dict]:
        ops: list[_StitchOpNode] = [
            i for i in self.items() if isinstance(i, _StitchOpNode)
        ]
        if not ops:
            return []

        id_map: dict[int, str] = {id(op): f"op_{k}" for k, op in enumerate(ops)}

        def _inputs_for(op: _StitchOpNode) -> list[str]:
            res = []
            for port in op.input_ports:
                for edge in port.edges:
                    src = edge.src.node
                    if isinstance(src, _SourceNode):
                        res.append(src.path)
                    elif isinstance(src, _StitchOpNode):
                        res.append(id_map[id(src)])
            return res

        # Kahn topological sort
        in_deg: dict[int, int] = {id(op): 0 for op in ops}
        deps: dict[int, list[_StitchOpNode]] = {id(op): [] for op in ops}
        for op in ops:
            for inp in _inputs_for(op):
                for dep in ops:
                    if id_map[id(dep)] == inp:
                        in_deg[id(op)] += 1
                        deps[id(dep)].append(op)

        queue = [op for op in ops if in_deg[id(op)] == 0]
        ordered: list[_StitchOpNode] = []
        while queue:
            cur = queue.pop(0)
            ordered.append(cur)
            for dep in deps[id(cur)]:
                in_deg[id(dep)] -= 1
                if in_deg[id(dep)] == 0:
                    queue.append(dep)

        return [
            {
                "id": id_map[id(op)],
                "name": op.step_name,
                "inputs": _inputs_for(op),
                "output": op.output_path,
            }
            for op in ordered
        ]


class _NodeView(QGraphicsView):
    """Zoomable / pannable view for the node graph canvas."""

    def __init__(self, scene: _NodeScene):
        super().__init__(scene)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)  # Managed in events
        self.setAcceptDrops(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#1a1a1e; border:none;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        added = False
        for i, url in enumerate(urls):
            fpath = url.toLocalFile()
            if not fpath:
                continue
            ext = os.path.splitext(fpath)[1].lower()
            if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
                # Offset multiple images slightly so they don't stack perfectly
                drop_pos = scene_pos + QPointF(i * 20, i * 20)
                self.scene().add_source(fpath, pos=drop_pos) # pyrefly: ignore[missing-attribute]
                added = True
        if added:
            event.acceptProposedAction()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if not item:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            else:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            f = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(f, f)
        else:
            super().wheelEvent(event)


__all__ = ["_NodeScene", "_NodeView"]
