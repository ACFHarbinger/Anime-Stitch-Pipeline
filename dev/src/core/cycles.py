"""Circular-dependency detection via Tarjan's strongly-connected-components
algorithm.

See moon/roadmaps/developer_tools.md D4 (ported concept from Image-Toolkit's
``check_circular_imports.py``, reimplemented against the shared
:class:`~dev.src.core.model.CodeGraph` instead of a Python-only import map).

Only SCCs of size > 1 are circular dependencies in the usual sense; a
self-loop (a node with an edge to itself) is also reported as a
single-element cycle.
"""

from __future__ import annotations

from core.model import CodeGraph


def find_cycles(graph: CodeGraph) -> list[list[str]]:
    """Find all circular-dependency groups in ``graph``.

    Args:
        graph: The graph to analyze. Only ``edges`` (as a directed adjacency
            structure over node ids) are used.

    Returns:
        A list of cycles, each a list of node ids forming a strongly
        connected component of size > 1, or a size-1 component with a
        self-loop. Order is Tarjan's discovery order, not guaranteed stable
        across runs with different edge orderings.
    """
    adjacency: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    for edge in graph.edges:
        adjacency.setdefault(edge.source_id, []).append(edge.target_id)
        adjacency.setdefault(edge.target_id, [])

    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    low_links: dict[str, int] = {}
    result: list[list[str]] = []

    def strongconnect(node_id: str) -> None:
        indices[node_id] = index_counter[0]
        low_links[node_id] = index_counter[0]
        index_counter[0] += 1
        stack.append(node_id)
        on_stack.add(node_id)

        for neighbor in adjacency.get(node_id, []):
            if neighbor not in indices:
                strongconnect(neighbor)
                low_links[node_id] = min(low_links[node_id], low_links[neighbor])
            elif neighbor in on_stack:
                low_links[node_id] = min(low_links[node_id], indices[neighbor])

        if low_links[node_id] == indices[node_id]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node_id:
                    break
            is_self_loop = len(component) == 1 and node_id in adjacency.get(node_id, [])
            if len(component) > 1 or is_self_loop:
                result.append(component)

    for node_id in adjacency:
        if node_id not in indices:
            strongconnect(node_id)

    return result
