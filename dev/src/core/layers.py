"""Layer classification and forbidden-direction violation detection.

Ported from the ``DEFAULT_LAYERS``/``FORBIDDEN_DIRECTIONS`` pattern
independently built in Build-Optimization's and WSmart-Route's
``visualize_module_graph.py`` (see moon/roadmaps/developer_tools.md D3),
generalized to operate on the shared :class:`~dev.src.core.model.CodeGraph`
instead of being Python-import-specific.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.model import CodeGraph, Edge


@dataclass(frozen=True)
class LayerRule:
    """One layer definition: a node-id prefix mapped to a layer label.

    Attributes:
        prefix: Node-id prefix this rule matches, e.g. ``"python:app.logic"``.
        label: Human-readable layer name, e.g. ``"logic"``.
    """

    prefix: str
    label: str


@dataclass(frozen=True)
class Violation:
    """A single forbidden cross-layer edge.

    Attributes:
        edge: The offending edge.
        source_layer: Resolved layer of ``edge.source_id``.
        target_layer: Resolved layer of ``edge.target_id``.
    """

    edge: Edge
    source_layer: str
    target_layer: str


def classify(graph: CodeGraph, rules: list[LayerRule]) -> CodeGraph:
    """Assign a ``layer`` to every node in ``graph``, longest-prefix-first.

    Args:
        graph: The graph to classify. Mutated in place (node ``layer``
            fields are set) and also returned for convenience.
        rules: Layer rules, checked longest-``prefix``-first so a more
            specific rule (e.g. ``"python:app.logic.io"``) wins over a
            broader one (e.g. ``"python:app.logic"``).

    Returns:
        The same ``graph`` instance, with ``node.layer`` populated.
    """
    ordered = sorted(rules, key=lambda r: len(r.prefix), reverse=True)
    for node in graph.nodes:
        for rule in ordered:
            if node.id.startswith(rule.prefix) or node.qualified_name.startswith(rule.prefix):
                node.layer = rule.label
                break
    return graph


def find_violations(
    graph: CodeGraph, forbidden_directions: list[tuple[str, str]]
) -> list[Violation]:
    """Find edges that cross a forbidden layer boundary.

    Args:
        graph: A graph whose nodes have already been :func:`classify`-d.
        forbidden_directions: Pairs of ``(from_layer, to_layer)`` that must
            never appear as an edge, e.g. ``[("test", "logic")]`` would flag
            test code importing from production logic modules... note the
            more common direction to forbid is the reverse (``logic`` must
            not depend on ``test``); pick pairs deliberately per project.

    Returns:
        One :class:`Violation` per matching edge, in graph edge order.
    """
    layer_by_id = {n.id: n.layer for n in graph.nodes}
    forbidden = set(forbidden_directions)
    violations: list[Violation] = []
    for edge in graph.edges:
        source_layer = layer_by_id.get(edge.source_id, "")
        target_layer = layer_by_id.get(edge.target_id, "")
        if (source_layer, target_layer) in forbidden:
            violations.append(
                Violation(edge=edge, source_layer=source_layer, target_layer=target_layer)
            )
    return violations
