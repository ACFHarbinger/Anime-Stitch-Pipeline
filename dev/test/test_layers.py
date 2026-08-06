from core.layers import LayerRule, classify, find_violations
from core.model import CodeGraph, Edge, EdgeKind, Node


def _node(node_id: str) -> Node:
    return Node(id=node_id, language="python", kind="module", qualified_name=node_id, file_path=f"{node_id}.py")


def test_classify_assigns_layer_by_longest_prefix_match():
    graph = CodeGraph(nodes=[_node("python:app.logic.io"), _node("python:app.test.io")])
    rules = [
        LayerRule(prefix="python:app.logic", label="logic"),
        LayerRule(prefix="python:app.test", label="test"),
    ]

    classify(graph, rules)

    layers = {n.id: n.layer for n in graph.nodes}
    assert layers["python:app.logic.io"] == "logic"
    assert layers["python:app.test.io"] == "test"


def test_find_violations_flags_forbidden_direction():
    graph = CodeGraph(
        nodes=[_node("logic_mod"), _node("test_mod")],
        edges=[Edge(source_id="logic_mod", target_id="test_mod", kind=EdgeKind.IMPORT)],
    )
    for node in graph.nodes:
        node.layer = "logic" if node.id == "logic_mod" else "test"

    violations = find_violations(graph, forbidden_directions=[("logic", "test")])

    assert len(violations) == 1
    assert violations[0].edge.source_id == "logic_mod"


def test_find_violations_allows_non_forbidden_direction():
    graph = CodeGraph(
        nodes=[_node("test_mod"), _node("logic_mod")],
        edges=[Edge(source_id="test_mod", target_id="logic_mod", kind=EdgeKind.IMPORT)],
    )
    for node in graph.nodes:
        node.layer = "test" if node.id == "test_mod" else "logic"

    violations = find_violations(graph, forbidden_directions=[("logic", "test")])

    assert violations == []
