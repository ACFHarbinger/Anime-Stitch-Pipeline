from core.cycles import find_cycles
from core.model import CodeGraph, Edge, EdgeKind, Node


def _node(node_id: str) -> Node:
    return Node(id=node_id, language="python", kind="module", qualified_name=node_id, file_path=f"{node_id}.py")


def test_find_cycles_detects_a_two_node_cycle():
    graph = CodeGraph(
        nodes=[_node("a"), _node("b"), _node("c")],
        edges=[
            Edge(source_id="a", target_id="b", kind=EdgeKind.IMPORT),
            Edge(source_id="b", target_id="a", kind=EdgeKind.IMPORT),
        ],
    )

    cycles = find_cycles(graph)

    assert len(cycles) == 1
    assert set(cycles[0]) == {"a", "b"}


def test_find_cycles_returns_empty_for_a_dag():
    graph = CodeGraph(
        nodes=[_node("a"), _node("b"), _node("c")],
        edges=[
            Edge(source_id="a", target_id="b", kind=EdgeKind.IMPORT),
            Edge(source_id="b", target_id="c", kind=EdgeKind.IMPORT),
        ],
    )

    assert find_cycles(graph) == []


def test_find_cycles_detects_a_self_loop():
    graph = CodeGraph(
        nodes=[_node("a")],
        edges=[Edge(source_id="a", target_id="a", kind=EdgeKind.IMPORT)],
    )

    cycles = find_cycles(graph)

    assert cycles == [["a"]]


def test_find_cycles_detects_a_three_node_cycle():
    graph = CodeGraph(
        nodes=[_node("a"), _node("b"), _node("c")],
        edges=[
            Edge(source_id="a", target_id="b", kind=EdgeKind.IMPORT),
            Edge(source_id="b", target_id="c", kind=EdgeKind.IMPORT),
            Edge(source_id="c", target_id="a", kind=EdgeKind.IMPORT),
        ],
    )

    cycles = find_cycles(graph)

    assert len(cycles) == 1
    assert set(cycles[0]) == {"a", "b", "c"}
