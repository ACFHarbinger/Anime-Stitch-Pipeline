from core.model import CodeGraph, Edge, EdgeKind, Node


def test_codegraph_to_dict_and_from_dict_roundtrip():
    graph = CodeGraph(
        nodes=[Node(id="python:a", language="python", kind="module", qualified_name="a", file_path="a.py")],
        edges=[Edge(source_id="python:a", target_id="python:b", kind=EdgeKind.IMPORT)],
        source_language="python",
    )
    payload = graph.to_dict()
    restored = CodeGraph.from_dict(payload)

    assert restored.source_language == "python"
    assert len(restored.nodes) == 1
    assert restored.nodes[0].id == "python:a"
    assert len(restored.edges) == 1
    assert restored.edges[0].source_id == "python:a"
    assert restored.edges[0].target_id == "python:b"
    assert restored.edges[0].kind == EdgeKind.IMPORT


def test_codegraph_merge_combines_nodes_edges_and_language():
    a = CodeGraph(nodes=[Node(id="python:a", language="python", kind="module", qualified_name="a", file_path="a.py")], source_language="python")
    b = CodeGraph(nodes=[Node(id="typescript:b", language="typescript", kind="module", qualified_name="b", file_path="b.ts")], source_language="typescript")

    merged = a.merge(b)

    assert len(merged.nodes) == 2
    assert merged.source_language == "python,typescript"
