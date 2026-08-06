from pathlib import Path

from cli import build_graph

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "sample_python_project"


def test_build_graph_classifies_and_merges_the_fixture_project():
    graph = build_graph(FIXTURE_ROOT)

    assert len(graph.nodes) == 4  # pkg, pkg.a, pkg.b, pkg.c
    assert len(graph.edges) == 2  # a<->b cycle
