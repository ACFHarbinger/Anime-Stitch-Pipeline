from pathlib import Path

from input.python.parser import parse_directory

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "sample_python_project"


def test_parse_directory_finds_all_modules():
    graph = parse_directory(FIXTURE_ROOT)

    node_ids = {n.id for n in graph.nodes}
    assert "python:pkg.a" in node_ids
    assert "python:pkg.b" in node_ids
    assert "python:pkg.c" in node_ids
    assert graph.source_language == "python"


def test_parse_directory_finds_the_ab_cycle_edges():
    graph = parse_directory(FIXTURE_ROOT)

    edge_pairs = {(e.source_id, e.target_id) for e in graph.edges}
    assert ("python:pkg.a", "python:pkg.b") in edge_pairs
    assert ("python:pkg.b", "python:pkg.a") in edge_pairs


def test_parse_directory_does_not_edge_to_c():
    graph = parse_directory(FIXTURE_ROOT)

    targets_from_a_or_b = {
        e.target_id for e in graph.edges if e.source_id in {"python:pkg.a", "python:pkg.b"}
    }
    assert "python:pkg.c" not in targets_from_a_or_b
