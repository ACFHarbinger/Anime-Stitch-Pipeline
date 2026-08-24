"""Focused tests for the default-off CleanCP local re-solve helper."""

from __future__ import annotations

import numpy as np
from asp_backend.core.pipeline._cleancp_recovery import (
    recover_clean_correspondence_edges,
)


def _edge(i: int, j: int, dy: float, *, point_outlier: bool = False) -> dict:
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, dy]], dtype=np.float32)
    pts_i = np.array([[x, y] for x in range(4) for y in range(2)], dtype=np.float32)
    pts_j = pts_i + matrix[:, 2]
    if point_outlier:
        pts_j[-1] += [300.0, -200.0]
    return {"i": i, "j": j, "M": matrix, "pts_i": pts_i, "pts_j": pts_j, "weight": 0.9}


def test_recovers_connected_consensus_and_removes_bad_control_point():
    raw_edges = [
        _edge(0, 1, 100.0),
        _edge(1, 2, 100.0, point_outlier=True),
        _edge(2, 3, 100.0),
        _edge(0, 3, 900.0),
    ]
    recovered, telemetry = recover_clean_correspondence_edges(raw_edges, [], 4)

    assert [(edge["i"], edge["j"]) for edge in recovered] == [(0, 1), (1, 2), (2, 3)]
    assert len(recovered[1]["pts_i"]) == 7
    assert telemetry["accepted"] is True
    assert telemetry["correspondences_removed"] == 1
    assert telemetry["outlier_candidates_removed"] == 1
    assert telemetry["components"]["after"] == [[0, 1, 2, 3]]
    assert telemetry["missing_adjacent_edge_count"] == {
        "before": 3,
        "candidate_consensus": 0,
        "after": 0,
    }


def test_keeps_filtered_edges_when_consensus_cannot_connect_graph():
    filtered = [_edge(0, 1, 100.0)]
    raw_edges = [
        _edge(0, 1, 100.0),
        _edge(1, 2, 500.0),
        _edge(2, 3, -300.0),
    ]
    recovered, telemetry = recover_clean_correspondence_edges(raw_edges, filtered, 4)

    assert recovered is filtered
    assert telemetry["accepted"] is False
    assert telemetry["stopped_reason"] == "consensus_graph_disconnected"
    assert telemetry["components"]["after"] == [[0, 1], [2], [3]]


def test_requires_three_raw_candidates_before_attempting_recovery():
    filtered = [_edge(0, 1, 100.0)]
    recovered, telemetry = recover_clean_correspondence_edges(filtered, filtered, 3)

    assert recovered is filtered
    assert telemetry["accepted"] is False
    assert telemetry["stopped_reason"] == "insufficient_raw_candidates"
