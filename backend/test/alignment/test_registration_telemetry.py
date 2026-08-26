from __future__ import annotations

import numpy as np

from asp_backend.alignment.registration_telemetry import collect_registration_telemetry, edge_graph_components


def _edge(i: int, j: int, dx: float) -> dict:
    return {"i": i, "j": j, "M": np.array([[1, 0, dx], [0, 1, 0]], dtype=np.float32), "registration_metrics": {"observed_correspondences": True, "ransac_inlier_count": 20}}


def test_components_include_isolates_and_telemetry_is_json_safe():
    edges = [_edge(0, 1, 10), _edge(1, 2, 10), _edge(0, 2, 20)]
    affines = [np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), np.array([[1, 0, -10], [0, 1, 0]], dtype=np.float32), np.array([[1, 0, -20], [0, 1, 0]], dtype=np.float32)]
    telemetry = collect_registration_telemetry(edges, edges, affines)
    assert edge_graph_components(edges, 4) == [[0, 1, 2], [3]]
    assert telemetry["ba_residual_rms"] == 0.0
    assert telemetry["cycle_count"] == 1
