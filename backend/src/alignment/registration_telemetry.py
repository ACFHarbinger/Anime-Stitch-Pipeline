"""JSON-safe, measurement-only registration diagnostics."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def edge_graph_components(edges: list[dict], n_frames: int) -> list[list[int]]:
    """Return sorted undirected components, including isolated frames."""
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    graph = [set() for _ in range(n_frames)]
    for edge in edges:
        i, j = int(edge.get("i", -1)), int(edge.get("j", -1))
        if 0 <= i < n_frames and 0 <= j < n_frames:
            graph[i].add(j)
            graph[j].add(i)
    unseen, components = set(range(n_frames)), []
    while unseen:
        stack, component = [min(unseen)], []
        unseen.remove(stack[0])
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in graph[node] & unseen:
                unseen.remove(neighbour)
                stack.append(neighbour)
        components.append(sorted(component))
    return components


def _edge_residual(edge: dict, affines: list[np.ndarray]) -> float | None:
    i, j = int(edge["i"]), int(edge["j"])
    if not (0 <= i < len(affines) and 0 <= j < len(affines)):
        return None
    predicted = np.array([
        float(affines[j][0, 2]) - float(affines[i][0, 2]),
        float(affines[j][1, 2]) - float(affines[i][1, 2]),
    ])
    return float(np.linalg.norm(predicted + np.asarray(edge["M"][:2, 2], dtype=float)))


def _cycle_errors(edges: list[dict]) -> list[float]:
    transforms, nodes = {}, set()
    for edge in edges:
        i, j = int(edge["i"]), int(edge["j"])
        vector = np.asarray(edge["M"][:2, 2], dtype=float)
        transforms[i, j], transforms[j, i] = vector, -vector
        nodes.update((i, j))
    return [
        float(np.linalg.norm(transforms[i, j] + transforms[j, k] + transforms[k, i]))
        for i, j, k in combinations(sorted(nodes), 3)
        if (i, j) in transforms and (j, k) in transforms and (k, i) in transforms
    ]


def collect_registration_telemetry(raw_edges: list[dict], filtered_edges: list[dict], affines: list[np.ndarray], pair_proposal: dict | None = None) -> dict:
    """Return solve-independent registration evidence for the M2 gate."""
    residuals = [value for edge in filtered_edges if (value := _edge_residual(edge, affines)) is not None]
    cycles = _cycle_errors(filtered_edges)
    pair_edges = filtered_edges or raw_edges
    return {
        "raw_edges": len(raw_edges), "filtered_edges": len(filtered_edges),
        "per_pair_source": "filtered" if filtered_edges else "raw",
        "per_pair": [{"i": int(edge["i"]), "j": int(edge["j"]), **{
            key: edge.get("registration_metrics", {}).get(key)
            for key in ("observed_correspondences", "ransac_inlier_count", "ransac_inlier_ratio", "reprojection_rms")
        }} for edge in pair_edges],
        "ba_residual_rms": round(float(np.sqrt(np.mean(np.square(residuals)))), 4) if residuals else None,
        "ba_residual_p95": round(float(np.percentile(residuals, 95)), 4) if residuals else None,
        "cycle_count": len(cycles),
        "cycle_error_rms": round(float(np.sqrt(np.mean(np.square(cycles)))), 4) if cycles else None,
        "cycle_error_p95": round(float(np.percentile(cycles, 95)), 4) if cycles else None,
        "pair_proposal": pair_proposal or {},
    }


__all__ = ["collect_registration_telemetry", "edge_graph_components"]
