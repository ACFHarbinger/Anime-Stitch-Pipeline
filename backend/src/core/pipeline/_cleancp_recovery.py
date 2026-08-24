"""Default-off CleanCP-style correspondence cleanup before a local BA retry."""

from __future__ import annotations

import numpy as np

_MIN_CORRESPONDENCES = 3
_MIN_CONSENSUS_EDGES = 3
_MIN_RESIDUAL_CUTOFF_PX = 5.0
_MIN_TRANSLATION_CUTOFF_PX = 20.0


def _missing_adjacent_edge_count(edges: list[dict], n_frames: int) -> int:
    """Count frame-neighbour links absent from an edge set."""
    present = {
        (min(int(edge.get("i", -1)), int(edge.get("j", -1))),
         max(int(edge.get("i", -1)), int(edge.get("j", -1))))
        for edge in edges
    }
    return sum((index, index + 1) not in present for index in range(n_frames - 1))


def _edge_graph_components(edges: list[dict], n_frames: int) -> list[list[int]]:
    """Return sorted undirected edge-graph components without telemetry imports."""
    neighbours = [set() for _ in range(n_frames)]
    for edge in edges:
        i, j = int(edge.get("i", -1)), int(edge.get("j", -1))
        if 0 <= i < n_frames and 0 <= j < n_frames:
            neighbours[i].add(j)
            neighbours[j].add(i)
    unseen = set(range(n_frames))
    components: list[list[int]] = []
    while unseen:
        pending = [min(unseen)]
        component: list[int] = []
        while pending:
            node = pending.pop()
            if node not in unseen:
                continue
            unseen.remove(node)
            component.append(node)
            pending.extend(neighbours[node] & unseen)
        components.append(sorted(component))
    return components


def _edge_with_clean_correspondences(edge: dict) -> tuple[dict | None, int]:
    """Drop MAD outlier control points while preserving the matcher's transform."""
    pts_i = np.asarray(edge.get("pts_i", ()), dtype=np.float32)
    pts_j = np.asarray(edge.get("pts_j", ()), dtype=np.float32)
    count = min(len(pts_i), len(pts_j))
    if count < _MIN_CORRESPONDENCES:
        return None, 0

    pts_i, pts_j = pts_i[:count], pts_j[:count]
    matrix = np.asarray(edge.get("M"), dtype=np.float32)
    if matrix.shape != (2, 3):
        return None, 0
    predicted = pts_i @ matrix[:, :2].T + matrix[:, 2]
    residuals = np.linalg.norm(pts_j - predicted, axis=1)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    cutoff = max(_MIN_RESIDUAL_CUTOFF_PX, median + 3.0 * 1.4826 * mad)
    keep = residuals <= cutoff
    if int(keep.sum()) < _MIN_CORRESPONDENCES:
        return None, count
    return dict(edge, pts_i=pts_i[keep], pts_j=pts_j[keep]), int((~keep).sum())


def _normalized_translation(edge: dict) -> np.ndarray:
    span = int(edge["j"]) - int(edge["i"])
    return np.asarray(edge["M"][:2, 2], dtype=float) / span


def recover_clean_correspondence_edges(
    raw_edges: list[dict],
    filtered_edges: list[dict],
    n_frames: int,
) -> tuple[list[dict], dict]:
    """Return a connected robust raw-edge subset, or the filtered edges unchanged.

    This is intentionally conservative: it needs three locally clean candidate
    edges to establish a translation consensus and only replaces the filtered
    set when that consensus connects the entire frame graph.
    """
    before_components = _edge_graph_components(filtered_edges, n_frames)
    telemetry = {
        "attempted": True,
        "accepted": False,
        "raw_candidates": len(raw_edges),
        "locally_clean_candidates": 0,
        "correspondences_removed": 0,
        "outlier_candidates_removed": 0,
        "components": {"before": before_components, "after": before_components},
        "missing_adjacent_edge_count": {
            "before": _missing_adjacent_edge_count(filtered_edges, n_frames),
            "after": _missing_adjacent_edge_count(filtered_edges, n_frames),
        },
    }
    if n_frames <= 1 or len(raw_edges) < _MIN_CONSENSUS_EDGES:
        telemetry["stopped_reason"] = "insufficient_raw_candidates"
        return filtered_edges, telemetry

    clean_edges: list[dict] = []
    for edge in raw_edges:
        i, j = int(edge.get("i", -1)), int(edge.get("j", -1))
        if not (0 <= i < j < n_frames):
            continue
        cleaned, removed = _edge_with_clean_correspondences(edge)
        telemetry["correspondences_removed"] += removed
        if cleaned is not None:
            clean_edges.append(cleaned)

    telemetry["locally_clean_candidates"] = len(clean_edges)
    if len(clean_edges) < _MIN_CONSENSUS_EDGES:
        telemetry["stopped_reason"] = "insufficient_clean_candidates"
        return filtered_edges, telemetry

    translations = np.asarray([_normalized_translation(edge) for edge in clean_edges])
    median = np.median(translations, axis=0)
    residuals = np.linalg.norm(translations - median, axis=1)
    residual_mad = float(np.median(np.abs(residuals - np.median(residuals))))
    cutoff = max(
        _MIN_TRANSLATION_CUTOFF_PX,
        3.0 * 1.4826 * residual_mad,
    )
    consensus_edges = [
        edge
        for edge, residual in zip(clean_edges, residuals, strict=True)
        if residual <= cutoff
    ]
    telemetry.update(
        {
            "median_span_translation": [round(float(value), 4) for value in median],
            "translation_cutoff_px": round(cutoff, 4),
            "outlier_candidates_removed": len(clean_edges) - len(consensus_edges),
            "components": {
                "before": before_components,
                "candidate_consensus": _edge_graph_components(consensus_edges, n_frames),
                "after": before_components,
            },
            "missing_adjacent_edge_count": {
                "before": _missing_adjacent_edge_count(filtered_edges, n_frames),
                "candidate_consensus": _missing_adjacent_edge_count(
                    consensus_edges, n_frames
                ),
                "after": _missing_adjacent_edge_count(filtered_edges, n_frames),
            },
        }
    )
    after_components = _edge_graph_components(consensus_edges, n_frames)
    if len(after_components) != 1:
        telemetry["stopped_reason"] = "consensus_graph_disconnected"
        return filtered_edges, telemetry

    telemetry["accepted"] = True
    telemetry["components"]["after"] = after_components
    telemetry["missing_adjacent_edge_count"]["after"] = _missing_adjacent_edge_count(
        consensus_edges, n_frames
    )
    return consensus_edges, telemetry


__all__ = ["recover_clean_correspondence_edges"]
