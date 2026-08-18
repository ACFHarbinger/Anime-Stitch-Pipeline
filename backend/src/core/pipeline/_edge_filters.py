"""Edge-graph filtering utilities used by ``AnimeStitchPipeline._filter_edges``
and ``run()``'s affine-validation retry chain."""

from __future__ import annotations

import numpy as np
try:
    from backend.src.constants.animation import (
        ADAPTIVE_MIN_DISP_FRAC,
        HIGH_CONF_EDGE_THRESH,
        STATIC_EDGE_MIN_DISP_PX,
    )
except ImportError:
    try:
        from backend.src.constants import (
            ADAPTIVE_MIN_DISP_FRAC,
            HIGH_CONF_EDGE_THRESH,
            STATIC_EDGE_MIN_DISP_PX,
        )
    except ImportError:
        ADAPTIVE_MIN_DISP_FRAC = 0.10
        HIGH_CONF_EDGE_THRESH = 0.65
        STATIC_EDGE_MIN_DISP_PX = 50.0



def _reject_static_edges(
    edges: list[dict],
    min_disp_px: float = STATIC_EDGE_MIN_DISP_PX,
) -> list[dict]:
    """§1.2A — Drop edges where |dx| < min_disp_px AND |dy| < min_disp_px.

    Rejects near-zero-2D-displacement matches for ALL edges (adjacent and
    skip-frame).  When such edges survive into bundle adjustment they anchor
    two frames at essentially the same canvas position, corrupting the global
    translation estimate for the rest of the sequence.

    A match is kept if EITHER axis displacement meets or exceeds the threshold,
    so valid diagonal-scroll edges (large |dx|, small |dy|) are preserved.
    """
    return [
        e
        for e in edges
        if abs(float(e["M"][0, 2])) >= min_disp_px
        or abs(float(e["M"][1, 2])) >= min_disp_px
    ]


def _compute_adaptive_min_disp(edges: list[dict]) -> float:
    """§1.2C — Content-adaptive minimum displacement threshold.

    Estimates the expected inter-frame step from the median of adjacent-edge
    displacements on the dominant scroll axis and returns
    ``max(STATIC_EDGE_MIN_DISP_PX, ADAPTIVE_MIN_DISP_FRAC * expected_step)``.

    For typical scroll sequences the floor dominates (step ≤ 500 px → 10% ≤
    50 px).  For high-resolution or fast-scroll content the adaptive value
    exceeds the floor and provides proportionally stronger rejection (e.g.,
    1 000 px/frame → threshold 100 px instead of 50 px).
    """
    adj_edges = [e for e in edges if e["j"] == e["i"] + 1]
    if not adj_edges:
        return float(STATIC_EDGE_MIN_DISP_PX)

    adx = np.array([abs(float(e["M"][0, 2])) for e in adj_edges])
    ady = np.array([abs(float(e["M"][1, 2])) for e in adj_edges])
    disps = adx if float(np.median(adx)) >= float(np.median(ady)) else ady

    expected_step = float(np.median(disps))
    return max(float(STATIC_EDGE_MIN_DISP_PX), ADAPTIVE_MIN_DISP_FRAC * expected_step)


def _filter_high_conf_edges(
    edges: list[dict],
    min_weight: float = HIGH_CONF_EDGE_THRESH,
) -> list[dict]:
    """§2.9C — Keep only edges whose match weight meets the high-confidence floor.

    LoFTR edges typically have ``weight`` in [0.7, 0.95]; template-match and
    phase-correlation fallbacks land in [0.15, 0.55].  When bundle adjustment
    produces a bad ratio (one outlier edge pulling frames together), filtering
    to high-confidence edges removes the low-quality fallback edges that are
    most likely to be wrong.

    Used as a pre-check before the existing Retry-1 (adjacent-only) path: if
    at least ``N-1`` high-confidence edges survive, re-solve the bundle.  If
    fewer survive, fall through to Retry 1 unchanged — no information is lost.
    """
    return [e for e in edges if float(e.get("weight", 0.0)) >= min_weight]


def _check_edge_graph_connectivity(
    edges: list[dict],
    n_frames: int,
) -> bool:
    """§1.15: Return True iff all frames 0..n_frames-1 are in one connected component.

    Uses iterative path-compression Union-Find (same algorithm as §1.1B spanning
    tree) to check graph connectivity after all edge filters have run.  A
    disconnected graph fed into bundle adjustment assigns wrong translations to
    isolated frames — catching this before BA allows an immediate fallback rather
    than a corrupt solve followed by a downstream validation failure.

    Trivially returns True when *n_frames* ≤ 1 (nothing to connect) or when
    *n_frames* − 1 edges already span all nodes (lower bound for connectivity).
    """
    if n_frames <= 1:
        return True

    parent = list(range(n_frames))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        ei, ej = int(e.get("i", -1)), int(e.get("j", -1))
        if not (0 <= ei < n_frames and 0 <= ej < n_frames):
            continue
        pi, pj = _find(ei), _find(ej)
        if pi != pj:
            parent[pi] = pj

    root = _find(0)
    return all(_find(f) == root for f in range(n_frames))


__all__ = [
    "_reject_static_edges",
    "_compute_adaptive_min_disp",
    "_filter_high_conf_edges",
    "_check_edge_graph_connectivity",
]
