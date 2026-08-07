"""Geometry/statistics helpers shared by the pairwise-matching strategies."""

from __future__ import annotations

import numpy as np


def _extract_similarity(M: np.ndarray) -> np.ndarray:
    """§1.3E — Project a full 2×3 affine to its best-fit 4-DOF similarity.

    A similarity transform has the form ``[[a, b, tx], [-b, a, ty]]``.  A
    general affine ``[[a, b, tx], [c, d, ty]]`` decomposes as::

        a_sym = (a + d) / 2   (average of diagonal — symmetric part)
        b_sym = (b - c) / 2   (antisymmetric off-diagonal — rotation/scale)

    This is the closed-form least-squares projection onto the similarity
    manifold (Procrustes for 2-D conformal maps).  Shear (asymmetric
    off-diagonal component) is discarded because feature matchers
    (LoFTR, RoMa) cannot reliably distinguish camera shear from
    perspective distortion at anime-panel scales.

    Parameters
    ----------
    M : (2, 3) float32 affine matrix.

    Returns
    -------
    (2, 3) float32 similarity matrix with the same translation as M.
    """
    a = float(M[0, 0])
    b = float(M[0, 1])
    c = float(M[1, 0])
    d = float(M[1, 1])
    a_sym = (a + d) / 2.0
    b_sym = (b - c) / 2.0
    out = np.eye(2, 3, dtype=np.float32)
    out[0, 0] = a_sym
    out[0, 1] = b_sym
    out[1, 0] = -b_sym
    out[1, 1] = a_sym
    out[0, 2] = float(M[0, 2])
    out[1, 2] = float(M[1, 2])
    return out


def _compute_translation_spread(
    pts_i: np.ndarray,
    pts_j: np.ndarray,
) -> tuple[float, float]:
    """§1.36: Per-axis MAD of LoFTR displacement estimates around their median.

    When LoFTR finds many correspondences but they disagree on the translation
    (e.g., bimodal distribution from foreground / background confusions), the median
    displacement is unreliable. A high MAD flags this ambiguity before the edge is
    committed to the graph.

    Parameters
    ----------
    pts_i, pts_j : (N, 2) float32 — matched keypoint coordinates in frames i and j.

    Returns
    -------
    (mad_dx, mad_dy) : pair of floats, each ≥ 0.
        0.0 when N ≤ 1 (no spread to compute).
    """
    if len(pts_i) <= 1:
        return 0.0, 0.0
    dxs = pts_j[:, 0] - pts_i[:, 0]
    dys = pts_j[:, 1] - pts_i[:, 1]
    mad_dx = float(np.median(np.abs(dxs - np.median(dxs))))
    mad_dy = float(np.median(np.abs(dys - np.median(dys))))
    return mad_dx, mad_dy


def _compute_bg_match_ratio(n_bg_pts: int, n_total_pts: int) -> float:
    """§1.38: Fraction of LoFTR matches that land on background pixels.

    When most LoFTR matches fall on foreground characters, the handful of
    surviving background matches produce a noisy median displacement estimate.
    This function quantifies how bg-clean the match set is so the caller can
    reject the edge when the ratio is too low.

    Parameters
    ----------
    n_bg_pts    : number of matches whose endpoints are both on background (mask > 127).
    n_total_pts : total LoFTR matches before bg filtering.

    Returns
    -------
    float in [0, 1].  Returns 0.0 when *n_total_pts* is 0 (avoids ZeroDivisionError).
    """
    if n_total_pts <= 0:
        return 0.0
    return float(n_bg_pts) / float(n_total_pts)


__all__ = [
    "_extract_similarity",
    "_compute_translation_spread",
    "_compute_bg_match_ratio",
]
