"""2D translation-domain wave correction — the analog of OpenCV's
``detail::waveCorrect`` for a cel-animation pan trajectory.

Chain drift ("wave") is the transverse deviation of the per-strip translation
trajectory from its dominant direction: pairwise registration errors
accumulate so the strip path bows across the panorama. OpenCV's ``waveCorrect``
straightens rotational panoramas by aligning every camera to the dominant
axis; the 2D translation analog here projects the strip positions onto the
dominant-direction line through the trajectory centroid, removing the
transverse bow while preserving each strip's along-axis coordinate exactly.

Consequence for validity (checked against the frozen M2 corpus): projection
never increases adjacent-frame Euclidean gaps — it shrinks them to their
along-axis component — so it cannot recover a ``min_gap`` rejection (those
gaps are already below the adaptive floor) and does not change
``no_valid_edges``/``disconnected_edge_graph`` (no BA output to correct).
Its plausible effect is on ``ratio`` (max/median gap) when the bow inflates
the max gap, and on output straightness for cases that already pass.

Default-off candidate behind ``ASP_WAVE_CORRECT`` — see the 2026-08-23
roadmap critique round. Isolated from the ``affine_invalid``/``min_gap``
rejection path: it only rewrites the translation slots of already-computed
affines; it never changes the 2x2 (rotation/scale) part and never runs the
validity gate itself.
"""

from __future__ import annotations

import numpy as np


def _to_translations(affines) -> tuple[np.ndarray, object]:
    """Extract per-strip translation vectors from the affine container.

    Returns ``(positions, reference)`` where ``positions`` is an (N, 2) array
    and ``reference`` is the first affine object (used to rebuild the list).
    """
    ref = affines[0]
    if isinstance(ref, np.ndarray):
        pts = np.array([[float(a[0, 2]), float(a[1, 2])] for a in affines])
    else:
        pts = np.array([[float(a.get("tx", 0.0)), float(a.get("ty", 0.0))] for a in affines])
    return pts, ref


def _rebuild(affines, pts: np.ndarray, ref) -> list:
    """Rebuild the affine container with corrected translations."""
    out = []
    for i, a in enumerate(affines):
        tx, ty = pts[i, 0], pts[i, 1]
        if isinstance(ref, np.ndarray):
            m = np.array(a, dtype=np.float32)
            m[0, 2] = tx
            m[1, 2] = ty
            out.append(m)
        else:
            out.append({**a, "tx": tx, "ty": ty})
    return out


def _dominant_direction(positions: np.ndarray) -> np.ndarray | None:
    """Principal direction of the strip trajectory (PCA of displacements)."""
    d = np.diff(positions, axis=0)
    if len(d) == 0 or not np.any(d):
        return None
    cov = d.T @ d
    try:
        _w, v = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    return v[:, -1]  # eigenvector of the largest eigenvalue


def wave_correct_affines(affines, kind: str = "auto"):
    """Straighten the strip trajectory's transverse wave (in place of a copy).

    Parameters
    ----------
    affines : list of (2, 3) float32 affine matrices (or dicts with
        ``tx``/``ty`` keys).
    kind : "auto" | "horizontal" | "vertical"
        For ``horizontal``/``vertical`` the dominant direction is assumed to
        be the given axis; ``auto`` uses PCA of the displacement vectors.

    Returns
    -------
    The corrected affine container (same type as input). Every strip's
    position is projected onto the dominant-direction line through the
    trajectory centroid: the along-axis coordinate is unchanged (adjacent
    gaps never grow) and the transverse deviation is removed.
    """
    if not affines or len(affines) <= 1:
        return affines

    positions, ref = _to_translations(affines)

    if kind == "horizontal":
        u = np.array([1.0, 0.0])
    elif kind == "vertical":
        u = np.array([0.0, 1.0])
    else:
        u = _dominant_direction(positions)
    if u is None:
        return affines

    u = u / np.linalg.norm(u)
    u_perp = np.array([-u[1], u[0]])

    across = positions @ u_perp  # (n,) transverse deviation (the wave)

    # Guard: already collinear — nothing to correct.
    if float(np.max(across) - np.min(across)) < 1e-6:
        return affines

    # Project every position onto the dominant line through the centroid.
    across_mean = float(np.mean(across))
    corrected = positions - np.outer(across - across_mean, u_perp)

    return _rebuild(affines, corrected, ref)


__all__ = ["wave_correct_affines"]
