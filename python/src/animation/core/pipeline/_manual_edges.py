"""HITL (human-in-the-loop) edge construction: manual displacement + landmark points."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _build_manual_edge(
    i: int,
    j: int,
    dx: float,
    dy: float,
    weight: float = 0.9,
) -> Dict:
    """§S89: Construct a pipeline-compatible edge dict from a user-supplied displacement.

    The affine M is a pure translation: [[1, 0, dx], [0, 1, dy]].
    pts_i / pts_j are set to a single centroid-estimate point so Bundle Adjust
    can process the edge without matched feature points.

    Args:
        i: Source frame index.
        j: Target frame index.
        dx: Horizontal pixel displacement (j relative to i).
        dy: Vertical pixel displacement (j relative to i).
        weight: Edge confidence weight in [0, 1]; default 0.9 (high confidence
                for manual edges since the user deliberately chose the value).

    Returns:
        Edge dict compatible with ``_bundle_adjust_affine`` and the HITL edge
        override path in ``StitchWorker``.
    """
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    pts_i = np.array([[0.0, 0.0]], dtype=np.float32)
    pts_j = np.array([[dx, dy]], dtype=np.float32)
    return {
        "i": i,
        "j": j,
        "M": M,
        "pts_i": pts_i,
        "pts_j": pts_j,
        "weight": float(np.clip(weight, 0.0, 1.0)),
        "method": "manual",
    }


def _build_landmark_affine(
    i: int,
    j: int,
    landmark_pairs: "List[Tuple[Tuple[float, float], Tuple[float, float]]]",
    weight: float = 0.95,
) -> Dict:
    """§2.9A: Build a pipeline edge dict from user-placed landmark point pairs.

    Constructs a least-squares affine (or partial-affine / translation) from
    the N landmark correspondences provided by the BigWarp landmark editor
    dialog and returns an edge dict compatible with ``_bundle_adjust_affine``.

    ``landmark_pairs`` is a list of ``((xi, yi), (xj, yj))`` tuples where
    ``(xi, yi)`` is the point in frame i and ``(xj, yj)`` is the corresponding
    point in frame j, both in pixel coordinates.

    Estimation strategy (by point count):
    - 1 pair  → pure translation (centroid-to-centroid displacement)
    - 2 pairs → ``cv2.estimateAffinePartial2D`` (4-DOF: tx, ty, rotation, scale)
    - 3+ pairs → ``cv2.estimateAffine2D`` (6-DOF general affine, LMEDS robust)

    Falls back to centroid translation if cv2 estimation returns None/fails.

    Args:
        i: Source frame index.
        j: Target frame index.
        landmark_pairs: At least 1 ``((xi, yi), (xj, yj))`` correspondence.
        weight: Edge confidence weight in [0, 1]; default 0.95.

    Returns:
        Edge dict compatible with ``_bundle_adjust_affine`` and the HITL edge
        override path in ``StitchWorker``.
    """
    if not landmark_pairs:
        raise ValueError("landmark_pairs must contain at least 1 point pair")

    pts_i = np.array([[p[0][0], p[0][1]] for p in landmark_pairs], dtype=np.float32)
    pts_j = np.array([[p[1][0], p[1][1]] for p in landmark_pairs], dtype=np.float32)

    M: Optional[np.ndarray] = None
    n = len(landmark_pairs)
    if n >= 3:
        M_est, inliers = cv2.estimateAffine2D(pts_i, pts_j, method=cv2.LMEDS)
        if M_est is not None:
            M = M_est.astype(np.float64)
    elif n == 2:
        M_est, inliers = cv2.estimateAffinePartial2D(pts_i, pts_j, method=cv2.LMEDS)
        if M_est is not None:
            M = M_est.astype(np.float64)

    if M is None:
        # Centroid translation fallback
        centroid_i = pts_i.mean(axis=0)
        centroid_j = pts_j.mean(axis=0)
        dx = float(centroid_j[0] - centroid_i[0])
        dy = float(centroid_j[1] - centroid_i[1])
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)

    return {
        "i": i,
        "j": j,
        "M": M,
        "pts_i": pts_i,
        "pts_j": pts_j,
        "weight": float(np.clip(weight, 0.0, 1.0)),
        "method": "landmark",
    }


__all__ = ["_build_manual_edge", "_build_landmark_affine"]
