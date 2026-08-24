"""Trapped-Ball segmentation for anime line-art and cel frames (Zhang et al. 2009).

Classical, deterministic cartoon segmentation designed for non-closed line art:
1. Detect line art / gradient edges via multi-scale morphological or Canny filtering.
2. Erode boundary gaps and perform ball-trapping morphology (closing by reconstruction
   with structuring element radius r) so the "ball" of radius r cannot pass through
   narrow line-art gaps (< 2r px), closing small ink fractures.
3. Flood-fill from borders / background seeds to produce clean, deterministic background
   regions even on flat-cel-shaded anime without neural network non-determinism.
4. Gated behind default-off ``ASP_TRAPPED_BALL=1`` and parameter ``ASP_TRAPPED_BALL_RADIUS=4``.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def trapped_ball_enabled() -> bool:
    """True when ASP_TRAPPED_BALL is enabled."""
    return os.environ.get("ASP_TRAPPED_BALL", "0") == "1"


def get_trapped_ball_radius() -> int:
    """Structuring element ball radius (px) for trapped-ball segmentation."""
    try:
        return max(1, int(os.environ.get("ASP_TRAPPED_BALL_RADIUS", "4")))
    except (ValueError, TypeError):
        return 4


def trapped_ball_segmentation(
    image_bgr: np.ndarray,
    *,
    ball_radius: int | None = None,
    edge_thresh_low: int = 40,
    edge_thresh_high: int = 120,
    border_seed_width: int = 2,
) -> np.ndarray:
    """Compute deterministic background mask using trapped-ball algorithm.

    Parameters
    ----------
    image_bgr : (H, W, 3) uint8 BGR frame.
    ball_radius : int, radius of the trapped ball (structuring element).
    edge_thresh_low : int, Canny lower threshold.
    edge_thresh_high : int, Canny upper threshold.
    border_seed_width : int, border width (px) used as background seeds for flood fill.

    Returns
    -------
    bg_mask : (H, W) uint8 mask where 255 = confirmed background, 0 = foreground/lines.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    r = ball_radius if ball_radius is not None else get_trapped_ball_radius()
    H, W = image_bgr.shape[:2]

    # 1. Edge detection for line art
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    edges = cv2.Canny(blurred, edge_thresh_low, edge_thresh_high)

    # 2. Trapped-ball gap closure:
    # Morphological dilation with circular structuring element of diameter 2r+1
    # bridges any line-art gap of size <= 2r.
    ball_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    thick_edges = cv2.dilate(edges, ball_se)

    # 3. Flood fill background starting from image borders
    # Free space where ball can move: inv_edges
    free_space = cv2.bitwise_not(thick_edges)

    # Create floodfill mask with 2px border per cv2.floodFill requirement
    ff_mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
    ff_mask[1:-1, 1:-1] = (free_space == 0).astype(np.uint8)

    bg_fill = np.zeros((H, W), dtype=np.uint8)

    # Seed floodfill along 4 borders where free_space is active
    seed_points: list[tuple[int, int]] = []
    step = max(1, r)
    for x in range(0, W, step):
        if free_space[0, x]:
            seed_points.append((x, 0))
        if free_space[H - 1, x]:
            seed_points.append((x, H - 1))
    for y in range(0, H, step):
        if free_space[y, 0]:
            seed_points.append((0, y))
        if free_space[y, W - 1]:
            seed_points.append((W - 1, y))

    for sx, sy in seed_points:
        if ff_mask[sy + 1, sx + 1] == 0:
            cv2.floodFill(
                free_space,
                ff_mask,
                (sx, sy),
                255,
                flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY,
            )

    # Background is all filled regions in ff_mask
    bg_flooded = (ff_mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255

    # 4. Morphological opening/erosion recovery to restore background boundaries up to the true edges
    # Dilate flooded background by ball radius so it meets the actual thin edges, not thick_edges
    if r > 0:
        recover_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r - 1, 2 * r - 1))
        bg_recovered = cv2.dilate(bg_flooded, recover_se)
        # Background must not overwrite original thin edges
        bg_mask = cv2.bitwise_and(bg_recovered, cv2.bitwise_not(edges))
    else:
        bg_mask = bg_flooded

    return bg_mask


def compute_trapped_ball_masks(
    frames: list[np.ndarray],
    *,
    ball_radius: int | None = None,
) -> list[np.ndarray]:
    """Batch trapped-ball segmentation over a list of frames."""
    return [
        trapped_ball_segmentation(f, ball_radius=ball_radius)
        for f in frames
    ]


__all__ = [
    "trapped_ball_enabled",
    "get_trapped_ball_radius",
    "trapped_ball_segmentation",
    "compute_trapped_ball_masks",
]
