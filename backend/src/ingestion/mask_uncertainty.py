"""P4: Segmentation uncertainty and Trapped-Ball alternate mask refinement.

Implements the P4 candidate contract from the ASP quality proposal:
1. Compute temporal disagreement between learned (BiRefNet) background masks
   after provisional alignment across adjacent frames.
2. In uncertain / disputed regions, compute classical deterministic trapped-ball
   line-art segmentation (Zhang et al. 2009) as an alternate structural candidate
   rather than replacing the learned mask globally.
3. Mark agreed background as safe (255), confirmed character cel as foreground (0),
   and disputed/unresolved regions as excluded/uncertain (128).
4. In downstream plate compositing (P1/P2), only confirmed background (255) is
   sampled for plate reconstruction, avoiding character ghosting from mask leaks.
5. Gated behind default-off ``ASP_MASK_UNCERTAINTY=1``.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from .trapped_ball import trapped_ball_segmentation

logger = logging.getLogger(__name__)


def mask_uncertainty_enabled() -> bool:
    """True when ASP_MASK_UNCERTAINTY is enabled."""
    return os.environ.get("ASP_MASK_UNCERTAINTY", "0") == "1"


def compute_pairwise_mask_disagreement(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    affine_b_to_a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mask disagreement between frame A and frame B warped to frame A.

    Parameters
    ----------
    mask_a : (H, W) uint8 mask for frame A (255=bg, 0=fg).
    mask_b : (H, W) uint8 mask for frame B (255=bg, 0=fg).
    affine_b_to_a : (2, 3) affine transform mapping coordinates in B to A.

    Returns
    -------
    (disagreement_mask, valid_overlap_mask) :
        disagreement_mask: (H, W) bool, True where masks disagree.
        valid_overlap_mask: (H, W) bool, True where frame B overlaps frame A.
    """
    H, W = mask_a.shape[:2]

    # Warp mask B into frame A's coordinate space
    warped_b = cv2.warpAffine(
        mask_b,
        affine_b_to_a,
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Track valid overlap of frame B on canvas of frame A
    ones_b = np.ones((mask_b.shape[0], mask_b.shape[1]), dtype=np.uint8) * 255
    valid_overlap = cv2.warpAffine(
        ones_b,
        affine_b_to_a,
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 127

    bg_a = mask_a > 127
    bg_b = warped_b > 127

    # Disagreement occurs where both frames overlap but their classifications differ
    disagreement = valid_overlap & (bg_a != bg_b)
    return disagreement, valid_overlap


def resolve_disputed_mask_region(
    frame_bgr: np.ndarray,
    birefnet_mask: np.ndarray,
    disagreement_mask: np.ndarray,
    *,
    ball_radius: int | None = None,
) -> np.ndarray:
    """Resolve disputed mask regions using trapped-ball structural segmentation.

    Parameters
    ----------
    frame_bgr : (H, W, 3) uint8 BGR image.
    birefnet_mask : (H, W) uint8 learned background mask.
    disagreement_mask : (H, W) bool mask where temporal disagreement was measured.
    ball_radius : int | None, trapped-ball radius.

    Returns
    -------
    refined_mask : (H, W) uint8 mask:
        255 = confirmed safe background
        128 = uncertain / disputed (excluded from plate samples)
        0   = confirmed character foreground
    """
    H, W = frame_bgr.shape[:2]
    refined = birefnet_mask.copy()

    if not disagreement_mask.any():
        return refined

    # Compute classical trapped-ball segmentation
    tb_mask = trapped_ball_segmentation(frame_bgr, ball_radius=ball_radius)
    tb_bg = tb_mask > 127
    bn_bg = birefnet_mask > 127

    # In disputed regions:
    # 1. If Trapped-Ball AND BiRefNet agree on background -> confirmed background (255)
    # 2. If Trapped-Ball AND BiRefNet agree on foreground -> confirmed foreground (0)
    # 3. If Trapped-Ball and BiRefNet disagree -> mark uncertain (128)
    disputed_pixels = disagreement_mask

    for y in range(H):
        for x in range(W):
            if not disputed_pixels[y, x]:
                continue
            b_val = bn_bg[y, x]
            t_val = tb_bg[y, x]
            if b_val == t_val:
                refined[y, x] = 255 if b_val else 0
            else:
                # Disagreement remains between classical and neural estimates -> mark uncertain
                refined[y, x] = 128

    return refined


def compute_temporal_mask_uncertainty(
    frames: list[np.ndarray],
    bg_masks: list[np.ndarray | None],
    affines: list[np.ndarray],
    *,
    ball_radius: int | None = None,
) -> list[np.ndarray | None]:
    """Compute temporal mask uncertainty across an image sequence.

    Parameters
    ----------
    frames : list of (H, W, 3) frames.
    bg_masks : list of (H, W) learned background masks.
    affines : list of (2, 3) or (3, 3) global affine alignment matrices.
    ball_radius : int | None, trapped-ball structuring element radius.

    Returns
    -------
    refined_masks : list of (H, W) uint8 refined ternary masks (255=bg, 128=uncertain, 0=fg).
    """
    N = len(frames)
    if N == 0:
        return []
    if N == 1 or all(m is None for m in bg_masks):
        return bg_masks

    refined_masks: list[np.ndarray | None] = []

    for i in range(N):
        mask_i = bg_masks[i]
        if mask_i is None:
            refined_masks.append(None)
            continue

        H_i, W_i = frames[i].shape[:2]
        accum_disagreement = np.zeros((H_i, W_i), dtype=bool)

        M_i_3x3 = np.eye(3, dtype=np.float64)
        M_i_3x3[:2, :3] = affines[i][:2, :3]
        inv_M_i = np.linalg.inv(M_i_3x3)

        # Check adjacent frame neighbors (i-1 and i+1)
        neighbors = [j for j in (i - 1, i + 1) if 0 <= j < N]
        for j in neighbors:
            mask_j = bg_masks[j]
            if mask_j is None:
                continue

            M_j_3x3 = np.eye(3, dtype=np.float64)
            M_j_3x3[:2, :3] = affines[j][:2, :3]

            # Affine mapping from frame j to frame i: M_{j->i} = M_i^{-1} @ M_j
            M_j_to_i_3x3 = inv_M_i @ M_j_3x3
            M_j_to_i = M_j_to_i_3x3[:2, :3].astype(np.float32)

            disagree, _ = compute_pairwise_mask_disagreement(mask_i, mask_j, M_j_to_i)
            accum_disagreement |= disagree

        # Resolve accumulated disagreement with Trapped-Ball
        refined_i = resolve_disputed_mask_region(
            frames[i],
            mask_i,
            accum_disagreement,
            ball_radius=ball_radius,
        )
        refined_masks.append(refined_i)

    return refined_masks


__all__ = [
    "mask_uncertainty_enabled",
    "compute_pairwise_mask_disagreement",
    "resolve_disputed_mask_region",
    "compute_temporal_mask_uncertainty",
]
