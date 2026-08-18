"""Hero-cel compositing onto background plate (#428).

Composites the extracted HeroCel at its registered canvas coordinate (via its
solved 2x3 affine matrix), preserving natural scene placement (Harbinger's decision).

Supports feathered alpha blending and Poisson seamless cloning (cv2.seamlessClone).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import cv2
import numpy as np

from ._hero_selector import HeroCel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CelCompositeResult:
    """Result of compositing the hero cel onto the background plate."""

    composite: np.ndarray  # (H_canvas, W_canvas, 3) uint8
    cel_canvas_mask: np.ndarray  # (H_canvas, W_canvas) bool
    hero_bbox_canvas: tuple[int, int, int, int]  # (x0, y0, x1, y1) in canvas space
    blend_method: str


def composite_hero_cel(
    plate: np.ndarray,
    hero_cel: HeroCel,
    hero_affine: np.ndarray,
    *,
    blend_mode: str = "feather",
    feather_px: int = 8,
) -> CelCompositeResult:
    """Rigidly composite the hero cel onto the background plate at registered position."""
    if plate.ndim != 3 or plate.shape[2] != 3:
        raise ValueError(f"Plate must be (H, W, 3) BGR image, got shape {plate.shape}.")
    if hero_affine.shape != (2, 3):
        raise ValueError(
            f"Hero affine matrix must be shape (2, 3), got {hero_affine.shape}."
        )

    canvas_h, canvas_w = plate.shape[:2]
    cel_bgr = hero_cel.cel_rgba[:, :, :3]
    cel_alpha = hero_cel.alpha_mask

    # 1. Warp hero cel BGR and alpha matte into canvas coordinates
    warped_bgr = cv2.warpAffine(
        cel_bgr,
        hero_affine.astype(np.float32),
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    warped_alpha_uint8 = cv2.warpAffine(
        cel_alpha,
        hero_affine.astype(np.float32),
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    cel_canvas_mask = warped_alpha_uint8 > 128

    # 2. Compute canvas-space bounding box
    ys, xs = np.where(cel_canvas_mask)
    if len(ys) > 0:
        hero_bbox_canvas = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
    else:
        hero_bbox_canvas = (0, 0, canvas_w, canvas_h)

    # 3. Perform blending
    composite = plate.copy()
    actual_method = blend_mode

    if blend_mode == "poisson" and len(ys) > 0:
        try:
            x0, y0, x1, y1 = hero_bbox_canvas
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            # Bounding check: cv2.seamlessClone requires center to be well-inside
            if 5 < cx < canvas_w - 5 and 5 < cy < canvas_h - 5:
                # Ensure mask is strictly 0 or 255
                bin_mask = (cel_canvas_mask.astype(np.uint8)) * 255
                composite = cv2.seamlessClone(
                    warped_bgr, plate, bin_mask, (cx, cy), cv2.NORMAL_CLONE
                )
                actual_method = "poisson"
            else:
                actual_method = "feather"
        except Exception as e:
            log.warning("Poisson seamlessClone failed, falling back to feathering: %s", e)
            actual_method = "feather"

    if actual_method == "feather" or actual_method == "alpha":
        alpha_f = warped_alpha_uint8.astype(np.float32) / 255.0
        if actual_method == "feather" and feather_px > 0:
            k = max(3, feather_px * 2 + 1)
            alpha_f = cv2.GaussianBlur(alpha_f, (k, k), 0)

        alpha_3d = np.repeat(alpha_f[:, :, np.newaxis], 3, axis=2)
        blended = (
            warped_bgr.astype(np.float32) * alpha_3d
            + plate.astype(np.float32) * (1.0 - alpha_3d)
        )
        composite = np.clip(blended, 0, 255).astype(np.uint8)

    return CelCompositeResult(
        composite=composite,
        cel_canvas_mask=cel_canvas_mask,
        hero_bbox_canvas=hero_bbox_canvas,
        blend_method=actual_method,
    )
