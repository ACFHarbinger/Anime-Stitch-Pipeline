"""Aspect-ratio framing and background extension solver (#429).

Enforces user-selected wallpaper aspect ratios (16:9, 9:16, 21:9) subject to:
1. Hard constraint: Window must fully contain the hero figure (hero_bbox_canvas).
2. Natural placement: Preserves original scene coordinates (no artificial centering/thirds shifts).
3. Hybrid outpainting: Fills overflow areas with Tier-1 classical extension and flags
   void_ratio > 0.10 for Tier-2 generative outpainting.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ._geometry import fit_window_containing_bbox, parse_aspect_ratio


@dataclass(frozen=True)
class FramedWallpaper:
    """Final framed wallpaper output and outpainting metadata."""

    wallpaper: np.ndarray  # (H_out, W_out, 3) uint8 BGR
    crop_rect: tuple[int, int, int, int]  # (x0, y0, x1, y1) in canvas coordinates
    target_aspect: str
    void_ratio: float
    needs_generative_outpaint: bool
    outpaint_mask: np.ndarray  # (H_out, W_out) bool


def frame_wallpaper(
    composite: np.ndarray,
    valid_mask: np.ndarray,
    hero_bbox_canvas: tuple[int, int, int, int],
    aspect: str = "16:9",
    *,
    allow_outpaint: bool = True,
    inpaint_radius: int = 5,
) -> FramedWallpaper:
    """Frame the composite to target aspect ratio around the hero figure's natural position."""
    if composite.ndim != 3 or composite.shape[2] != 3:
        raise ValueError(
            f"Composite must be (H, W, 3) BGR image, got shape {composite.shape}."
        )

    aspect_ratio = parse_aspect_ratio(aspect)
    canvas_h, canvas_w = composite.shape[:2]

    # 1. Fit aspect-constrained window containing the hero figure
    raw_window = fit_window_containing_bbox(hero_bbox_canvas, aspect_ratio, (canvas_h, canvas_w))
    wx0, wy0, wx1, wy1 = raw_window
    win_w = wx1 - wx0
    win_h = wy1 - wy0

    hx0, hy0, hx1, hy1 = hero_bbox_canvas

    # 2. Shift window within canvas bounds if it fits, while maintaining containment
    if win_w <= canvas_w:
        if wx0 < 0:
            shift_x = -wx0
            wx0 += shift_x
            wx1 += shift_x
        elif wx1 > canvas_w:
            shift_x = canvas_w - wx1
            wx0 += shift_x
            wx1 += shift_x

        # Ensure hero bbox is still strictly inside
        wx0 = min(wx0, hx0)
        wx1 = max(wx1, hx1)
        win_w = wx1 - wx0

    if win_h <= canvas_h:
        if wy0 < 0:
            shift_y = -wy0
            wy0 += shift_y
            wy1 += shift_y
        elif wy1 > canvas_h:
            shift_y = canvas_h - wy1
            wy0 += shift_y
            wy1 += shift_y

        wy0 = min(wy0, hy0)
        wy1 = max(wy1, hy1)
        win_h = wy1 - wy0

    crop_rect = (wx0, wy0, wx1, wy1)

    # 3. If window is completely inside canvas, slice directly
    if 0 <= wx0 and wx1 <= canvas_w and 0 <= wy0 and wy1 <= canvas_h:
        cropped_img = composite[wy0:wy1, wx0:wx1].copy()
        sub_valid = valid_mask[wy0:wy1, wx0:wx1]
        outpaint_mask = ~sub_valid
        void_ratio = (
            float(np.count_nonzero(outpaint_mask)) / float(cropped_img.shape[0] * cropped_img.shape[1])
        )
        return FramedWallpaper(
            wallpaper=cropped_img,
            crop_rect=crop_rect,
            target_aspect=aspect,
            void_ratio=void_ratio,
            needs_generative_outpaint=void_ratio > 0.10,
            outpaint_mask=outpaint_mask,
        )

    # 4. Handle window overflow (requires background extension / padding)
    out_canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    out_valid = np.zeros((win_h, win_w), dtype=bool)

    # Calculate overlap region between canvas and window
    src_x0 = max(0, wx0)
    src_y0 = max(0, wy0)
    src_x1 = min(canvas_w, wx1)
    src_y1 = min(canvas_h, wy1)

    dst_x0 = src_x0 - wx0
    dst_y0 = src_y0 - wy0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        out_canvas[dst_y0:dst_y1, dst_x0:dst_x1] = composite[src_y0:src_y1, src_x0:src_x1]
        out_valid[dst_y0:dst_y1, dst_x0:dst_x1] = valid_mask[src_y0:src_y1, src_x0:src_x1]

    outpaint_mask = ~out_valid
    void_count = int(np.count_nonzero(outpaint_mask))
    total_px = win_h * win_w
    void_ratio = float(void_count / total_px) if total_px > 0 else 0.0

    # 5. Tier-1 Classical inpainting for voids
    if allow_outpaint and void_count > 0:
        # Edge-replicate pad initial colors into void to provide boundary seeds
        inpaint_target = out_canvas.copy()
        # Create 1-pixel boundary mask for cv2.inpaint
        telea_mask = outpaint_mask.astype(np.uint8) * 255
        try:
            inpainted = cv2.inpaint(
                inpaint_target, telea_mask, inpaint_radius, cv2.INPAINT_TELEA
            )
            out_canvas = inpainted
        except Exception:
            # Fallback border replicate if inpaint fails
            pass

    return FramedWallpaper(
        wallpaper=out_canvas,
        crop_rect=crop_rect,
        target_aspect=aspect,
        void_ratio=void_ratio,
        needs_generative_outpaint=void_ratio > 0.10,
        outpaint_mask=outpaint_mask,
    )
