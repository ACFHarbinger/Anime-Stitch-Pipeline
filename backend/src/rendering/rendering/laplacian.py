"""Perfect Seamless Blender: Sequential Laplacian with Optimal Seams."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from animation.core.stateless import _laplacian_blend

from . import _native

logger = logging.getLogger(__name__)


def _render_laplacian(
    frames: List[np.ndarray],
    affines: List[np.ndarray],
    bg_masks: List[Optional[np.ndarray]],
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Perfect Seamless Blender: Sequential Laplacian with Optimal Seams.
    """
    N = len(frames)

    # Phase 5f: C++ parallel warpAffine for the laplacian renderer warp step.
    warped_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    if _native._BATCH_RENDER:
        try:
            affines_f32 = [np.ascontiguousarray(a, dtype=np.float32) for a in affines]
            _gpu_kw: dict = {"try_gpu": True} if _native._BATCH_GPU else {}
            warped_list = list(
                _native._batch_render.canvas.warp_frames_to_canvas(
                    [np.ascontiguousarray(f) for f in frames],
                    affines_f32, H, W, **_gpu_kw,
                )
            )
            mask_list = [
                (w.max(axis=2) > 0).astype(np.uint8) * 255 for w in warped_list
            ]
        except Exception as _e:
            logger.debug("[Stitch] _render_laplacian batch warp fallback: %s", _e)
            warped_list = []
            mask_list = []

    if not warped_list:
        for img, M in zip(frames, affines, strict=False):
            w = cv2.warpAffine(
                img,
                M,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            warped_list.append(w)
            mask_list.append((w.max(axis=2) > 0).astype(np.uint8) * 255)

    # ── Color Matching (Sequential: each frame matched to the previous) ────
    # Chaining to the adjacent frame gives a better reference than anchoring
    # everything to frame 0, which may have very little spatial overlap with
    # frames far along the pan.
    colour_matched = [warped_list[0].astype(np.float32)]
    for i in range(1, N):
        src = warped_list[i].astype(np.float32)
        ref = colour_matched[i - 1]
        ref_m = mask_list[i - 1] > 0
        vm = mask_list[i] > 0
        overlap = vm & ref_m
        if overlap.sum() > 5000:
            out = src.copy()
            for c in range(3):
                ref_std = ref[overlap, c].std() + 1e-6
                src_std = src[overlap, c].std() + 1e-6
                ref_mean = ref[overlap, c].mean()
                src_mean = src[overlap, c].mean()

                gain = ref_std / src_std
                gain = np.clip(gain, 0.85, 1.18)

                bias = ref_mean - (src_mean * gain)
                bias = np.clip(bias, -15.0, 15.0)

                out[..., c] = np.clip(src[..., c] * gain + bias, 0, 255)
            colour_matched.append(out)
        else:
            colour_matched.append(src)

    # ── Sequential Seamless Blend (ROI-Based Distance Transform) ───────────
    canvas = colour_matched[0].copy()
    canvas_m = mask_list[0].copy()

    for i in range(1, N):
        img = colour_matched[i]
        m_i = mask_list[i]
        overlap = (canvas_m > 0) & (m_i > 0)
        if not overlap.any():
            canvas[m_i > 0] = img[m_i > 0]
            canvas_m[m_i > 0] = 255
            continue

        ys, xs = np.where(overlap)
        y0_ov, y1_ov = int(ys.min()), int(ys.max()) + 1
        x0_ov, x1_ov = int(xs.min()), int(xs.max()) + 1

        feather = 40
        y0_ov = max(0, y0_ov - feather)
        y1_ov = min(H, y1_ov + feather)
        x0_ov = max(0, x0_ov - feather)
        x1_ov = min(W, x1_ov + feather)

        H_roi = y1_ov - y0_ov
        W_roi = x1_ov - x0_ov

        canvas_roi = canvas[y0_ov:y1_ov, x0_ov:x1_ov].copy()
        img_roi = img[y0_ov:y1_ov, x0_ov:x1_ov]
        m_i_roi = m_i[y0_ov:y1_ov, x0_ov:x1_ov]
        canvas_m_roi = canvas_m[y0_ov:y1_ov, x0_ov:x1_ov]

        canvas_roi[m_i_roi > 0] = img_roi[m_i_roi > 0]
        canvas_roi[canvas_m_roi > 0] = canvas[y0_ov:y1_ov, x0_ov:x1_ov][
            canvas_m_roi > 0
        ]

        img_cross = img_roi.copy()
        img_cross[canvas_m_roi > 0] = canvas[y0_ov:y1_ov, x0_ov:x1_ov][canvas_m_roi > 0]
        img_cross[m_i_roi > 0] = img_roi[m_i_roi > 0]

        # Distance transform for soft weight map
        mask_roi = np.zeros((H_roi, W_roi), dtype=np.float32)
        mask_roi[feather : H_roi - feather, feather : W_roi - feather] = 1.0
        dist_mask = cv2.distanceTransform(
            (mask_roi * 255).astype(np.uint8), cv2.DIST_L2, 3
        )
        weight_roi = dist_mask / (dist_mask.max() + 1e-6)

        weight_roi[(m_i_roi > 0) & (canvas_m_roi == 0)] = 1.0
        weight_roi[(canvas_m_roi > 0) & (m_i_roi == 0)] = 0.0

        if bg_masks[i] is not None:
            fg_i = bg_masks[i] < 127
            w_fg_i = cv2.warpAffine(
                fg_i.astype(np.uint8) * 255,
                affines[i],
                (W, H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            w_fg_roi = w_fg_i[y0_ov:y1_ov, x0_ov:x1_ov]
            weight_roi[w_fg_roi > 127] = 1.0

        blended_roi = _laplacian_blend(img_cross, canvas_roi, weight_roi)

        update_mask = (m_i_roi > 0) | (canvas_m_roi > 0)
        canvas[y0_ov:y1_ov, x0_ov:x1_ov][update_mask] = blended_roi[update_mask]
        canvas_m |= m_i

    warped_fgs = []
    for _i, (M, bg) in enumerate(zip(affines, bg_masks, strict=False)):
        if bg is not None:
            fg = (bg < 127).astype(np.uint8) * 255
            w_fg = cv2.warpAffine(
                fg,
                M,
                (W, H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            warped_fgs.append(w_fg)
        else:
            warped_fgs.append(np.zeros((H, W), np.uint8))

    return (
        canvas.astype(np.uint8),
        canvas_m,
        [c.astype(np.uint8) for c in colour_matched],
        warped_fgs,
    )


__all__ = ["_render_laplacian"]
