"""Simple first-frame-wins renderer."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from . import _native

logger = logging.getLogger(__name__)


def _render_first(
    frames: list[np.ndarray],
    affines: list[np.ndarray],
    H: int,
    W: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple first-frame-wins renderer."""
    canvas = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.uint8)

    # §Phase5: C++ parallel warpAffine — first-frame-wins compositing in reverse order
    if _native._BATCH_RENDER:
        try:
            affines_f32 = [np.ascontiguousarray(a, dtype=np.float32) for a in affines]
            _gpu_kw: dict = {"try_gpu": True} if _native._BATCH_GPU else {}
            warped = _native._batch_render.canvas.warp_frames_to_canvas(
                [np.ascontiguousarray(f) for f in frames],
                affines_f32, H, W, **_gpu_kw,
            )
            for w in reversed(warped):
                m = (w.max(axis=2) > 0).astype(np.uint8) * 255
                canvas[m > 0] = w[m > 0]
                mask |= m
            return canvas, mask
        except Exception as _e:
            logger.debug(f"[Stitch] batch.canvas.warp_frames_to_canvas fallback: {_e}")
            canvas[:] = 0
            mask[:] = 0

    _frame_masks = [np.full(f.shape[:2], 255, dtype=np.uint8) for f in frames]
    for img, M, f_mask in reversed(list(zip(frames, affines, _frame_masks, strict=False))):
        w = cv2.warpAffine(
            img,
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        w_mask = cv2.warpAffine(
            f_mask,
            M,
            (W, H),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        m = (w_mask > 0).astype(np.uint8) * 255
        canvas[m > 0] = w[m > 0]
        mask |= m
    return canvas, mask


__all__ = ["_render_first"]
