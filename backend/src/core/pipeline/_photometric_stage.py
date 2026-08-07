"""Stage 4.5/4.5b: background-referenced + per-segment photometric normalisation.

Extracted from ``AnimeStitchPipeline.run()`` as a standalone function -- pure
code motion, no logic change -- so ``run()`` fits closer to this codebase's
500-code-line file-size convention (§5.17). No early-return/fallback paths
run through this block in the original, so it's a low-risk, purely
computational extraction (unlike most of the rest of run(), which is left
untouched given the lack of an end-to-end regression test for this
benchmark-sensitive method -- see run_stage.py's module docstring).
"""

from __future__ import annotations

import cv2
import numpy as np


def _apply_background_photometric_normalization(
    frames: list[np.ndarray],
    bg_masks: list[np.ndarray | None],
    N: int,
) -> list[np.ndarray]:
    """Normalise per-frame ambient lighting using background-pixel statistics.

    Stage 4.5: computes each frame's mean background color (bg_mask > 127)
    and normalises every frame to the same median background level, removing
    frame-to-frame ambient lighting variation (anime cel flicker) that would
    otherwise appear as horizontal color seams in the temporal median.

    Stage 4.5b (P2.6): refines correction at the connected-component level --
    the frame-level gain above is a single scalar, but anime assigns
    different exposure levels to different colour regions (sky vs costume vs
    background props). Matches each background segment to the reference
    (frame 0) segment with the closest colour, removing per-region flicker
    independently.

    Returns the corrected frame list (frames may also be mutated in place,
    matching the pre-split behaviour).
    """
    bg_frame_means: list[np.ndarray | None] = []
    for _i, (_frame, _mask) in enumerate(zip(frames, bg_masks, strict=False)):
        if _mask is not None:
            _bg_px = _frame[_mask > 127].astype(np.float32)
            if len(_bg_px) >= 1000:
                bg_frame_means.append(_bg_px.mean(axis=0))
                continue
        bg_frame_means.append(None)

    _valid_means = [m for m in bg_frame_means if m is not None]
    if len(_valid_means) >= 3:
        _ref_mean = np.median(_valid_means, axis=0)  # (3,) BGR reference
        for _i in range(N):
            _bg_mean_i = bg_frame_means[_i]
            if _bg_mean_i is None:
                continue
            _gain = _ref_mean / np.maximum(_bg_mean_i, 1.0)
            _ref_lum_scalar = float(np.dot(_ref_mean, [0.114, 0.587, 0.299]))
            _gain_lo, _gain_hi = (
                (0.80, 1.25) if _ref_lum_scalar < 80.0 else (0.88, 1.14)
            )
            _gain = np.clip(_gain, _gain_lo, _gain_hi)
            if not np.allclose(_gain, 1.0, atol=0.01):
                frames[_i] = np.clip(
                    frames[_i].astype(np.float32) * _gain, 0, 255
                ).astype(np.uint8)

    # P2.6 — Per-segment photometric correction.
    # The global gain above applies one scalar per frame.  Anime assigns
    # different exposure levels to different colour regions (sky vs costume
    # vs background props), so a single gain is a poor approximation.
    # This pass refines correction at the connected-component level,
    # matching each background segment to the reference (frame 0) segment
    # with the closest colour, removing per-region flicker independently.
    for _i in range(1, N):
        bg_mask_i = bg_masks[_i]
        if bg_mask_i is None:
            continue
        bm = bg_mask_i > 127
        if bm.sum() < 1000:
            continue
        # Quick color-region segmentation via quantization (no SAM needed)
        img_small = cv2.resize(
            frames[_i],
            (frames[_i].shape[1] // 4, frames[_i].shape[0] // 4),
            interpolation=cv2.INTER_AREA,
        )
        flat = img_small.reshape(-1, 3).astype(np.float32)
        _, labels_flat, centers = cv2.kmeans(
            flat,
            min(8, len(np.unique(flat.reshape(-1)))),
            None,  # type: ignore[call-overload] # cv2-stub gap: bestLabels=None is the standard idiom
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            2,
            cv2.KMEANS_PP_CENTERS,
        )
        seg_map = labels_flat.reshape(img_small.shape[:2])
        seg_map_full = cv2.resize(
            seg_map.astype(np.uint8),
            (frames[_i].shape[1], frames[_i].shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        # Reference: frame 0 colour clusters
        img0_small = cv2.resize(
            frames[0], img_small.shape[:2][::-1], interpolation=cv2.INTER_AREA
        )
        flat0 = img0_small.reshape(-1, 3).astype(np.float32)
        ref_centers = cv2.kmeans(
            flat0,
            min(8, len(np.unique(flat0.reshape(-1)))),
            None,  # type: ignore[call-overload] # cv2-stub gap: bestLabels=None is the standard idiom
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            2,
            cv2.KMEANS_PP_CENTERS,
        )[2]

        gain_map = np.ones(frames[_i].shape[:2], dtype=np.float32)
        for _k in range(int(seg_map_full.max()) + 1):
            _seg_px = (seg_map_full == _k) & bm
            if _seg_px.sum() < 200:
                continue
            _seg_mean = frames[_i][_seg_px].astype(np.float32).mean(axis=0)  # (3,)
            # Find closest reference cluster by colour distance
            _dists = np.linalg.norm(ref_centers - _seg_mean[np.newaxis], axis=1)
            _ref_seg = ref_centers[int(np.argmin(_dists))]
            _gain_seg = np.clip(_ref_seg / np.maximum(_seg_mean, 1.0), 0.88, 1.12)
            gain_map[_seg_px] = _gain_seg.mean()

        frames[_i] = np.clip(
            frames[_i].astype(np.float32) * gain_map[..., np.newaxis], 0, 255
        ).astype(np.uint8)

    return frames


__all__ = ["_apply_background_photometric_normalization"]
