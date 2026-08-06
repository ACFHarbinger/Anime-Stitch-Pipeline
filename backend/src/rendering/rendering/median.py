"""Memory-efficient, FAST Temporal Median Render.

Tries the C++/GPU fast path (``_median_fastpath.py``) first; falls back to
a chunked Python loop (kept here) when the fast path's preconditions aren't
met or the native call fails. Animation-phase re-render (shared with the
fast path) is delegated to ``_animation_repaint.py``.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import List, Optional, Tuple

import cv2
import numpy as np

from asp_backend.alignment.canvas import _detect_scroll_axis
from backend.src.constants import LANCZOS_BLEED, MAX_SAFE_GAIN_DEV, RENDERING_FADE_ROWS

from ._animation_repaint import _render_animation_repaint
from ._gain_correction import _compute_sequential_color_gains
from ._gpu_median import _gpu_nanmedian
from ._median_fastpath import _render_median_gpu_fastpath

logger = logging.getLogger(__name__)

# A5 — Foreground-excluded temporal median.  When enabled, foreground (character)
# pixels are excluded from the per-pixel median so the background PLATE never
# averages the character's differing animation poses into a translucent ghost.
# Where a canvas pixel has NO background sample across any frame (the character
# is always there), the median falls back to all geometrically-valid pixels so
# no holes appear.  Stage 11 then composites the re-posed foreground on top.
# Set ASP_FG_EXCLUDE_MEDIAN=0 to disable (A/B comparison).
_FG_EXCLUDE_MEDIAN = os.environ.get("ASP_FG_EXCLUDE_MEDIAN", "1") != "0"

# §2.5 — Overmix-style background sub-pixel averaging.
# When ON (and A5 fg-exclusion is active, so samples are confirmed-background),
# canvas pixels where >=3 frames agree on background lean toward a mean
# instead of the median: MPEG DCT block noise cancels out by sqrt(N) for a
# mean, which the median (picking one sample, or interpolating between two
# for even counts) doesn't provide. Median remains the sole choice for
# count==2, where there's no averaging benefit and a misclassified fg sample
# would ghost the plate. Default OFF. Enable: ASP_BG_AVERAGE=1.
#
# 2026-07-27 fix (measured harmful, S221/#24): the original version switched
# abruptly at the count==2/count==3 boundary. Canvas "sample count" varies
# geographically with frame-overlap geometry (higher near the middle of a
# frame stack, lower near entry/exit edges), so count==2 and count>=3 zones
# form contiguous geographic bands — an abrupt statistic switch at that
# boundary produced visible strip-banding (median and mean can differ
# meaningfully with residual misalignment/warp noise). Now blends mean and
# median proportionally across count in [2, _BG_AVERAGE_FULL_AT], so there is
# no discontinuity at the count boundary itself.
_BG_AVERAGE: bool = os.environ.get("ASP_BG_AVERAGE", "0") != "0"
_BG_AVERAGE_FULL_AT: int = int(os.environ.get("ASP_BG_AVERAGE_FULL_AT", "5"))

# §1.87 — Masked-Median Background Plate.
# When enabled, changes the A5 fg-exclusion fallback for pixels where every frame
# has a foreground sample (all_fg): instead of averaging ALL valid samples (which
# ghost-averages different animation poses), the per-pixel fallback is suppressed
# so those pixels stay zero until bg_complete fills them.  Pairs with ASP_BG_COMPLETE
# to inpaint the zero-coverage holes cleanly.  Enable: ASP_MASKED_MEDIAN=1.
_MASKED_MEDIAN: bool = os.environ.get("ASP_MASKED_MEDIAN", "0") != "0"


def _render_median(  # noqa: C901
    frames: List[np.ndarray],
    affines: List[np.ndarray],
    bg_masks: List[Optional[np.ndarray]],
    H: int,
    W: int,
    _baselines: Optional[List[float]] = None,
    _skip_anim: bool = False,
    confidence_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Memory-efficient and FAST Temporal Median Render.
    Avoids float32 conversion and nanmedian where possible.

    confidence_weights : (N,) float32 per-frame matching confidence [0, 1].
        When provided and any frame has confidence < 0.70, multi-sample pixels
        use a confidence-weighted average instead of an unweighted median.
        Frames aligned via LoFTR (conf ~0.9) outweigh Template Match frames
        (conf ~0.55), reducing blur from low-quality fallback edges (P1.3).
    """
    N = len(frames)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    valid_mask = np.zeros((H, W), dtype=np.uint8)

    # Pre-compute sequential color corrections to eliminate frame-boundary seams.
    # Matches each frame's overlap-zone photometry to its predecessor, chained
    # from frame 0.  This implements the "histogram matching" that photometric.py
    # defers to this renderer.
    _cg, _cb = _compute_sequential_color_gains(frames, affines, bg_masks=bg_masks)
    # Only apply correction when gains are small (< 5% per-channel) — large gains
    # indicate foreground contamination and would make the seam worse.

    _cg_safe = np.where(np.abs(_cg - 1.0) <= MAX_SAFE_GAIN_DEV, _cg, np.ones_like(_cg))
    _cb_safe = np.where(np.abs(_cg - 1.0) <= MAX_SAFE_GAIN_DEV, _cb, np.zeros_like(_cb))
    _need_color_corr = not (
        np.allclose(_cg_safe, 1.0, atol=0.005) and np.allclose(_cb_safe, 0.0, atol=0.5)
    )
    _cg, _cb = _cg_safe, _cb_safe

    scroll_axis = _detect_scroll_axis(affines)

    # Canvas entry/exit bounds for each frame (for fade-in/out)
    _frame_ty = np.array([float(affines[i][1, 2]) for i in range(N)], dtype=np.float64)
    _frame_bot = np.array(
        [_frame_ty[i] + frames[i].shape[0] for i in range(N)], dtype=np.float64
    )
    _frame_tx = np.array([float(affines[i][0, 2]) for i in range(N)], dtype=np.float64)
    _frame_right = np.array(
        [_frame_tx[i] + frames[i].shape[1] for i in range(N)], dtype=np.float64
    )

    # Precompute geometric masks for each frame to avoid confusing black pixels with borders
    _frame_masks = [np.full(f.shape[:2], 255, dtype=np.uint8) for f in frames]

    # A5 — per-frame BACKGROUND masks (uint8, 255 = background) for fg-excluded median.
    _exclude_fg = _FG_EXCLUDE_MEDIAN and any(m is not None for m in bg_masks)
    _frame_bg_u8: List[Optional[np.ndarray]] = []
    for i in range(N):
        bm = bg_masks[i] if i < len(bg_masks) else None
        if _exclude_fg and bm is not None:
            # Normalise to frame size, 255 where background.
            bm_u8 = (bm > 127).astype(np.uint8) * 255
            if bm_u8.shape[:2] != frames[i].shape[:2]:
                bm_u8 = cv2.resize(
                    bm_u8,
                    (frames[i].shape[1], frames[i].shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            _frame_bg_u8.append(bm_u8)
        else:
            _frame_bg_u8.append(None)
    if _exclude_fg:
        logger.info(
            "[Stitch]   A5: foreground-excluded temporal median ENABLED (clean bg plate)."
        )

    # Phase 5e: C++ fast path — parallel warpAffine + OpenMP nth_element median.
    # See _median_fastpath.py for the precondition gate and implementation.
    fastpath_result = _render_median_gpu_fastpath(
        frames, affines, bg_masks, H, W, N,
        scroll_axis, _frame_ty, _frame_bot, _frame_tx, _frame_right,
        _exclude_fg, _need_color_corr, _baselines, confidence_weights, _skip_anim,
    )
    if fastpath_result is not None:
        canvas, valid_mask = fastpath_result
        return canvas, valid_mask, [], []

    # Determine chunk size. We want to keep stack size < 1GB
    chunk_size = max(1, min(1024, (1024 * 1024 * 1024) // (N * W * 3 + 1)))

    logger.info(
        "[Stitch]   Rendering %d frames in chunks of %dpx height...", N, chunk_size
    )

    for y0 in range(0, H, chunk_size):
        y1 = min(y0 + chunk_size, H)
        ch = y1 - y0

        stack = np.zeros((N, ch, W, 3), dtype=np.uint8)
        masks = np.zeros((N, ch, W), dtype=bool)
        bg_canvas = np.zeros((N, ch, W), dtype=bool) if _exclude_fg else None

        for i in range(N):
            M_strip = affines[i].copy()
            M_strip[1, 2] -= y0
            w_strip = cv2.warpAffine(
                frames[i],
                M_strip,
                (W, ch),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            w_mask = cv2.warpAffine(
                _frame_masks[i],
                M_strip,
                (W, ch),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            valid_px = w_mask > 0
            if _exclude_fg and _frame_bg_u8[i] is not None:
                w_bg = cv2.warpAffine(
                    _frame_bg_u8[i],
                    M_strip,
                    (W, ch),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                bg_canvas[i] = (w_bg > 127) & valid_px
            elif _exclude_fg:
                # No mask for this frame → treat all valid pixels as background.
                bg_canvas[i] = valid_px
            if _baselines is not None:
                b_i = _baselines[i]
                if b_i < 0.90:
                    scale = min(1.0 / max(b_i, 0.5), 1.25)
                    w_strip = np.clip(
                        w_strip.astype(np.float32) * scale, 0, 255
                    ).astype(np.uint8)
            if _need_color_corr:
                g_i = _cg[i]
                bc_i = _cb[i]
                if not (
                    np.allclose(g_i, 1.0, atol=0.01)
                    and np.allclose(bc_i, 0.0, atol=1.0)
                ):
                    w_f32 = w_strip.astype(np.float32)
                    for c in range(3):
                        w_f32[:, :, c] = np.clip(
                            w_f32[:, :, c] * g_i[c] + bc_i[c], 0, 255
                        )
                    w_strip = w_f32.astype(np.uint8)
                    w_strip[~valid_px] = 0
            stack[i] = w_strip
            masks[i] = valid_px

        geo_count = masks.sum(axis=0)  # geometric coverage (for valid_mask, fades)

        # A5 — effective masks for the median: prefer BACKGROUND samples; where a
        # pixel has no background sample anywhere, fall back to all valid samples
        # (default) or leave as zero when _MASKED_MEDIAN is enabled (§1.87).
        if _exclude_fg:
            bg_count = bg_canvas.sum(axis=0)  # (ch, W)
            use_bg = bg_count >= 1  # pixel has ≥1 background sample
            if _MASKED_MEDIAN:
                # §1.87: no-bg pixels stay zero — avoids ghost-averaging fg poses.
                eff_masks = np.where(
                    use_bg[None, :, :], bg_canvas, np.zeros_like(masks)
                )
            else:
                eff_masks = np.where(use_bg[None, :, :], bg_canvas, masks)
        else:
            eff_masks = masks

        count = eff_masks.sum(axis=0)

        # Case 1: pixels with exactly 1 sample
        m1 = count == 1
        if m1.any():
            idx1 = eff_masks[:, m1].argmax(axis=0)
            rows1, cols1 = np.where(m1)
            canvas[y0:y1][rows1, cols1] = stack[idx1, rows1, cols1]

        # Case 2: pixels with > 1 samples
        m_gt1 = count > 1
        if m_gt1.any():
            canvas_strip = canvas[y0:y1]
            s_gt1 = stack.reshape(N, -1, 3)[:, m_gt1.flatten(), :]
            masks_gt1 = eff_masks.reshape(N, -1)[:, m_gt1.flatten()]

            s_gt1_f = s_gt1.astype(np.float32)

            # P1.3 — Confidence-weighted average for low-quality edges (W3).
            # When any frame has matching confidence < 0.70 (Template Match or
            # Phase Correlation fallback), replace the pure median with a
            # weighted average so high-confidence LoFTR frames dominate.
            _use_weighted = (
                confidence_weights is not None
                and float(confidence_weights.min()) < 0.70
            )
            if _use_weighted:
                # Build weight matrix: (N, P) — zero for out-of-bounds pixels
                w_mat = np.where(
                    masks_gt1,
                    confidence_weights[:, np.newaxis],
                    0.0,
                ).astype(np.float32)
                w_sum = w_mat.sum(axis=0)  # (P,)
                safe_w = np.where(w_sum > 0, w_sum, 1.0)
                # Weighted average: (P, 3)
                med = (s_gt1_f * w_mat[:, :, np.newaxis]).sum(axis=0) / safe_w[
                    :, np.newaxis
                ]
            else:
                s_gt1_f[~masks_gt1] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    med = _gpu_nanmedian(s_gt1_f)

                    # §2.5 — Overmix-style background averaging (fixed
                    # 2026-07-27, #24): where confirmed-background samples
                    # accumulate (only meaningful under A5 fg-exclusion —
                    # otherwise samples aren't confirmed-bg and averaging
                    # risks ghosting a differing animation pose into the
                    # plate), blend toward the mean for the sqrt(N) noise
                    # reduction it gives over the median. count==2 stays
                    # pure median (no averaging benefit there); the blend
                    # weight ramps linearly from 0 at count==2 to 1 at
                    # count>=_BG_AVERAGE_FULL_AT, so there is no value
                    # discontinuity at the count boundary itself — count
                    # varies geographically with frame-overlap geometry, so
                    # an abrupt switch previously produced visible strip
                    # banding along those geographic contours.
                    if _BG_AVERAGE and _exclude_fg:
                        count_gt1 = count[m_gt1].astype(np.float32)
                        full_at = max(3, _BG_AVERAGE_FULL_AT)
                        blend_w = np.clip((count_gt1 - 2.0) / (full_at - 2.0), 0.0, 1.0)
                        if (blend_w > 0.0).any():
                            mean_ = np.nanmean(s_gt1_f, axis=0)
                            w = blend_w[:, None]
                            med = (1.0 - w) * med + w * mean_

            canvas_strip.reshape(-1, 3)[m_gt1.flatten()] = np.clip(med, 0, 255).astype(
                np.uint8
            )

        # ── Fade-in / fade-out post-pass ────────────────────────────────────
        # For each frame whose entry or exit boundary falls inside this chunk,
        # smoothly ramp its median contribution over RENDERING_FADE_ROWS rows/cols.
        if scroll_axis != "horizontal":
            for i in range(N):
                for fade_start, fade_end, is_entry in [
                    (
                        _frame_ty[i] - LANCZOS_BLEED,
                        _frame_ty[i] + RENDERING_FADE_ROWS,
                        True,
                    ),
                    (
                        _frame_bot[i] - RENDERING_FADE_ROWS,
                        _frame_bot[i] + LANCZOS_BLEED,
                        False,
                    ),
                ]:
                    if fade_end <= y0 or fade_start >= y1:
                        continue  # fade zone not in this chunk

                    local_start = max(0, int(np.floor(fade_start)) - y0)
                    local_end = min(ch, int(np.ceil(fade_end)) - y0)
                    if local_start >= local_end:
                        continue

                    s_f_full = stack[:, local_start:local_end, :, :].astype(np.float32)
                    # A5: fade uses the same fg-excluded effective masks as the
                    # main median so the entry/exit ramp stays background-clean.
                    m_full = eff_masks[:, local_start:local_end, :]  # (N, rows, W)

                    s_f_no_i = s_f_full.copy()
                    m_no_i = m_full.copy()
                    m_no_i[i] = False
                    s_f_no_i[~m_no_i] = np.nan
                    s_f_full[~m_full] = np.nan

                    count_no_i = m_no_i.sum(axis=0)  # (rows, W)
                    i_present = m_full[i]  # (rows, W)
                    affected = i_present & (count_no_i >= 1)

                    if not affected.any():
                        continue

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        med_with = _gpu_nanmedian(s_f_full)  # (rows, W, 3)
                        med_without = _gpu_nanmedian(s_f_no_i)  # (rows, W, 3)

                    canvas_ys = np.arange(
                        y0 + local_start, y0 + local_end, dtype=np.float64
                    )
                    if is_entry:
                        alphas = np.clip(
                            (canvas_ys - fade_start) / RENDERING_FADE_ROWS, 0.0, 1.0
                        )
                    else:
                        alphas = np.clip(
                            (fade_end - canvas_ys) / RENDERING_FADE_ROWS, 0.0, 1.0
                        )
                    alphas = alphas[:, np.newaxis, np.newaxis]  # (rows, 1, 1)

                    blended = (1.0 - alphas) * med_without + alphas * med_with

                    canvas_rows = canvas[y0 + local_start : y0 + local_end]
                    aff3 = np.stack([affected] * 3, axis=-1)  # (rows, W, 3)
                    canvas_rows[aff3] = np.clip(blended[aff3], 0, 255).astype(np.uint8)
        else:
            for i in range(N):
                for fade_start, fade_end, is_entry in [
                    (
                        _frame_tx[i] - LANCZOS_BLEED,
                        _frame_tx[i] + RENDERING_FADE_ROWS,
                        True,
                    ),
                    (
                        _frame_right[i] - RENDERING_FADE_ROWS,
                        _frame_right[i] + LANCZOS_BLEED,
                        False,
                    ),
                ]:
                    local_start = max(0, int(np.floor(fade_start)))
                    local_end = min(W, int(np.ceil(fade_end)))
                    if local_start >= local_end:
                        continue

                    s_f_full = stack[:, :, local_start:local_end, :].astype(np.float32)
                    m_full = eff_masks[:, :, local_start:local_end]  # (N, ch, cols)

                    s_f_no_i = s_f_full.copy()
                    m_no_i = m_full.copy()
                    m_no_i[i] = False
                    s_f_no_i[~m_no_i] = np.nan
                    s_f_full[~m_full] = np.nan

                    count_no_i = m_no_i.sum(axis=0)  # (ch, cols)
                    i_present = m_full[i]  # (ch, cols)
                    affected = i_present & (count_no_i >= 1)

                    if not affected.any():
                        continue

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        med_with = _gpu_nanmedian(s_f_full)  # (ch, cols, 3)
                        med_without = _gpu_nanmedian(s_f_no_i)  # (ch, cols, 3)

                    canvas_xs = np.arange(local_start, local_end, dtype=np.float64)
                    if is_entry:
                        alphas = np.clip(
                            (canvas_xs - fade_start) / RENDERING_FADE_ROWS, 0.0, 1.0
                        )
                    else:
                        alphas = np.clip(
                            (fade_end - canvas_xs) / RENDERING_FADE_ROWS, 0.0, 1.0
                        )
                    alphas = alphas[np.newaxis, :, np.newaxis]  # (1, cols, 1)

                    blended = (1.0 - alphas) * med_without + alphas * med_with

                    canvas_cols = canvas[y0:y1, local_start:local_end]
                    aff3 = np.stack([affected] * 3, axis=-1)  # (ch, cols, 3)
                    canvas_cols[aff3] = np.clip(blended[aff3], 0, 255).astype(np.uint8)

        valid_mask[y0:y1][geo_count > 0] = 255

    if not _skip_anim:
        canvas = _render_animation_repaint(
            canvas, frames, affines, bg_masks, H, W,
            _baselines=_baselines, confidence_weights=confidence_weights,
        )

    return canvas, valid_mask, [], []


__all__ = ["_render_median"]
