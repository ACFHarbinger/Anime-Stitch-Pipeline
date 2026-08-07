"""C++/GPU fast path for ``_render_median`` (Phase 5e).

Fires when: no FG exclusion, no sequential colour correction, no baselines,
no confidence weighting, and the full-canvas stack fits within 1 GB.
Extracted from ``_render_median`` as a standalone function -- pure code
motion, no logic change -- so that function fits under the codebase's
500-code-line file-size convention (§5.17).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from backend.src.constants import LANCZOS_BLEED, RENDERING_FADE_ROWS

from . import _native
from ._animation_repaint import _render_animation_repaint
from ._gpu_median import _gpu_nanmedian

logger = logging.getLogger(__name__)


def _render_median_gpu_fastpath(  # noqa: C901
    frames: list[np.ndarray],
    affines: list[np.ndarray],
    bg_masks: list[np.ndarray | None],
    H: int,
    W: int,
    N: int,
    scroll_axis: str,
    frame_ty: np.ndarray,
    frame_bot: np.ndarray,
    frame_tx: np.ndarray,
    frame_right: np.ndarray,
    exclude_fg: bool,
    need_color_corr: bool,
    baselines: list[float] | None,
    confidence_weights: np.ndarray | None,
    skip_anim: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(canvas, valid_mask)`` on success, or ``None`` when the fast
    path's preconditions aren't met or the native call raised (caller should
    fall back to the chunked Python path in either case).
    """
    fast_path_mem_bytes = 2 * N * H * W * 3  # warped frames + fade stack
    if not (
        _native._BATCH_RENDER
        and not exclude_fg
        and not need_color_corr
        and baselines is None
        and confidence_weights is None
        and fast_path_mem_bytes <= 1024 * 1024 * 1024
    ):
        return None

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    valid_mask = np.zeros((H, W), dtype=np.uint8)

    try:
        affines_f32 = [np.ascontiguousarray(a, dtype=np.float32) for a in affines]
        _gpu_kw: dict = {"try_gpu": True} if _native._BATCH_GPU else {}
        warped = _native._batch_render.canvas.warp_frames_to_canvas(
            [np.ascontiguousarray(f) for f in frames],
            affines_f32, H, W, **_gpu_kw,
        )
        canvas = np.ascontiguousarray(_native._batch_render.canvas.render_median(warped, **_gpu_kw))
        for w in warped:
            valid_mask[w.max(axis=2) > 0] = 255
        # Fade pass — uses the full-canvas stack (no chunking needed here).
        _warped_stack = np.stack(warped, axis=0)  # (N, H, W, 3)
        _masks_stack = _warped_stack.max(axis=3) > 0  # (N, H, W)
        del warped  # free list; _warped_stack holds the same data
        if scroll_axis != "horizontal":
            for i in range(N):
                for fade_start, fade_end, is_entry in [
                    (
                        frame_ty[i] - LANCZOS_BLEED,
                        frame_ty[i] + RENDERING_FADE_ROWS,
                        True,
                    ),
                    (
                        frame_bot[i] - RENDERING_FADE_ROWS,
                        frame_bot[i] + LANCZOS_BLEED,
                        False,
                    ),
                ]:
                    local_start = max(0, int(np.floor(fade_start)))
                    local_end = min(H, int(np.ceil(fade_end)))
                    if local_start >= local_end:
                        continue
                    s_f_full = _warped_stack[:, local_start:local_end, :, :].astype(
                        np.float32
                    )
                    m_full = _masks_stack[:, local_start:local_end, :]
                    s_f_no_i = s_f_full.copy()
                    m_no_i = m_full.copy()
                    m_no_i[i] = False
                    s_f_no_i[~m_no_i] = np.nan
                    s_f_full[~m_full] = np.nan
                    count_no_i = m_no_i.sum(axis=0)
                    i_present = m_full[i]
                    affected = i_present & (count_no_i >= 1)
                    if not affected.any():
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        med_with = _gpu_nanmedian(s_f_full)
                        med_without = _gpu_nanmedian(s_f_no_i)
                    canvas_ys = np.arange(local_start, local_end, dtype=np.float64)
                    if is_entry:
                        alphas = np.clip(
                            (canvas_ys - fade_start) / RENDERING_FADE_ROWS, 0.0, 1.0
                        )
                    else:
                        alphas = np.clip(
                            (fade_end - canvas_ys) / RENDERING_FADE_ROWS, 0.0, 1.0
                        )
                    alphas = alphas[:, np.newaxis, np.newaxis]
                    blended = (1.0 - alphas) * med_without + alphas * med_with
                    canvas_rows = canvas[local_start:local_end]
                    aff3 = np.stack([affected] * 3, axis=-1)
                    canvas_rows[aff3] = np.clip(blended[aff3], 0, 255).astype(np.uint8)
        else:
            for i in range(N):
                for fade_start, fade_end, is_entry in [
                    (
                        frame_tx[i] - LANCZOS_BLEED,
                        frame_tx[i] + RENDERING_FADE_ROWS,
                        True,
                    ),
                    (
                        frame_right[i] - RENDERING_FADE_ROWS,
                        frame_right[i] + LANCZOS_BLEED,
                        False,
                    ),
                ]:
                    local_start = max(0, int(np.floor(fade_start)))
                    local_end = min(W, int(np.ceil(fade_end)))
                    if local_start >= local_end:
                        continue
                    s_f_full = _warped_stack[:, :, local_start:local_end, :].astype(
                        np.float32
                    )
                    m_full = _masks_stack[:, :, local_start:local_end]
                    s_f_no_i = s_f_full.copy()
                    m_no_i = m_full.copy()
                    m_no_i[i] = False
                    s_f_no_i[~m_no_i] = np.nan
                    s_f_full[~m_full] = np.nan
                    count_no_i = m_no_i.sum(axis=0)
                    i_present = m_full[i]
                    affected = i_present & (count_no_i >= 1)
                    if not affected.any():
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        med_with = _gpu_nanmedian(s_f_full)
                        med_without = _gpu_nanmedian(s_f_no_i)
                    canvas_xs = np.arange(local_start, local_end, dtype=np.float64)
                    if is_entry:
                        alphas = np.clip(
                            (canvas_xs - fade_start) / RENDERING_FADE_ROWS, 0.0, 1.0
                        )
                    else:
                        alphas = np.clip(
                            (fade_end - canvas_xs) / RENDERING_FADE_ROWS, 0.0, 1.0
                        )
                    alphas = alphas[np.newaxis, :, np.newaxis]
                    blended = (1.0 - alphas) * med_without + alphas * med_with
                    canvas_cols = canvas[:, local_start:local_end]
                    aff3 = np.stack([affected] * 3, axis=-1)
                    canvas_cols[aff3] = np.clip(blended[aff3], 0, 255).astype(np.uint8)
        # Animation re-render (same logic as the chunked path).
        if not skip_anim:
            canvas = _render_animation_repaint(
                canvas, frames, affines, bg_masks, H, W,
                _baselines=None, confidence_weights=None,
            )
        return canvas, valid_mask
    except Exception as _e:
        logger.debug("[Stitch] render_median batch fast-path failed: %s", _e)
        return None


__all__ = ["_render_median_gpu_fastpath"]
