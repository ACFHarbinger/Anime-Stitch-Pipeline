"""Canvas warping and per-frame background-referenced luminance normalisation."""

from __future__ import annotations

import cv2
import numpy as np
from backend.src.constants import LUMINANCE_WEIGHTS

from ._flags import _BG_NORM_MIN_PX
from ._gain_compensation import _bg_gain_unclamped


def _has_sufficient_bg(
    bg_sel: np.ndarray,
    min_px: int = 200,
) -> bool:
    """§1.27: Return True iff the background mask has at least *min_px* True pixels.

    The normalisation loop requires enough background pixels to compute a
    reliable mean luminance for gain estimation.  Below *min_px* the sample
    is too sparse and the estimated gain is noisy — particularly when the
    character fills most of the frame (portrait shots, tight cropping).

    Parameters
    ----------
    bg_sel:
        Boolean or uint8 mask; True/nonzero = background pixel.
    min_px:
        Minimum number of background pixels required for reliable estimation.
        Defaults to 200 (the historical hardcoded floor).

    Returns
    -------
    bool
        True when the background coverage is sufficient for normalisation.
    """
    if bg_sel is None:
        return False
    count = int(np.count_nonzero(bg_sel))
    return count >= max(1, min_px)


def _coherence_skip_mask(
    order: np.ndarray,
    frame_lums: list[float | None],
    coherence_limit: float = 20.0,
) -> list[bool]:
    """Per-frame normalization-skip mask from adjacent-strip coherence check (S18).

    Marks both frames in an adjacent pair as skip-normalization when their
    background luminance differs by more than *coherence_limit*.  Only the
    bad pair's frames are excluded — other frames proceed normally.  This
    replaces the former global-skip approach that penalised every frame when
    a single scene-change pair exceeded the limit.

    Returns a list of bool, one entry per frame index (not per order slot).
    """
    N = len(order)
    skip: list[bool] = [False] * N
    lum_by_order = [frame_lums[int(order[k])] for k in range(N)]
    for k in range(N - 1):
        la, lb = lum_by_order[k], lum_by_order[k + 1]
        if la is not None and lb is not None and abs(la - lb) > coherence_limit:
            skip[int(order[k])] = True
            skip[int(order[k + 1])] = True
    return skip


def _warp_inputs(
    frames: list,
    affines: list,
    bg_masks: list,
    H: int,
    W: int,
    N: int,
    *,
    include_valid: bool = False,
    preserve_ternary: bool = False,
) -> tuple:
    # Warp every frame to the full canvas.
    warped_list = []
    warped_valid = []
    for i in range(N):
        wf = cv2.warpAffine(
            frames[i],
            affines[i],
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_list.append(wf)
        source_valid = np.full(frames[i].shape[:2], 255, dtype=np.uint8)
        valid = cv2.warpAffine(
            source_valid,
            affines[i],
            (W, H),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 127
        warped_valid.append(valid)

    # Warp bg_masks to canvas space (True = background pixel).
    warped_bg = []
    for i in range(N):
        if bg_masks[i] is not None:
            m_u8 = bg_masks[i].astype(np.uint8)
            if bg_masks[i].dtype == bool:
                m_u8 = m_u8 * 255
            wm = cv2.warpAffine(
                m_u8,
                affines[i],
                (W, H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            # P1/P2 needs the uncertainty state to exclude it from both plate
            # and foreground ownership. The legacy compositor consumes bools.
            if preserve_ternary and (bg_masks[i] == 128).any():
                warped_bg.append(wm)
            else:
                thresh = 200 if (bg_masks[i] == 128).any() else 127
                warped_bg.append(wm > thresh)
        else:
            warped_bg.append(None)
    if include_valid:
        return warped_list, warped_bg, warped_valid
    return warped_list, warped_bg


def _compute_frame_lums(
    warped_list: list,
    warped_bg: list,
    N: int,
) -> list[float | None]:
    frame_lums: list[float | None] = []
    for i in range(N):
        if warped_bg[i] is not None:
            bg_sel = warped_bg[i] & (warped_list[i].max(axis=2) > 10)
            bg_px = warped_list[i][bg_sel]
            if len(bg_px) >= 200:
                frame_lums.append(
                    float(bg_px.astype(np.float32).dot(LUMINANCE_WEIGHTS).mean())
                )
                continue
        frame_lums.append(None)
    return frame_lums


def _compute_skip_normalization_mask(
    order: np.ndarray,
    frame_lums: list,
    N: int,
) -> list:
    _COHERENCE_LIMIT = 20.0
    valid_lums = [lum for lum in frame_lums if lum is not None]
    _skip_norm = _coherence_skip_mask(order, frame_lums, _COHERENCE_LIMIT)
    if len(valid_lums) >= 2:
        lum_by_order = [frame_lums[int(order[k])] for k in range(N)]
        adj_diffs = [
            abs(lum_by_order[k + 1] - lum_by_order[k])
            for k in range(len(lum_by_order) - 1)
            if lum_by_order[k] is not None and lum_by_order[k + 1] is not None
        ]
        _max_adj_diff = max(adj_diffs) if adj_diffs else 0.0
        _n_skipped = sum(_skip_norm)
        if _n_skipped:
            print(
                f"[Stitch]   Color coherence gate (per-pair): max adj diff={_max_adj_diff:.1f}"
                f" → skipping normalization for {_n_skipped}/{N} frames in bad pairs."
            )
        else:
            print(
                f"[Stitch]   Color coherence OK (max adj diff={_max_adj_diff:.1f}). "
                "Applying normalization."
            )
    return _skip_norm


def _normalize_single_frame(
    i: int,
    canvas: np.ndarray,
    warped_list: list,
    warped_bg: list,
    _skip_norm: list,
    global_ref_lum: float | None,
    frame_lums: list,
) -> tuple:
    if (
        not _skip_norm[i]
        and global_ref_lum is not None
        and warped_bg[i] is not None
    ):
        bg_sel = warped_bg[i] & (warped_list[i].max(axis=2) > 10)
        _bg_min = _BG_NORM_MIN_PX if _BG_NORM_MIN_PX > 0 else 200
        if _has_sufficient_bg(bg_sel, _bg_min) and frame_lums[i] is not None:
            f32 = warped_list[i].astype(np.float32)
            gain = _bg_gain_unclamped(global_ref_lum, frame_lums[i])
            f32[bg_sel] = np.clip(f32[bg_sel] * gain, 0, 255)
            print(f"[Stitch]     Frame {i}: lum_gain={gain:.3f} (bg-only)")
            return f32.astype(np.uint8), gain
    return warped_list[i], 1.0


def _normalize_warped_frames(
    canvas: np.ndarray,
    warped_list: list,
    warped_bg: list,
    order: np.ndarray,
    N: int,
    H: int,
    W: int,
) -> tuple:
    print("[Stitch]   Normalising warped frames to global temporal-median reference...")
    union_bg = np.zeros((H, W), dtype=bool)
    for wb in warped_bg:
        if wb is not None:
            union_bg |= wb

    global_ref_lum = None
    ref_px = canvas[union_bg & (canvas.max(axis=2) > 10)]
    if len(ref_px) >= 500:
        global_ref_lum = float(ref_px.astype(np.float32).dot(LUMINANCE_WEIGHTS).mean())

    frame_lums = _compute_frame_lums(warped_list, warped_bg, N)
    _skip_norm = _compute_skip_normalization_mask(order, frame_lums, N)

    frame_gains = [1.0] * N
    warped_norm = []
    for i in range(N):
        wn, gain = _normalize_single_frame(
            i, canvas, warped_list, warped_bg, _skip_norm, global_ref_lum, frame_lums
        )
        warped_norm.append(wn)
        frame_gains[i] = gain

    return warped_norm, frame_gains


__all__ = [
    "_has_sufficient_bg",
    "_coherence_skip_mask",
    "_warp_inputs",
    "_compute_frame_lums",
    "_compute_skip_normalization_mask",
    "_normalize_single_frame",
    "_normalize_warped_frames",
]
