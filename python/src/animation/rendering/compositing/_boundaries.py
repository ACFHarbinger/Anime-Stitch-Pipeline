"""Ownership-boundary placement and feather-width computation between adjacent
warped frames."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.src.constants import (
    FEATHER_MAX,
    FEATHER_MIN,
    FEATHER_TABLE,
    SEARCH_RANGE,
    SEARCH_SLAB,
)

from ._native import BATCH_AVAILABLE, batch


def _diff_to_feather(diff: float) -> int:
    for threshold, feather in FEATHER_TABLE:
        if diff <= threshold:
            return feather
    return FEATHER_MIN


def _gain_to_min_feather(gain_diff: float) -> int:
    """§1.6B: Minimum feather width from luminance gain difference (S22).

    When adjacent frames required significantly different gain corrections the
    normalization residual leaves a brightness step proportional to
    |gain_A − gain_B|.  Returns a floor feather wide enough to blend it:
    max(40, int(gain_diff × 300)), capped at 120 px.
    """
    return min(120, max(40, int(gain_diff * 300)))


def _find_optimal_boundaries(
    warped_list: List[np.ndarray],
    order: np.ndarray,
    initial_boundaries: np.ndarray,
    H: int,
    W: int,
    bg_masks: Optional[List[Optional[np.ndarray]]] = None,
    affines: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Move each boundary to the y-position (within ±SEARCH_RANGE of the midpoint)
    where the two adjacent warped frames are most photometrically similar.

    When bg_masks and affines are provided the score is computed over background
    pixels (static, unaffected by character pose).  Falls back to all-pixel diff
    when background pixel count is insufficient.

    Returns (optimised_boundaries, diff_scores), both shape (N-1,).
    """
    # Phase 5d: C++ fast path — GIL-released inner pixel loops, ~10–30× speedup
    if BATCH_AVAILABLE and hasattr(batch, "compositing") and hasattr(
        batch.compositing, "find_optimal_boundaries"
    ):
        try:
            _bg = None
            if bg_masks is not None:
                _bg = [
                    np.asarray(m, dtype=np.uint8) if m is not None else None
                    for m in bg_masks
                ]
            _bounds, _diffs = batch.compositing.find_optimal_boundaries(
                [np.ascontiguousarray(f) for f in warped_list],
                np.asarray(order, dtype=np.int64),
                np.asarray(initial_boundaries, dtype=np.float64),
                H, W,
                SEARCH_RANGE, SEARCH_SLAB,
                _bg, affines,
            )
            return np.asarray(_bounds), np.asarray(_diffs)
        except Exception:
            pass

    len(order)
    optimised = initial_boundaries.copy()
    diffs = np.full(len(initial_boundaries), float("inf"))

    # S17: Adaptive boundary search range.  For pure vertical-scroll sequences
    # (horizontal tx spread < 5 px) the optimal boundary is always very close
    # to the midpoint — a ±100 px window finds it safely and cuts 60 % of the
    # candidate evaluations for sparse sequences.  For diagonal/2D motion the
    # full ±SEARCH_RANGE is needed to reach the similarity minimum.
    _effective_range = SEARCH_RANGE
    if affines is not None and len(order) >= 2:
        _txs = np.array(
            [float(affines[int(order[i])][0, 2]) for i in range(len(order))],
            dtype=np.float32,
        )
        if float(np.ptp(_txs)) < 5.0:
            _effective_range = 100

    warped_bgs: List[Optional[np.ndarray]] = [None] * (max(order) + 1)
    if bg_masks is not None and affines is not None:
        for i in range(len(order)):
            fi = int(order[i])
            if bg_masks[fi] is not None:
                warped_bgs[fi] = (
                    cv2.warpAffine(
                        bg_masks[fi].astype(np.uint8),
                        affines[fi],
                        (W, H),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    > 127
                )

    for k, by in enumerate(initial_boundaries):
        fi_a = int(order[k])
        fi_b = int(order[k + 1])

        lo_limit = int(optimised[k - 1]) + 2 * SEARCH_SLAB + 1 if k > 0 else SEARCH_SLAB
        hi_limit = (
            int(initial_boundaries[k + 1]) - 2 * SEARCH_SLAB - 1
            if k < len(initial_boundaries) - 1
            else H - SEARCH_SLAB - SEARCH_SLAB
        )

        y_lo = max(lo_limit, int(by) - _effective_range)
        y_hi = min(hi_limit, int(by) + _effective_range)

        best_y = int(by)
        best_diff = float("inf")
        best_score = float("inf")

        bg_a = warped_bgs[fi_a]
        bg_b = warped_bgs[fi_b]

        for y_cand in range(y_lo, min(y_hi, H - SEARCH_SLAB)):
            slab_a = warped_list[fi_a][y_cand : y_cand + SEARCH_SLAB].astype(np.float32)
            slab_b = warped_list[fi_b][y_cand : y_cand + SEARCH_SLAB].astype(np.float32)
            all_valid = (slab_a.max(axis=2) > 0) & (slab_b.max(axis=2) > 0)
            if all_valid.sum() < 50:
                continue

            all_d = float(np.abs(slab_a - slab_b).mean(axis=2)[all_valid].mean())

            bg_d = None
            if bg_a is not None and bg_b is not None:
                bg_cand = (
                    bg_a[y_cand : y_cand + SEARCH_SLAB]
                    & bg_b[y_cand : y_cand + SEARCH_SLAB]
                    & all_valid
                )
                if bg_cand.sum() >= 50:
                    bg_d = float(np.abs(slab_a - slab_b).mean(axis=2)[bg_cand].mean())

            score = (0.4 * bg_d + 0.6 * all_d) if bg_d is not None else all_d

            if score < best_score:
                best_score = score
                best_diff = bg_d if bg_d is not None else all_d
                best_y = y_cand + SEARCH_SLAB // 2

        half = SEARCH_SLAB // 2
        y0_f = max(0, best_y - half)
        y1_f = min(H - 1, best_y + half)
        sa = warped_list[fi_a][y0_f:y1_f].astype(np.float32)
        sb = warped_list[fi_b][y0_f:y1_f].astype(np.float32)
        av = (sa.max(axis=2) > 0) & (sb.max(axis=2) > 0)
        total_diff = (
            float(np.abs(sa - sb).mean(axis=2)[av].mean())
            if av.sum() >= 10
            else best_diff
        )
        feather_metric = (
            best_diff if (best_diff < 20.0 and total_diff < 20.0) else total_diff
        )

        optimised[k] = float(best_y)
        diffs[k] = feather_metric
        feather = _diff_to_feather(feather_metric)
        moved = best_y - int(by)
        print(
            f"[Stitch]     Boundary {k} (frames {fi_a}/{fi_b}): "
            f"{int(by)} → {best_y} (Δ={moved:+d}, bg_diff={best_diff:.1f}, "
            f"total_diff={total_diff:.1f}, feather={feather}px)"
        )

    return optimised, diffs


def _compute_initial_boundaries(
    affines: List[np.ndarray],
    frames: List[np.ndarray],
) -> np.ndarray:
    """Return midpoint y-coordinates between adjacent frame strip centres.

    Used by HITL checkpoint 3.5 to pre-compute boundary candidates for the
    boundary editor dialog before `_composite_foreground` is called.
    """
    N = len(frames)
    strip_center_ys = np.array(
        [float(affines[i][1, 2]) + frames[i].shape[0] / 2.0 for i in range(N)],
        dtype=np.float64,
    )
    order = np.argsort(strip_center_ys)
    sorted_centers = strip_center_ys[order]
    return (sorted_centers[:-1] + sorted_centers[1:]) / 2.0


def _dominant_frame_in_band(
    by: float,
    fi_a: int,
    fi_b: int,
    fg_a: np.ndarray,
    fg_b: np.ndarray,
    feathers: np.ndarray,
    k: int,
    H: int,
) -> int:
    """Which of the two seam frames carries more foreground in the seam band —
    used whenever a seam is escalated straight to single-pose (user override,
    §2.3 phase boundary, or the A6 ghost-prevention threshold)."""
    _by_int = int(by)
    _half = min(20, int(feathers[k]))
    _y0 = max(0, _by_int - _half)
    _y1 = min(H, _by_int + _half)
    _fg_a_cnt = int(fg_a[_y0:_y1].sum())
    _fg_b_cnt = int(fg_b[_y0:_y1].sum())
    return fi_a if _fg_a_cnt >= _fg_b_cnt else fi_b


def _optimize_boundaries_and_feathers(
    warped_norm: list,
    order: np.ndarray,
    initial_boundaries: np.ndarray,
    bg_masks: list,
    affines: list,
    frames: list,
    frame_gains: list,
    warped_bg: list,
    H: int,
    W: int,
    N: int,
) -> tuple:
    print("[Stitch]   Optimising boundary placement...")
    boundaries, diff_scores = _find_optimal_boundaries(
        warped_norm,
        order,
        initial_boundaries,
        H,
        W,
        bg_masks=bg_masks,
        affines=affines,
    )
    feathers = np.array([_diff_to_feather(d) for d in diff_scores], dtype=np.int64)

    # Cap feathers by natural frame overlap so they never extend past real content
    n_b = len(boundaries)
    max_feathers = []
    for k in range(n_b):
        fi_a = int(order[k])
        fi_b = int(order[k + 1])
        ty_a = float(affines[fi_a][1, 2])
        ty_b = float(affines[fi_b][1, 2])
        H_a = frames[fi_a].shape[0]
        H_b = frames[fi_b].shape[0]
        nat_overlap = max(0, int(min(ty_a + H_a, ty_b + H_b) - max(ty_a, ty_b)))
        max_feather = max(5, min(nat_overlap // 2, FEATHER_MAX))
        max_feathers.append(max_feather)
        if feathers[k] > max_feather:
            feathers[k] = max_feather
    print(
        "[Stitch]   Feathers (overlap-capped): "
        + " ".join(f"B{k}={int(feathers[k])}px" for k in range(n_b))
    )

    # §1.6B: Minimum feather from luminance gain difference at each boundary.
    _feather_gain_widened = False
    for k in range(n_b):
        fi_a = int(order[k])
        fi_b = int(order[k + 1])
        gain_diff = abs(frame_gains[fi_a] - frame_gains[fi_b])
        min_fk = _gain_to_min_feather(gain_diff)
        if feathers[k] < min_fk:
            feathers[k] = min(min_fk, max_feathers[k])
            _feather_gain_widened = True
    if _feather_gain_widened:
        print(
            "[Stitch]   Feathers (§1.6B gain-adjusted): "
            + " ".join(f"B{k}={int(feathers[k])}px" for k in range(n_b))
        )

    return boundaries, feathers


__all__ = [
    "_diff_to_feather",
    "_gain_to_min_feather",
    "_find_optimal_boundaries",
    "_compute_initial_boundaries",
    "_dominant_frame_in_band",
    "_optimize_boundaries_and_feathers",
]
