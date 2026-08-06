"""Foreground pose re-registration and single-pose seam-color matching."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from animation.alignment.fg_register import register_foreground_at_seam

from ._boundaries import _dominant_frame_in_band
from ._flags import _FG_REGISTER_ENABLED, _PHASE_COMPOSITE


def _single_pose_soft_edge(
    dom_zone: np.ndarray,
    oth_zone: np.ndarray,
    path_local: np.ndarray,
    apply_mask: np.ndarray,
    sp_soft_px: int,
) -> np.ndarray:
    """Narrow path-guided blend at a single-pose seam boundary (S15).

    Applies a linear feather of half-width *sp_soft_px* centred on
    *path_local* to smooth the hard color step between the dominant frame
    and the fill frame.  The zone is intentionally narrow (≤ 12 px at the
    default of 6) so pose-gap ghosts are not perceptible.

    Only pixels where BOTH frames have foreground content (non-zero) AND
    *apply_mask* is True AND distance to *path_local* < *sp_soft_px* are
    modified.  All other pixels are returned unchanged.

    Returns a (zone_h, W, C) uint8 copy of dom_zone with the blend applied.
    """
    zone_h, W = dom_zone.shape[:2]
    out = dom_zone.copy()
    if sp_soft_px <= 0:
        return out
    both_have = (dom_zone.max(axis=2) > 0) & (oth_zone.max(axis=2) > 0) & apply_mask
    if not both_have.any():
        return out
    _row_idx = np.arange(zone_h, dtype=np.float32)[:, np.newaxis]
    _path_row = path_local[np.newaxis, :].astype(np.float32)
    _dist = np.abs(_row_idx - _path_row)  # (zone_h, W)
    _in_band = (_dist < sp_soft_px) & both_have
    if not _in_band.any():
        return out
    _w_oth = (np.clip(1.0 - _dist / sp_soft_px, 0.0, 1.0) * 0.5)[:, :, np.newaxis]
    _blended = np.clip(
        dom_zone.astype(np.float32) * (1.0 - _w_oth)
        + oth_zone.astype(np.float32) * _w_oth,
        0,
        255,
    ).astype(np.uint8)
    out[_in_band] = _blended[_in_band]
    return out


def _seam_color_match(
    dom_zone: np.ndarray,
    oth_zone: np.ndarray,
    path_local: np.ndarray,
    band_px: int,
) -> np.ndarray:
    """Shift *oth_zone* channel means to match *dom_zone* inside the seam band (S16).

    Computes per-channel mean luminance of content pixels within `band_px` rows
    of the seam path in each zone, then adds the per-channel delta to oth_zone's
    band pixels.  Pixels outside the band are returned unchanged.

    This reduces the color step at the seam centre from `post_warp_diff` lum
    units toward zero before the S15 linear ramp blend is applied, making the
    composite transition far less perceptible.  Safe to call even when
    `band_px == sp_soft_px` because the blend ramp weight approaches 0 at the
    band edge — the shift is smoothed away by the blend itself.

    Returns a (zone_h, W, C) uint8 copy of oth_zone with the band shifted.
    Unchanged copy returned when fewer than 10 content pixels exist in either
    zone's band (degenerate zone).
    """
    if band_px <= 0:
        return oth_zone.copy()
    zone_h, W = oth_zone.shape[:2]
    _row_idx = np.arange(zone_h, dtype=np.float32)[:, np.newaxis]
    _path_row = path_local[np.newaxis, :].astype(np.float32)
    _in_band = np.abs(_row_idx - _path_row) < band_px  # (zone_h, W) bool

    dom_content_mask = (dom_zone.max(axis=2) > 0) & _in_band
    oth_content_mask = (oth_zone.max(axis=2) > 0) & _in_band

    if dom_content_mask.sum() < 10 or oth_content_mask.sum() < 10:
        return oth_zone.copy()

    dom_mean = dom_zone[dom_content_mask].astype(np.float32).mean(axis=0)  # (C,)
    oth_mean = oth_zone[oth_content_mask].astype(np.float32).mean(axis=0)  # (C,)
    delta = dom_mean - oth_mean  # (C,)

    out = oth_zone.copy()
    shifted = oth_zone[_in_band].astype(np.float32) + delta
    out[_in_band] = np.clip(shifted, 0, 255).astype(np.uint8)
    return out


def _check_preemptive_escalations(
    k: int,
    by: float,
    fi_a: int,
    fi_b: int,
    fg_a: np.ndarray,
    fg_b: np.ndarray,
    affines: list,
    warped_norm: list,
    feathers: np.ndarray,
    seam_overrides: dict,
    seam_single_pose: dict,
    seam_post_diffs: dict,
    H: int,
    phase_ids: Optional[List[int]] = None,
) -> bool:
    # §2.4A: User seam override — force single-pose for this seam.
    _ov_k = (seam_overrides or {}).get(k, {})
    if _ov_k.get("force_single_pose"):
        seam_single_pose[k] = _dominant_frame_in_band(
            by, fi_a, fi_b, fg_a, fg_b, feathers, k, H
        )
        seam_post_diffs[k] = 99.0  # sentinel: user-forced single-pose
        return True

    # §2.3 — Phase-consistent compositing: never midpoint-warp across an
    # animation-phase boundary. Checked before any registration attempt, not
    # as a post-hoc residual escalation — this makes the failure mode
    # structurally impossible for phase-labelled seams, per the roadmap's
    # coherence-first rationale.
    if (
        _PHASE_COMPOSITE
        and phase_ids is not None
        and fi_a < len(phase_ids)
        and fi_b < len(phase_ids)
        and phase_ids[fi_a] != phase_ids[fi_b]
    ):
        dom = _dominant_frame_in_band(by, fi_a, fi_b, fg_a, fg_b, feathers, k, H)
        seam_single_pose[k] = dom
        seam_post_diffs[k] = 98.0  # sentinel: phase-boundary single-pose
        print(
            f"[Stitch]     Seam B{k} (frames {fi_a}/{fi_b}): phase boundary "
            f"({phase_ids[fi_a]}→{phase_ids[fi_b]}) → single-pose (frame {dom})"
        )
        return True

    return False


def _apply_foreground_registration(
    k: int,
    by: float,
    fi_a: int,
    fi_b: int,
    fg_a: np.ndarray,
    fg_b: np.ndarray,
    warped_norm: list,
    warped_bg: list,
    feathers: np.ndarray,
    seam_single_pose: dict,
    seam_post_diffs: dict,
    reg_axis: int,
    alpha_a: float,
    alpha_b: float,
    ref_fi: int,
    _flow_ov: Optional[np.ndarray],
    H: int,
) -> None:
    adj_a, adj_b, info = register_foreground_at_seam(
        warped_norm[fi_a],
        warped_norm[fi_b],
        fg_a,
        fg_b,
        seam_pos=int(by),
        axis=reg_axis,
        alpha_a=alpha_a,
        alpha_b=alpha_b,
        flow_override=_flow_ov,
    )
    if info["warped"]:
        warped_norm[fi_a] = adj_a
        warped_norm[fi_b] = adj_b
        post_diff = info.get("post_warp_diff", 0.0)
        seam_post_diffs[k] = post_diff
        # A6 ghost-prevention: escalate to single-pose when the post-warp
        # residual is still large (fixed 22-lum threshold, benchmarked default).
        _sp_thresh = 22.0
        if post_diff > _sp_thresh:
            dom = fi_a if info["dominant"] == "a" else fi_b
            seam_single_pose[k] = dom
            print(
                f"[Stitch]     FG-register B{k} (frames {fi_a}/{fi_b}): "
                f"residual={info['residual']:.1f}px post_diff={post_diff:.1f} "
                f"→ re-posed BUT escalated to single-pose (ghost prevention)"
            )
        else:
            print(
                f"[Stitch]     FG-register B{k} (frames {fi_a}/{fi_b}): "
                f"residual={info['residual']:.1f}px α=({alpha_a:.2f},{alpha_b:.2f}) "
                f"post_diff={post_diff:.1f} fg_px={info['fg_pixels']} → re-posed"
            )
    elif info.get("fallback"):
        dom = fi_a if info["dominant"] == "a" else fi_b
        seam_single_pose[k] = dom
        seam_post_diffs[k] = float(info.get("residual", 0.0))
        print(
            "[Stitch]     FG-register B{k} (frames {fi_a}/{fi_b}): "
            f"residual={info['residual']:.1f}px too large → "
            f"single-pose fallback (frame {dom})"
        )


def _register_foreground_poses(
    warped_norm: list,
    warped_bg: list,
    order: np.ndarray,
    boundaries: np.ndarray,
    feathers: np.ndarray,
    affines: list,
    frames: list,
    seam_overrides: dict,
    ty_range: float,
    tx_range: float,
    H: int,
    W: int,
    N: int,
    phase_ids: Optional[List[int]] = None,
) -> tuple:
    seam_single_pose = {}
    seam_post_diffs = {}  # k → post-warp diff score (residual if fallback)
    seam_synthesized = {}  # retained for signature compatibility (always empty)
    if _FG_REGISTER_ENABLED and N >= 2:
        try:
            scroll_is_h = tx_range > 0 and ty_range / max(tx_range, 1.0) < 0.1
            reg_axis = 1 if scroll_is_h else 0

            # Reference index: the temporally-central frame in the sorted order.
            ref_idx_in_order = len(order) // 2
            ref_fi = int(order[ref_idx_in_order])

            for k, by in enumerate(boundaries):
                fi_a = int(order[k])
                fi_b = int(order[k + 1])
                if warped_bg[fi_a] is None or warped_bg[fi_b] is None:
                    continue
                fg_a = ~warped_bg[fi_a]
                fg_b = ~warped_bg[fi_b]

                if _check_preemptive_escalations(
                    k, by, fi_a, fi_b, fg_a, fg_b, affines, warped_norm, feathers, seam_overrides, seam_single_pose, seam_post_diffs, H, phase_ids
                ):
                    continue

                alpha_a = 0.5
                alpha_b = 0.5

                _apply_foreground_registration(
                    k, by, fi_a, fi_b, fg_a, fg_b, warped_norm, warped_bg, feathers,
                    seam_single_pose, seam_post_diffs, reg_axis, alpha_a, alpha_b, ref_fi, None, H
                )

        except Exception as _fg_exc:
            print(f"[Stitch]   FG pose registration skipped ({_fg_exc}).")

    return seam_single_pose, seam_post_diffs, seam_synthesized


__all__ = [
    "_single_pose_soft_edge",
    "_seam_color_match",
    "_check_preemptive_escalations",
    "_apply_foreground_registration",
    "_register_foreground_poses",
]
