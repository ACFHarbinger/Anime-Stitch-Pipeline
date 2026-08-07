"""Per-seam blend/single-pose fill and hard-partition/black-pixel fallback fill."""

from __future__ import annotations

import os

import numpy as np
from asp_backend.core.stateless import _laplacian_blend

from ._fg_pose import _seam_color_match, _single_pose_soft_edge
from ._flags import _BLOCKS_GAIN_COMP, _BLOCKS_LUM_COMP
from ._gain_compensation import _blocks_gain_compensate, _blocks_lum_compensate
from ._seam_cache import _get_or_compute_path_local
from ._seam_cost import _soft_seam_weight


def _initial_hard_partition_fill(
    canvas: np.ndarray,
    warped_norm: list[np.ndarray],
    warped_bg: list[np.ndarray | None],
    order: np.ndarray,
    boundaries: np.ndarray,
    N: int,
    H: int,
) -> np.ndarray:
    result = canvas.copy()
    for k in range(N):
        fi = int(order[k])
        y_start = 0 if k == 0 else int(boundaries[k - 1])
        y_end = H if k == N - 1 else int(boundaries[k])
        src = warped_norm[fi][y_start:y_end]
        has_content = src.max(axis=2) > 0
        _bg_fi = warped_bg[fi]
        if _bg_fi is not None:
            is_fg = ~_bg_fi[y_start:y_end]
            replace = has_content & is_fg
        else:
            replace = has_content
        result[y_start:y_end][replace] = src[replace]
    return result


def _fill_single_pose(
    k: int,
    y0_f: int,
    y1_f: int,
    zone_h: int,
    result: np.ndarray,
    fi_a: int,
    fi_b: int,
    warped_norm: list[np.ndarray],
    _single: int,
    seam_post_diffs: dict,
    path_local: np.ndarray,
    apply_mask: np.ndarray,
    feather: int,
):
    dom_zone = warped_norm[_single][y0_f:y1_f]
    oth = fi_b if _single == fi_a else fi_a
    oth_zone = warped_norm[oth][y0_f:y1_f]
    dom_has = dom_zone.max(axis=2) > 0
    fg_apply = apply_mask
    take_dom = fg_apply & dom_has
    take_oth = fg_apply & (~dom_has) & (oth_zone.max(axis=2) > 0)
    result[y0_f:y1_f][take_dom] = dom_zone[take_dom]
    result[y0_f:y1_f][take_oth] = oth_zone[take_oth]

    _eff_sp_soft_px = int(os.environ.get("ASP_SP_SOFT_PX", "6"))
    _band_px_sp = _eff_sp_soft_px + 4
    _oth_matched = _seam_color_match(
        dom_zone, oth_zone, path_local, _band_px_sp
    )
    _sp_zone = _single_pose_soft_edge(
        dom_zone, _oth_matched, path_local, fg_apply, _eff_sp_soft_px
    )
    _both_for_sp = dom_has & (oth_zone.max(axis=2) > 0) & fg_apply
    if _both_for_sp.any():
        result[y0_f:y1_f][_both_for_sp] = _sp_zone[_both_for_sp]

    print(
        f"[Stitch]   Single-pose B{k} (frame {_single}): "
        f"zone=[{y0_f}–{y1_f}] fg_px={int(fg_apply.sum())} "
        f"soft_px={_eff_sp_soft_px}"
    )


def _blend_or_single_pose_fill(
    k: int,
    y0_f: int,
    y1_f: int,
    zone_h: int,
    W: int,
    result: np.ndarray,
    fa_zone: np.ndarray,
    fb_zone: np.ndarray,
    fi_a: int,
    fi_b: int,
    warped_norm: list[np.ndarray],
    warped_bg: list[np.ndarray | None],
    seam_single_pose: dict,
    seam_post_diffs: dict,
    seam_synthesized: dict,
    seam_canonical_crops: dict,
    path_local: np.ndarray,
    blended: np.ndarray,
    apply_mask: np.ndarray,
    is_fg: np.ndarray | None,
    feather: int,
):
    _single = seam_single_pose.get(k)
    if _single is not None and is_fg is not None:
        _fill_single_pose(
            k, y0_f, y1_f, zone_h, result, fi_a, fi_b, warped_norm,
            _single, seam_post_diffs, path_local, apply_mask, feather
        )
    else:
        has_a = fa_zone.max(axis=2) > 0
        has_b = fb_zone.max(axis=2) > 0
        both_content = has_a & has_b & apply_mask
        only_a = has_a & (~has_b) & apply_mask
        only_b = (~has_a) & has_b & apply_mask
        if both_content.any():
            result[y0_f:y1_f][both_content] = blended[both_content]
        if only_a.any():
            result[y0_f:y1_f][only_a] = fa_zone[only_a]
        if only_b.any():
            result[y0_f:y1_f][only_b] = fb_zone[only_b]
        print(
            f"[Stitch]   Blended B{k} (frames {fi_a}/{fi_b}, laplacian): "
            f"zone=[{y0_f}–{y1_f}] feather={feather}px "
            f"seam=[{int(path_local.min())}–{int(path_local.max())}]"
        )


def _process_single_seam(
    k: int,
    by: float,
    result: np.ndarray,
    order: np.ndarray,
    feathers: np.ndarray,
    warped_norm: list[np.ndarray],
    warped_bg: list[np.ndarray | None],
    seam_single_pose: dict,
    seam_post_diffs: dict,
    seam_synthesized: dict,
    seam_canonical_crops: dict,
    seam_overrides: dict | None,
    _precomp_paths: dict,
    _eff_exclusion: list[np.ndarray] | None,
    seam_meta_out: dict | None,
    H: int,
    W: int,
):
    fi_a = int(order[k])
    fi_b = int(order[k + 1])
    feather = int(feathers[k])

    y0_f = max(0, int(by) - feather)
    y1_f = min(H, int(by) + feather + 1)
    zone_h = y1_f - y0_f
    if zone_h < 4:
        return

    fa_zone = warped_norm[fi_a][y0_f:y1_f]
    fb_zone = warped_norm[fi_b][y0_f:y1_f]
    _bg_a = warped_bg[fi_a]
    _bg_b = warped_bg[fi_b]
    bg_a_zone = _bg_a[y0_f:y1_f] if _bg_a is not None else None
    bg_b_zone = _bg_b[y0_f:y1_f] if _bg_b is not None else None

    path_local, sem_cost = _get_or_compute_path_local(
        k, y0_f, y1_f, zone_h, W, fa_zone, fb_zone, bg_a_zone, bg_b_zone, result[y0_f:y1_f],
        _precomp_paths, _eff_exclusion, seam_overrides
    )

    mask_float = _soft_seam_weight(
        fa_zone,
        fb_zone,
        path_local,
        zone_h,
        W,
        bg_mask_a=bg_a_zone,
        bg_mask_b=bg_b_zone,
    )

    _fb_for_blend = fb_zone
    if k not in seam_single_pose:
        if _BLOCKS_GAIN_COMP:
            _fb_for_blend = _blocks_gain_compensate(fa_zone, _fb_for_blend)
        if _BLOCKS_LUM_COMP:
            _fb_for_blend = _blocks_lum_compensate(fa_zone, _fb_for_blend)

    blended = _laplacian_blend(fa_zone, _fb_for_blend, mask_float)

    has_a = fa_zone.max(axis=2) > 0
    has_b = fb_zone.max(axis=2) > 0
    has_any = has_a | has_b

    if bg_a_zone is not None and bg_b_zone is not None:
        is_fg = ~(bg_a_zone & bg_b_zone)
        apply_mask = has_any & is_fg
    else:
        is_fg = None
        apply_mask = has_any

    _blend_or_single_pose_fill(
        k, y0_f, y1_f, zone_h, W, result, fa_zone, fb_zone, fi_a, fi_b,
        warped_norm, warped_bg, seam_single_pose, seam_post_diffs,
        seam_synthesized, seam_canonical_crops, path_local, blended,
        apply_mask, is_fg, feather
    )


def _fill_still_black_pixels(
    result: np.ndarray,
    warped_norm: list[np.ndarray],
):
    still_black = result.max(axis=2) == 0
    if still_black.any():
        for wn in warped_norm:
            has_content = (wn.max(axis=2) > 0) & still_black
            if has_content.any():
                result[has_content] = wn[has_content]
                still_black = result.max(axis=2) == 0
                if not still_black.any():
                    break


__all__ = [
    "_initial_hard_partition_fill",
    "_fill_single_pose",
    "_blend_or_single_pose_fill",
    "_process_single_seam",
    "_fill_still_black_pixels",
]
