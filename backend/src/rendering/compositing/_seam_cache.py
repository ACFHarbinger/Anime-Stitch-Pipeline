"""Seam-path caching keys, per-seam job preparation, and parallel precompute."""

from __future__ import annotations

import numpy as np
from backend.src.constants import SEAM_CROP_BAND_PX

from ._flags import _GRAPHCUT_SEAM, _get_seam_pool
from ._seam_cost import _build_seam_cost_map
from ._seam_cut import _seam_cut


def _get_seam_cost_flags() -> tuple:
    """§1.5D: Snapshot of module-level flags that affect seam cost computation."""
    return (_GRAPHCUT_SEAM,)


def _make_seam_cache_key(
    frame_keys: tuple[str, ...] | None,
    k: int,
    cost_flags: tuple,
) -> tuple | None:
    """§1.5D: Hashable cache key for seam boundary *k*.

    Returns *None* when *frame_keys* is None, disabling cache lookup/insertion.
    The key encodes frame identity, boundary index, and active cost flags so
    that changing a flag (e.g. enabling Poisson) correctly bypasses the cache.
    """
    if frame_keys is None:
        return None
    return (frame_keys, k, cost_flags)


def _extract_seam_crops(
    canvas: np.ndarray,
    boundaries: np.ndarray,
    band_px: int = SEAM_CROP_BAND_PX,
) -> dict[int, np.ndarray]:
    """§2.4C — Crop ±band_px rows around each seam boundary from *canvas*.

    Returns a dict mapping seam index k → cropped subarray.  The crop is
    clamped to canvas bounds so edge seams produce narrower crops rather than
    raising an error.  Returns an empty dict when *boundaries* is empty or
    *canvas* has zero area.
    """
    result: dict[int, np.ndarray] = {}
    if canvas.size == 0 or len(boundaries) == 0:
        return result
    H = canvas.shape[0]
    for k, by in enumerate(boundaries):
        y = int(by)
        y0 = max(0, y - band_px)
        y1 = min(H, y + band_px)
        result[k] = canvas[y0:y1].copy()
    return result


def _prepare_seam_jobs(
    boundaries: np.ndarray,
    order: np.ndarray,
    feathers: np.ndarray,
    warped_norm: list[np.ndarray],
    warped_bg: list[np.ndarray | None],
    frame_keys: tuple[str, ...] | None,
    seam_path_cache: dict | None,
    seam_overrides: dict | None,
    _eff_exclusion: list[np.ndarray] | None,
    _seam_cost_flags: dict,
    result: np.ndarray,
    H: int,
    W: int,
    _precomp_paths: dict,
) -> list[tuple]:
    _seam_jobs = []
    for _k, _by in enumerate(boundaries):
        _ck = _make_seam_cache_key(frame_keys, _k, _seam_cost_flags)
        if _ck is not None and seam_path_cache is not None and _ck in seam_path_cache:
            _precomp_paths[_k] = seam_path_cache[_ck]
            continue
        _fi_a = int(order[_k])
        _fi_b = int(order[_k + 1])
        _f = int(feathers[_k])
        _y0 = max(0, int(_by) - _f)
        _y1 = min(H, int(_by) + _f + 1)
        if _y1 - _y0 < 4:
            continue
        _fa_z = warped_norm[_fi_a][_y0:_y1].copy()
        _fb_z = warped_norm[_fi_b][_y0:_y1].copy()
        _bg_a_z = warped_bg[_fi_a][_y0:_y1] if warped_bg[_fi_a] is not None else None
        _bg_b_z = warped_bg[_fi_b][_y0:_y1] if warped_bg[_fi_b] is not None else None
        _em_zone = [
            em[_y0:_y1]
            for em in (_eff_exclusion or [])
            if em is not None and em.shape[0] >= _y1
        ]
        _sem = _build_seam_cost_map(
            result[_y0:_y1].copy(),
            ((_bg_a_z.astype(np.uint8) * 255) if _bg_a_z is not None else None),
            ((_bg_b_z.astype(np.uint8) * 255) if _bg_b_z is not None else None),
            exclusion_masks=_em_zone or None,
        )
        _ov_wps_raw = (seam_overrides or {}).get(_k, {}).get("waypoints")
        _ov_wps = None
        if _ov_wps_raw:
            _ov_wps = [
                (int(x), int(y) - _y0)
                for x, y in _ov_wps_raw
                if 0 <= int(y) - _y0 < _y1 - _y0
            ]
        _seam_jobs.append((_k, _fa_z, _fb_z, _sem, W, _y1 - _y0, _ov_wps))
    return _seam_jobs


def _precompute_seam_paths(
    result: np.ndarray,
    boundaries: np.ndarray,
    order: np.ndarray,
    feathers: np.ndarray,
    warped_norm: list[np.ndarray],
    warped_bg: list[np.ndarray | None],
    frame_keys: tuple[str, ...] | None,
    seam_path_cache: dict | None,
    exclusion_masks: list[np.ndarray] | None,
    seam_overrides: dict | None,
    paint_mask: np.ndarray | None,
    H: int,
    W: int,
) -> tuple[dict, list[np.ndarray] | None]:
    def _seam_job(job_args):
        _k, _fa_z, _fb_z, _sem, _W, _zh, _wps = job_args
        _both = (_fa_z.max(axis=2) > 0) & (_fb_z.max(axis=2) > 0)
        if int(_both.sum()) > _zh * _W // 20:
            try:
                return _k, _seam_cut(_fa_z, _fb_z, sem_cost=_sem, waypoints=_wps)
            except Exception:
                pass
        return _k, np.full(_W, _zh // 2, dtype=np.int32)

    _eff_exclusion = list(exclusion_masks or [])
    if paint_mask is not None and paint_mask.shape[0] == H and paint_mask.shape[1] == W:
        _eff_exclusion.append(paint_mask)
    _eff_exclusion = _eff_exclusion or None

    _seam_cost_flags = _get_seam_cost_flags()
    _precomp_paths: dict = {}
    _seam_jobs = _prepare_seam_jobs(
        boundaries, order, feathers, warped_norm, warped_bg, frame_keys,
        seam_path_cache, seam_overrides, _eff_exclusion, _seam_cost_flags,
        result, H, W, _precomp_paths
    )

    if len(_seam_jobs) > 1:
        _pool = _get_seam_pool()
        for _k, _path in _pool.map(_seam_job, _seam_jobs):
            _precomp_paths[_k] = _path
    elif _seam_jobs:
        _k, _path = _seam_job(_seam_jobs[0])
        _precomp_paths[_k] = _path

    if frame_keys is not None and seam_path_cache is not None:
        for _k, _path in _precomp_paths.items():
            _ck = _make_seam_cache_key(frame_keys, _k, _seam_cost_flags)
            if _ck not in seam_path_cache:
                seam_path_cache[_ck] = _path

    return _precomp_paths, _eff_exclusion


def _get_or_compute_path_local(
    k: int,
    y0_f: int,
    y1_f: int,
    zone_h: int,
    W: int,
    fa_zone: np.ndarray,
    fb_zone: np.ndarray,
    bg_a_zone: np.ndarray | None,
    bg_b_zone: np.ndarray | None,
    result_zone: np.ndarray,
    _precomp_paths: dict,
    _eff_exclusion: list[np.ndarray] | None,
    seam_overrides: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    path_local = _precomp_paths.get(k)
    _em_zone_fb = [
        em[y0_f:y1_f]
        for em in (_eff_exclusion or [])
        if em is not None and em.shape[0] >= y1_f
    ]
    _sem_cost = _build_seam_cost_map(
        result_zone,
        (
            (bg_a_zone.astype(np.uint8) * 255)
            if bg_a_zone is not None
            else None
        ),
        (
            (bg_b_zone.astype(np.uint8) * 255)
            if bg_b_zone is not None
            else None
        ),
        exclusion_masks=_em_zone_fb or None,
    )
    if path_local is None:
        both = (fa_zone.max(axis=2) > 0) & (fb_zone.max(axis=2) > 0)
        if int(both.sum()) > zone_h * W // 20:
            try:
                _fb_wps_raw = (seam_overrides or {}).get(k, {}).get("waypoints")
                _fb_wps = None
                if _fb_wps_raw:
                    _fb_wps = [
                        (int(_wx), int(_wy) - y0_f)
                        for _wx, _wy in _fb_wps_raw
                        if 0 <= int(_wy) - y0_f < zone_h
                    ]
                path_local = _seam_cut(
                    fa_zone, fb_zone, sem_cost=_sem_cost, waypoints=_fb_wps
                )
            except Exception:
                path_local = np.full(W, zone_h // 2, dtype=np.int32)
        else:
            path_local = np.full(W, zone_h // 2, dtype=np.int32)

    return path_local, _sem_cost


__all__ = [
    "_get_seam_cost_flags",
    "_make_seam_cache_key",
    "_extract_seam_crops",
    "_prepare_seam_jobs",
    "_precompute_seam_paths",
    "_get_or_compute_path_local",
]
