"""Post-composite seam luminance audit, adaptive feather refinement, and
seam-metadata annotation."""

from __future__ import annotations

import numpy as np
from backend.src.constants import FEATHER_MAX, FEATHER_MIN

from ._fill import _fill_still_black_pixels
from ._flags import _POST_SEAM_WARN_THRESH
from ._seam_cache import _extract_seam_crops


def _audit_seam_lum_steps(
    result: np.ndarray,
    boundaries: np.ndarray,
    band_px: int = 5,
    warn_thresh: float = 8.0,
) -> dict[int, float]:
    """§1.106: Post-composite per-boundary luminance step audit (S152).

    For each boundary, measures mean absolute lum difference in a ±band_px
    row band around the boundary in *result*.  Logs a warning when any step
    exceeds *warn_thresh*.  Returns a dict {boundary_idx: lum_step}.
    """
    H = result.shape[0]
    steps: dict = {}
    for k, by_f in enumerate(boundaries):
        by = int(by_f)
        above_y0 = max(0, by - band_px)
        above_y1 = max(0, by)
        below_y0 = min(H, by)
        below_y1 = min(H, by + band_px)
        if above_y1 <= above_y0 or below_y1 <= below_y0:
            steps[k] = 0.0
            continue
        above_lum = float(
            result[above_y0:above_y1]
            .astype(np.float32)
            .dot(np.array([0.114, 0.587, 0.299], dtype=np.float32))
            .mean()
        )
        below_lum = float(
            result[below_y0:below_y1]
            .astype(np.float32)
            .dot(np.array([0.114, 0.587, 0.299], dtype=np.float32))
            .mean()
        )
        step = abs(above_lum - below_lum)
        steps[k] = step
        if step > warn_thresh:
            print(
                f"[Stitch] §1.106 seam-step WARNING: B{k} lum_step={step:.1f} "
                f"> {warn_thresh:.1f} at y={by}"
            )
    return steps


def _adapt_feathers_and_synthesize(
    seam_post_diffs: dict,
    seam_single_pose: dict,
    seam_synthesized: dict,
    feathers: np.ndarray,
    boundaries: np.ndarray,
    order: np.ndarray,
    affines: list[np.ndarray],
    frames: list[np.ndarray],
    warped_norm: list[np.ndarray],
    H: int,
    W: int,
) -> dict:
    _feather_adapted = False
    n_b = len(boundaries)
    for _k, _pdiff in seam_post_diffs.items():
        if _k in seam_single_pose:
            continue
        if _pdiff < 8.0:
            feathers[_k] = min(int(feathers[_k] * 1.5), FEATHER_MAX)
            _feather_adapted = True
        elif _pdiff > 16.0:
            feathers[_k] = max(int(feathers[_k] * 0.75), FEATHER_MIN)
            _feather_adapted = True

    if _feather_adapted:
        for _k in range(n_b):
            _fi_a = int(order[_k])
            _fi_b = int(order[_k + 1])
            _ty_a = float(affines[_fi_a][1, 2])
            _ty_b = float(affines[_fi_b][1, 2])
            _H_a = frames[_fi_a].shape[0]
            _H_b = frames[_fi_b].shape[0]
            _nat_ov = max(0, int(min(_ty_a + _H_a, _ty_b + _H_b) - max(_ty_a, _ty_b)))
            _max_f = max(5, min(_nat_ov // 2, FEATHER_MAX))
            if feathers[_k] > _max_f:
                feathers[_k] = _max_f
        print(
            "[Stitch]   Feathers (post-FG-reg adapted): "
            + " ".join(f"B{_k}={int(feathers[_k])}px" for _k in range(n_b))
        )

    seam_canonical_crops: dict = {}
    return seam_canonical_crops


def _audit_and_annotate_composite(
    result: np.ndarray,
    boundaries: np.ndarray,
    order: np.ndarray,
    feathers: np.ndarray,
    warped_norm: list[np.ndarray],
    _precomp_paths: dict,
    seam_post_diffs: dict,
    seam_single_pose: dict,
    seam_meta_out: dict | None,
) -> np.ndarray:
    _fill_still_black_pixels(result, warped_norm)

    _seam_lum_steps = _audit_seam_lum_steps(
        result, boundaries, band_px=5, warn_thresh=_POST_SEAM_WARN_THRESH
    )
    _max_step = max(_seam_lum_steps.values()) if _seam_lum_steps else 0.0

    # §2.4A/C: Populate seam metadata dict for HITL checkpoint 4.6.
    if seam_meta_out is not None:
        seam_meta_out.update(
            {
                "seam_lum_steps": _seam_lum_steps,
                "max_seam_lum_step": _max_step,
                "boundaries": (
                    boundaries.tolist()
                    if hasattr(boundaries, "tolist")
                    else list(boundaries)
                ),
                "seam_post_diffs": dict(seam_post_diffs),
                "seam_single_pose": dict(seam_single_pose),
                "seam_crops": _extract_seam_crops(result, boundaries),
            }
        )

    return result


__all__ = [
    "_audit_seam_lum_steps",
    "_adapt_feathers_and_synthesize",
    "_audit_and_annotate_composite",
]
