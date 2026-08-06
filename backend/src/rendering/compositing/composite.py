"""``_composite_foreground`` -- the Stage 11 Laplacian-blend composite entry point."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ._audit import _adapt_feathers_and_synthesize, _audit_and_annotate_composite
from ._boundaries import _optimize_boundaries_and_feathers
from ._fg_pose import _register_foreground_poses
from ._fill import _initial_hard_partition_fill, _process_single_seam
from ._flags import _GLOBAL_GAIN_COMP, _JOINT_GAIN_SOLVE
from ._gain_compensation import _apply_joint_gain_solve, _equalize_warped_gains
from ._graphcut import _try_global_seam_composite
from ._normalization import _normalize_warped_frames, _warp_inputs
from ._seam_cache import _precompute_seam_paths


def _composite_foreground(
    warped_corr: List[np.ndarray],
    warped_fgs: List[np.ndarray],
    canvas: np.ndarray,
    H: int,
    W: int,
    frames: List[np.ndarray],
    affines: List[np.ndarray],
    bg_masks: List[Optional[np.ndarray]],
    frame_keys: Optional[Tuple[str, ...]] = None,
    seam_path_cache: Optional[Dict] = None,
    exclusion_masks: Optional[List[np.ndarray]] = None,
    preset_boundaries: Optional[np.ndarray] = None,
    paint_mask: Optional[np.ndarray] = None,
    seam_meta_out: Optional[dict] = None,
    seam_overrides: Optional[dict] = None,
    phase_ids: Optional[List[int]] = None,
) -> np.ndarray:
    N = len(frames)
    print("[Stitch]   Laplacian-blend composite (foreground-only deghost)...")

    # Scroll check
    tys = np.array([float(affines[i][1, 2]) for i in range(N)])
    txs = np.array([float(affines[i][0, 2]) for i in range(N)])
    ty_range = float(tys.max() - tys.min())
    tx_range = float(txs.max() - txs.min())
    if tx_range > 0 and ty_range / max(tx_range, 1.0) < 0.1:
        print(
            "[Stitch]   Horizontal scroll — temporal median is already optimal, skipping zone composite."
        )
        return canvas.copy()

    # Strip centres and ownership ordering
    strip_center_ys = np.array(
        [float(affines[i][1, 2]) + frames[i].shape[0] / 2.0 for i in range(N)],
        dtype=np.float64,
    )
    order = np.argsort(strip_center_ys)
    sorted_centers = strip_center_ys[order]
    initial_boundaries = (sorted_centers[:-1] + sorted_centers[1:]) / 2.0
    if preset_boundaries is not None and len(preset_boundaries) == N - 1:
        initial_boundaries = np.asarray(preset_boundaries, dtype=np.float64)

    # Warp inputs
    warped_list, warped_bg = _warp_inputs(frames, affines, bg_masks, H, W, N)

    # Normalise warped frames
    warped_norm, frame_gains = _normalize_warped_frames(
        canvas, warped_list, warped_bg, order, N, H, W
    )

    # Optimize boundary placement and feathers
    boundaries, feathers = _optimize_boundaries_and_feathers(
        warped_norm, order, initial_boundaries, bg_masks, affines, frames, frame_gains, warped_bg, H, W, N
    )

    # Foreground pose registration
    seam_single_pose, seam_post_diffs, seam_synthesized = _register_foreground_poses(
        warped_norm, warped_bg, order, boundaries, feathers, affines, frames, seam_overrides, ty_range, tx_range, H, W, N, phase_ids
    )

    # Adaptive feather refinement and canonical crop synthesis
    seam_canonical_crops = _adapt_feathers_and_synthesize(
        seam_post_diffs, seam_single_pose, seam_synthesized, feathers, boundaries, order, affines, frames, warped_norm, H, W
    )

    # Equalise inter-frame luminance before seam finding.
    if _GLOBAL_GAIN_COMP and len(warped_norm) >= 2:
        if _JOINT_GAIN_SOLVE:
            warped_norm = _apply_joint_gain_solve(warped_norm, warped_bg)
        else:
            warped_norm = _equalize_warped_gains(warped_norm, block_size=32)

    # Try global GraphCut / Canvas-space DP composite
    global_res = _try_global_seam_composite(
        warped_norm, warped_bg, canvas, H, W, N, boundaries,
        seam_post_diffs, seam_single_pose, seam_meta_out
    )
    if global_res is not None:
        return global_res

    # Hard-partition fill
    result = _initial_hard_partition_fill(
        canvas, warped_norm, warped_bg, order, boundaries, N, H
    )

    # Pre-compute seam DP paths
    _precomp_paths, _eff_exclusion = _precompute_seam_paths(
        result, boundaries, order, feathers, warped_norm, warped_bg,
        frame_keys, seam_path_cache, exclusion_masks, seam_overrides, paint_mask, H, W
    )

    # Laplacian blend at each boundary seam zone
    for k in range(len(boundaries)):
        by = boundaries[k]
        _process_single_seam(
            k, by, result, order, feathers, warped_norm, warped_bg,
            seam_single_pose, seam_post_diffs, seam_synthesized, seam_canonical_crops,
            seam_overrides, _precomp_paths, _eff_exclusion, seam_meta_out, H, W
        )

    # Post-composite audit and annotations
    result = _audit_and_annotate_composite(
        result, boundaries, order, feathers, warped_norm, _precomp_paths,
        seam_post_diffs, seam_single_pose, seam_meta_out
    )

    return result


__all__ = ["_composite_foreground"]
