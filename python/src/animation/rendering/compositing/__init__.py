"""
Laplacian-pyramid composite for animated vertical-scroll stitches (§5.17 split).

For each canvas row the frame whose strip-centre is nearest supplies all
pixels (hard partition).  At ownership boundaries a Laplacian pyramid blend
routes the seam through flat cel-shaded regions via a DP energy path.
No photometric gain corrections are applied — the temporal median (Stage 9)
already ghost-removes frames and the pyramid blend is immune to the
brightness-banding artefacts produced by the earlier per-boundary LS gain
machinery.

Optimal boundary placement: for each adjacent frame pair the boundary is
moved (within ±SEARCH_RANGE rows of the midpoint) to the y-position where the
two warped frames are most photometrically similar, guided by background pixels
when available.

Adaptive feathering: the Laplacian blend half-width is scaled from the
photometric similarity score at the boundary, then capped by the natural
overlap between the two adjacent frames.

Split by role. `_composite_foreground` (composite.py) is already a thin
orchestrator over these pieces in the pre-split file -- this split mostly
just relocates each already-separate helper to its own topic file:
  _native.py            -- shared base.compositing/base.seam C++ extension probe
  _flags.py              -- env-var flags + shared seam ThreadPoolExecutor
  _seam_cut.py            -- DP/C++ seam-cut path finding
  _seam_cost.py           -- per-pixel seam cost map + soft blend weight
  _gain_compensation.py  -- per-block/joint-solve luminance gain matching
  _boundaries.py          -- ownership boundary placement + feather widths
  _normalization.py       -- canvas warp + bg-referenced luminance normalisation
  _fg_pose.py              -- foreground pose re-registration (Stage 8.5 seam use)
  _seam_cache.py           -- seam-path caching + parallel precompute
  _graphcut.py             -- GraphCut global multi-image seam (§4.2)
  _fill.py                 -- per-seam blend/single-pose fill
  _audit.py                -- post-composite seam-step audit + metadata
  composite.py             -- _composite_foreground, the public entry point
"""

from ._audit import (
    _adapt_feathers_and_synthesize,
    _audit_and_annotate_composite,
    _audit_seam_lum_steps,
)
from ._boundaries import (
    _compute_initial_boundaries,
    _diff_to_feather,
    _dominant_frame_in_band,
    _find_optimal_boundaries,
    _gain_to_min_feather,
    _optimize_boundaries_and_feathers,
)
from ._fg_pose import (
    _apply_foreground_registration,
    _check_preemptive_escalations,
    _register_foreground_poses,
    _seam_color_match,
    _single_pose_soft_edge,
)
from ._fill import (
    _blend_or_single_pose_fill,
    _fill_single_pose,
    _fill_still_black_pixels,
    _initial_hard_partition_fill,
    _process_single_seam,
)
from ._flags import (
    _BG_NORM_MIN_PX,
    _BLOCKS_GAIN_COMP,
    _BLOCKS_LUM_COMP,
    _FG_REGISTER_ENABLED,
    _GC_FEATHER_PX,
    _GLOBAL_GAIN_COMP,
    _GRAPHCUT_SEAM,
    _POST_SEAM_WARN_THRESH,
    _SEAM_POOL,
    _get_seam_pool,
)
from ._gain_compensation import (
    _adaptive_gain_clamp,
    _apply_joint_gain_solve,
    _bg_gain_unclamped,
    _blocks_gain_compensate,
    _blocks_lum_compensate,
    _equalize_warped_gains,
    _joint_gain_solve,
)
from ._graphcut import (
    _execute_graphcut_composite,
    _feather_gc_boundaries,
    _try_global_seam_composite,
)
from ._native import BATCH_AVAILABLE, batch
from ._normalization import (
    _coherence_skip_mask,
    _compute_frame_lums,
    _compute_skip_normalization_mask,
    _has_sufficient_bg,
    _normalize_single_frame,
    _normalize_warped_frames,
    _warp_inputs,
)
from ._seam_cache import (
    _extract_seam_crops,
    _get_or_compute_path_local,
    _get_seam_cost_flags,
    _make_seam_cache_key,
    _precompute_seam_paths,
    _prepare_seam_jobs,
)
from ._seam_cost import _apply_exclusion_masks, _build_seam_cost_map, _soft_seam_weight
from ._seam_cut import (
    _compute_seam_energy,
    _prepare_waypoints,
    _seam_cut,
    _seam_cut_batch,
    _seam_cut_python,
)
from .composite import _composite_foreground

__all__ = [
    "_BG_NORM_MIN_PX",
    "_BLOCKS_GAIN_COMP",
    "_BLOCKS_LUM_COMP",
    "_FG_REGISTER_ENABLED",
    "_GC_FEATHER_PX",
    "_GLOBAL_GAIN_COMP",
    "_GRAPHCUT_SEAM",
    "_POST_SEAM_WARN_THRESH",
    "_SEAM_POOL",
    "_adapt_feathers_and_synthesize",
    "_adaptive_gain_clamp",
    "_apply_exclusion_masks",
    "_apply_foreground_registration",
    "_audit_and_annotate_composite",
    "_audit_seam_lum_steps",
    "_bg_gain_unclamped",
    "_blend_or_single_pose_fill",
    "_blocks_gain_compensate",
    "_blocks_lum_compensate",
    "_build_seam_cost_map",
    "_check_preemptive_escalations",
    "_coherence_skip_mask",
    "_composite_foreground",
    "_compute_frame_lums",
    "_compute_initial_boundaries",
    "_compute_seam_energy",
    "_compute_skip_normalization_mask",
    "_diff_to_feather",
    "_equalize_warped_gains",
    "_execute_graphcut_composite",
    "_extract_seam_crops",
    "_feather_gc_boundaries",
    "_fill_single_pose",
    "_fill_still_black_pixels",
    "_find_optimal_boundaries",
    "_gain_to_min_feather",
    "_get_or_compute_path_local",
    "_get_seam_cost_flags",
    "_get_seam_pool",
    "_has_sufficient_bg",
    "_initial_hard_partition_fill",
    "_make_seam_cache_key",
    "_normalize_single_frame",
    "_normalize_warped_frames",
    "_optimize_boundaries_and_feathers",
    "_precompute_seam_paths",
    "_prepare_seam_jobs",
    "_prepare_waypoints",
    "_process_single_seam",
    "_register_foreground_poses",
    "_seam_color_match",
    "_seam_cut",
    "_seam_cut_batch",
    "_seam_cut_python",
    "_single_pose_soft_edge",
    "_soft_seam_weight",
    "_try_global_seam_composite",
    "_warp_inputs",
]
