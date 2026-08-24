"""
Pose-consistent smart frame selection for the Anime Stitch Pipeline (§5.17 split).

Problem
-------
Pan-shot anime contains two superimposed motions:
  T_camera  — rigid background translation (the camera panning)
  A_animation — non-rigid character articulation

Selecting frames based solely on camera displacement picks arbitrary animation
phases.  When consecutive selected frames show the character in different poses,
the seam-registration stage (Stage 8.5) must warp one pose toward the other.
The warp is only approximate, leaving residual edge-doubling ("ghost") artifacts.
The SSIM ceiling (test09: 0.787, test27: 0.709) is caused by this animation
timing mismatch, not by compositing quality.

Solution (§6.1 of the Upgrade Research report)
----------------------------------------------
Anime is animated "on twos" or "on threes" — the same character cel is held for
2–3 consecutive video frames.  Within these runs, the background has advanced by
the inter-frame camera step while the character pose is identical.  By detecting
these runs and selecting frames that match the previous anchor pose, we assemble
a panorama from geometrically coherent inputs before rendering begins.

The pose similarity metric is the mean absolute pixel difference on the
foreground region (BiRefNet-masked, per-frame gain-normalised).  Background
pixels are hard-thresholded out so camera-panning background structure
contributes nothing to the score.  Falls back to gradient-magnitude L1 on
the central 50% crop when BiRefNet masks are unavailable.

Algorithm
---------
1. Load all frames at thumbnail scale (no GPU, I/O-bound).
2. Phase-correlate consecutive thumbnails → cumulative canvas positions.
3. Greedy forward-selection with a pose-consistent lookahead window:
   - Accumulate candidates within [min_step_px, min_step_px + pose_window_px]
     of the last selected frame.
   - From the window, pick the candidate with the lowest central-crop L1
     distance to the last selected thumbnail (most pose-similar).
4. Always include the first and last frames.

Configuration
-------------
``ASP_POSE_WINDOW_PX`` (env, default "80") — window width in canvas pixels.
Set to "0" to revert to v1 first-past-threshold behaviour.
``ASP_TWO_CHANNEL_SELECT`` (env, default "0") — enable BiRefNet background mask
for camera displacement estimation (currently disabled; see note below).

Split by role
-------------
  _native.py           -- shared base.frame_selection C++ extension probe
  _hold_detection.py   -- "on twos/threes" hold-block detection (MAD/dHash)
  _quality_filters.py  -- temporal-variance/blur/contrast/near-dup filters
                          plus default-off shot-boundary pre-filter
  _thumbs.py             -- thumbnail I/O + per-pair Otsu bg masking
  _pose.py                -- fg-masked pixel L1 + DINOv2 pose similarity
  _hold_average.py     -- ECC-aligned hold-block sub-pixel averaging
  _biref_probes.py     -- BiRefNet probe masks (smart_select_frames step 2)
  _phase_correlate.py  -- pairwise phase correlation (step 3)
  _pose_refine.py       -- Pass 2 pose-consistent refinement (step 7)
  selector.py            -- smart_select_frames(), the public entry point
  phases.py               -- animation-phase clustering (§2.2, downstream of
                            selection, not part of smart_select_frames itself)
"""

from ._hold_average import _hold_block_average
from ._hold_detection import (
    _compute_dhash,
    _detect_hold_blocks,
    _detect_hold_blocks_dhash,
    _drop_exact_dhash_duplicates,
    _estimate_background_plate,
    _refine_hold_ids_by_response,
    _select_hold_keyframes_dp,
    _separate_character_cels,
)
from ._pose import _DINOV2_CACHE, _compute_dinov2_features, _fg_center_diff
from ._quality_filters import (
    _filter_shot_boundaries,
    _near_dup_luma_filter,
    _reject_blurry_frames,
    _reject_low_contrast_frames,
    _temporal_variance_filter,
)
from ._thumbs import _load_thumbs_parallel, _otsu_bg_mask_pair
from .phases import _phase_ids_from_hashes, detect_animation_phases, phase_spans
from .selector import smart_select_frames

__all__ = [
    "smart_select_frames",
    "detect_animation_phases",
    "phase_spans",
    "_phase_ids_from_hashes",
    "_estimate_background_plate",
    "_separate_character_cels",
    "_detect_hold_blocks",
    "_select_hold_keyframes_dp",
    "_detect_hold_blocks_dhash",
    "_compute_dhash",
    "_drop_exact_dhash_duplicates",
    "_refine_hold_ids_by_response",
    "_temporal_variance_filter",
    "_reject_blurry_frames",
    "_reject_low_contrast_frames",
    "_near_dup_luma_filter",
    "_filter_shot_boundaries",
    "_compute_dinov2_features",
    "_fg_center_diff",
    "_load_thumbs_parallel",
    "_otsu_bg_mask_pair",
    "_hold_block_average",
    "_DINOV2_CACHE",
]
