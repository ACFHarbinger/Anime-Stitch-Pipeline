"""``smart_select_frames`` — pose-consistent smart frame selection (§6.1).

See the package docstring (``frame_selection/__init__.py``) for the problem
statement and algorithm overview.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from ._biref_probes import _compute_biref_probe_masks
from ._hold_average import _hold_block_average
from ._hold_detection import (
    _DHASH_EXACT_DROP,
    _HIGH_HOLD_RESPONSE,
    _HOLD_DHASH_THRESHOLD,
    _HOLD_THRESHOLD,
    _MAX_SKIPPABLE_HOLD_SIZE,
    _detect_hold_blocks,
    _detect_hold_blocks_dhash,
    _drop_exact_dhash_duplicates,
    _refine_hold_ids_by_response,
)
from ._phase_correlate import _pairwise_phase_correlate
from ._pose_refine import _pass2_pose_refine
from ._quality_filters import (
    _BLUR_REJECT_THRESH,
    _CONTRAST_REJECT_THRESH,
    _NEAR_DUP_LUMA,
    _TEMPORAL_VAR_THRESH,
    _near_dup_luma_filter,
    _reject_blurry_frames,
    _reject_low_contrast_frames,
    _temporal_variance_filter,
)
from ._thumbs import _OTSU_BG_CORR, _SELECTOR_THUMB_LONG, _load_thumbs_parallel

# Pose-consistent refinement is disabled by default.  Session-5 upgraded the
# metric from gradient L1 (confounded by background scrolling) to fg-masked
# pixel L1 (background-invariant).  However GT-coupling still causes
# regressions on some tests: any frame substitution that diverges from the
# GT's temporal reference penalises SSIM.  Enable via ASP_POSE_WINDOW_PX=80
# for targeted experiments or when GT-SSIM is not the primary quality metric.
try:
    _POSE_WINDOW_PX = float(os.environ.get("ASP_POSE_WINDOW_PX", "0"))
except ValueError:
    _POSE_WINDOW_PX = 0.0

# §3.12A: Overmix hold-block sub-pixel averaging.
# After hold detection, align and stack-average all frames within each hold
# block using ECC (MOTION_TRANSLATION).  Produces one high-SNR representative
# per block; MPEG DCT compression noise cancels out by √N.
# Default OFF.  Enable with ASP_HOLD_AVERAGE=1.
_HOLD_AVERAGE: bool = os.environ.get("ASP_HOLD_AVERAGE", "0") != "0"

# §2.4: phase-aware frame selection. Pass 2 already penalises candidates that
# stay within the *same hold block* as the previous anchor (identical pose,
# zero ARAP benefit). This is the coarser, opposite-direction bias: penalise
# candidates that would jump to a *different animation phase* (a new pose/
# action loop, à la Overmix's AnimationSeparator) when a same-phase candidate
# is available, since cross-phase pairs are harder to align/composite than
# within-phase ones (§2.3 already refuses to midpoint-warp across a phase
# boundary at composite time — this reduces how often selection forces that
# case in the first place). Default OFF pending A/B; enable with
# ASP_PHASE_AWARE_SELECT=1.
_PHASE_AWARE_SELECT: bool = os.environ.get("ASP_PHASE_AWARE_SELECT", "0") != "0"
try:
    _PHASE_CROSS_PENALTY = float(os.environ.get("ASP_PHASE_CROSS_PENALTY", "0.05"))
except ValueError:
    _PHASE_CROSS_PENALTY = 0.05


def smart_select_frames(  # noqa: C901
    frames_paths: List[str],
    min_step_px: float = 25.0,
    min_phase_response: float = 0.04,
    high_anim_mad: float = 0.10,
    tiny_step_px: float = 8.0,
    pose_window_px: Optional[float] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Return a pose-consistent subset of ``frames_paths`` for the stitch pipeline.

    Parameters
    ----------
    frames_paths :
        Sorted list of input frame paths (any order is accepted; the function
        determines the dominant scroll direction from the phase-correlation data).
    min_step_px :
        Minimum camera displacement (full-resolution canvas pixels) between
        consecutive selected frames.  Default 25px.
    min_phase_response :
        Phase-correlation quality threshold.  Pairs below this are rejected
        (motion blur, scene cut, unreliable displacement estimate).
    high_anim_mad :
        MAD threshold for the high-animation / low-movement gate.  Frames
        where the camera barely moved but the thumbnail changed a lot (character
        is animating in place) are discarded.
    tiny_step_px :
        Camera movement threshold below which the high-animation gate is active.
    pose_window_px :
        Width of the pose-consistency lookahead window (canvas pixels).  Defaults
        to the ``ASP_POSE_WINDOW_PX`` env var (80px).  Set to 0 to revert to
        first-past-threshold (v1) behaviour.
    verbose :
        Print diagnostic messages.

    Returns
    -------
    List[str]
        Subset of ``frames_paths`` with near-duplicates, backward-direction
        frames, and pose-inconsistent frames removed.
    """
    N = len(frames_paths)
    if N <= 2:
        return frames_paths

    pw = _POSE_WINDOW_PX if pose_window_px is None else pose_window_px

    # ── 1. Load thumbnails ─────────────────────────────────────────────────
    thumbs = _load_thumbs_parallel(frames_paths)

    # ── 0. §1.64: Exact-duplicate dHash guard ─────────────────────────────
    # Drop consecutive frames whose 64-bit dHash is bit-for-bit identical —
    # MPEG still frames that survived INTER_AREA downscale unchanged.  Runs
    # before all heavier filters so they operate on a deduplicated sequence.
    if _DHASH_EXACT_DROP and N > 2:
        thumbs, frames_paths, _n_dedup_drop = _drop_exact_dhash_duplicates(
            thumbs, frames_paths
        )
        N = len(frames_paths)
        if verbose and _n_dedup_drop > 0:
            print(
                f"  [ExactDedup] §1.64: dropped {_n_dedup_drop} exact-duplicate "
                f"frames → {N} remain"
            )

    # ── 1a. §1.2D: Temporal variance pre-filter ───────────────────────────
    # Drop interior frames whose mean per-pixel variance across the (i-1,i,i+1)
    # triplet is below _TEMPORAL_VAR_THRESH.  Zero camera motion AND zero
    # character animation → frame carries no new canvas information.
    if _TEMPORAL_VAR_THRESH > 0.0 and N > 2:
        thumbs, frames_paths, _n_tvf_drop = _temporal_variance_filter(
            thumbs, frames_paths, _TEMPORAL_VAR_THRESH
        )
        N = len(frames_paths)
        if verbose and _n_tvf_drop > 0:
            print(
                f"  [TemporalVar] Dropped {_n_tvf_drop} static frames "
                f"(thresh={_TEMPORAL_VAR_THRESH:.4f}) → {N} remain"
            )

    # ── 1a-b. §1.2E: Blur/artifact frame pre-rejection ────────────────────
    if _BLUR_REJECT_THRESH > 0.0 and N > 2:
        thumbs, frames_paths, _n_blur_drop = _reject_blurry_frames(
            thumbs, frames_paths, _BLUR_REJECT_THRESH
        )
        N = len(frames_paths)
        if verbose and _n_blur_drop > 0:
            print(
                f"  [BlurReject] Dropped {_n_blur_drop} blurry frames "
                f"(thresh={_BLUR_REJECT_THRESH:.1f}) → {N} remain"
            )

    # ── 1b-a. §1.46: Low-contrast frame pre-rejection ────────────────────
    if _CONTRAST_REJECT_THRESH > 0.0 and N > 2:
        thumbs, frames_paths, _n_contrast_drop = _reject_low_contrast_frames(
            thumbs, frames_paths, _CONTRAST_REJECT_THRESH
        )
        N = len(frames_paths)
        if verbose and _n_contrast_drop > 0:
            print(
                f"  [ContrastReject] Dropped {_n_contrast_drop} low-contrast frames "
                f"(thresh={_CONTRAST_REJECT_THRESH:.1f}) → {N} remain"
            )

    # ── 1b. Hold-block detection (FD-Means preprocessing, §1.11 / §3.4) ───
    # Detect animation "on twos / on threes" hold blocks.  Each block
    # represents one unique character cel.  We record hold_ids[i] so that
    # Pass 2 can prefer candidates from a new hold block (different pose)
    # over candidates within the same hold block (identical pose, zero ARAP
    # benefit).  Hold detection also surfaces the block boundary count as
    # a diagnostic for predicted ARAP workload.
    hold_ids: List[int] = [0] * N  # hold block ID for each frame (0-indexed)
    n_hold_blocks = 1
    _use_dhash_hold = _HOLD_DHASH_THRESHOLD > 0
    if _use_dhash_hold or _HOLD_THRESHOLD > 0.0:
        if _use_dhash_hold:
            hold_reps = _detect_hold_blocks_dhash(
                thumbs, distance_threshold=_HOLD_DHASH_THRESHOLD
            )
            _hd_label = f"dHash(d≤{_HOLD_DHASH_THRESHOLD})"
        else:
            hold_reps = _detect_hold_blocks(thumbs, hold_threshold=_HOLD_THRESHOLD)
            _hd_label = f"MAD(t={_HOLD_THRESHOLD:.3f})"
        _block_id = 0
        _rep_set = set(hold_reps)
        for i in range(N):
            if i in _rep_set and i > 0:
                _block_id += 1
            hold_ids[i] = _block_id
        n_hold_blocks = _block_id + 1
        if verbose:
            print(
                f"  [HoldDetect/{_hd_label}] {n_hold_blocks} hold blocks from {N} frames "
                f"(avg {N / n_hold_blocks:.1f} frames/block)"
            )

        # Real animation holds are 2-6 frames. A block far larger than that is
        # a detector false positive (e.g. a slow scroll whose per-frame MAD
        # never trips the threshold) — treat each of its frames as its own
        # block rather than one "hold", or hold-averaging would blur dozens
        # of distinct poses together and the phase-correlation skip below
        # would zero out real camera motion across the whole span.
        _sizes: Dict[int, int] = {}
        for hid in hold_ids:
            _sizes[hid] = _sizes.get(hid, 0) + 1
        if any(sz > _MAX_SKIPPABLE_HOLD_SIZE for sz in _sizes.values()):
            _next_id = n_hold_blocks
            _capped_ids: List[int] = []
            for hid in hold_ids:
                if _sizes[hid] > _MAX_SKIPPABLE_HOLD_SIZE:
                    _capped_ids.append(_next_id)
                    _next_id += 1
                else:
                    _capped_ids.append(hid)
            hold_ids = _capped_ids
            n_hold_blocks = len(set(hold_ids))
            if verbose:
                print(
                    f"  [HoldDetect] {sum(1 for sz in _sizes.values() if sz > _MAX_SKIPPABLE_HOLD_SIZE)} "
                    f"oversized block(s) (>{_MAX_SKIPPABLE_HOLD_SIZE} frames) split back to "
                    f"singletons — likely detector false positives, not real holds."
                )

    # ── 1b-c. §3.12A: Hold-block sub-pixel averaging ──────────────────────
    # Runs immediately after hold detection, before phase correlation, so
    # every downstream array (raw_dx/dy, responses, frame_mads) is sized to
    # the post-averaging frame count rather than the original one.
    if _HOLD_AVERAGE and _HOLD_THRESHOLD > 0.0 and n_hold_blocks < N:
        thumbs, frames_paths = _hold_block_average(thumbs, hold_ids, frames_paths)
        N = len(thumbs)
        hold_ids = list(range(N))
        n_hold_blocks = N
        if verbose:
            print(f"  [HoldAverage] compressed to {N} hold-averaged frames")

    img0 = cv2.imread(frames_paths[0])
    if img0 is not None:
        full_h, full_w = img0.shape[:2]
        th0, tw0 = thumbs[0].shape[:2]
        scale_y = full_h / max(th0, 1)
        scale_x = full_w / max(tw0, 1)
    else:
        scale_y = scale_x = float(_SELECTOR_THUMB_LONG)

    # ── 1c. DINOv2 Features (Pass 2) ───────────────────────────────────────
    dinov2_features = None
    if pw > 0:
        if verbose:
            print("  [SmartSelect] Computing DINOv2 pose features...")
        from ._pose import _compute_dinov2_features

        dinov2_features = _compute_dinov2_features(frames_paths)
        if dinov2_features is not None and verbose:
            print("  [SmartSelect] DINOv2 features loaded successfully.")

    # ── 2. BiRefNet probe masks for camera displacement and pose similarity ──
    bg_thumb_mask, fg_thumb_mask = _compute_biref_probe_masks(
        N, thumbs, frames_paths, pw, dinov2_features, verbose
    )

    # ── 3. Pairwise phase-correlation ──────────────────────────────────────
    raw_dx, raw_dy, responses, frame_mads = _pairwise_phase_correlate(
        N, thumbs, hold_ids, _HOLD_THRESHOLD, _OTSU_BG_CORR, bg_thumb_mask, scale_x, scale_y
    )

    # ── 3b. §1.11C: Response-based hold refinement ────────────────────────
    # Pairs where phaseCorrelate response >= _HIGH_HOLD_RESPONSE are near-
    # identical frames (same cel) that MAD-based detection split due to MPEG
    # noise.  Merge their hold blocks now so Pass 2 treats them as one pose.
    if _HOLD_THRESHOLD > 0.0 and _HIGH_HOLD_RESPONSE > 0.0:
        hold_ids, n_hold_blocks = _refine_hold_ids_by_response(
            hold_ids, responses, _HIGH_HOLD_RESPONSE
        )
        if verbose:
            print(
                f"  [HoldRefine] {n_hold_blocks} hold blocks after response refinement"
            )

    # ── 4. Dominant scroll axis ────────────────────────────────────────────
    med_dy = float(np.median(raw_dy))
    med_dx = float(np.median(raw_dx))
    if abs(med_dy) >= abs(med_dx):
        axis_steps = raw_dy
        dominant_sign = int(np.sign(med_dy)) if abs(med_dy) > 2.0 else 0
    else:
        axis_steps = raw_dx
        dominant_sign = int(np.sign(med_dx)) if abs(med_dx) > 2.0 else 0

    _chan = "2ch" if bg_thumb_mask is not None else "1ch"
    if verbose:
        _hold_info = f"  hold_blocks={n_hold_blocks}" if _HOLD_THRESHOLD > 0.0 else ""
        print(
            f"  [SmartSelect] N={N}  axis={'y' if abs(med_dy) >= abs(med_dx) else 'x'}"
            f"  sign={dominant_sign:+d}"
            f"  med_step={abs(med_dy if abs(med_dy) >= abs(med_dx) else med_dx):.1f}px"
            f"  mode={_chan}  pose_window={pw:.0f}px{_hold_info}"
        )

    # ── 5. Pre-compute cumulative canvas positions ─────────────────────────
    cumpos: List[float] = [0.0] * N
    for i in range(N - 1):
        step = axis_steps[i]
        rejected = responses[i] < min_phase_response or (
            abs(step) < tiny_step_px and frame_mads[i] > high_anim_mad
        )
        cumpos[i + 1] = cumpos[i] + (0.0 if rejected else step)

    # ── 6. Pass 1 — v1 greedy selection (first-past-threshold) ───────────────
    selected_v1: List[int] = [0]
    last_pos_v1: float = 0.0

    for i in range(1, N):
        adv = cumpos[i] - last_pos_v1
        nf = adv * dominant_sign if dominant_sign != 0 else abs(adv)
        if nf >= min_step_px:
            selected_v1.append(i)
            last_pos_v1 = cumpos[i]

    if selected_v1[-1] != N - 1:
        selected_v1.append(N - 1)

    # ── 7. Pass 2 — pose-consistent local refinement ──────────────────────
    selected = _pass2_pose_refine(
        selected_v1,
        N,
        thumbs,
        dinov2_features,
        fg_thumb_mask,
        cumpos,
        dominant_sign,
        min_step_px,
        hold_ids,
        _HOLD_THRESHOLD,
        pw,
        _PHASE_AWARE_SELECT,
        _PHASE_CROSS_PENALTY,
        verbose,
    )

    if verbose:
        print(
            f"  [SmartSelect] Selected {len(selected)}/{N} frames"
            f"  (dropped {N - len(selected)})."
        )

    # ── 8. §1.2B Near-duplicate luma post-filter ───────────────────────────
    # Drop consecutive selected frames whose mean grayscale diff is below
    # _NEAR_DUP_LUMA (default 0.0 = disabled).  Thumbnail-scale check is
    # sufficient — a 5-luma unit diff at thumbnail scale reliably separates
    # genuine content advance from camera-barely-moved redundancy.
    _sel_paths = [frames_paths[i] for i in selected]
    if _NEAR_DUP_LUMA > 0.0 and len(_sel_paths) > 2:
        _sel_thumbs = [thumbs[i] for i in selected]
        _sel_paths_filt = _near_dup_luma_filter(
            _sel_thumbs, _sel_paths, threshold=_NEAR_DUP_LUMA
        )
        if verbose and len(_sel_paths_filt) < len(_sel_paths):
            print(
                f"  [NearDup] §1.2B: {len(_sel_paths) - len(_sel_paths_filt)} "
                f"near-dup frame(s) dropped (threshold={_NEAR_DUP_LUMA:.1f} luma)."
            )
        _sel_paths = _sel_paths_filt

    return _sel_paths


__all__ = ["smart_select_frames"]
