"""Pass 2 pose-consistent local refinement for ``smart_select_frames``.

Extracted from ``smart_select_frames`` step 7 as a standalone, individually
testable function -- pure code motion, no logic change.

For each interior frame, checks whether a nearby frame (within +-2 slots,
subject to a minimum/maximum advance constraint) has >=10% better pose
similarity to the previous selected frame. Frame count is preserved.

Similarity metric priority:
  1. DINOv2 cosine distance (§3.3): loaded lazily; handles holds natively
     because identical-pose frames map to the same point in feature space.
  2. fg-masked pixel L1 (_fg_center_diff): falls back to this when DINOv2
     weights are unavailable or torch.hub cannot reach HuggingFace.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from ._hold_detection import _compute_dhash
from ._pose import _compute_dinov2_features, _fg_center_diff
from .phases import _phase_ids_from_hashes

# Exposed as ASP advanced options (see core/config.py _CONFIG_SCHEMA's
# "pose_refine" section).
try:
    _LOOK_RANGE = int(os.environ.get("ASP_POSE_REFINE_LOOK_RANGE", "2"))
except ValueError:
    _LOOK_RANGE = 2
try:
    _MIN_GAIN = float(os.environ.get("ASP_POSE_REFINE_MIN_GAIN", "0.10"))
except ValueError:
    _MIN_GAIN = 0.10
try:
    _MIN_ADV_FRAC = float(os.environ.get("ASP_POSE_REFINE_MIN_ADV_FRAC", "0.50"))
except ValueError:
    _MIN_ADV_FRAC = 0.50
try:
    _MAX_ADV_FRAC = float(os.environ.get("ASP_POSE_REFINE_MAX_ADV_FRAC", "2.50"))
except ValueError:
    _MAX_ADV_FRAC = 2.50
try:
    _SAME_HOLD_PENALTY = float(os.environ.get("ASP_POSE_REFINE_SAME_HOLD_PENALTY", "0.05"))
except ValueError:
    _SAME_HOLD_PENALTY = 0.05


def _pass2_pose_refine(  # noqa: C901
    selected_v1: List[int],
    N: int,
    thumbs: List[np.ndarray],
    dinov2_features: Optional[np.ndarray],
    fg_thumb_mask: Optional[np.ndarray],
    cumpos: List[float],
    dominant_sign: int,
    min_step_px: float,
    hold_ids: List[int],
    hold_threshold: float,
    pw: float,
    phase_aware_select: bool,
    phase_cross_penalty: float,
    verbose: bool,
) -> List[int]:
    """Refine ``selected_v1`` (Pass 1's greedy selection) toward pose-consistent
    neighbours. Returns ``selected_v1`` unchanged when Pass 2 is inactive
    (``pw <= 0``) or there are too few interior frames to refine.
    """
    if not (pw > 0 and len(selected_v1) > 2):
        return selected_v1

    # §2.4: candidate-level phase clustering, computed on this pass's own
    # thumbs (post pre-filters, post optional hold-averaging) so indices line
    # up with `candidates`/`s_prev` below. This is a proxy for the real §2.2
    # phase_ids (which run later, on the final selected set) — good enough
    # for a same-vs-different-phase tie-break signal at selection time.
    _cand_phase_ids: Optional[List[int]] = None
    if phase_aware_select:
        _cand_hashes = [_compute_dhash(t) for t in thumbs]
        _cand_phase_ids = _phase_ids_from_hashes(_cand_hashes)
        if verbose:
            print(
                f"  [PhaseSelect] {len(set(_cand_phase_ids))} candidate "
                f"phase(s) across {len(thumbs)} pre-selection frames."
            )

    # Try to compute DINOv2 features when Pass 2 is active.
    _dino_feats: Optional[np.ndarray] = _compute_dinov2_features(thumbs)
    if _dino_feats is not None and verbose:
        print(f"  [PoseSelect] DINOv2 features: {_dino_feats.shape} loaded.")
    elif verbose:
        print("  [PoseSelect] DINOv2 unavailable; using fg pixel L1.")

    def _pose_dist(i: int, j: int) -> float:
        """Pose dissimilarity between frame i and frame j (lower = more similar)."""
        if _dino_feats is not None:
            return 1.0 - float(np.dot(_dino_feats[i], _dino_feats[j]))
        return _fg_center_diff(thumbs[i], thumbs[j], fg_thumb_mask)

    refined: List[int] = [selected_v1[0]]
    n_subs = 0
    for k in range(1, len(selected_v1) - 1):
        s_prev = refined[-1]
        s_curr = selected_v1[k]
        lo = max(s_prev + 1, s_curr - _LOOK_RANGE)
        hi = min(N - 1, s_curr + _LOOK_RANGE)

        candidates = []
        for c in range(lo, hi + 1):
            adv = cumpos[c] - cumpos[s_prev]
            nf = adv * dominant_sign if dominant_sign != 0 else abs(adv)
            if _MIN_ADV_FRAC * min_step_px <= nf <= _MAX_ADV_FRAC * min_step_px:
                candidates.append(c)
        if not candidates:
            refined.append(s_curr)
            continue
        if dinov2_features is not None:
            last_t = dinov2_features[s_prev]
            curr_score = 1.0 - float(np.dot(last_t, dinov2_features[s_curr]))
            scores = [
                1.0 - float(np.dot(last_t, dinov2_features[c])) for c in candidates
            ]
        else:
            last_t = thumbs[s_prev]
            curr_score = _fg_center_diff(last_t, thumbs[s_curr], fg_thumb_mask)
            scores = [
                _fg_center_diff(last_t, thumbs[c], fg_thumb_mask)
                for c in candidates
            ]
        # Hold-block tie-breaking: candidates from the same hold block as
        # s_prev have pose identical to the previous anchor frame.  Their
        # pixel L1 is near-zero not because the pose is good but because
        # the character hasn't moved.  Apply a small penalty to prefer
        # cross-hold candidates.  (DINOv2 handles this naturally — same-hold
        # frames map to the same feature vector — so the penalty is a no-op
        # when DINOv2 is active, but kept for consistency.)
        scores_adj = [
            s
            + (
                _SAME_HOLD_PENALTY
                if hold_threshold > 0 and hold_ids[c] == hold_ids[s_prev]
                else 0.0
            )
            # §2.4: opposite-direction bias — penalise a candidate that
            # would cross into a different animation phase than s_prev
            # when a same-phase candidate is available, since cross-phase
            # pairs are harder to align/composite than within-phase ones.
            + (
                phase_cross_penalty
                if _cand_phase_ids is not None
                and _cand_phase_ids[c] != _cand_phase_ids[s_prev]
                else 0.0
            )
            for s, c in zip(scores, candidates, strict=False)
        ]
        best_local = int(np.argmin(scores_adj))
        best = candidates[best_local]
        best_score = scores[best_local]
        if best != s_curr and best_score < curr_score * (1.0 - _MIN_GAIN):
            refined.append(best)
            n_subs += 1
            if verbose:
                print(
                    f"  [PoseSelect] Slot {k}: {s_curr}→{best} "
                    f"(score {curr_score:.3f}→{best_score:.3f})"
                )
        else:
            refined.append(s_curr)
    refined.append(selected_v1[-1])
    if verbose and n_subs > 0:
        print(f"  [PoseSelect] {n_subs}/{len(selected_v1) - 2} slots refined.")
    return refined


__all__ = ["_pass2_pose_refine"]
