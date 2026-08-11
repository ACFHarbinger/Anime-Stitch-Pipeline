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

import numpy as np

from ._hold_detection import _compute_dhash
from ._pose import _fg_center_diff
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

# Global candidate-path selection is deliberately experimental.  The existing
# pass is greedy: a locally good replacement can make the next slot impossible
# or create a large pose jump later.  This flag enables a small dynamic program
# over the same bounded candidate windows without changing the default path.
_PATH_SELECT = os.environ.get("ASP_POSE_PATH_SELECT", "0") != "0"
_PATH_SAFE = os.environ.get("ASP_POSE_PATH_SAFE", "0") != "0"


def _path_structurally_safe(
    baseline: list[int],
    path: list[int],
    cumpos: list[float],
    dominant_sign: int,
    min_step_px: float,
    phase_ids: list[int] | None,
) -> bool:
    """Reject global paths that improve pose similarity by breaking structure."""
    if len(path) != len(baseline) or len(path) < 3:
        return False
    if path[0] != baseline[0] or path[-1] != baseline[-1]:
        return False
    if any(a >= b for a, b in zip(path, path[1:], strict=False)):
        return False

    advances = [
        (cumpos[b] - cumpos[a]) * dominant_sign
        if dominant_sign
        else abs(cumpos[b] - cumpos[a])
        for a, b in zip(path, path[1:], strict=False)
    ]
    positive = [advance for advance in advances if advance > 0]
    if not positive or min(positive) < max(1.0, min_step_px * 0.35):
        return False

    substitutions = sum(a != b for a, b in zip(baseline, path, strict=False))
    if substitutions > max(2, int(len(baseline) * 0.35)):
        return False

    if phase_ids is not None:
        baseline_crossings = sum(
            phase_ids[a] != phase_ids[b]
            for a, b in zip(baseline, baseline[1:], strict=False)
        )
        path_crossings = sum(
            phase_ids[a] != phase_ids[b]
            for a, b in zip(path, path[1:], strict=False)
        )
        if path_crossings > baseline_crossings + 1:
            return False
    return True


def _pose_distance(
    a: int,
    b: int,
    thumbs: list[np.ndarray],
    dinov2_features: np.ndarray | None,
    fg_thumb_mask: np.ndarray | None,
) -> float:
    """Return the comparable pose distance for two candidate frames."""
    if dinov2_features is not None:
        return max(0.0, 1.0 - float(np.dot(dinov2_features[a], dinov2_features[b])))
    return _fg_center_diff(thumbs[a], thumbs[b], fg_thumb_mask)


def _global_pose_path(
    selected_v1: list[int],
    N: int,
    thumbs: list[np.ndarray],
    dinov2_features: np.ndarray | None,
    fg_thumb_mask: np.ndarray | None,
    cumpos: list[float],
    dominant_sign: int,
    min_step_px: float,
    hold_ids: list[int],
    hold_threshold: float,
    phase_ids: list[int] | None,
    verbose: bool,
) -> list[int]:
    """Choose a globally coherent path through bounded frame candidates.

    Each Pass-1 slot keeps its original camera-progress neighbourhood.  The
    dynamic program minimizes pose discontinuity plus a small camera-progress
    penalty, so a locally attractive substitution cannot strand a later slot.
    Endpoints remain fixed, matching the existing selector contract.
    """
    if len(selected_v1) <= 2:
        return selected_v1

    layers: list[list[int]] = [[selected_v1[0]]]
    for slot in selected_v1[1:-1]:
        lo = max(1, slot - _LOOK_RANGE)
        hi = min(N - 2, slot + _LOOK_RANGE)
        candidates = list(range(lo, hi + 1))
        if not candidates:
            return selected_v1
        layers.append(candidates)
    layers.append([selected_v1[-1]])

    # The target step is the median Pass-1 camera advance.  It prevents the
    # pose objective from selecting nearly stationary frames just because they
    # look similar.
    advances = []
    for a, b in zip(selected_v1, selected_v1[1:], strict=False):
        step = (cumpos[b] - cumpos[a]) * dominant_sign if dominant_sign else abs(cumpos[b] - cumpos[a])
        if step > 0:
            advances.append(step)
    target_step = float(np.median(advances)) if advances else max(min_step_px, 1.0)
    min_advance = max(min_step_px * _MIN_ADV_FRAC, target_step * 0.35)
    max_advance = max(min_advance, target_step * _MAX_ADV_FRAC)
    progress_weight = 0.15

    costs: list[dict[int, float]] = [{layers[0][0]: 0.0}]
    parents: list[dict[int, int]] = [{}]
    for layer_idx in range(1, len(layers)):
        current_cost: dict[int, float] = {}
        current_parent: dict[int, int] = {}
        for c in layers[layer_idx]:
            best_cost = float("inf")
            best_parent: int | None = None
            for p, prior_cost in costs[-1].items():
                if c <= p:
                    continue
                advance = (cumpos[c] - cumpos[p]) * dominant_sign if dominant_sign else abs(cumpos[c] - cumpos[p])
                if advance < min_advance or advance > max_advance:
                    continue
                pose = _pose_distance(p, c, thumbs, dinov2_features, fg_thumb_mask)
                progress = abs(advance - target_step) / max(target_step, 1.0)
                penalty = 0.0
                if hold_threshold > 0 and hold_ids[p] == hold_ids[c]:
                    penalty += _SAME_HOLD_PENALTY
                if phase_ids is not None and phase_ids[p] != phase_ids[c]:
                    penalty += 0.05
                total = prior_cost + pose + progress_weight * progress + penalty
                if total < best_cost:
                    best_cost = total
                    best_parent = p
            if best_parent is not None:
                current_cost[c] = best_cost
                current_parent[c] = best_parent
        if not current_cost:
            if verbose:
                print(f"  [PosePath] no valid transitions at slot {layer_idx}; keeping greedy path")
            return selected_v1
        costs.append(current_cost)
        parents.append(current_parent)

    path = [layers[-1][0]]
    for layer_idx in range(len(layers) - 1, 0, -1):
        path.append(parents[layer_idx][path[-1]])
    path.reverse()
    if verbose:
        substitutions = sum(a != b for a, b in zip(selected_v1, path, strict=False))
        print(f"  [PosePath] global path selected ({substitutions} substitutions, {len(path)} frames)")
    return path


def _pass2_pose_refine(  # noqa: C901
    selected_v1: list[int],
    N: int,
    thumbs: list[np.ndarray],
    dinov2_features: np.ndarray | None,
    fg_thumb_mask: np.ndarray | None,
    cumpos: list[float],
    dominant_sign: int,
    min_step_px: float,
    hold_ids: list[int],
    hold_threshold: float,
    pw: float,
    phase_aware_select: bool,
    phase_cross_penalty: float,
    verbose: bool,
) -> list[int]:
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
    _cand_phase_ids: list[int] | None = None
    if phase_aware_select:
        _cand_hashes = [_compute_dhash(t) for t in thumbs]
        _cand_phase_ids = _phase_ids_from_hashes(_cand_hashes)
        if verbose:
            print(
                f"  [PhaseSelect] {len(set(_cand_phase_ids))} candidate "
                f"phase(s) across {len(thumbs)} pre-selection frames."
            )

    if _PATH_SELECT:
        path = _global_pose_path(
            selected_v1,
            N,
            thumbs,
            dinov2_features,
            fg_thumb_mask,
            cumpos,
            dominant_sign,
            min_step_px,
            hold_ids,
            hold_threshold,
            _cand_phase_ids,
            verbose,
        )
        # A failed global transition search returns the original Pass-1 path.
        # Continue into the established greedy refinement in that case; the
        # experimental path must never disable the safer existing behavior.
        if path != selected_v1 and (
            not _PATH_SAFE
            or _path_structurally_safe(
                selected_v1,
                path,
                cumpos,
                dominant_sign,
                min_step_px,
                _cand_phase_ids,
            )
        ):
            return path
        if path != selected_v1 and verbose and _PATH_SAFE:
            print("  [PosePath] structural-risk veto; keeping greedy path")

    if verbose:
        if dinov2_features is not None:
            print(f"  [PoseSelect] DINOv2 features: {dinov2_features.shape} loaded.")
        else:
            print("  [PoseSelect] DINOv2 unavailable; using fg pixel L1.")

    refined: list[int] = [selected_v1[0]]
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
            curr_score = _pose_distance(s_prev, s_curr, thumbs, dinov2_features, fg_thumb_mask)
            scores = [
                _pose_distance(s_prev, c, thumbs, dinov2_features, fg_thumb_mask)
                for c in candidates
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


__all__ = ["_pass2_pose_refine", "_global_pose_path", "_path_structurally_safe"]
