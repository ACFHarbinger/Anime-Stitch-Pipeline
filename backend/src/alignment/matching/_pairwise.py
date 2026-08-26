"""Fallback-chain orchestration: try each matcher strategy per frame pair,
then build the full pairwise correspondence-edge list for bundle adjustment.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
import torch
from backend.src.constants import MATCH_EDGE_CROP, MAX_DX_DRIFT_RATIO

from ._matchers import _phase_correlate, _segment_guided_match, _template_match
from ._math import _compute_bg_match_ratio, _compute_translation_spread, _extract_similarity
from ._sampling import _sample_bg_points_grid
from ._estimators import estimate_affine2d, estimate_affine_partial2d

logger = logging.getLogger(__name__)

# §1.3E — Similarity-mode flag.  When ON, matched affines are projected to
# their best-fit 4-DOF similarity (scale + rotation + translation, no shear)
# instead of being stripped to translation-only.  Useful for zoom-pan sequences
# where the camera both pans and zooms simultaneously (e.g. test5).
# Default OFF to preserve backward-compatible translation-only behaviour.
_SIMILARITY_MODE: bool = os.environ.get("ASP_SIMILARITY_MODE", "0") != "0"

# §1.36: LoFTR translation consensus spread filter.
# Rejects the LoFTR translation estimate when per-match displacements have high MAD
# (median absolute deviation) around the median — indicative of texture confusion
# between repeated background elements, or character motion polluting matches.
# Set to 0.0 to disable (default); recommend 30.0 for real sequences.
_MATCH_SPREAD_CEIL: float = float(os.environ.get("ASP_MATCH_SPREAD_CEIL", "0.0"))

# §1.38: LoFTR background match ratio gate.
# Rejects the LoFTR edge when background keypoints are too small a fraction of all
# LoFTR matches — indicates a foreground-dominated scene where the surviving bg
# keypoints are sparse and their median displacement is noisy.
# Set to 0.0 to disable (default); recommend 0.15 for real sequences.
_LOFTR_BG_RATIO_MIN: float = float(os.environ.get("ASP_LOFTR_BG_RATIO_MIN", "0.0"))

# M4: Background-masked matching. Feeds inverted BiRefNet masks (~fg_mask)
# directly into feature detectors/matchers to avoid cel motion contamination.
_BG_MASKED_MATCHING: bool = os.environ.get("ASP_BG_MASKED_MATCHING", "0") == "1"

# asp_test83 hung 1+ hour after "Loading weights: 100%" with no further
# logs: 18×1080p frames × (adj+skip1+skip2) pairs × LoFTR/LG/RoMa looks
# like a stall. Budget stops matching and keeps whatever edges exist so
# the pipeline can fall back to SCANS instead of spinning forever.
# 0 disables the budget.
_MATCH_BUDGET_SEC: float = float(os.environ.get("ASP_MATCH_BUDGET_SEC", "180"))


@dataclass(frozen=True)
class TemporalPairProposal:
    """One deterministic Stage 5–6 candidate pair and its selection reason."""

    i: int
    j: int
    span: int
    reason: str


def propose_temporal_pairs(N: int, range_width: int = 3) -> list[TemporalPairProposal]:
    """Return the deterministic temporal pair policy used before matching.

    ``range_width=3`` exactly preserves the historical ordering: all adjacent
    pairs, then all span-two pairs, then all span-three pairs.  The proposal
    is deliberately pure so P0 can measure it without changing matchers or
    graph filtering.  Adjacent pairs are explicitly marked as the connectivity
    backbone for later policy experiments.
    """
    if N < 0:
        raise ValueError("N must be non-negative")
    if range_width < 1:
        raise ValueError("range_width must be at least 1")

    proposals: list[TemporalPairProposal] = []
    for span in range(1, min(range_width, N - 1) + 1):
        reason = "adjacent_backbone" if span == 1 else "temporal_skip"
        proposals.extend(
            TemporalPairProposal(i=i, j=i + span, span=span, reason=reason)
            for i in range(N - span)
        )
    return proposals



def _ransac_metrics(
    points_i: np.ndarray | None, points_j: np.ndarray | None
) -> dict[str, float | int | bool | None]:
    """Measure RANSAC agreement for real matcher correspondences only."""
    if points_i is None or points_j is None or len(points_i) < 3:
        return {
            "observed_correspondences": False,
            "ransac_inlier_count": None,
            "ransac_inlier_ratio": None,
            "reprojection_rms": None,
        }
    matrix, inliers = estimate_affine_partial2d(points_i, points_j, ransacReprojThreshold=5.0)
    if matrix is None or inliers is None:
        return {
            "observed_correspondences": True,
            "ransac_inlier_count": 0,
            "ransac_inlier_ratio": 0.0,
            "reprojection_rms": None,
        }
    mask = inliers.ravel().astype(bool)
    count = int(mask.sum())
    if not count:
        return {
            "observed_correspondences": True,
            "ransac_inlier_count": 0,
            "ransac_inlier_ratio": 0.0,
            "reprojection_rms": None,
        }
    projected = points_i[mask] @ matrix[:, :2].T + matrix[:, 2]
    residuals = np.linalg.norm(projected - points_j[mask], axis=1)
    return {
        "observed_correspondences": True,
        "ransac_inlier_count": count,
        "ransac_inlier_ratio": round(count / len(points_i), 4),
        "reprojection_rms": round(float(np.sqrt(np.mean(np.square(residuals)))), 4),
    }


def _match_pair(  # noqa: C901
    frames: list[np.ndarray],
    bg_masks: list[np.ndarray | None],
    i: int,
    j: int,
    H: int,
    W: int,
    loftr_wrapper=None,
    use_loftr: bool = True,
    motion_model: str = "translation",
    aliked_wrapper=None,
    roma_wrapper=None,
    bg_masked_matching: bool = False,
) -> dict | None:
    """
    Try to match frame i to frame j. Optimized for vertical anime pans.
    """
    img_i, img_j = frames[i], frames[j]
    m_i = bg_masks[i]
    m_j = bg_masks[j]

    # ── Pre-match Edge Crop (Discard distortion) ──
    ec_h = int(H * MATCH_EDGE_CROP)
    ec_w = int(W * MATCH_EDGE_CROP)

    match_img_i = img_i[ec_h:-ec_h, ec_w:-ec_w]
    match_img_j = img_j[ec_h:-ec_h, ec_w:-ec_w]
    match_m_i = m_i[ec_h:-ec_h, ec_w:-ec_w] if m_i is not None else None
    match_m_j = m_j[ec_h:-ec_h, ec_w:-ec_w] if m_j is not None else None

    # M4: Background-masked matching restricts input to static pixels if enabled
    match_in_i = match_img_i
    match_in_j = match_img_j
    if bg_masked_matching or _BG_MASKED_MATCHING:
        if match_m_i is not None and (match_m_i <= 127).any():
            match_in_i = match_img_i.copy()
            match_in_i[match_m_i <= 127] = 0
        if match_m_j is not None and (match_m_j <= 127).any():
            match_in_j = match_img_j.copy()
            match_in_j[match_m_j <= 127] = 0

    def _is_valid(M):
        if M is None:
            return False
        dx = abs(M[0, 2])
        return not dx > W * MAX_DX_DRIFT_RATIO

    M: np.ndarray | None = None
    mean_conf = 0.0
    actual_pts_i: np.ndarray | None = None
    actual_pts_j: np.ndarray | None = None
    observed_pts_i: np.ndarray | None = None
    observed_pts_j: np.ndarray | None = None
    _loftr_bg_pts: int = 0  # track how many BG keypoints LoFTR found (for 1b trigger)

    # ── Attempt 1: LoFTR ───────────────────────────────────────────────────
    if use_loftr and loftr_wrapper is not None:
        _matcher_name = type(loftr_wrapper).__name__
        _match_started = time.perf_counter()
        logger.info("[Stitch]   %d→%d: %s matching started.", i, j, _matcher_name)
        try:
            pts1, pts2, conf = loftr_wrapper.match(match_in_i, match_in_j)
            print(
                f"[Stitch]   {i}→{j}: {_matcher_name} matching finished "
                f"in {time.perf_counter() - _match_started:.1f}s ({len(pts1)} points).",
                flush=True,
            )

            logger.info(
                "[Stitch]   %d→%d: %s matching finished (%d points).",
                i,
                j,
                _matcher_name,
                len(pts1),
            )
            if len(pts1) >= 30:
                n_loftr_total = len(pts1)  # capture before bg filtering (§1.38)
                if match_m_i is not None and match_m_j is not None:
                    y1, x1 = pts1[:, 1].astype(int), pts1[:, 0].astype(int)
                    y2, x2 = pts2[:, 1].astype(int), pts2[:, 0].astype(int)
                    h, w = match_m_i.shape[:2]
                    valid = (
                        (x1 >= 0)
                        & (x1 < w)
                        & (y1 >= 0)
                        & (y1 < h)
                        & (x2 >= 0)
                        & (x2 < w)
                        & (y2 >= 0)
                        & (y2 < h)
                    )
                    if valid.any():
                        m1_vals = match_m_i[y1[valid], x1[valid]]
                        m2_vals = match_m_j[y2[valid], x2[valid]]
                        bg_mask = (m1_vals > 127) & (m2_vals > 127)
                        indices = np.where(valid)[0][bg_mask]
                        pts1, pts2, conf = (
                            pts1[indices],
                            pts2[indices],
                            conf[indices],
                        )
                _loftr_bg_pts = len(pts1)
                # §1.38: Reject LoFTR edge when bg matches are a small fraction of
                # total matches — fg-dominated pairs produce noisy median displacement.
                if _LOFTR_BG_RATIO_MIN > 0.0:
                    _bg_ratio = _compute_bg_match_ratio(_loftr_bg_pts, n_loftr_total)
                    if _bg_ratio < _LOFTR_BG_RATIO_MIN:
                        logger.debug(
                            f"[Stitch]   {i}→{j}: LoFTR rejected "
                            f"(bg_ratio={_bg_ratio:.2f} < {_LOFTR_BG_RATIO_MIN:.2f}, "
                            f"bg_pts={_loftr_bg_pts}/{n_loftr_total})"
                        )
                        pts1 = np.empty((0, 2), np.float32)

                if len(pts1) >= 20:
                    if motion_model == "translation":
                        dxs = pts2[:, 0] - pts1[:, 0]
                        dys = pts2[:, 1] - pts1[:, 1]
                        dx, dy = np.median(dxs), np.median(dys)
                        M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
                        mean_conf = float(conf.mean())
                        # §1.36: Reject when per-match displacement spread is too high —
                        # high MAD means LoFTR matches disagree on the translation
                        # (foreground/background confusion, bimodal distribution).
                        if _MATCH_SPREAD_CEIL > 0.0:
                            _mad_dx, _mad_dy = _compute_translation_spread(pts1, pts2)
                            if max(_mad_dx, _mad_dy) > _MATCH_SPREAD_CEIL:
                                M = None
                                logger.debug(
                                    f"[Stitch]   {i}→{j}: LoFTR rejected "
                                    f"(spread mad_dx={_mad_dx:.1f} mad_dy={_mad_dy:.1f} "
                                    f"> {_MATCH_SPREAD_CEIL:.0f}px)"
                                )
                    else:
                        M_raw, inliers = estimate_affine2d(
                            pts1, pts2, ransacReprojThreshold=5.0
                        )
                        if _is_valid(M_raw):
                            inl = inliers.ravel().astype(bool)
                            if inl.sum() >= 15:
                                M, mean_conf = (
                                    M_raw.astype(np.float32),
                                    float(conf[inl].mean()),
                                )

                    if M is not None:
                        actual_pts_i = pts1 + [ec_w, ec_h]
                        actual_pts_j = pts2 + [ec_w, ec_h]
                        observed_pts_i = actual_pts_i
                        observed_pts_j = actual_pts_j
                        logger.debug(
                            f"[Stitch]   {i}→{j}: LoFTR dx={M[0, 2]:.1f} dy={M[1, 2]:.1f} conf={mean_conf:.3f} (pts={len(pts1)})"
                        )

        except Exception as _match_error:
            logger.warning(
                "[Stitch]   %d→%d: %s matching failed after %.1fs: %s",
                i,
                j,
                _matcher_name,
                time.perf_counter() - _match_started,
                _match_error,
            )

    # ── Attempt 1b: ALIKED + LightGlue (P2.3) ─────────────────────────────
    # Trigger when LoFTR returned < 20 background keypoints on a flat/sparse
    # scene.  ALIKED's deformable descriptor head detects keypoints at anime
    # line-art edges that LoFTR misses in low-texture regions.
    if M is None and aliked_wrapper is not None and _loftr_bg_pts < 20:
        _aliked_started = time.perf_counter()
        print(f"[Stitch]   {i}→{j}: ALIKED+LightGlue fallback started...", flush=True)
        try:
            M_alg, c_alg, pts_alg_i, pts_alg_j = aliked_wrapper.get_translation(
                match_in_i, match_in_j, match_m_i, match_m_j
            )
            print(
                f"[Stitch]   {i}→{j}: ALIKED+LightGlue fallback finished "
                f"in {time.perf_counter() - _aliked_started:.1f}s.",
                flush=True,
            )
            if M_alg is not None and _is_valid(M_alg) and len(pts_alg_i) >= 15:
                M, mean_conf = M_alg, c_alg
                actual_pts_i = pts_alg_i + [ec_w, ec_h]
                actual_pts_j = pts_alg_j + [ec_w, ec_h]
                observed_pts_i = actual_pts_i
                observed_pts_j = actual_pts_j
                logger.debug(
                    f"[Stitch]   {i}→{j}: ALIKED+LG dx={M[0, 2]:.1f} dy={M[1, 2]:.1f} "
                    f"conf={mean_conf:.3f} (pts={len(pts_alg_i)})"
                )
        except Exception as _aliked_error:
            print(
                f"[Stitch]   {i}→{j}: ALIKED+LightGlue fallback failed "
                f"after {time.perf_counter() - _aliked_started:.1f}s: {_aliked_error}",
                flush=True,
            )

    # ── Attempt 2: Template Match (Fallback) ───────────────────────────────
    if M is None:
        M_tm, c_tm = _template_match(
            match_in_i, match_in_j, match_m_i, match_m_j, match_in_i.shape[0]
        )
        if M_tm is not None and c_tm > 0.6:
            M, mean_conf = M_tm, c_tm
            logger.debug(
                f"[Stitch]   {i}→{j}: TemplateMatch dy={M[1, 2]:.1f} conf={mean_conf:.3f}"
            )

    # ── Attempt 3a: Masked phase correlation ───────────────────────────────
    if M is None:
        M_pc, c_pc = _phase_correlate(
            match_in_i, match_in_j, match_m_i, match_m_j, use_mask=True
        )
        if M_pc is not None and _is_valid(M_pc) and c_pc > 0.25:
            M, mean_conf = M_pc, c_pc
            logger.debug(
                f"[Stitch]   {i}→{j}: PhaseCorr(masked) dx={M[0, 2]:.1f} dy={M[1, 2]:.1f} conf={mean_conf:.3f}"
            )

    # ── Attempt 3b: Unmasked phase correlation (uniform-bg fallback) ──────
    if M is None:
        M_pc2, c_pc2 = _phase_correlate(
            match_img_i, match_img_j, None, None, use_mask=False
        )
        if M_pc2 is not None and _is_valid(M_pc2) and c_pc2 > 0.15:
            M, mean_conf = M_pc2, c_pc2
            logger.debug(
                f"[Stitch]   {i}→{j}: PhaseCorr(unmasked) dx={M[0, 2]:.1f} dy={M[1, 2]:.1f} conf={mean_conf:.3f}"
            )

    # ── Attempt 4: Segment-guided matching (P2.9, AnimeInterp technique) ──
    # Segment both frames into flat-color regions via mean-shift + connected
    # components, match regions by colour/position proximity, and take the
    # median centroid displacement as the translation estimate.  Robust on
    # low-texture anime cells where all above methods fail.
    if M is None:
        try:
            M_sg, c_sg = _segment_guided_match(
                match_in_i, match_in_j, match_m_i, match_m_j
            )
            if M_sg is not None and _is_valid(M_sg):
                M, mean_conf = M_sg, c_sg
                logger.debug(
                    f"[Stitch]   {i}→{j}: SegmentGuided dx={M[0, 2]:.1f} dy={M[1, 2]:.1f} conf={mean_conf:.3f}"
                )
        except Exception:
            pass

    # ── Attempt 5: RoMa v2 dense warp (P2.8) ─────────────────────────────
    # DINOv2 features are style-agnostic and work on flat anime cells where
    # all other matchers fail.  Last resort before declaring the edge dead.
    if M is None and roma_wrapper is not None:
        try:
            M_roma, c_roma = roma_wrapper.match_translation(
                match_in_i, match_in_j, match_m_i, match_m_j
            )
            if M_roma is not None and _is_valid(M_roma):
                M, mean_conf = M_roma, c_roma
                logger.debug(
                    f"[Stitch]   {i}→{j}: RoMa dx={M[0, 2]:.1f} dy={M[1, 2]:.1f} conf={mean_conf:.3f}"
                )
        except Exception:
            pass

    if M is None:
        logger.info(f"[Stitch]   {i}→{j}: all methods failed — skipping edge.")
        return None

    # §1.3E: when ASP_SIMILARITY_MODE=1, project to best-fit 4-DOF similarity
    # (scale + rotation + translation, shear discarded).  Default: strip to
    # translation-only to preserve backward-compatible behaviour.
    if _SIMILARITY_MODE:
        M = _extract_similarity(M)
    else:
        M_transl = np.eye(2, 3, dtype=np.float32)
        M_transl[0, 2] = M[0, 2]
        M_transl[1, 2] = M[1, 2]
        M = M_transl

    # Build anchor points for the BA residuals.
    # Convention: M[1,2] = dy where dy = y_j - y_i (forward-shift: LoFTR/PC).
    # Canvas placement: ty_j = ty_i - dy, so residual pi_global = pj_global
    # requires pts_j = pts_i + M[:2, 2].
    if actual_pts_i is not None and actual_pts_j is not None:
        pts_i = actual_pts_i
        pts_j = actual_pts_j
    else:
        # P1.5: use spatially-distributed grid sampling (4×4, n=50) for non-LoFTR edges
        # to avoid centre-biased random anchor points that dilute the BA signal (W7).
        pts_i = _sample_bg_points_grid(m_i, H, W, n=50, grid=(4, 4))
        pts_j = pts_i + M[:2, 2]

    return {
        "i": i,
        "j": j,
        "M": M,
        "pts_i": pts_i,
        "pts_j": pts_j,
        "weight": mean_conf,
        "registration_metrics": _ransac_metrics(observed_pts_i, observed_pts_j),
    }


def _pairwise_match(
    frames: list[np.ndarray],
    bg_masks: list[np.ndarray | None],
    loftr_wrapper=None,
    use_loftr: bool = True,
    motion_model: str = "translation",
    aliked_wrapper=None,
    roma_wrapper=None,
    bg_masked_matching: bool = False,
    proposal_telemetry: dict[str, Any] | None = None,
    extra_proposals: list[TemporalPairProposal] | None = None,
) -> list[dict]:
    """
    Build pairwise correspondence edges using LoFTR -> template match -> PC fallback.
    Default proposals are consecutive (i->i+1) plus spans two and three.
    ``extra_proposals`` (P2 connectivity, default-off) are appended additively —
    never used to remove or reorder the temporal backbone.
    """
    N = len(frames)
    H, W = frames[0].shape[:2]

    rw = int(os.environ.get('ASP_TEMPORAL_RANGE', '3'))
    proposals = propose_temporal_pairs(N, range_width=rw)
    if extra_proposals:
        existing_keys = {(p.i, p.j) for p in proposals}
        proposals = proposals + [
            p for p in extra_proposals if (p.i, p.j) not in existing_keys
        ]
    if proposal_telemetry is not None:
        proposal_telemetry.update(
            {
                "range_width": rw,
                "candidates": [asdict(proposal) for proposal in proposals],
                "extra_proposals_added": len(extra_proposals) if extra_proposals else 0,
            }
        )

    edges: list[dict] = []
    t0 = time.perf_counter()
    for _idx, proposal in enumerate(proposals):
        i, j = proposal.i, proposal.j
        elapsed = time.perf_counter() - t0
        if _MATCH_BUDGET_SEC > 0 and elapsed > _MATCH_BUDGET_SEC:
            logger.warning(
                "[Stitch] match budget %.0fs exceeded after %d/%d pairs "
                "(%d edges) — stopping so SCANS can run (asp_test83 hang class).",
                _MATCH_BUDGET_SEC,
                _idx,
                len(proposals),
                len(edges),
            )
            break
        print(
            f"[Stitch]   Pair {_idx + 1}/{len(proposals)} ({i}→{j}, {elapsed:.0f}s elapsed)...",
            flush=True,
        )
        edge = _match_pair(
            frames,
            bg_masks,
            i,
            j,
            H,
            W,
            loftr_wrapper=loftr_wrapper,
            use_loftr=use_loftr,
            motion_model=motion_model,
            aliked_wrapper=aliked_wrapper,
            roma_wrapper=roma_wrapper,
            bg_masked_matching=bg_masked_matching,
        )

        if edge is not None:
            edges.append(edge)

    if proposal_telemetry is not None:
        proposal_telemetry["matched_edges"] = len(edges)

    # Moving tensors back to CPU in the matcher already synchronizes the work.
    # Avoid forcing a device-wide synchronize and allocator flush after every
    # pair: on CUDA this serializes the fallback chain and can present as a
    # second "hang" immediately after model loading. Callers that need the old
    # diagnostic behaviour can opt in for one run.
    if os.environ.get("ASP_MATCH_CUDA_SYNC", "0") == "1" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return edges


__all__ = ["_match_pair", "_pairwise_match"]
