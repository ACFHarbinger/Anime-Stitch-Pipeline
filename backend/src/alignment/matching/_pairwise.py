"""Fallback-chain orchestration: try each matcher strategy per frame pair,
then build the full pairwise correspondence-edge list for bundle adjustment.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from backend.src.constants import MATCH_EDGE_CROP, MAX_DX_DRIFT_RATIO

from ._matchers import _phase_correlate, _segment_guided_match, _template_match
from ._math import _compute_bg_match_ratio, _compute_translation_spread, _extract_similarity
from ._sampling import _sample_bg_points_grid

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


def _match_pair(  # noqa: C901
    frames: List[np.ndarray],
    bg_masks: List[Optional[np.ndarray]],
    i: int,
    j: int,
    H: int,
    W: int,
    loftr_wrapper=None,
    use_loftr: bool = True,
    motion_model: str = "translation",
    aliked_wrapper=None,
    roma_wrapper=None,
) -> Optional[Dict]:
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

    def _is_valid(M):
        if M is None:
            return False
        dx = abs(M[0, 2])
        return not dx > W * MAX_DX_DRIFT_RATIO

    M: Optional[np.ndarray] = None
    mean_conf = 0.0
    actual_pts_i: Optional[np.ndarray] = None
    actual_pts_j: Optional[np.ndarray] = None
    _loftr_bg_pts: int = 0  # track how many BG keypoints LoFTR found (for 1b trigger)

    # ── Attempt 1: LoFTR ───────────────────────────────────────────────────
    if use_loftr and loftr_wrapper is not None:
        try:
            pts1, pts2, conf = loftr_wrapper.match(match_img_i, match_img_j)
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
                        M_raw, inliers = cv2.estimateAffine2D(
                            pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0
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
                        logger.debug(
                            f"[Stitch]   {i}→{j}: LoFTR dx={M[0, 2]:.1f} dy={M[1, 2]:.1f} conf={mean_conf:.3f} (pts={len(pts1)})"
                        )

        except Exception:
            pass

    # ── Attempt 1b: ALIKED + LightGlue (P2.3) ─────────────────────────────
    # Trigger when LoFTR returned < 20 background keypoints on a flat/sparse
    # scene.  ALIKED's deformable descriptor head detects keypoints at anime
    # line-art edges that LoFTR misses in low-texture regions.
    if M is None and aliked_wrapper is not None and _loftr_bg_pts < 20:
        try:
            M_alg, c_alg, pts_alg_i, pts_alg_j = aliked_wrapper.get_translation(
                match_img_i, match_img_j, match_m_i, match_m_j
            )
            if M_alg is not None and _is_valid(M_alg) and len(pts_alg_i) >= 15:
                M, mean_conf = M_alg, c_alg
                actual_pts_i = pts_alg_i + [ec_w, ec_h]
                actual_pts_j = pts_alg_j + [ec_w, ec_h]
                logger.debug(
                    f"[Stitch]   {i}→{j}: ALIKED+LG dx={M[0, 2]:.1f} dy={M[1, 2]:.1f} "
                    f"conf={mean_conf:.3f} (pts={len(pts_alg_i)})"
                )
        except Exception:
            pass

    # ── Attempt 2: Template Match (Fallback) ───────────────────────────────
    if M is None:
        M_tm, c_tm = _template_match(
            match_img_i, match_img_j, match_m_i, match_m_j, match_img_i.shape[0]
        )
        if M_tm is not None and c_tm > 0.6:
            M, mean_conf = M_tm, c_tm
            logger.debug(
                f"[Stitch]   {i}→{j}: TemplateMatch dy={M[1, 2]:.1f} conf={mean_conf:.3f}"
            )

    # ── Attempt 3a: Masked phase correlation ───────────────────────────────
    if M is None:
        M_pc, c_pc = _phase_correlate(
            match_img_i, match_img_j, match_m_i, match_m_j, use_mask=True
        )
        if _is_valid(M_pc) and c_pc > 0.25:
            M, mean_conf = M_pc, c_pc
            logger.debug(
                f"[Stitch]   {i}→{j}: PhaseCorr(masked) dx={M[0, 2]:.1f} dy={M[1, 2]:.1f} conf={mean_conf:.3f}"
            )

    # ── Attempt 3b: Unmasked phase correlation (uniform-bg fallback) ──────
    if M is None:
        M_pc2, c_pc2 = _phase_correlate(
            match_img_i, match_img_j, None, None, use_mask=False
        )
        if _is_valid(M_pc2) and c_pc2 > 0.15:
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
                match_img_i, match_img_j, match_m_i, match_m_j
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
                match_img_i, match_img_j, match_m_i, match_m_j
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
    }


def _pairwise_match(
    frames: List[np.ndarray],
    bg_masks: List[Optional[np.ndarray]],
    loftr_wrapper=None,
    use_loftr: bool = True,
    motion_model: str = "translation",
    aliked_wrapper=None,
    roma_wrapper=None,
) -> List[Dict]:
    """
    Build pairwise correspondence edges using LoFTR -> template match -> PC fallback.
    Adds consecutive (i->i+1) plus skip-pair (i->i+2, i->i+3) edges.
    """
    N = len(frames)
    H, W = frames[0].shape[:2]

    # Build list of (i, j) pairs to try
    pairs: List[Tuple[int, int]] = []
    for i in range(N - 1):
        pairs.append((i, i + 1))
    for i in range(N - 2):
        pairs.append((i, i + 2))  # skip-1
    for i in range(N - 3):
        pairs.append((i, i + 3))  # skip-2

    edges: List[Dict] = []
    for _idx, (i, j) in enumerate(pairs):
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
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if edge is not None:
            edges.append(edge)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return edges


__all__ = ["_match_pair", "_pairwise_match"]
