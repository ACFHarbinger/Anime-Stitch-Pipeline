"""Module-level env-var flags for the Laplacian-blend composite, plus the
shared session-level seam ``ThreadPoolExecutor``.
"""

from __future__ import annotations

import concurrent.futures as _cf
import os
from typing import Optional

from ._native import BATCH_AVAILABLE

# §3.11 — Session-level seam ThreadPoolExecutor.  Creating a new pool per
# _composite_foreground call (311 tests × up to 4 workers) causes ~1 200
# pthread_create/join cycles that stall the Linux CFS scheduler.  One shared
# pool, created once on first use, eliminates all thread lifecycle overhead.
_SEAM_POOL: Optional["_cf.ThreadPoolExecutor"] = None


def _get_seam_pool() -> "_cf.ThreadPoolExecutor":
    global _SEAM_POOL
    if _SEAM_POOL is None:
        _SEAM_POOL = _cf.ThreadPoolExecutor(max_workers=4)
    return _SEAM_POOL


# Stage 8.5 foreground pose registration toggle (see fg_register.py).
# Enabled by default; set ASP_FG_REGISTER=0 to disable for A/B comparison.
_FG_REGISTER_ENABLED = os.environ.get("ASP_FG_REGISTER", "1") != "0"

# §2.3 — Phase-consistent compositing (roadmap Phase 2.2/2.3, S216). When a
# seam's two frames belong to different animation phases (per
# frame_selection.detect_animation_phases), never midpoint-warp them
# together — that IS the "body part assembled from two poses" failure mode
# the critical evaluation identifies. Escalate straight to single-pose from
# the dominant (more-complete) phase instead, promoting the A6 single-pose
# fallback from a residual-threshold escape hatch to policy at phase
# boundaries. Default OFF pending A/B measurement.
_PHASE_COMPOSITE: bool = os.environ.get("ASP_PHASE_COMPOSITE", "0") != "0"

# Phase 4 — cv::detail::GraphCutSeamFinder global multi-image seam.
# Default OFF (2026-07-09): the first full measurement of this path (5-test
# verify, post-trim) showed seam_visibility 20–80 vs the pairwise-DP path's
# ~26 S160 average — the hard ownership cut + ±8px feather without per-seam
# blocks gain compensation produces *more* visible seams than the wide-feather
# DP blend, not fewer.  Re-enable via ASP_GRAPHCUT_SEAM=1 only together with
# work on GC-boundary photometric correction, and benchmark before defaulting.
_GRAPHCUT_SEAM: bool = BATCH_AVAILABLE and os.environ.get("ASP_GRAPHCUT_SEAM", "0") != "0"

# §3.33 / §1.3 post-mortem (2026-07-27) — Feather width (px) at GraphCut
# ownership boundaries. The original 8px default was an order of magnitude
# narrower than the DP path's typical 100-300px feathers (see Feathers logs
# in any benchmark run) — a likely dominant contributor to GraphCut's first
# measurement (seam_visibility 20-80 vs the DP path's 2-16). Widened to 96px
# to bring it onto the same scale; still configurable via ASP_GC_FEATHER_PX.
# Set ASP_GC_FEATHER_PX=0 to disable feathering entirely.
_GC_FEATHER_PX: int = int(os.environ.get("ASP_GC_FEATHER_PX", "96"))

# §1.27: Background pixel coverage minimum for normalisation.  The normalisation loop
# already guards with `len(bg_px) >= 200` before applying gain correction.  This flag
# makes that 200-pixel floor configurable — useful when the character fills most of the
# frame (sparse-bg scenes) and a tighter or looser threshold is needed.
# Default 0 → falls back to the built-in 200-pixel floor.
_BG_NORM_MIN_PX: int = int(os.environ.get("ASP_BG_NORM_MIN_PX", "0"))

# §1.106 — Post-composite seam luminance step audit (S152).
# After all seams are composited, measures the mean absolute lum step at each
# boundary row in the final output and logs warnings for large steps.
# ASP_POST_SEAM_WARN_THRESH sets the warning threshold (default 8.0 lum units).
# Always runs when boundaries are available (negligible overhead).
_POST_SEAM_WARN_THRESH: float = float(
    os.environ.get("ASP_POST_SEAM_WARN_THRESH", "8.0")
)

# §4.1 — Spatial blocks gain compensation (S160).
# After per-frame luminance normalisation, divide the blend zone into 32×32
# blocks and compute a per-block BGR gain ratio (fa/fb).  A bilinear-resized
# gain map (clamped [0.5, 2.0]) is applied to fb_zone before blending.
# Targets strip-level banding that the global scalar gain cannot handle.
# Default ON (§4.10 pre-seam equalization covers GraphCut; DP path also corrects).
_BLOCKS_GAIN_COMP: bool = os.environ.get("ASP_BLOCKS_GAIN_COMP", "1") != "0"

# §4.4 — Per-channel luminance blocks gain compensation (S160).
# Like §4.1 but uses the LAB L-channel ratio as a scalar gain applied to all
# BGR channels — avoids color cast from near-zero individual channel means.
# Default ON — LAB L-channel complement to BGR gain (no colour-cast risk).
_BLOCKS_LUM_COMP: bool = os.environ.get("ASP_BLOCKS_LUM_COMP", "1") != "0"

# §4.10 — Pre-seam global gain equalization (S165).
# Applied to ALL warped frames before GraphCut / DP seam finding.  Sequential
# pairwise _blocks_gain_compensate calls equalize inter-frame luminance to
# reduce strip_banding_score and seam_visibility.  Frame 0 (reference) unchanged.
# Only corrects pixels where BOTH adjacent frames have valid content (non-black).
# Default ON.  Set ASP_GLOBAL_GAIN_COMP=0 to disable.
_GLOBAL_GAIN_COMP: bool = os.environ.get("ASP_GLOBAL_GAIN_COMP", "1") != "0"

# §3.1 — Joint canvas-space blocks-gain solve (Brown-Lowe 2007, research §9.3).
# Alternative to _equalize_warped_gains's sequential pairwise chain, which
# anchors each frame's correction to its already-corrected predecessor and so
# drifts over long chains. This solves one linear least-squares system over
# ALL overlapping frame pairs' bg-only mean luminance simultaneously, with a
# gain-prior term regularizing each frame toward gain=1.0 — the same
# formulation cv2's own detail::GainCompensator implements for classical
# panorama stitching. Default OFF pending A/B against the sequential chain.
_JOINT_GAIN_SOLVE: bool = os.environ.get("ASP_JOINT_GAIN_SOLVE", "0") != "0"
_JOINT_GAIN_SIGMA_N: float = float(os.environ.get("ASP_JOINT_GAIN_SIGMA_N", "10.0"))
_JOINT_GAIN_SIGMA_G: float = float(os.environ.get("ASP_JOINT_GAIN_SIGMA_G", "0.1"))


__all__ = [
    "_get_seam_pool",
    "_FG_REGISTER_ENABLED",
    "_PHASE_COMPOSITE",
    "_GRAPHCUT_SEAM",
    "_GC_FEATHER_PX",
    "_BG_NORM_MIN_PX",
    "_POST_SEAM_WARN_THRESH",
    "_BLOCKS_GAIN_COMP",
    "_BLOCKS_LUM_COMP",
    "_GLOBAL_GAIN_COMP",
    "_JOINT_GAIN_SOLVE",
    "_JOINT_GAIN_SIGMA_N",
    "_JOINT_GAIN_SIGMA_G",
]
