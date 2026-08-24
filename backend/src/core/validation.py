"""
Affine matrix validation — post-bundle-adjustment health checks.

Checks the output of Stage 7 (bundle adjustment) for pathological patterns
that would cascade into catastrophic Stage 9 ghosting.  When validation fails,
the pipeline should fall back to OpenCV SCANS stitch.

Issue categories covered:
  C — Alignment failure: frame clustering (min_gap=0), high ratio (>3×).
  D/G — Affine rotation/scale mismatch (test18: good ty/tx, bad rotation).
"""

from __future__ import annotations

# --------------------------------
import os
from typing import NamedTuple

import numpy as np

# --- Relocated Nested Imports ---
from asp_backend.alignment.canvas import _detect_scroll_axis

try:
    import base as _batch
    if (
        getattr(_batch, "__file__", None) is None
        or not hasattr(_batch, "validation")
    ):
        raise ImportError("compiled base.validation extension not available")
    _HAS_BATCH: bool = True
except ImportError:
    _batch = None  # type: ignore[assignment]
    _HAS_BATCH = False

# §0.5D — tight/loose rotation and scale thresholds used by
# _compute_adaptive_rot_scale.  "Loose" applies when all frames share a
# near-identical rotation/scale (systematic camera property); "tight" when
# frame-to-frame variance is high (BA noise).
# All six thresholds below are exposed as ASP advanced options (see
# backend/src/animation/core/config.py _CONFIG_SCHEMA's "validation" section).
try:
    _ROT_TIGHT: float = float(os.environ.get("ASP_ROT_TIGHT", "0.10"))
except ValueError:
    _ROT_TIGHT = 0.10

# §1.12 — Minimum |Kendall τ| for the translation-monotonicity check.
# A value of 0.4 allows up to ~30 % discordant frame pairs before flagging
# the solution as catastrophically disordered.  Good BA solutions typically
# score ≥ 0.85; random permutations of N=14 score ≈ 0.27 on average.
try:
    _MONO_TAU_MIN: float = float(os.environ.get("ASP_MONO_TAU_MIN", "0.4"))
except ValueError:
    _MONO_TAU_MIN = 0.4
try:
    _ROT_LOOSE: float = float(os.environ.get("ASP_ROT_LOOSE", "0.15"))
except ValueError:
    _ROT_LOOSE = 0.15
try:
    _SC_TIGHT: float = float(os.environ.get("ASP_SC_TIGHT", "0.10"))
except ValueError:
    _SC_TIGHT = 0.10
try:
    _SC_LOOSE: float = float(os.environ.get("ASP_SC_LOOSE", "0.15"))
except ValueError:
    _SC_LOOSE = 0.15
try:
    _ROT_SCALE_CONSISTENCY_THRESH: float = float(
        os.environ.get("ASP_ROT_SCALE_CONSISTENCY_THRESH", "0.02")
    )
except ValueError:
    _ROT_SCALE_CONSISTENCY_THRESH = 0.02


def _check_translation_monotonicity(
    affines: list[np.ndarray],
    primary_axis: int = 1,
    min_tau_abs: float = _MONO_TAU_MIN,
) -> tuple[bool, float]:
    """§1.12: Verify that frame translations are spatially ordered consistently
    with their temporal (frame-index) order, measured by |Kendall τ|.

    For a valid scrolling sequence the camera moves monotonically in the scroll
    direction, so the frame captured at time *t_i* should be spatially ahead of
    the frame captured at time *t_{i-1}*.  If bundle adjustment places frames in
    a scrambled spatial order while preserving uniform spacing (ratio ≈ 1), the
    existing ratio / min-gap checks will not catch the failure — this check does.

    **Algorithm**: compute the Kendall rank-correlation coefficient τ between the
    temporal index sequence [0, 1, …, N-1] and the primary-axis translation
    sequence.  |τ| = 1 for perfectly monotone sequences (forward *and* backward
    scrolling both pass), |τ| ≈ 0 for random permutations.

    Parameters
    ----------
    affines : list of (2, 3) float32 affine matrices in **temporal** order.
    primary_axis : 0 for horizontal scroll, 1 for vertical (default).
    min_tau_abs : minimum acceptable |τ|.  Default 0.4 — flags sequences where
                  more than ~30 % of frame pairs are out of spatial order.

    Returns
    -------
    (is_monotone, tau_abs)
        *is_monotone* is ``True`` when the ordering is acceptable.
        *tau_abs* is the measured |Kendall τ| (0 = random, 1 = perfectly ordered).

    Notes
    -----
    Requires at least 4 frames; shorter sequences always return ``(True, 1.0)``.
    """
    N = len(affines)
    if N < 4:
        return True, 1.0
    vals = np.array([float(M[primary_axis, 2]) for M in affines], dtype=np.float64)
    concordant = discordant = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            d = vals[j] - vals[i]
            if d > 0.0:
                concordant += 1
            elif d < 0.0:
                discordant += 1
    n_pairs = concordant + discordant
    if n_pairs == 0:
        return True, 1.0
    tau = (concordant - discordant) / n_pairs
    tau_abs = abs(tau)
    return tau_abs >= min_tau_abs, tau_abs


class AffineHealth(NamedTuple):
    """Result of affine matrix validation."""

    valid: bool
    ratio: float
    min_gap: float
    max_rotation: float
    max_scale_dev: float
    reason: str


def _affine_gap_stats(affines: list[np.ndarray]) -> dict[str, float]:
    """Return primary-order Euclidean gap diagnostics for telemetry."""
    if len(affines) < 2:
        return {"min_gap": 0.0, "median_gap": 0.0, "max_gap": 0.0}

    txs = np.array([float(a[0, 2]) for a in affines])
    tys = np.array([float(a[1, 2]) for a in affines])
    scroll_axis = _detect_scroll_axis(affines)
    if scroll_axis == "horizontal":
        order = np.argsort(txs)
    elif scroll_axis == "diagonal":
        order = np.argsort(np.hypot(txs - txs[0], tys - tys[0]))
    else:
        order = np.argsort(tys)

    gaps = np.hypot(np.diff(txs[order]), np.diff(tys[order]))
    return {
        "min_gap": float(gaps.min()),
        "median_gap": float(np.median(gaps)),
        "max_gap": float(gaps.max()),
    }


def _validate_affines(
    affines: list[np.ndarray],
    min_step: float = 25.0,
    max_ratio: float = 3.0,
    max_rotation: float = 0.1,
    max_scale_dev: float = 0.1,
) -> AffineHealth:
    """
    Full affine health check — determines whether bundle-adjusted affines are
    safe to use for canvas construction and rendering.

    A frame set is considered INVALID if ANY of the following hold:
      1. max_gap / median_gap > max_ratio  (uneven spacing / clustering)
      2. min_gap < min_step                (co-located frames)
      3. any off-diagonal element > max_rotation  (rotation mismatch — test18)
      4. any diagonal element deviates from 1.0 by > max_scale_dev (scale drift)

    Gap computation uses Euclidean vector magnitude sqrt(dy²+dx²) so that
    mixed-axis scrolls (diagonal, or vertical with horizontal drift) are not
    spuriously rejected when the per-axis component is below min_step but the
    total displacement is well above it.

    Parameters
    ----------
    affines : list of (2, 3) float32 affine matrices.
    min_step : minimum Euclidean displacement between adjacent sorted frames (px).
               Default 25px — lower than the old 50px to accommodate slower-scroll
               sequences in the 94-test corpus without rejecting well-aligned sets.
    max_ratio : maximum allowed max_gap / median_gap ratio.
    max_rotation : maximum allowed off-diagonal (rotation) element magnitude.
    max_scale_dev : maximum allowed deviation of diagonal elements from 1.0.

    Returns
    -------
    AffineHealth named tuple with validation result and diagnostic metrics.
    """
    if _HAS_BATCH:
        valid, reason, ratio, min_gap, max_rot, max_sc = _batch.validation.validate_affines(
            [np.asarray(a, dtype=np.float32) for a in affines],
            min_step=min_step, max_ratio=max_ratio,
            max_rotation=max_rotation, max_scale_dev=max_scale_dev,
        )
        return AffineHealth(valid, ratio, min_gap, max_rot, max_sc, reason)

    N = len(affines)
    if N < 2:
        return AffineHealth(True, 1.0, 0.0, 0.0, 0.0, "single frame")

    # relocated: from asp_backend.alignment.canvas import _detect_scroll_axis
    scroll_axis = _detect_scroll_axis(affines)

    txs = np.array([float(a[0, 2]) for a in affines])
    tys = np.array([float(a[1, 2]) for a in affines])

    # Sort frames by their primary scroll axis, then compute Euclidean gaps
    # between consecutive sorted positions.
    if scroll_axis == "horizontal":
        order = np.argsort(txs)
    elif scroll_axis == "diagonal":
        # Diagonal: sort by cumulative distance from frame 0
        dists = np.sqrt((txs - txs[0]) ** 2 + (tys - tys[0]) ** 2)
        order = np.argsort(dists)
    else:
        # Vertical (default)
        order = np.argsort(tys)

    s_txs = txs[order]
    s_tys = tys[order]
    gaps = np.sqrt(np.diff(s_txs) ** 2 + np.diff(s_tys) ** 2)

    if len(gaps) == 0:
        return AffineHealth(False, float("inf"), 0.0, 0.0, 0.0, "all frames at same position")

    median_gap = float(np.median(gaps))
    max_gap = float(gaps.max())
    min_gap = float(gaps.min())
    ratio = max_gap / max(median_gap, 1.0)

    # Rotation: magnitude of off-diagonal elements in the 2×2 rotation block
    max_rot = max(max(abs(float(a[0, 1])), abs(float(a[1, 0]))) for a in affines)
    # Scale: deviation of diagonal elements from identity (1.0)
    max_sc = max(max(abs(float(a[0, 0]) - 1.0), abs(float(a[1, 1]) - 1.0)) for a in affines)

    if ratio > max_ratio:
        return AffineHealth(False, ratio, min_gap, max_rot, max_sc, f"ratio={ratio:.1f} > {max_ratio}")
    if min_gap < min_step:
        return AffineHealth(False, ratio, min_gap, max_rot, max_sc, f"min_gap={min_gap:.1f}px < {min_step}px")
    if max_rot > max_rotation:
        return AffineHealth(False, ratio, min_gap, max_rot, max_sc, f"rotation={max_rot:.3f} > {max_rotation}")
    if max_sc > max_scale_dev:
        return AffineHealth(False, ratio, min_gap, max_rot, max_sc, f"scale_dev={max_sc:.3f} > {max_scale_dev}")

    # §1.12: translation monotonicity — fires only for vertical/horizontal scroll
    if scroll_axis in ("vertical", "horizontal"):
        primary_axis_int = 0 if scroll_axis == "horizontal" else 1
        ok_mono, tau_val = _check_translation_monotonicity(affines, primary_axis_int)
        if not ok_mono:
            return AffineHealth(
                False, ratio, min_gap, max_rot, max_sc,
                f"monotonicity={tau_val:.2f} < {_MONO_TAU_MIN}"
            )

    return AffineHealth(True, ratio, min_gap, max_rot, max_sc, "ok")


def _compute_adaptive_min_gap(affines: list[np.ndarray]) -> float:
    """§0.5C/§0.7 — Content-adaptive minimum-gap threshold for _validate_affines.

    Returns ``max(20.0, canvas_span / (N × 3))`` where ``canvas_span`` is:

    * **vertical** scroll — ``dy_span``
    * **horizontal** scroll — ``dx_span``
    * **diagonal** scroll — ``sqrt(dy_span² + dx_span²)`` (§0.7: vector
      magnitude gives the actual path length; ``max(dy, dx)`` underestimates it
      by up to 1.41× at 45°, causing the threshold to be too loose for fast
      diagonal pans)

    Rationale: for slow-scroll sequences (canvas_span ≈ 200 px, N=10) the
    expected inter-frame step is ~20 px, so the fixed 25 px threshold is too
    strict.  For fast-scroll / 4K sequences (canvas_span ≈ 4000 px, N=10) a
    25 px gap represents a near-duplicate frame that should be rejected; the
    adaptive threshold rises to 133 px.

    The floor of 20.0 px matches Retry-3's ``min_step=20.0`` so that very
    slow-scroll sequences always pass at least that check.
    """
    if _HAS_BATCH:
        return float(_batch.validation.compute_adaptive_min_gap(
            [np.asarray(a, dtype=np.float32) for a in affines]
        ))

    N = len(affines)
    if N < 2:
        return 20.0
    tys = np.array([float(a[1, 2]) for a in affines])
    txs = np.array([float(a[0, 2]) for a in affines])
    dy_span = float(tys.max() - tys.min())
    dx_span = float(txs.max() - txs.min())
    # §0.7: use axis-appropriate span so diagonal sequences get the correct path length.
    scroll_axis = _detect_scroll_axis(affines)
    if scroll_axis == "horizontal":
        canvas_span = dx_span
    elif scroll_axis == "diagonal":
        canvas_span = float(np.sqrt(dy_span ** 2 + dx_span ** 2))
    else:
        canvas_span = dy_span
    if canvas_span < 1.0:
        return 20.0
    adaptive = canvas_span / (N * 3.0)
    return max(20.0, adaptive)


def _compute_adaptive_rot_scale(
    affines: list[np.ndarray],
) -> tuple[float, float]:
    """§0.5D — Adaptive rotation/scale thresholds for _validate_affines.

    Returns ``(max_rotation, max_scale_dev)``.  When rotation (or scale) is
    *consistent* across frames — i.e. all frames carry nearly the same value,
    as happens for a physically real constant-zoom or fixed-tilt camera — the
    frame-to-frame standard deviation is low and the dominant signal is a
    systematic camera property rather than noisy per-frame BA drift.  In that
    case a looser acceptance window is safe.

    Decision rule (applied independently for rotation and scale)::

        σ < _ROT_SCALE_CONSISTENCY_THRESH  →  use loose threshold (0.15)
        σ ≥ _ROT_SCALE_CONSISTENCY_THRESH  →  use tight threshold (0.10)

    Rationale for the 0.02 consistency threshold: BA typically introduces
    random frame-to-frame rotation noise of ±0.01–0.03 rad.  If σ < 0.02 the
    noise is well below the tight acceptance window and the dominant component
    is a systematic lens/camera offset — widening to 0.15 recovers borderline
    zoom-pan sequences (e.g. test5: max_rot ≈ 0.111, consistent across frames)
    without admitting sequences with wild per-frame BA noise.
    """
    if _HAS_BATCH:
        return tuple(_batch.validation.compute_adaptive_rot_scale(
            [np.asarray(a, dtype=np.float32) for a in affines]
        ))

    if len(affines) < 2:
        return _ROT_TIGHT, _SC_TIGHT

    rotations = np.array(
        [max(abs(float(a[0, 1])), abs(float(a[1, 0]))) for a in affines]
    )
    scales = np.array(
        [max(abs(float(a[0, 0]) - 1.0), abs(float(a[1, 1]) - 1.0)) for a in affines]
    )

    max_rot = _ROT_LOOSE if float(rotations.std()) < _ROT_SCALE_CONSISTENCY_THRESH else _ROT_TIGHT
    max_sc = _SC_LOOSE if float(scales.std()) < _ROT_SCALE_CONSISTENCY_THRESH else _SC_TIGHT

    return max_rot, max_sc


__all__ = [
    "AffineHealth",
    "_affine_gap_stats",
    "_validate_affines",
    "_compute_adaptive_min_gap",
    "_compute_adaptive_rot_scale",
    "_check_translation_monotonicity",
]
