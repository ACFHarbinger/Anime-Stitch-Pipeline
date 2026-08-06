"""
Tests for affine matrix validation — verifies the behavior of the production
``_validate_affines`` function in ``src.animation.validation``.

Issue categories covered:
  C — Alignment failure: frame clustering (min_gap=0), high ratio (>3×).
  D/G — Affine rotation/scale mismatch (test18: good ty/tx, bad rotation).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

from backend.src.animation.core.validation import (  # noqa: E402
    AffineHealth,  # noqa: F401
    _check_translation_monotonicity,
    _compute_adaptive_min_gap,
    _compute_adaptive_rot_scale,
    _validate_affines,
)
from conftest import make_rotation_affine, make_translation_affine  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Clean affines — should pass
# ---------------------------------------------------------------------------


class TestGoodAffinesPass:
    def test_perfect_chain(self):
        """Well-spaced, pure-translation affines should be marked valid."""
        affines = [make_translation_affine(ty=i * 300.0) for i in range(5)]
        h = _validate_affines(affines)
        assert h.valid, f"Expected valid, got: {h.reason}"

    def test_test11_baseline_pattern(self):
        """test11 baseline: ratio=1.1×, min_gap=59px → valid."""
        tys = [0, 59, 130, 200, 275, 345, 420]
        affines = [make_translation_affine(ty=float(ty)) for ty in tys]
        h = _validate_affines(affines, min_step=50.0)
        assert h.valid, f"test11-like affines should be valid: {h.reason}"
        assert h.ratio < 2.0

    def test_test6_baseline_pattern(self):
        """test6 baseline: ratio=1.4×, min_gap=126px → valid."""
        gaps = [174, 177, 178, 163, 238, 133, 137, 126]
        tys = np.cumsum([0] + gaps).tolist()
        affines = [make_translation_affine(ty=float(ty)) for ty in tys]
        h = _validate_affines(affines, min_step=50.0)
        assert h.valid, f"test6-like affines should be valid: {h.reason}"

    def test_borderline_ratio_large_min_gap(self):
        """test12: ratio=2.9×, min_gap=173px → valid (min_gap large enough)."""
        tys = [0, 173, 400, 600, 780, 1280]
        affines = [make_translation_affine(ty=float(ty)) for ty in tys]
        h = _validate_affines(affines, min_step=50.0, max_ratio=3.0)
        assert h.valid, (
            f"Borderline ratio with large min_gap should be valid: {h.reason}"
        )
        assert h.min_gap >= 173.0


# ---------------------------------------------------------------------------
# 2. High ratio detection (frame clustering)
# ---------------------------------------------------------------------------


class TestHighRatioDetection:
    def test_test8_pattern_ratio_59x(self):
        """test8: ratio=5.9×, min_gap=16px → invalid."""
        tys = [1085, 243, 762, 259, 1166, 1785, 0, 333, 726, 1263, 1806]
        affines = [make_translation_affine(ty=float(ty)) for ty in tys]
        h = _validate_affines(affines)
        assert not h.valid, f"test8-like affines should be INVALID: ratio={h.ratio:.1f}"
        assert h.ratio > 3.0

    def test_test9_pattern_ratio_118x(self):
        """test9: ratio=11.8× → invalid."""
        tys = [680, 654, 628, 624, 471, 459, 451, 145, 0]
        affines = [make_translation_affine(ty=float(ty)) for ty in tys]
        h = _validate_affines(affines)
        assert not h.valid

    def test_test16_pattern_ratio_61x(self):
        """test16: ratio=6.1×, min_gap=12px → invalid."""
        tys = [430, 0, 746, 843, 418, 370, 1021, 902, 807, 565]
        affines = [make_translation_affine(ty=float(ty)) for ty in tys]
        h = _validate_affines(affines)
        assert not h.valid
        assert h.min_gap < 50.0

    def test_single_2x_outlier_ratio_threshold(self):
        """One gap at 3.5× median → exceeds threshold → invalid."""
        gaps = [200, 200, 700, 200, 200]
        tys = np.cumsum([0] + gaps).tolist()
        affines = [make_translation_affine(ty=float(ty)) for ty in tys]
        h = _validate_affines(affines, max_ratio=3.0)
        assert not h.valid
        assert h.ratio == pytest.approx(3.5, abs=0.1)


# ---------------------------------------------------------------------------
# 3. Min gap detection (co-located frames)
# ---------------------------------------------------------------------------


class TestMinGapDetection:
    def test_test21_colocated_triplet(self):
        """test21: 3 frames at ty=0 → min_gap=0 → invalid."""
        tys = [0, 177, 355, 532, 710, 887, 1064, 1242, 0, 0]
        affines = [make_translation_affine(ty=float(ty)) for ty in tys]
        h = _validate_affines(affines)
        assert not h.valid
        assert h.min_gap == pytest.approx(0.0, abs=1.0)

    def test_zero_gap_two_identical_frames(self):
        """Two frames at the same ty → min_gap=0 → invalid."""
        affines = [make_translation_affine(ty=float(ty)) for ty in [0, 0, 300, 600]]
        h = _validate_affines(affines)
        assert not h.valid

    def test_small_gap_below_min_step(self):
        """Gap of 10px < min_step=50px → invalid."""
        affines = [make_translation_affine(ty=float(ty)) for ty in [0, 10, 300, 600]]
        h = _validate_affines(affines, min_step=50.0)
        assert not h.valid
        assert h.min_gap < 50.0


# ---------------------------------------------------------------------------
# 4. Rotation / scale mismatch (test18 pattern — Category G)
# ---------------------------------------------------------------------------


class TestRotationScaleMismatch:
    def test_small_rotation_detected(self):
        """
        test18 pattern: ty/tx look healthy but affine matrices contain significant
        rotation components → canvas placement will misalign warped frames.
        """
        affines = [
            make_rotation_affine(tx=0.0, ty=float(i * 300), angle_deg=8.0)
            for i in range(5)
        ]
        h = _validate_affines(affines, max_rotation=0.1)
        assert not h.valid, (
            f"8° rotation should be flagged; max_rotation={h.max_rotation:.3f}"
        )
        assert h.max_rotation > 0.05

    def test_pure_translation_no_rotation_warning(self):
        """Pure translation affines have off-diagonal=0 → rotation check passes."""
        affines = [make_translation_affine(ty=i * 300.0) for i in range(5)]
        h = _validate_affines(affines, max_rotation=0.1)
        assert h.max_rotation == pytest.approx(0.0, abs=1e-6)

    def test_scale_deviation_detected(self):
        """Scale drift (a[0][0] != 1.0) should be flagged."""
        affines = [
            np.array([[1.2, 0.0, 0.0], [0.0, 1.2, float(i * 300)]], dtype=np.float32)
            for i in range(4)
        ]
        h = _validate_affines(affines, max_scale_dev=0.1)
        assert not h.valid
        assert h.max_scale_dev > 0.1

    def test_tiny_rotation_within_tolerance(self):
        """Sub-pixel rotation (< 0.5°) should not trigger the rotation check."""
        affines = [
            make_rotation_affine(tx=0.0, ty=float(i * 300), angle_deg=0.3)
            for i in range(5)
        ]
        h = _validate_affines(affines, max_rotation=0.1)
        assert h.max_rotation < 0.1


# ---------------------------------------------------------------------------
# 5. Non-monotonic order (test2, test7)
# ---------------------------------------------------------------------------


class TestNonMonotonicOrder:
    def test_non_monotonic_tys_still_sorted_for_gap_computation(self):
        """
        Even if affines have non-monotonic ty (by frame index), gap computation
        sorts them first. Non-monotonicity is detectable via a separate check.
        """
        tys_by_frame = [
            0,
            49,
            99,
            130,
            165,
            335,
            538,
            465,
            265,
            816,
            771,
            821,
            991,
            1040,
        ]
        affines = [make_translation_affine(ty=float(ty)) for ty in tys_by_frame]

        # sorted_tys = sorted(tys_by_frame)
        # gaps = np.diff(sorted_tys)
        # ratio = float(gaps.max()) / float(np.median(gaps))

        h = _validate_affines(affines)
        assert not h.valid, f"test7 non-monotonic affines should be invalid: {h.reason}"
        assert h.ratio > 3.0

    def test_decreasing_tys_valid_if_evenly_spaced(self):
        """
        test9 affines are monotonically decreasing (reverse order) — the gap
        computation should handle this correctly.
        """
        tys = [900, 700, 500, 300, 100]
        affines = [make_translation_affine(ty=float(ty)) for ty in tys]
        h = _validate_affines(affines, min_step=50.0)
        assert h.valid, f"Evenly spaced decreasing tys should be valid: {h.reason}"
        assert h.ratio == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# 6. Alignment statistics helper
# ---------------------------------------------------------------------------


class TestAlignmentStats:
    def test_stats_for_positive_baseline(self):
        """test22: ratio=1.0×, min_gap=83px → ratio near 1.0."""
        tys = [0, 83, 166, 249, 332, 415, 498, 581, 664, 747, 830]
        affines = [make_translation_affine(ty=float(ty)) for ty in tys]
        h = _validate_affines(affines)
        assert h.ratio == pytest.approx(1.0, abs=0.05)
        assert h.min_gap == pytest.approx(83.0, abs=1.0)
        assert h.valid

    def test_ratio_calculation_matches_report_values(self):
        """Verify ratio formula: max_gap / median_gap matches the report."""
        gaps = [143, 129, 151, 167, 400, 154, 443, 170, 477, 103, 109, 135, 114]
        tys = np.cumsum([0] + sorted(gaps)).tolist()
        affines = [make_translation_affine(ty=float(ty)) for ty in tys]
        h = _validate_affines(affines)
        assert h.ratio == pytest.approx(477 / 151, abs=0.5)


# ---------------------------------------------------------------------------
# TestAdaptiveMinGap — §0.5C adaptive validation threshold (S36)
# ---------------------------------------------------------------------------


class TestAdaptiveMinGap:
    """
    _compute_adaptive_min_gap returns max(20.0, canvas_span / (N × 3)).

    Slow-scroll: span ≈ 200 px, N=10 → 200/(10×3)=6.7 → floor 20.0 returned.
    Fast-scroll: span = 3000 px, N=10 → 3000/30=100 px returned.
    Dominant axis: uses the larger of dy_span and dx_span.
    """

    def _aff(self, ty: float = 0.0, tx: float = 0.0) -> np.ndarray:
        a = np.eye(2, 3, dtype=np.float32)
        a[0, 2] = tx
        a[1, 2] = ty
        return a

    def test_slow_scroll_returns_floor(self):
        """Canvas span 200 px / (10 × 3) = 6.7 → floor 20.0 returned."""
        affines = [
            self._aff(ty=float(i * 20)) for i in range(11)
        ]  # 11 frames, span=200
        result = _compute_adaptive_min_gap(affines)
        assert result == pytest.approx(20.0)

    def test_fast_scroll_exceeds_fixed_threshold(self):
        """Canvas span 3000 px, N=10 → 3000/(10×3)=100.0 → adaptive > default 25 px."""
        # 10 frames: ty = 0, 333, 666, ..., 2997 → span = 2997 px ≈ 3000 px
        # Use explicit span: first=0, last=3000 with 10 frames total
        tys = [(i * (3000 / 9)) for i in range(10)]  # span=3000, N=10
        affines = [self._aff(ty=ty) for ty in tys]
        result = _compute_adaptive_min_gap(affines)
        assert result == pytest.approx(3000.0 / (10 * 3.0), abs=0.1)
        assert result > 25.0

    def test_single_frame_returns_floor(self):
        """Degenerate N=1 case: no gaps possible → return floor 20.0."""
        result = _compute_adaptive_min_gap([self._aff(ty=0.0)])
        assert result == pytest.approx(20.0)

    def test_dominant_axis_is_max_span(self):
        """Horizontal scroll (dx >> dy): uses dx_span as canvas_span."""
        # 10 frames with tx spanning 1500 px, ty spanning 30 px
        affines = [self._aff(ty=float(i * 3), tx=float(i * 150)) for i in range(11)]
        result = _compute_adaptive_min_gap(affines)
        # dx_span=1500, N=11, adaptive=1500/(11×3)=45.45...
        assert result == pytest.approx(1500.0 / (11 * 3.0), abs=0.1)
        assert result > 25.0  # exceeds fixed default

    def test_wired_into_pipeline_initial_call(self):
        """Fast-scroll affines should use adaptive threshold on the first validation call.

        A frame set with span=3000 px (N=10, adaptive_min_gap=100 px) where one
        pair has a gap of only 30 px should be rejected at the first call, even
        though 30 px > the fixed 25 px default.  This verifies that the pipeline
        now applies the content-adaptive threshold rather than the fixed one.
        """
        # 10 frames at ~300 px apart, except frames 5-6 which are only 30 px apart
        tys = [float(i * 300) for i in range(10)]
        tys[6] = tys[5] + 30.0  # near-duplicate pair
        affines = [self._aff(ty=ty) for ty in tys]
        adaptive = _compute_adaptive_min_gap(affines)
        assert adaptive > 25.0, (
            "adaptive threshold should exceed fixed default for fast scroll"
        )
        health = _validate_affines(affines, min_step=adaptive)
        # The 30 px gap is below the adaptive threshold → validation fails
        assert not health.valid
        assert "min_gap" in health.reason


# ---------------------------------------------------------------------------
# §0.5D — Adaptive rotation/scale thresholds
# ---------------------------------------------------------------------------


class TestAdaptiveRotScale:
    """
    _compute_adaptive_rot_scale returns (max_rotation, max_scale_dev).

    Consistent rotation/scale (σ < 0.02) → loose thresholds (0.15).
    Inconsistent rotation/scale (σ ≥ 0.02) → tight thresholds (0.10).
    """

    def _rot_affine(self, rot: float, ty: float = 0.0) -> np.ndarray:
        """Affine with explicit off-diagonal rotation element and translation ty."""
        a = np.eye(2, 3, dtype=np.float32)
        a[0, 1] = rot
        a[1, 0] = -rot
        a[1, 2] = ty
        return a

    def _sc_affine(self, scale_dev: float, ty: float = 0.0) -> np.ndarray:
        """Affine with diagonal scale deviation from 1.0."""
        a = np.eye(2, 3, dtype=np.float32)
        a[0, 0] = 1.0 + scale_dev
        a[1, 1] = 1.0 + scale_dev
        a[1, 2] = ty
        return a

    def test_consistent_rotation_returns_loose_threshold(self):
        """All frames with near-identical rotation → σ≈0 → loose threshold 0.15 returned."""
        affines = [self._rot_affine(rot=0.111, ty=float(i * 200)) for i in range(6)]
        max_rot, max_sc = _compute_adaptive_rot_scale(affines)
        assert max_rot == pytest.approx(0.15), (
            "Consistent rotation should yield loose threshold 0.15"
        )

    def test_inconsistent_rotation_returns_tight_threshold(self):
        """Varying per-frame rotation (σ > 0.02) → tight threshold 0.10 returned."""
        rots = [0.0, 0.05, 0.0, 0.10, 0.0, 0.08]  # std ≈ 0.04
        affines = [
            self._rot_affine(rot=r, ty=float(i * 200)) for i, r in enumerate(rots)
        ]
        max_rot, _ = _compute_adaptive_rot_scale(affines)
        assert max_rot == pytest.approx(0.10), (
            "Inconsistent rotation should yield tight threshold 0.10"
        )

    def test_consistent_scale_returns_loose_threshold(self):
        """All frames with near-identical scale deviation → loose threshold 0.15 returned."""
        affines = [self._sc_affine(scale_dev=0.12, ty=float(i * 200)) for i in range(6)]
        _, max_sc = _compute_adaptive_rot_scale(affines)
        assert max_sc == pytest.approx(0.15), (
            "Consistent scale should yield loose threshold 0.15"
        )

    def test_inconsistent_scale_returns_tight_threshold(self):
        """Varying per-frame scale deviation (σ > 0.02) → tight threshold 0.10 returned."""
        devs = [0.0, 0.05, 0.0, 0.10, 0.0, 0.08]  # std ≈ 0.04
        affines = [
            self._sc_affine(scale_dev=d, ty=float(i * 200)) for i, d in enumerate(devs)
        ]
        _, max_sc = _compute_adaptive_rot_scale(affines)
        assert max_sc == pytest.approx(0.10), (
            "Inconsistent scale should yield tight threshold 0.10"
        )

    def test_single_frame_returns_tight_defaults(self):
        """N=1 edge case: cannot compute std → return tight defaults (0.10, 0.10)."""
        affines = [self._rot_affine(rot=0.111)]
        max_rot, max_sc = _compute_adaptive_rot_scale(affines)
        assert max_rot == pytest.approx(0.10)
        assert max_sc == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# TestTranslationMonotonicity — §1.12 Kendall-τ frame ordering check (S52)
# ---------------------------------------------------------------------------


class TestTranslationMonotonicity:
    """
    _check_translation_monotonicity verifies that the temporal order of frames
    matches their spatial (scroll-axis) order via |Kendall τ| ≥ 0.4.

    τ = 1.0  → perfect monotone sequence (forward or backward scroll).
    τ = 0.0  → random permutation — catastrophic BA failure.
    """

    def _make(self, tys: list, axis: int = 1) -> list:
        """Build affines with the given primary-axis translations."""
        affines = []
        for v in tys:
            M = np.eye(2, 3, dtype=np.float32)
            M[axis, 2] = float(v)
            affines.append(M)
        return affines

    def test_perfectly_monotone_sequence_passes(self):
        """Strictly increasing tys → τ=1.0 → passes."""
        affines = self._make([0, 100, 200, 300, 400])
        ok, tau = _check_translation_monotonicity(affines, primary_axis=1)
        assert ok, f"Monotone sequence should pass; τ={tau:.3f}"
        assert tau == pytest.approx(1.0, abs=1e-6)

    def test_strictly_reversed_sequence_passes(self):
        """Uniformly decreasing tys (backward scroll) → |τ|=1.0 → passes."""
        affines = self._make([400, 300, 200, 100, 0])
        ok, tau = _check_translation_monotonicity(affines, primary_axis=1)
        assert ok, f"Reversed monotone sequence should pass; τ={tau:.3f}"
        assert tau == pytest.approx(1.0, abs=1e-6)

    def test_catastrophically_shuffled_fails(self):
        """Heavily shuffled tys → |τ|≈0.2 < 0.4 → fails."""
        # [0, 400, 100, 300, 200]: many pairs out of order (τ=0.2)
        affines = self._make([0, 400, 100, 300, 200])
        ok, tau = _check_translation_monotonicity(affines, primary_axis=1)
        assert not ok, f"Shuffled sequence should fail; τ={tau:.3f}"
        assert tau < 0.4

    def test_single_out_of_order_frame_passes(self):
        """One slightly reversed frame (99% of pairs concordant) → passes."""
        # [0, 100, 250, 230, 300, 400]: frame 3 slightly behind frame 2
        affines = self._make([0, 100, 250, 230, 300, 400])
        ok, tau = _check_translation_monotonicity(affines, primary_axis=1)
        assert ok, f"One transposed frame should pass; τ={tau:.3f}"
        assert tau >= 0.6

    def test_fewer_than_four_frames_always_passes(self):
        """N < 4 → always returns (True, 1.0) regardless of order."""
        affines = self._make([300, 100, 200])  # 3 frames, clearly non-monotone
        ok, tau = _check_translation_monotonicity(affines, primary_axis=1)
        assert ok
        assert tau == pytest.approx(1.0)
