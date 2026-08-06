"""
Tests for pairwise matching helpers — specifically the §1.3E similarity-mode
extraction function ``_extract_similarity``.

All tests use synthetic numpy matrices; no GPU or image files required.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from backend.src.animation.alignment.matching import (
    _compute_bg_match_ratio,
    _compute_translation_spread,
    _extract_similarity,
)  # noqa: E402

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

# ---------------------------------------------------------------------------
# §1.3E — _extract_similarity
# ---------------------------------------------------------------------------


class TestExtractSimilarity:
    """
    _extract_similarity(M) projects a full 2×3 affine to the best-fit 4-DOF
    similarity [[a, b, tx], [-b, a, ty]].

    Closed-form:  a_sym = (M[0,0] + M[1,1]) / 2
                  b_sym = (M[0,1] - M[1,0]) / 2
    Translation terms are copied unchanged.
    """

    def _pure_translation(self, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
        M = np.eye(2, 3, dtype=np.float32)
        M[0, 2] = tx
        M[1, 2] = ty
        return M

    def _rotation_affine(
        self, theta_deg: float, tx: float = 0.0, ty: float = 0.0
    ) -> np.ndarray:
        theta = math.radians(theta_deg)
        M = np.array(
            [
                [math.cos(theta), -math.sin(theta), tx],
                [math.sin(theta), math.cos(theta), ty],
            ],
            dtype=np.float32,
        )
        return M

    def _scale_translation(
        self, scale: float, tx: float = 0.0, ty: float = 0.0
    ) -> np.ndarray:
        M = np.array(
            [
                [scale, 0.0, tx],
                [0.0, scale, ty],
            ],
            dtype=np.float32,
        )
        return M

    def test_pure_translation_unchanged(self):
        """Identity rotation/scale → output matches input exactly."""
        M = self._pure_translation(tx=30.0, ty=120.0)
        out = _extract_similarity(M)
        assert out.shape == (2, 3)
        assert out[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert out[0, 1] == pytest.approx(0.0, abs=1e-5)
        assert out[1, 0] == pytest.approx(0.0, abs=1e-5)
        assert out[1, 1] == pytest.approx(1.0, abs=1e-5)
        assert out[0, 2] == pytest.approx(30.0, abs=1e-4)
        assert out[1, 2] == pytest.approx(120.0, abs=1e-4)

    def test_rotation_preserved(self):
        """A pure rotation matrix → similarity projection recovers rotation terms exactly."""
        theta = 6.35
        M = self._rotation_affine(theta, tx=0.0, ty=200.0)
        out = _extract_similarity(M)
        # For a proper rotation: a=cos θ, b=-sin θ, c=sin θ, d=cos θ
        # a_sym = (a+d)/2 = cos θ; b_sym = (b-c)/2 = (-sin θ - sin θ)/2 = -sin θ
        expected_a = math.cos(math.radians(theta))
        expected_b = -math.sin(math.radians(theta))
        assert out[0, 0] == pytest.approx(expected_a, abs=1e-5)
        assert out[0, 1] == pytest.approx(expected_b, abs=1e-5)
        assert out[1, 0] == pytest.approx(-expected_b, abs=1e-5)
        assert out[1, 1] == pytest.approx(expected_a, abs=1e-5)
        assert out[0, 2] == pytest.approx(0.0, abs=1e-4)
        assert out[1, 2] == pytest.approx(200.0, abs=1e-4)

    def test_uniform_scale_preserved(self):
        """Uniform scale (s×I) → similarity output has a_sym=s, b_sym=0."""
        s = 1.121
        M = self._scale_translation(scale=s, tx=50.0, ty=300.0)
        out = _extract_similarity(M)
        assert out[0, 0] == pytest.approx(s, abs=1e-5)
        assert out[0, 1] == pytest.approx(0.0, abs=1e-5)
        assert out[1, 0] == pytest.approx(0.0, abs=1e-5)
        assert out[1, 1] == pytest.approx(s, abs=1e-5)
        assert out[0, 2] == pytest.approx(50.0, abs=1e-4)
        assert out[1, 2] == pytest.approx(300.0, abs=1e-4)

    def test_shear_component_eliminated(self):
        """A matrix with asymmetric off-diagonals (shear) → shear discarded, similarity kept.

        M = [[1, 0.3, 0], [0.1, 1, 0]] has shear (c=0.1 ≠ -b=-0.3).
        Symmetric: a_sym=(1+1)/2=1, b_sym=(0.3-0.1)/2=0.1.
        """
        M = np.array([[1.0, 0.3, 0.0], [0.1, 1.0, 0.0]], dtype=np.float32)
        out = _extract_similarity(M)
        assert out[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert out[0, 1] == pytest.approx(0.1, abs=1e-5)  # b_sym = (0.3-0.1)/2
        assert out[1, 0] == pytest.approx(-0.1, abs=1e-5)  # -b_sym
        assert out[1, 1] == pytest.approx(1.0, abs=1e-5)

    def test_output_satisfies_similarity_constraint(self):
        """For any input M, the output must satisfy the similarity constraint:
        out[1, 0] == -out[0, 1] and out[1, 1] == out[0, 0].
        """
        rng = np.random.default_rng(42)
        for _ in range(20):
            M = rng.standard_normal((2, 3)).astype(np.float32)
            out = _extract_similarity(M)
            assert out[1, 0] == pytest.approx(-out[0, 1], abs=1e-5), (
                "Similarity constraint -b == c violated"
            )
            assert out[1, 1] == pytest.approx(out[0, 0], abs=1e-5), (
                "Similarity constraint a == d violated"
            )


# ---------------------------------------------------------------------------
# §1.36 — _compute_translation_spread (S100)
# ---------------------------------------------------------------------------


class TestComputeTranslationSpread:
    """_compute_translation_spread returns MAD of per-match dx/dy around their median."""

    def _pts(self, dxs, dys):
        """Build synthetic pts_i (all zeros) and pts_j (offsets by given dxs/dys)."""
        n = len(dxs)
        pts_i = np.zeros((n, 2), dtype=np.float32)
        pts_j = np.column_stack([dxs, dys]).astype(np.float32)
        return pts_i, pts_j

    def test_identical_displacements_zero_spread(self):
        """All matches agree on the same translation → MAD = 0."""
        pts_i, pts_j = self._pts(np.full(20, -100.0), np.full(20, 50.0))
        mad_dx, mad_dy = _compute_translation_spread(pts_i, pts_j)
        assert mad_dx == pytest.approx(0.0, abs=1e-4)
        assert mad_dy == pytest.approx(0.0, abs=1e-4)

    def test_spread_matches_known_mad(self):
        """With known displacements, MAD equals the expected value."""
        # dxs = [0, 10, 20, 30, 40], median = 20, |dxs - 20| = [20,10,0,10,20], MAD = 10
        dxs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        pts_i, pts_j = self._pts(dxs, np.zeros_like(dxs))
        mad_dx, mad_dy = _compute_translation_spread(pts_i, pts_j)
        assert mad_dx == pytest.approx(10.0, abs=1e-4)
        assert mad_dy == pytest.approx(0.0, abs=1e-4)

    def test_high_spread_detected(self):
        """Bimodal distribution of displacements yields high MAD."""
        # Half at dx=-100, half at dx=-200 → MAD = 50
        dxs = np.concatenate([np.full(10, -100.0), np.full(10, -200.0)])
        pts_i, pts_j = self._pts(dxs, np.zeros(20))
        mad_dx, _ = _compute_translation_spread(pts_i, pts_j)
        assert mad_dx > 30.0, f"expected high spread, got mad_dx={mad_dx}"

    def test_single_point_returns_zero(self):
        """N ≤ 1 → (0.0, 0.0) — no spread to compute."""
        pts_i = np.zeros((1, 2), dtype=np.float32)
        pts_j = np.array([[50.0, 30.0]], dtype=np.float32)
        mad_dx, mad_dy = _compute_translation_spread(pts_i, pts_j)
        assert mad_dx == 0.0 and mad_dy == 0.0

    def test_dx_and_dy_independent(self):
        """Spread in dx and dy are reported independently; dy spread is zero when all agree."""
        dxs = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
        dys = np.full(5, -75.0)
        pts_i, pts_j = self._pts(dxs, dys)
        mad_dx, mad_dy = _compute_translation_spread(pts_i, pts_j)
        assert mad_dx > 0.0, "expected nonzero dx spread"
        assert mad_dy == pytest.approx(0.0, abs=1e-4), "expected zero dy spread"


class TestComputeBgMatchRatio:
    """§1.38: LoFTR background match ratio gate."""

    def test_all_background_returns_one(self):
        """When every LoFTR match is on background, ratio must be 1.0."""
        assert _compute_bg_match_ratio(n_bg_pts=200, n_total_pts=200) == pytest.approx(
            1.0
        )

    def test_no_background_returns_zero(self):
        """When no LoFTR matches survive bg filtering, ratio is 0.0."""
        assert _compute_bg_match_ratio(n_bg_pts=0, n_total_pts=150) == pytest.approx(
            0.0
        )

    def test_known_ratio_computed_correctly(self):
        """20 bg pts out of 100 total → ratio 0.20."""
        result = _compute_bg_match_ratio(n_bg_pts=20, n_total_pts=100)
        assert result == pytest.approx(0.20, abs=1e-6)

    def test_zero_total_returns_zero_no_division_error(self):
        """When n_total_pts is 0 (no LoFTR matches at all), returns 0.0 without raising."""
        assert _compute_bg_match_ratio(n_bg_pts=0, n_total_pts=0) == pytest.approx(0.0)

    def test_half_bg_half_fg_returns_point_five(self):
        """Exactly half the matches on background → ratio 0.5."""
        assert _compute_bg_match_ratio(n_bg_pts=50, n_total_pts=100) == pytest.approx(
            0.5, abs=1e-6
        )
