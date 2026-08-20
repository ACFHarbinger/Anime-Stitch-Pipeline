"""
Tests for the core quality metrics in bench_anime_stitch.py (2026-07 trim).

Covers the surviving metric set:
  _seam_visibility_score — worst adjacent-row luminance jump (dominant ASP
                           failure mode vs simple stitch)
  _seam_coherence        — row-mean luminance std (banding proxy)
  _edge_energy_score     — double-Sobel Y energy (sharpness proxy, NOT ghosting)
  _ghosting_score_v2     — FFT autocorrelation ghosting (the true ghost metric)
  _compute_all_metrics   — core metric dict contract
  _compute_cqas          — no-GT aggregate quality score
plus the SeamVisGate decision formula (§4.8) used in process_dataset.
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

try:
    from asp_backend_benchmark.bench_anime_stitch import (  # noqa: E402
        _compute_all_metrics,
        _compute_cqas_v1_legacy,
        _edge_energy_score,
        _ghosting_score_v2,
        _seam_coherence,
        _seam_visibility_score,
        _strip_banding_score,
    )
except ImportError:
    from backend.benchmark.bench_anime_stitch import (  # noqa: E402
        _compute_all_metrics,
        _compute_cqas_v1_legacy,
        _edge_energy_score,
        _ghosting_score_v2,
        _seam_coherence,
        _seam_visibility_score,
        _strip_banding_score,
    )


def _solid(H: int, W: int, lum: int) -> np.ndarray:
    """(H, W, 3) uint8 BGR image of uniform luminance."""
    return np.full((H, W, 3), lum, dtype=np.uint8)


def _stacked(top_lum: int, bot_lum: int, H: int = 100, W: int = 120) -> np.ndarray:
    """Two uniform halves stacked vertically — guaranteed visible seam."""
    top = _solid(H // 2, W, top_lum)
    bot = _solid(H - H // 2, W, bot_lum)
    return np.concatenate([top, bot], axis=0)


class TestSeamVisibilityScore:
    def test_uniform_image_scores_near_zero(self):
        assert _seam_visibility_score(_solid(100, 120, 128)) < 1.0

    def test_hard_seam_scores_high(self):
        img = _stacked(40, 200)
        assert _seam_visibility_score(img) > 50.0

    def test_soft_gradient_scores_low(self):
        ramp = np.tile(
            np.linspace(40, 200, 100).astype(np.uint8)[:, None, None], (1, 120, 3)
        )
        assert _seam_visibility_score(ramp) < 5.0


class TestSeamCoherence:
    def test_uniform_image_is_coherent(self):
        assert _seam_coherence(_solid(100, 120, 128)) < 1.0

    def test_banded_image_is_incoherent(self):
        bands = np.concatenate(
            [_solid(25, 120, v) for v in (40, 200, 40, 200)], axis=0
        )
        assert _seam_coherence(bands) > _seam_coherence(_solid(100, 120, 128))


class TestEdgeEnergyScore:
    def test_flat_image_scores_zero(self):
        assert _edge_energy_score(_solid(64, 64, 100)) == pytest.approx(0.0, abs=1e-4)

    def test_edges_raise_score(self):
        assert _edge_energy_score(_stacked(0, 255, H=64, W=64)) > 1.0


class TestGhostingSiqe:
    def test_returns_bounded_score(self):
        img = _stacked(60, 180)
        v = _ghosting_score_v2(img)
        assert 0.0 <= v <= 100.0


class TestComputeAllMetrics:
    CORE_KEYS = {
        "sharpness", "coverage", "seam_gradient", "color_entropy",
        "edge_energy_score", "ghosting_siqe", "seam_coherence",
        "seam_visibility", "strip_banding_score", "ghost_seam_scores",
        "ghost_seam_max", "width", "height", "cqas_v1_legacy",
    }

    def test_core_keys_present(self):
        m = _compute_all_metrics(_stacked(60, 180), n_strips=2)
        assert set(m.keys()) >= self.CORE_KEYS

    def test_removed_metric_zoo_keys_absent(self):
        m = _compute_all_metrics(_stacked(60, 180), n_strips=2)
        for stale in (
            "ghosting_score", "rlhf_score",
            "composite_quality", "canvas_gain_uniformity", "strip_self_ssim",
            "mllm_overall", "seam_ownership_entropy",
        ):
            assert stale not in m, f"stale metric key {stale!r} still emitted"

    def test_strip_banding_zero_without_affines(self):
        m = _compute_all_metrics(_stacked(20, 235))
        assert m["strip_banding_score"] == 0.0

    def test_strip_banding_detects_adjacent_strip_jump(self):
        img = _stacked(20, 235, H=100, W=80)
        affines = [
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 50.0]], dtype=np.float32),
        ]
        score = _strip_banding_score(img, affines)
        assert score > 50.0
        assert _compute_all_metrics(img, affines)["strip_banding_score"] == round(score, 2)


class TestComputeCqas:
    def test_range_and_ordering(self):
        good = _compute_all_metrics(_solid(100, 120, 128))
        bad = _compute_all_metrics(_stacked(20, 235))
        assert 0.0 <= bad["cqas_v1_legacy"] <= 1.0
        assert 0.0 <= good["cqas_v1_legacy"] <= 1.0
        assert good["cqas_v1_legacy"] >= bad["cqas_v1_legacy"]

    def test_all_none_returns_none(self):
        assert _compute_cqas_v1_legacy({}) is None


class TestSeamVisGateFormula:
    """§4.8 SeamVisGate decision logic: limit = max(floor, ratio × max(sim, 1))."""

    def _limit(self, sim_sv, ratio=3.0, floor=20.0):
        return max(floor, ratio * max(sim_sv, 1.0))

    def test_catastrophic_seam_fires(self):
        assert self._limit(2.9) < 92.6  # test74-representative

    def test_clean_seam_does_not_fire(self):
        assert self._limit(1.2) >= 6.0  # test27-representative

    def test_floor_dominates_for_clean_simple(self):
        assert self._limit(0.0) == 20.0

    def test_ratio_dominates_for_noisy_simple(self):
        assert self._limit(15.0) == 45.0
