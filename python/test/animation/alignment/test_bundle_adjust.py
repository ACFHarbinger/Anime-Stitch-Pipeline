"""
Tests for the global bundle adjustment stage (Stage 7).

Issue categories covered:
  C — Alignment failure: frame clustering, wrong order, non-monotonic ty values.

All tests use synthetic edges with no GPU dependency.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Repo root is 4 dirname() calls up from backend/test/animation/test_*.py
_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

from backend.src.animation.alignment.bundle_adjust import (  # noqa: E402
    _bundle_adjust_affine,
    _compute_adaptive_f_scale,
    _gnc_weights_geman_mcclure,
    _spanning_tree_inlier_filter,
)
from conftest import compute_ty_gaps, make_edge  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tys(affines):
    return np.array([float(a[1, 2]) for a in affines])


# ---------------------------------------------------------------------------
# 1. Basic correctness
# ---------------------------------------------------------------------------


class TestPerfectChainTranslation:
    """A perfect sequential chain should yield monotonically spaced ty values."""

    def test_3frame_chain_ty_values(self):
        edges = [make_edge(i, i + 1, dy=300.0) for i in range(2)]
        affines = _bundle_adjust_affine(edges, 3, use_affine=False)
        tys = _tys(affines)
        gaps = np.diff(sorted(tys))
        assert len(gaps) == 2
        assert np.allclose(gaps, 300.0, atol=5.0), f"gaps={gaps}"

    def test_5frame_chain_monotonic(self):
        edges = [make_edge(i, i + 1, dy=250.0) for i in range(4)]
        affines = _bundle_adjust_affine(edges, 5, use_affine=False)
        tys = _tys(affines)
        sorted_tys = sorted(tys)
        gaps = np.diff(sorted_tys)
        assert all(g > 0 for g in gaps), f"Non-positive gaps: {gaps}"

    def test_uniform_dy_produces_uniform_gaps(self):
        dy = 200.0
        N = 6
        edges = [make_edge(i, i + 1, dy=dy) for i in range(N - 1)]
        affines = _bundle_adjust_affine(edges, N, use_affine=False)
        gaps = compute_ty_gaps(affines)
        assert np.allclose(gaps, dy, atol=10.0), f"gaps={gaps}"


class TestAnchorFrame:
    """Frame 0 must be pinned near identity (ty≈0) to prevent global drift."""

    def test_frame0_ty_near_zero(self):
        edges = [make_edge(i, i + 1, dy=300.0) for i in range(3)]
        affines = _bundle_adjust_affine(edges, 4, use_affine=False)
        assert abs(float(affines[0][1, 2])) < 5.0, (
            f"frame0 ty={affines[0][1, 2]:.2f} should be near 0"
        )

    def test_frame0_tx_near_zero_translation_mode(self):
        edges = [make_edge(i, i + 1, dx=5.0, dy=300.0) for i in range(3)]
        affines = _bundle_adjust_affine(edges, 4, use_affine=False)
        assert abs(float(affines[0][0, 2])) < 5.0


class TestTranslationOnlyMode:
    """use_affine=False should produce pure translation matrices."""

    def test_returns_translation_matrices(self):
        edges = [make_edge(i, i + 1, dy=300.0) for i in range(2)]
        affines = _bundle_adjust_affine(edges, 3, use_affine=False)
        for i, M in enumerate(affines):
            assert M.shape == (2, 3), f"frame{i}: shape={M.shape}"
            assert np.allclose(M[:2, :2], np.eye(2), atol=1e-4), (
                f"frame{i}: rotation block deviated from identity: {M[:2, :2]}"
            )

    def test_returns_n_affines(self):
        N = 7
        edges = [make_edge(i, i + 1, dy=150.0) for i in range(N - 1)]
        affines = _bundle_adjust_affine(edges, N, use_affine=False)
        assert len(affines) == N


class TestAffineMode:
    """use_affine=True should return 2×3 matrices (may include small rotation)."""

    def test_returns_n_affines_affine_mode(self):
        N = 4
        edges = [make_edge(i, i + 1, dy=300.0) for i in range(N - 1)]
        affines = _bundle_adjust_affine(edges, N, use_affine=True)
        assert len(affines) == N

    def test_affine_mode_ty_close_to_translation_mode(self):
        """When edges are pure translation, affine and translation modes agree closely."""
        N = 4
        edges = [make_edge(i, i + 1, dy=300.0) for i in range(N - 1)]
        aff_t = _bundle_adjust_affine(edges, N, use_affine=False)
        aff_a = _bundle_adjust_affine(edges, N, use_affine=True)
        tys_t = _tys(aff_t)
        tys_a = _tys(aff_a)
        assert np.allclose(sorted(tys_t), sorted(tys_a), atol=10.0), (
            f"translation mode tys={sorted(tys_t)}, affine mode tys={sorted(tys_a)}"
        )


# ---------------------------------------------------------------------------
# 2. Robustness to bad edges (documents missing RANSAC — Issues C)
# ---------------------------------------------------------------------------


class TestFrameClusteringPattern:
    """
    Near-zero dy edges cause multiple frames to land at the same canvas position.
    This is the root cause of test8 and test9 catastrophic failures.

    These tests document the CURRENT behavior (clustering occurs) and the
    EXPECTED behavior after the Priority-1 fix (clusters are rejected/corrected).
    """

    def test_near_zero_edges_inverted_outlier_rejection(self):
        """
        When most edges have near-zero dy and only one is normal, the BA outlier
        rejection correctly identifies the normal edge as a statistical outlier
        and prunes it. The remaining near-zero edges produce a low-ratio (but
        physically incorrect) result.

        This demonstrates why the min-step guard in _filter_edges is required
        BEFORE bundle adjustment — to prevent the inverted consensus scenario.
        """
        edges = [
            make_edge(0, 1, dy=5.0),  # near-zero
            make_edge(1, 2, dy=8.0),  # near-zero
            make_edge(2, 3, dy=6.0),  # near-zero
            make_edge(3, 4, dy=300.0),  # one normal — pruned as outlier!
        ]
        affines = _bundle_adjust_affine(edges, 5, use_affine=False)
        gaps = compute_ty_gaps(affines)

        max_gap = float(gaps.max())
        median_gap = float(np.median(gaps))
        ratio = max_gap / max(median_gap, 1.0)

        # The 300px edge is pruned, leaving only near-zero edges.
        # The ratio of max (8) to median (6) is < 3.0.
        assert ratio < 3.0, (
            f"Expected the normal edge to be pruned as an outlier (ratio < 3), "
            f"but got ratio={ratio:.1f}."
        )

    def test_all_near_zero_edges_degenerate_output(self):
        """All near-zero edges → all frames cluster at ~0; canvas height collapses."""
        edges = [make_edge(i, i + 1, dy=3.0) for i in range(4)]
        affines = _bundle_adjust_affine(edges, 5, use_affine=False)
        tys = _tys(affines)
        canvas_extent = float(max(tys) - min(tys))
        assert canvas_extent < 50.0, (
            f"All near-zero dy edges should produce a tiny canvas extent, "
            f"got {canvas_extent:.1f}px"
        )

    def test_single_outlier_edge_rejected_by_ba(self):
        """
        A single edge with 5× the expected dy should be pruned by the
        residual-based outlier rejection, keeping the result near inlier dy.
        """
        inlier_dy = 200.0
        edges = [
            make_edge(0, 1, dy=inlier_dy),
            make_edge(1, 2, dy=inlier_dy),
            make_edge(2, 3, dy=inlier_dy * 5),  # outlier
            make_edge(3, 4, dy=inlier_dy),
        ]
        affines = _bundle_adjust_affine(edges, 5, use_affine=False)
        gaps = compute_ty_gaps(affines)
        max_gap = float(gaps.max())
        median_gap = float(np.median(gaps))
        ratio = max_gap / max(median_gap, 1.0)

        # After outlier rejection, the 5× outlier edge should be pruned
        # and the remaining inlier edges should produce a ratio near 1.0.
        assert ratio < 3.0, (
            f"Outlier should have been pruned by BA residual rejection "
            f"(ratio={ratio:.1f}), but it still distorts the result."
        )


class TestNonMonotonicFrameOrder:
    """
    Non-monotonic ty values (test2, test7 pattern) indicate wrong-direction matches.
    The C++ path uses outlier rejection to prune reversed edges; the Python path
    keeps them and produces non-monotonic output (caught by _filter_edges later).
    """

    def test_reversed_edge_result_is_consistent(self):
        """One reversed edge (dy<0 among otherwise positive chain).

        With the C++ fast path, the outlier rejection prunes the reversed edge
        and produces a monotonic result.  With the Python-only path the result
        is non-monotonic.  Either way the output must be a list of N valid
        affines with frame 0 near identity.
        """
        edges = [
            make_edge(0, 1, dy=300.0),
            make_edge(1, 2, dy=-280.0),  # reversed — wrong direction
            make_edge(2, 3, dy=300.0),
        ]
        affines = _bundle_adjust_affine(edges, 4, use_affine=False)
        assert len(affines) == 4
        assert abs(float(affines[0][1, 2])) < 5.0, "frame 0 must be near identity"

    def test_compute_ty_gap_ratio(self):
        """Utility: compute alignment health ratio from affine list."""
        affines = [
            np.array([[1, 0, 0], [0, 1, ty]], dtype=np.float32)
            for ty in [0, 300, 600, 900]
        ]
        gaps = compute_ty_gaps(affines)
        assert gaps.max() / np.median(gaps) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_two_frames_minimal_input(self):
        edges = [make_edge(0, 1, dy=200.0)]
        affines = _bundle_adjust_affine(edges, 2, use_affine=False)
        assert len(affines) == 2
        assert abs(float(affines[0][1, 2])) < 5.0

    def test_skip_pair_edge_included(self):
        """Skip-pair edges (non-adjacent) are valid inputs."""
        edges = [
            make_edge(0, 1, dy=300.0),
            make_edge(1, 2, dy=300.0),
            make_edge(0, 2, dy=600.0),  # skip-pair: sum of adj edges
        ]
        affines = _bundle_adjust_affine(edges, 3, use_affine=False)
        gaps = compute_ty_gaps(affines)
        assert all(g > 0 for g in gaps)

    def test_dx_component_propagates(self):
        """Horizontal tx offsets (diagonal scroll) should be preserved."""
        edges = [make_edge(i, i + 1, dx=50.0, dy=200.0) for i in range(2)]
        affines = _bundle_adjust_affine(edges, 3, use_affine=False)
        txs = [float(a[0, 2]) for a in affines]
        tx_range = max(txs) - min(txs)
        assert tx_range > 10.0, (
            f"Expected non-zero tx range for dx=50 edges, got tx_range={tx_range:.1f}"
        )


# ---------------------------------------------------------------------------
# 4. GNC robust loss (§1.1C) — Cauchy loss makes BA resist outlier edges
# ---------------------------------------------------------------------------


class TestGNCRobustLoss:
    """
    The bundle adjust now uses a Cauchy (GNC) robust loss.  Verify that:
      - Clean edge chains still produce accurate ty gaps (loss doesn't distort inliers)
      - A single extreme outlier edge (10× inlier dy) is absorbed without
        catastrophically distorting the inlier chain.
    """

    def test_clean_chain_unaffected_by_gnc(self):
        """GNC Cauchy loss should not distort a chain of good edges."""
        dy = 250.0
        N = 5
        edges = [make_edge(i, i + 1, dy=dy) for i in range(N - 1)]
        affines = _bundle_adjust_affine(edges, N, use_affine=False)
        gaps = compute_ty_gaps(affines)
        # GNC should not distort good edges; gaps should stay near dy
        assert np.allclose(gaps, dy, atol=15.0), (
            f"GNC distorted clean chain: gaps={gaps}, expected ≈{dy}px"
        )

    def test_extreme_outlier_absorbed_by_gnc(self):
        """
        A single edge with 10× the consensus dy should not destroy the inlier
        chain under GNC — the Cauchy loss down-weights it sufficiently.
        After optional post-solve pruning, the remaining gaps should be close
        to the inlier consensus.
        """
        inlier_dy = 200.0
        N = 6
        edges = [make_edge(i, i + 1, dy=inlier_dy) for i in range(N - 1)]
        # Replace one edge with a massive outlier
        edges[2] = make_edge(2, 3, dy=inlier_dy * 10.0)
        affines = _bundle_adjust_affine(edges, N, use_affine=False)
        gaps = compute_ty_gaps(affines)
        # Even with the outlier, at least 3 of 5 gaps should be near the inlier dy
        near_inlier = int(np.sum(np.abs(gaps - inlier_dy) < inlier_dy * 0.5))
        assert near_inlier >= 3, (
            f"GNC should protect most inlier gaps; only {near_inlier}/5 were "
            f"near {inlier_dy}px: gaps={gaps}"
        )

    def test_gnc_f_scale_env_var_overridden(self):
        """ASP_BA_F_SCALE env var changes f_scale without breaking imports."""
        import importlib
        import os as _os

        original = _os.environ.get("ASP_BA_F_SCALE")
        try:
            _os.environ["ASP_BA_F_SCALE"] = "5.0"
            import backend.src.animation.alignment.bundle_adjust as _ba

            importlib.reload(_ba)
            assert abs(_ba._BA_F_SCALE - 5.0) < 1e-6
        finally:
            if original is None:
                _os.environ.pop("ASP_BA_F_SCALE", None)
            else:
                _os.environ["ASP_BA_F_SCALE"] = original
            importlib.reload(_ba)


# ---------------------------------------------------------------------------
# §1.1D — Adaptive GNC f_scale  (S30)
# ---------------------------------------------------------------------------


class TestAdaptiveFScale:
    """
    _compute_adaptive_f_scale: data-driven Cauchy loss scale from post-solve
    edge residuals.  Returns max(floor, 2 × median_residual_px).
    """

    def test_floor_dominates_for_perfect_solution(self):
        """A BA solution that perfectly fits all edges → residuals ≈ 0 → floor returned."""
        dy = 200.0
        N = 4
        edges = [make_edge(i, i + 1, dy=dy) for i in range(N - 1)]
        affines = _bundle_adjust_affine(edges, N, use_affine=False)
        scale = _compute_adaptive_f_scale(edges, affines, floor=10.0)
        # Residuals ≈ 0 on a clean chain → 2×0 < floor → floor=10.0
        assert scale == pytest.approx(10.0, abs=2.0)

    def test_widens_when_solution_does_not_fit_edges(self):
        """Manual mismatch: affines predict near-zero displacement but edges say 100 px."""
        # N = 3
        edges = [make_edge(0, 1, dy=100.0), make_edge(1, 2, dy=100.0)]
        # Deliberately wrong: all frames at canvas row 0 (ty=0)
        fake_affines = [
            np.eye(2, 3, dtype=np.float32),  # ty=0
            np.eye(2, 3, dtype=np.float32),  # ty=0
            np.eye(2, 3, dtype=np.float32),  # ty=0
        ]
        # obs_dy = -100 for both; pred_dy = 0-0 = 0 → residual = 100 px each
        # adaptive = max(10, 2×100) = 200
        scale = _compute_adaptive_f_scale(edges, fake_affines, floor=10.0)
        assert scale > 10.0
        assert scale == pytest.approx(200.0, abs=5.0)

    def test_empty_edges_returns_floor(self):
        """No edges → floor returned without error."""
        scale = _compute_adaptive_f_scale([], [], floor=7.5)
        assert scale == pytest.approx(7.5)

    def test_floor_respected_for_tiny_residuals(self):
        """2 × median < floor → floor is returned (no shrinkage)."""
        N = 3
        # Perfect edges: after BA the residuals should be near zero
        edges = [make_edge(0, 1, dy=300.0), make_edge(1, 2, dy=300.0)]
        affines = _bundle_adjust_affine(edges, N, use_affine=False)
        # Use a large floor — adaptive should not go below it
        scale = _compute_adaptive_f_scale(edges, affines, floor=50.0)
        assert scale >= 50.0

    def test_single_edge_computes_correctly(self):
        """One edge with a known residual gives the right adaptive scale."""
        N = 2
        edges = [make_edge(0, 1, dy=80.0)]
        # Affines: both frames at ty=0 (100% wrong prediction for dy=80)
        fake_affines = [np.eye(2, 3, dtype=np.float32)] * N
        # obs_dy=-80, pred_dy=0 → residual=80; adaptive=max(10, 160)=160
        scale = _compute_adaptive_f_scale(edges, fake_affines, floor=10.0)
        assert scale == pytest.approx(160.0, abs=2.0)


# ---------------------------------------------------------------------------
# §1.1B  Spanning-tree consensus pre-filter
# ---------------------------------------------------------------------------


class TestSpanningTreeInlierFilter:
    """Unit tests for _spanning_tree_inlier_filter (§1.1B).

    The filter builds a max-weight spanning tree and removes edges whose
    observed dx/dy disagrees with the tree reference by > inlier_threshold.
    Spanning-tree edges themselves always pass (residual = 0), so the graph
    stays connected after filtering.
    """

    def test_all_consistent_chain_edges_returned(self):
        """Perfect chain: all edges agree with the spanning tree → all kept."""
        N = 4
        edges = [
            make_edge(0, 1, dy=300.0),
            make_edge(1, 2, dy=300.0),
            make_edge(2, 3, dy=300.0),
        ]
        result = _spanning_tree_inlier_filter(edges, N, inlier_threshold=50.0)
        assert len(result) == len(edges)

    def test_inconsistent_skip_edge_removed(self):
        """Skip edge (0→2) with dy inconsistent with chain solution is removed."""
        N = 3
        # Chain: 0→1 (dy=300), 1→2 (dy=300) → frame 2 at canvas ty = -600
        # Skip edge: 0→2 should have dy≈600, but we set it to 200 (wrong by 400px)
        edges = [
            make_edge(0, 1, dy=300.0, weight=0.9),
            make_edge(1, 2, dy=300.0, weight=0.9),
            make_edge(0, 2, dy=200.0, weight=0.5),  # inconsistent — should be 600
        ]
        result = _spanning_tree_inlier_filter(edges, N, inlier_threshold=50.0)
        # The bad skip edge (0→2, dy=200) should be removed; chain edges kept
        result_pairs = [(e["i"], e["j"]) for e in result]
        assert (0, 2) not in result_pairs
        assert (0, 1) in result_pairs
        assert (1, 2) in result_pairs

    def test_consistent_skip_edge_kept(self):
        """Skip edge (0→2) with dy matching chain prediction is retained."""
        N = 3
        # Chain dy=300 per step → skip 0→2 should be dy=600
        edges = [
            make_edge(0, 1, dy=300.0, weight=0.9),
            make_edge(1, 2, dy=300.0, weight=0.9),
            make_edge(0, 2, dy=600.0, weight=0.5),  # consistent
        ]
        result = _spanning_tree_inlier_filter(edges, N, inlier_threshold=50.0)
        result_pairs = [(e["i"], e["j"]) for e in result]
        assert (0, 2) in result_pairs

    def test_fallback_on_disconnected_graph(self):
        """Disconnected graph: spanning tree can't reach all frames → original returned."""
        # Frames {0,1} and {2,3} with no cross-component edge
        N = 4
        edges = [make_edge(0, 1, dy=300.0), make_edge(2, 3, dy=300.0)]
        result = _spanning_tree_inlier_filter(edges, N, inlier_threshold=50.0)
        assert result is edges or result == edges

    def test_low_weight_bad_edge_does_not_corrupt_spanning_tree(self):
        """A low-weight bad edge is not chosen for the spanning tree when a
        high-weight good edge connects the same pair — bad edge is then removed."""
        N = 3
        # Two edges between 0→1: high-weight correct, low-weight wrong dy
        # High-weight (0.95) wins the spanning tree; low-weight (0.1) is non-tree
        # Non-tree residual vs spanning tree reference: |200 - 300| = 100px > 50 → removed
        edges = [
            make_edge(0, 1, dy=300.0, weight=0.95),
            make_edge(0, 1, dy=200.0, weight=0.10),  # inconsistent copy
            make_edge(1, 2, dy=300.0, weight=0.90),
        ]
        result = _spanning_tree_inlier_filter(edges, N, inlier_threshold=50.0)
        # Only 2 unique consistent edges should survive (the correct 0→1 and 1→2)
        # The low-weight wrong 0→1 (dy=200) should be removed
        result_dy = [
            abs(float(e["M"][1, 2])) for e in result if e["i"] == 0 and e["j"] == 1
        ]
        assert all(d == pytest.approx(300.0, abs=1.0) for d in result_dy)


# ---------------------------------------------------------------------------
# §1.17 — GNC-TLS Geman-McClure weights
# ---------------------------------------------------------------------------


class TestGNCWeightsGemanMcclure:
    """Unit tests for _gnc_weights_geman_mcclure (Yang et al. 2020, §1.17)."""

    def test_unit_weights_large_mu(self):
        """At large μ (convex regime) all weights approach 1.0."""
        r_sq = np.array([0.0, 25.0, 100.0, 10000.0], dtype=np.float64)
        c_sq = 100.0  # c = 10px
        mu = 1e6
        w = _gnc_weights_geman_mcclure(r_sq, mu, c_sq)
        assert w.shape == r_sq.shape
        assert np.all(w > 0.999), f"weights at large μ should be ≈1, got {w}"

    def test_zero_residual_weight_one(self):
        """An edge with zero residual always receives weight exactly 1.0."""
        r_sq = np.array([0.0], dtype=np.float64)
        w = _gnc_weights_geman_mcclure(r_sq, mu=1.0, c_sq=100.0)
        assert float(w[0]) == pytest.approx(1.0, abs=1e-9)

    def test_high_residual_suppressed(self):
        """Residual >> c at μ=1 yields weight << 1 (outlier suppression)."""
        c_sq = 100.0  # c = 10px
        r_sq = np.array([100.0**2], dtype=np.float64)  # 100px >> 10px
        w = _gnc_weights_geman_mcclure(r_sq, mu=1.0, c_sq=c_sq)
        assert float(w[0]) < 0.01, f"outlier should be near-zero weight, got {w[0]}"

    def test_weights_in_valid_range(self):
        """All returned weights lie in [0, 1] for arbitrary residuals and μ."""
        rng = np.random.default_rng(42)
        r_sq = rng.uniform(0, 1e6, size=50)
        w = _gnc_weights_geman_mcclure(r_sq, mu=2.0, c_sq=100.0)
        assert np.all(w >= 0.0), "weights must be non-negative"
        assert np.all(w <= 1.0 + 1e-9), f"weights must be ≤ 1, max={w.max()}"

    def test_higher_residual_lower_weight(self):
        """Weights are strictly monotone decreasing in residual magnitude."""
        r_sq = np.array([0.0, 1.0, 25.0, 100.0, 1000.0], dtype=np.float64)
        w = _gnc_weights_geman_mcclure(r_sq, mu=1.0, c_sq=100.0)
        assert np.all(np.diff(w) <= 0), f"weights should decrease with residual: {w}"
