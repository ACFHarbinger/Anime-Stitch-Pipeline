"""
Tests for the _filter_edges method of AnimeStitchPipeline (Stage 5-6 post-processing).

Issue categories covered:
  C — Alignment failure: wrong-direction matches, gross outliers, frame clustering.

The direction consensus filter is only active when len(edges) >= 3 and there are
at least 3 adjacent edges. Tests use synthetic timestamp filenames to enable
velocity-based outlier replacement (avoiding template-match fallback).
"""

from __future__ import annotations

import os
import sys

import pytest

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

from backend.src.animation.core.pipeline import (  # noqa: E402
    AnimeStitchPipeline,
    _compute_adaptive_min_disp,
    _reject_static_edges,
)
from backend.src.constants.animation import STATIC_EDGE_MIN_DISP_PX  # noqa: E402
from conftest import make_edge, make_frame  # noqa: E402

# §3.13B — all tests in this module create 480×640 frame arrays (≥900 KB each);
# the gc_heavy marker triggers gc.collect() after each test.
pytestmark = [pytest.mark.gc_heavy]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline() -> AnimeStitchPipeline:
    return AnimeStitchPipeline(
        use_basic=False,
        use_birefnet=False,
        use_loftr=False,
        use_ecc=False,
    )


def _ts_path(frame_idx: int, t_ms: int) -> str:
    """Fake image path with embedded timestamp (enables velocity-based recovery)."""
    return f"/fake/frame{frame_idx:02d}_{t_ms}ms.png"


def _adj_edges(edges: list) -> list:
    return [e for e in edges if e["j"] == e["i"] + 1]


# ---------------------------------------------------------------------------
# 1. Clean edges pass through unchanged
# ---------------------------------------------------------------------------


class TestCleanEdgesPreserved:
    def test_uniform_dy_all_kept(self):
        pipeline = _make_pipeline()
        N = 5
        edges = [make_edge(i, i + 1, dy=300.0) for i in range(N - 1)]
        frames = [make_frame(480, 640) for _ in range(N)]
        masks = [None] * N
        paths = [f"/fake/frame{i}.png" for i in range(N)]

        result = pipeline._filter_edges(edges, paths, 480, 640, frames, masks)
        adj_result = _adj_edges(result)
        assert len(adj_result) == N - 1, (
            f"All clean adjacent edges should be preserved; kept {len(adj_result)}/{N - 1}"
        )

    def test_slightly_varying_dy_all_kept(self):
        """Edges within ±15% of median should not be flagged as outliers."""
        pipeline = _make_pipeline()
        N = 5
        dys = [290.0, 310.0, 295.0, 305.0]
        edges = [make_edge(i, i + 1, dy=dy) for i, dy in enumerate(dys)]
        frames = [make_frame(480, 640) for _ in range(N)]
        masks = [None] * N
        paths = [f"/fake/frame{i}.png" for i in range(N)]

        result = pipeline._filter_edges(edges, paths, 480, 640, frames, masks)
        adj_result = _adj_edges(result)
        assert len(adj_result) == len(dys)

    def test_two_edges_no_consensus_filter(self):
        """Less than 3 edges → direction consensus filter is not applied."""
        pipeline = _make_pipeline()
        edges = [make_edge(0, 1, dy=300.0), make_edge(1, 2, dy=300.0)]
        frames = [make_frame(480, 640) for _ in range(3)]
        masks = [None] * 3
        paths = [f"/fake/frame{i}.png" for i in range(3)]

        result = pipeline._filter_edges(edges, paths, 480, 640, frames, masks)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 2. Wrong-sign detection (test2, test7 pattern)
# ---------------------------------------------------------------------------


class TestWrongSignDetection:
    def test_wrong_sign_edge_replaced_with_velocity(self):
        """
        One edge with reversed dy (negative when majority is positive) should be
        replaced by velocity estimate when timestamp filenames are provided.
        """
        pipeline = _make_pipeline()
        t_ms = [0, 500, 1000, 1500, 2000]
        paths = [_ts_path(i, t_ms[i]) for i in range(5)]
        N = 5
        frames = [make_frame(480, 640) for _ in range(N)]
        masks = [None] * N

        consensus_dy = 300.0
        edges = [
            make_edge(0, 1, dy=consensus_dy),
            make_edge(1, 2, dy=-280.0),  # wrong sign
            make_edge(2, 3, dy=consensus_dy),
            make_edge(3, 4, dy=consensus_dy),
        ]
        result = pipeline._filter_edges(edges, paths, 480, 640, frames, masks)
        adj = _adj_edges(result)
        assert len(adj) == 4, f"Expected 4 adj edges after correction, got {len(adj)}"
        edge_1_2 = next((e for e in adj if e["i"] == 1 and e["j"] == 2), None)
        assert edge_1_2 is not None
        assert float(edge_1_2["M"][1, 2]) > 0, (
            f"Edge 1→2 dy={float(edge_1_2['M'][1, 2]):.1f} should be positive after correction"
        )

    def test_wrong_sign_without_timestamps_falls_back_to_median(self):
        """Without timestamps, wrong-sign edges fall back to median dy correction."""
        pipeline = _make_pipeline()
        N = 5
        frames = [make_frame(480, 640) for _ in range(N)]
        masks = [None] * N
        paths = [f"/fake/frame{i}.png" for i in range(N)]

        consensus_dy = 250.0
        edges = [
            make_edge(0, 1, dy=consensus_dy),
            make_edge(1, 2, dy=-200.0),  # wrong sign
            make_edge(2, 3, dy=consensus_dy),
            make_edge(3, 4, dy=consensus_dy),
        ]
        result = pipeline._filter_edges(edges, paths, 480, 640, frames, masks)
        adj = _adj_edges(result)
        assert len(adj) == 4
        edge_1_2 = next((e for e in adj if e["i"] == 1 and e["j"] == 2), None)
        assert edge_1_2 is not None
        assert float(edge_1_2["M"][1, 2]) > 0


# ---------------------------------------------------------------------------
# 3. Gross outlier detection
# ---------------------------------------------------------------------------


class TestGrossOutlierDetection:
    def test_gross_outlier_replaced_with_velocity(self):
        """Edge with dy >> 2× median should be replaced by velocity estimate."""
        pipeline = _make_pipeline()
        t_ms = [0, 500, 1000, 1500, 2000]
        paths = [_ts_path(i, t_ms[i]) for i in range(5)]
        N = 5
        frames = [make_frame(480, 640) for _ in range(N)]
        masks = [None] * N

        consensus_dy = 300.0
        edges = [
            make_edge(0, 1, dy=consensus_dy),
            make_edge(1, 2, dy=consensus_dy),
            make_edge(2, 3, dy=2000.0),  # gross outlier
            make_edge(3, 4, dy=consensus_dy),
        ]
        result = pipeline._filter_edges(edges, paths, 480, 640, frames, masks)
        adj = _adj_edges(result)
        assert len(adj) == 4

        edge_2_3 = next((e for e in adj if e["i"] == 2 and e["j"] == 3), None)
        assert edge_2_3 is not None
        corrected_dy = abs(float(edge_2_3["M"][1, 2]))
        assert corrected_dy < 1000.0, (
            f"Gross outlier edge dy={corrected_dy:.1f} should be corrected closer to median"
        )

    def test_gross_outlier_threshold_2x_median_plus_200(self):
        """
        Gross outlier condition: |dy| > 2×|median_dy| AND |dy - median_dy| > 200.
        An edge borderline over the 200px gap threshold is still an outlier.
        """
        pipeline = _make_pipeline()
        N = 5
        frames = [make_frame(480, 640) for _ in range(N)]
        masks = [None] * N
        paths = [f"/fake/frame{i}.png" for i in range(N)]

        median_dy = 200.0
        edges = [
            make_edge(0, 1, dy=median_dy),
            make_edge(1, 2, dy=median_dy),
            make_edge(2, 3, dy=median_dy * 2.1),
            make_edge(3, 4, dy=median_dy),
        ]
        result = pipeline._filter_edges(edges, paths, 480, 640, frames, masks)
        adj = _adj_edges(result)
        assert len(adj) >= 3, "At least 3 of 4 adjacent edges should survive"


# ---------------------------------------------------------------------------
# 4. Geometric consistency filter
# ---------------------------------------------------------------------------


class TestGeometricConsistency:
    def test_consistent_skip_pair_kept(self):
        """
        Skip-pair edge (0→2) whose dy matches the sum of adjacent edges (0→1 + 1→2)
        within 15px tolerance should be kept.
        """
        pipeline = _make_pipeline()
        N = 3
        frames = [make_frame(480, 640) for _ in range(N)]
        masks = [None] * N
        paths = [f"/fake/frame{i}.png" for i in range(N)]

        adj_dy = 300.0
        edges = [
            make_edge(0, 1, dy=adj_dy),
            make_edge(1, 2, dy=adj_dy),
            make_edge(0, 2, dy=adj_dy * 2),  # consistent: 0→2 ≈ 0→1 + 1→2
        ]
        result = pipeline._filter_edges(edges, paths, 480, 640, frames, masks)
        skip_edges = [e for e in result if e["i"] == 0 and e["j"] == 2]
        assert len(skip_edges) == 1, "Consistent skip-pair edge should be kept"

    def test_inconsistent_skip_pair_removed(self):
        """
        Skip-pair edge (0→2) with dy that disagrees with adj edges by > 15px
        should be rejected by the geometric consistency filter.
        """
        pipeline = _make_pipeline()
        N = 3
        frames = [make_frame(480, 640) for _ in range(N)]
        masks = [None] * N
        paths = [f"/fake/frame{i}.png" for i in range(N)]

        adj_dy = 300.0
        edges = [
            make_edge(0, 1, dy=adj_dy),
            make_edge(1, 2, dy=adj_dy),
            make_edge(0, 2, dy=adj_dy * 2 + 100),  # inconsistent: 100px off
        ]
        result = pipeline._filter_edges(edges, paths, 480, 640, frames, masks)
        skip_edges = [e for e in result if e["i"] == 0 and e["j"] == 2]
        assert len(skip_edges) == 0, (
            f"Inconsistent skip-pair edge should be removed; kept {len(skip_edges)}"
        )

    def test_skip_pair_without_adj_path_kept(self):
        """
        Skip-pair edge where intermediate adj edges are missing cannot be verified
        → kept by default.
        """
        pipeline = _make_pipeline()
        N = 4
        frames = [make_frame(480, 640) for _ in range(N)]
        masks = [None] * N
        paths = [f"/fake/frame{i}.png" for i in range(N)]

        edges = [
            make_edge(0, 1, dy=300.0),
            make_edge(2, 3, dy=300.0),
            make_edge(0, 2, dy=600.0),  # cannot verify (1→2 missing)
        ]
        result = pipeline._filter_edges(edges, paths, 480, 640, frames, masks)
        skip_edges = [e for e in result if e["i"] == 0 and e["j"] == 2]
        assert len(skip_edges) == 1, (
            "Unverifiable skip-pair edge should be kept by default"
        )


# ---------------------------------------------------------------------------
# 5. Near-zero dy pattern (test8, test9, test16, test21)
# ---------------------------------------------------------------------------


class TestNearZeroDyPattern:
    """
    Documents the current filter behavior for near-zero dy edges (frame clustering).

    When ALL adjacent edges have near-zero dy, the median_dy is also near-zero,
    so the direction consensus filter cannot distinguish good from bad edges.

    Priority-1 fix requires adding a min_step guard:
      reject edges with |dy| < min_expected_step (e.g., 50px)
    """

    def test_near_zero_edges_rejected_by_min_step(self):
        """
        When all edges have near-zero dy (<50px), the min-step guard rejects
        them before the consensus filter runs.  This is the fix for the
        test8/test9/test16 clustering failure mode.
        """
        pipeline = _make_pipeline()
        N = 5
        frames = [make_frame(480, 640) for _ in range(N)]
        masks = [None] * N
        paths = [f"/fake/frame{i}.png" for i in range(N)]

        edges = [make_edge(i, i + 1, dy=float(3 + i)) for i in range(N - 1)]
        result = pipeline._filter_edges(edges, paths, 480, 640, frames, masks)
        adj = _adj_edges(result)
        assert len(adj) == 0, (
            f"All near-zero edges should be rejected by min-step guard; "
            f"kept {len(adj)}/{N - 1}"
        )

    def test_mixed_near_zero_and_normal_preserves_normal(self):
        """
        3 near-zero edges + 1 normal-dy edge: the min-step guard rejects
        the near-zero edges, preserving the normal edge.  This is the fix
        for the inverted-consensus pattern (test8 failure mode).
        """
        pipeline = _make_pipeline()
        N = 5
        frames = [make_frame(480, 640) for _ in range(N)]
        masks = [None] * N
        t_ms = [0, 100, 200, 300, 400]
        paths = [_ts_path(i, t_ms[i]) for i in range(N)]

        edges = [
            make_edge(0, 1, dy=5.0),  # near-zero → rejected
            make_edge(1, 2, dy=3.0),  # near-zero → rejected
            make_edge(2, 3, dy=300.0),  # normal → preserved
            make_edge(3, 4, dy=8.0),  # near-zero → rejected
        ]
        result = pipeline._filter_edges(edges, paths, 480, 640, frames, masks)
        adj = _adj_edges(result)
        # Only edge 2→3 (dy=300) survives the min-step guard
        assert len(adj) == 1, f"Expected 1 surviving edge, got {len(adj)}"
        edge_2_3 = next((e for e in adj if e["i"] == 2 and e["j"] == 3), None)
        assert edge_2_3 is not None, "Edge 2→3 (dy=300) should survive min-step guard"
        assert abs(float(edge_2_3["M"][1, 2]) - 300.0) < 1.0, (
            f"Edge 2→3 dy should be ~300, got {float(edge_2_3['M'][1, 2]):.1f}"
        )


# ---------------------------------------------------------------------------
# _reject_static_edges — §1.2A pre-bundle static edge rejection (S32)
# ---------------------------------------------------------------------------


class TestRejectStaticEdges:
    """
    _reject_static_edges(edges, min_disp_px) drops edges where BOTH |dx| and
    |dy| are below the threshold.  This is distinct from the min-step guard in
    `_filter_edges`, which only checks the primary axis on adjacent edges.
    """

    def test_normal_edges_all_kept(self):
        """Edges with large dy always survive regardless of dx."""
        edges = [make_edge(i, i + 1, dy=300.0) for i in range(4)]
        result = _reject_static_edges(edges, min_disp_px=50.0)
        assert len(result) == 4

    def test_both_axes_below_threshold_rejected(self):
        """Edge where |dx|=10 AND |dy|=10 — both below 50px — is dropped."""
        edge = make_edge(0, 1, dx=10.0, dy=10.0)
        result = _reject_static_edges([edge], min_disp_px=50.0)
        assert len(result) == 0

    def test_one_axis_above_threshold_kept(self):
        """Edge where |dy|=10 but |dx|=80 (diagonal scroll) — kept because dx >= 50."""
        edge = make_edge(0, 1, dx=80.0, dy=10.0)
        result = _reject_static_edges([edge], min_disp_px=50.0)
        assert len(result) == 1

    def test_skip_edge_with_small_displacement_rejected(self):
        """Skip edge (j > i+1) with small 2D displacement is also rejected."""
        edge = make_edge(0, 3, dx=20.0, dy=30.0)
        result = _reject_static_edges([edge], min_disp_px=50.0)
        assert len(result) == 0

    def test_empty_edge_list(self):
        """Empty input → empty output."""
        result = _reject_static_edges([], min_disp_px=50.0)
        assert result == []


# ---------------------------------------------------------------------------
# _compute_adaptive_min_disp — §1.2C adaptive min-step threshold (S34)
# ---------------------------------------------------------------------------


class TestComputeAdaptiveMinDisp:
    """
    _compute_adaptive_min_disp(edges) returns
    max(STATIC_EDGE_MIN_DISP_PX, ADAPTIVE_MIN_DISP_FRAC * median_adjacent_step).
    The floor (STATIC_EDGE_MIN_DISP_PX=50) dominates for slow-scroll sequences;
    the adaptive value exceeds it for large-step / high-resolution sequences.
    """

    def test_floor_dominates_for_small_steps(self):
        """Steps of 30 px → expected=30, 10 %=3 → floor (50) returned."""
        edges = [make_edge(i, i + 1, dy=30.0) for i in range(5)]
        result = _compute_adaptive_min_disp(edges)
        assert result == pytest.approx(STATIC_EDGE_MIN_DISP_PX)

    def test_adaptive_exceeds_floor_for_large_steps(self):
        """Steps of 1 000 px → 10 %=100 → 100 returned (above floor 50)."""
        edges = [make_edge(i, i + 1, dy=1000.0) for i in range(4)]
        result = _compute_adaptive_min_disp(edges)
        assert result == pytest.approx(100.0)

    def test_empty_edges_returns_floor(self):
        """No edges at all → static floor returned."""
        assert _compute_adaptive_min_disp([]) == pytest.approx(STATIC_EDGE_MIN_DISP_PX)

    def test_dominant_axis_x_selected(self):
        """Horizontal-scroll sequence (dx >> dy) — x displacements drive the threshold."""
        edges = [make_edge(i, i + 1, dx=800.0, dy=5.0) for i in range(4)]
        result = _compute_adaptive_min_disp(edges)
        assert result == pytest.approx(80.0)

    def test_no_adjacent_edges_returns_floor(self):
        """Skip edges (j != i+1) are ignored; no adjacent pairs → floor."""
        edges = [make_edge(0, 3, dy=1000.0), make_edge(1, 4, dy=1000.0)]
        assert _compute_adaptive_min_disp(edges) == pytest.approx(
            STATIC_EDGE_MIN_DISP_PX
        )


# ---------------------------------------------------------------------------
# _triangular_consistency_filter — §2.14 triangular consistency filter (S93)
# ---------------------------------------------------------------------------


