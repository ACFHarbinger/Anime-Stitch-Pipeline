"""
Tests for pipeline.py module-level functions:
  §1.9A — _spatial_dedup_frames (scans_frames sync bug fix)
  §2.9C — _filter_high_conf_edges (high-confidence BA re-solve)
  §1.9C — _reload_scans_frames (on-demand SCANS frame reload)
  §1.13 — _reject_scene_change_edges (scene-change luma gate)
  §1.3C — _normalize_frame_scales (scale normalisation before BA)
  §4.7  — _compute_dy_cv (step-size CV gate for SCANS fallback)
  §5.8  — _compute_adaptive_dy_cv_max (adaptive dy_cv ceiling for large-N sequences)
"""

import cv2
import numpy as np
import pytest
from asp_backend.core.pipeline import (  # noqa: E402
    _apply_hires_keyframes,
    _check_edge_graph_connectivity,
    _compute_adaptive_dy_cv_max,
    _compute_dy_cv,
    _filter_high_conf_edges,
    _reload_scans_frames,
    _sort_frames_by_index,
    _spatial_dedup_frames,
    run_stage,
)
from asp_backend.core.pipeline.run_stage import _panorama_fallback_allowed, _refine_masks_for_plate
from backend.src.constants.animation import (  # noqa: E402
    HIGH_CONF_EDGE_THRESH,
)


def _make_affine(dy: float, dx: float = 0.0) -> np.ndarray:
    """2×3 affine matrix with given translation."""
    M = np.eye(2, 3, dtype=np.float32)
    M[0, 2] = dx
    M[1, 2] = dy
    return M


def _make_frame(h: int = 4, w: int = 4, val: int = 128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def _make_mask(h: int = 4, w: int = 4) -> np.ndarray:
    return np.ones((h, w), dtype=np.uint8)


def _make_edge(i: int, j: int, dy: float, dx: float = 0.0) -> dict:
    return {"i": i, "j": j, "M": _make_affine(dy=dy, dx=dx)}


class TestPanoramaFallbackSafety:
    def test_default_limit_rejects_large_sequences(self, monkeypatch):
        monkeypatch.delenv("ASP_PANORAMA_MAX_FRAMES", raising=False)
        assert _panorama_fallback_allowed(12)
        assert not _panorama_fallback_allowed(13)

    def test_limit_can_be_overridden_or_disabled(self, monkeypatch):
        monkeypatch.setenv("ASP_PANORAMA_MAX_FRAMES", "24")
        assert _panorama_fallback_allowed(22)
        monkeypatch.setenv("ASP_PANORAMA_MAX_FRAMES", "0")
        assert not _panorama_fallback_allowed(2)


class TestP4MaskRefinement:
    def test_is_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("ASP_MASK_UNCERTAINTY", raising=False)
        masks = [_make_mask(), _make_mask()]
        affines = [_make_affine(0), _make_affine(1)]

        refined, uncertain_px = _refine_masks_for_plate(
            [_make_frame(), _make_frame()], masks, affines
        )

        assert refined is masks
        assert uncertain_px == 0

    def test_refines_after_alignment_when_enabled(self, monkeypatch):
        monkeypatch.setenv("ASP_MASK_UNCERTAINTY", "1")
        masks = [_make_mask(), _make_mask()]
        refined_mask = _make_mask()
        refined_mask[1:3, 1:3] = 128
        monkeypatch.setattr(
            run_stage,
            "compute_temporal_mask_uncertainty",
            lambda frames, original_masks, affines: [refined_mask, masks[1]],
        )

        refined, uncertain_px = _refine_masks_for_plate(
            [_make_frame(), _make_frame()], masks, [_make_affine(0), _make_affine(1)]
        )

        assert refined[0] is refined_mask
        assert uncertain_px == 4


class TestSpatialDedupFrames:
    def test_no_drop_when_displacement_above_threshold(self):
        frames = [_make_frame(val=v) for v in [10, 20, 30]]
        scans = [_make_frame(val=v) for v in [10, 20, 30]]
        masks = [_make_mask() for _ in range(3)]
        paths = ["a.png", "b.png", "c.png"]
        edges = [_make_edge(0, 1, dy=50.0), _make_edge(1, 2, dy=50.0)]

        out_frames, out_scans, _, _, out_edges, n = _spatial_dedup_frames(
            frames, scans, masks, paths, edges, min_displacement_px=25.0
        )

        assert n == 0
        assert len(out_frames) == 3
        assert len(out_scans) == 3
        assert len(out_edges) == 2

    def test_drops_near_static_adjacent_frame(self):
        frames = [_make_frame(val=v) for v in [10, 20, 30]]
        scans = [_make_frame(val=v) for v in [10, 20, 30]]
        masks = [_make_mask() for _ in range(3)]
        paths = ["a.png", "b.png", "c.png"]
        # Frame 1 is near-static (dy=5px < threshold=25px)
        edges = [_make_edge(0, 1, dy=5.0), _make_edge(1, 2, dy=50.0)]

        out_frames, out_scans, _, out_paths, out_edges, n = _spatial_dedup_frames(
            frames, scans, masks, paths, edges, min_displacement_px=25.0
        )

        assert n == 1
        assert len(out_frames) == 2
        # Frame 1 (val=20) should be gone; frames 0 and 2 remain
        assert out_frames[0][0, 0, 0] == 10
        assert out_frames[1][0, 0, 0] == 30

    def test_scans_frames_synced_with_frames_after_drop(self):
        """§1.9A: scans_frames must have same length and entries as frames."""
        frames = [_make_frame(val=v) for v in [10, 20, 30]]
        scans = [_make_frame(val=v + 1) for v in [10, 20, 30]]  # offset to distinguish
        masks = [_make_mask() for _ in range(3)]
        paths = ["a.png", "b.png", "c.png"]
        edges = [_make_edge(0, 1, dy=5.0), _make_edge(1, 2, dy=50.0)]

        out_frames, out_scans, _, _, _, n = _spatial_dedup_frames(
            frames, scans, masks, paths, edges, min_displacement_px=25.0
        )

        assert n == 1
        assert len(out_scans) == len(out_frames) == 2
        # scans_frames[0] should correspond to original frame 0 (val+1=11)
        assert out_scans[0][0, 0, 0] == 11
        # scans_frames[1] should correspond to original frame 2 (val+1=31)
        assert out_scans[1][0, 0, 0] == 31

    def test_edges_reindexed_after_drop(self):
        """After frame 1 is dropped, edge i/j indices must be remapped."""
        frames = [_make_frame() for _ in range(4)]
        scans = [_make_frame() for _ in range(4)]
        masks = [_make_mask() for _ in range(4)]
        paths = ["a.png", "b.png", "c.png", "d.png"]
        # Frame 1 near-static; keep frames 0, 2, 3
        edges = [
            _make_edge(0, 1, dy=3.0),
            _make_edge(1, 2, dy=60.0),
            _make_edge(2, 3, dy=60.0),
        ]

        _, _, _, _, out_edges, n = _spatial_dedup_frames(
            frames, scans, masks, paths, edges, min_displacement_px=25.0
        )

        assert n == 1
        # Only the surviving edges (not involving dropped frame 1) should remain,
        # reindexed: old (2→3) becomes new (1→2)
        assert len(out_edges) == 1
        surviving = out_edges[0]
        assert surviving["i"] == 1
        assert surviving["j"] == 2

    def test_first_frame_never_dropped(self):
        """Frame 0 is an anchor in the adjacency chain and must never be dropped."""
        frames = [_make_frame(val=v) for v in [10, 20, 30]]
        scans = [_make_frame(val=v) for v in [10, 20, 30]]
        masks = [_make_mask() for _ in range(3)]
        paths = ["a.png", "b.png", "c.png"]
        # Even if frame 0→1 displacement is tiny, only frame 1 (j) can be dropped
        edges = [_make_edge(0, 1, dy=2.0), _make_edge(1, 2, dy=2.0)]

        out_frames, _, _, _, _, n = _spatial_dedup_frames(
            frames, scans, masks, paths, edges, min_displacement_px=25.0
        )

        # Frame 0 must survive; frame 1 is dropped; frame 2 has i=1 which is in
        # drop, so it is skipped by the guard and NOT dropped (it becomes orphaned
        # but kept since only j-targets of non-dropped sources are removed).
        assert out_frames[0][0, 0, 0] == 10
        assert n >= 1


# ---------------------------------------------------------------------------
# TestFilterHighConfEdges — §2.9C high-confidence BA re-solve (S37)
# ---------------------------------------------------------------------------


def _make_edge_with_weight(i: int, j: int, weight: float, dy: float = 50.0) -> dict:
    return {"i": i, "j": j, "M": _make_affine(dy=dy), "weight": weight}


class TestFilterHighConfEdges:
    """
    _filter_high_conf_edges keeps only edges whose ``weight`` meets or
    exceeds the HIGH_CONF_EDGE_THRESH floor (default 0.65).

    LoFTR edges (weight ~0.7–0.95) are kept; TM/PC fallback edges
    (weight ~0.15–0.55) are removed so that a single bad low-confidence
    edge cannot corrupt the bundle adjustment solution.
    """

    def test_high_weight_edges_kept(self):
        """Edges at or above threshold must survive the filter."""
        edges = [
            _make_edge_with_weight(0, 1, weight=0.90),
            _make_edge_with_weight(
                1, 2, weight=HIGH_CONF_EDGE_THRESH
            ),  # exactly at floor
            _make_edge_with_weight(2, 3, weight=0.75),
        ]
        result = _filter_high_conf_edges(edges)
        assert len(result) == 3

    def test_low_weight_edges_removed(self):
        """Edges below threshold must be filtered out."""
        edges = [
            _make_edge_with_weight(0, 1, weight=0.90),
            _make_edge_with_weight(1, 2, weight=0.40),  # TM fallback
            _make_edge_with_weight(0, 2, weight=0.15),  # PC fallback
        ]
        result = _filter_high_conf_edges(edges)
        assert len(result) == 1
        assert result[0]["i"] == 0 and result[0]["j"] == 1

    def test_empty_edges_returns_empty(self):
        """Empty input list must produce empty output without error."""
        assert _filter_high_conf_edges([]) == []

    def test_all_below_threshold_returns_empty(self):
        """When all edges are low-quality the result is empty (triggers fallthrough)."""
        edges = [
            _make_edge_with_weight(0, 1, weight=0.50),
            _make_edge_with_weight(1, 2, weight=0.30),
        ]
        result = _filter_high_conf_edges(edges)
        assert len(result) == 0

    def test_missing_weight_field_treated_as_zero(self):
        """Edges without a ``weight`` key must not raise and must be filtered out."""
        edges = [
            {"i": 0, "j": 1, "M": _make_affine(dy=50.0)},  # no weight key
            _make_edge_with_weight(1, 2, weight=0.80),
        ]
        result = _filter_high_conf_edges(edges)
        # weight defaults to 0.0 → below threshold → only the second edge survives
        assert len(result) == 1
        assert result[0]["i"] == 1 and result[0]["j"] == 2


# ---------------------------------------------------------------------------
# TestReloadScansFrames — §1.9C on-demand SCANS frame reload (S41)
# ---------------------------------------------------------------------------


def _write_rgb_png(path, h: int = 8, w: int = 8, val: int = 128) -> None:
    """Write a small solid-colour BGR image to *path*."""
    img = np.full((h, w, 3), val, dtype=np.uint8)
    cv2.imwrite(str(path), img)


class TestReloadScansFrames:
    """
    _reload_scans_frames(paths) reads frames from disk, skips unreadable
    files, and returns all surviving frames width-normalised to the first
    frame's width — exactly matching the Stage-1/2 pipeline output.
    """

    def test_returns_loaded_frames_for_valid_paths(self, tmp_path):
        """Two readable PNG files → two frames returned."""
        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        _write_rgb_png(p1, val=50)
        _write_rgb_png(p2, val=200)
        result = _reload_scans_frames([str(p1), str(p2)])
        assert len(result) == 2
        assert result[0].shape == result[1].shape  # widths normalised

    def test_empty_paths_returns_empty_list(self):
        """No input paths → empty list, no exception."""
        assert _reload_scans_frames([]) == []

    def test_unreadable_path_skipped(self, tmp_path):
        """Non-existent path is skipped; the one valid frame is returned."""
        p_good = tmp_path / "good.png"
        _write_rgb_png(p_good, val=100)
        result = _reload_scans_frames([str(p_good), "/nonexistent/ghost.png"])
        assert len(result) == 1

    def test_all_frames_normalised_to_first_width(self, tmp_path):
        """Frames with different widths are resized to match the first frame."""
        p1 = tmp_path / "wide.png"
        p2 = tmp_path / "narrow.png"
        cv2.imwrite(str(p1), np.zeros((8, 16, 3), dtype=np.uint8))
        cv2.imwrite(str(p2), np.zeros((8, 8, 3), dtype=np.uint8))
        result = _reload_scans_frames([str(p1), str(p2)])
        assert len(result) == 2
        assert result[0].shape[1] == result[1].shape[1]  # same width

    def test_all_unreadable_returns_empty(self, tmp_path):
        """When all paths fail to load, an empty list is returned without error."""
        result = _reload_scans_frames(["/no/such/file.png", "/another/bad.png"])
        assert result == []




# ---------------------------------------------------------------------------
# TestNormalizeFrameScales — §1.3C scale normalisation before BA (S54)
# ---------------------------------------------------------------------------


def _make_scale_affine(scale: float, dy: float = 50.0) -> np.ndarray:
    """2×3 similarity affine with given uniform scale and vertical translation."""
    M = np.eye(2, 3, dtype=np.float32)
    M[0, 0] = scale
    M[1, 1] = scale
    M[1, 2] = dy
    return M




# ---------------------------------------------------------------------------
# §1.13B — _reject_scene_change_edges with use_bgr=True
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §1.15 — _check_edge_graph_connectivity
# ---------------------------------------------------------------------------


class TestCheckEdgeGraphConnectivity:
    """§1.15 — Union-Find edge graph connectivity check before bundle adjustment."""

    @staticmethod
    def _edge(i: int, j: int) -> dict:
        return {"i": i, "j": j, "M": np.eye(2, 3, dtype=np.float32), "weight": 1.0}

    def test_chain_graph_is_connected(self):
        """0→1→2 chain covers all 3 frames → True."""
        edges = [self._edge(0, 1), self._edge(1, 2)]
        assert _check_edge_graph_connectivity(edges, n_frames=3) is True

    def test_isolated_frame_is_disconnected(self):
        """N=3, edge only 0→1 → frame 2 isolated → False."""
        edges = [self._edge(0, 1)]
        assert _check_edge_graph_connectivity(edges, n_frames=3) is False

    def test_single_frame_trivially_connected(self):
        """N=1 → trivially connected regardless of edges."""
        assert _check_edge_graph_connectivity([], n_frames=1) is True

    def test_complete_graph_is_connected(self):
        """All pairs connected (complete graph on N=4) → True."""
        edges = [
            self._edge(0, 1),
            self._edge(0, 2),
            self._edge(0, 3),
            self._edge(1, 2),
            self._edge(1, 3),
            self._edge(2, 3),
        ]
        assert _check_edge_graph_connectivity(edges, n_frames=4) is True

    def test_no_edges_multiple_frames_disconnected(self):
        """N=3, empty edge list → all frames isolated → False."""
        assert _check_edge_graph_connectivity([], n_frames=3) is False


# ---------------------------------------------------------------------------
# §1.16 — _compute_mst_weight (minimum spanning tree weight gate)
# ---------------------------------------------------------------------------




# §1.17 — _compute_canvas_span_utilization (canvas span utilisation gate)
# Tests cover: edge cases (N<2), perfect monotone sequence, collapsed BA
# (all frames same position), over-utilised (frames spread wider than median
# step × (N-1)), and dominant-axis selection (horizontal vs vertical).






# ── TestDetectStaticInput — §1.29 static input detection gate (S73) ──────────




# ---------------------------------------------------------------------------
# §S89 — _build_manual_edge
# ---------------------------------------------------------------------------


class TestBuildManualEdge:
    """Tests for _build_manual_edge (§S89 manual edge entry at HITL checkpoint 2)."""

    def setup_method(self):
        from asp_backend.core.pipeline import _build_manual_edge

        self._fn = _build_manual_edge

    def test_m_is_pure_translation(self):
        edge = self._fn(0, 1, dx=50.0, dy=100.0)
        M = edge["M"]
        assert M.shape == (2, 3)
        np.testing.assert_allclose(M[:, :2], np.eye(2), atol=1e-9)
        assert M[0, 2] == pytest.approx(50.0)
        assert M[1, 2] == pytest.approx(100.0)

    def test_indices_stored(self):
        edge = self._fn(2, 5, dx=0.0, dy=200.0)
        assert edge["i"] == 2
        assert edge["j"] == 5

    def test_method_is_manual(self):
        edge = self._fn(0, 1, dx=10.0, dy=20.0)
        assert edge["method"] == "manual"

    def test_weight_clipped_to_unit_interval(self):
        over = self._fn(0, 1, dx=0.0, dy=0.0, weight=2.5)
        assert over["weight"] == pytest.approx(1.0)
        under = self._fn(0, 1, dx=0.0, dy=0.0, weight=-0.5)
        assert under["weight"] == pytest.approx(0.0)

    def test_pts_shape_is_compatible_with_bundle_adjust(self):
        edge = self._fn(0, 1, dx=30.0, dy=60.0)
        assert edge["pts_i"].shape == (1, 2)
        assert edge["pts_j"].shape == (1, 2)
        np.testing.assert_allclose(
            edge["pts_j"] - edge["pts_i"], [[30.0, 60.0]], atol=1e-6
        )


# ---------------------------------------------------------------------------
# §1.37 — _compute_bg_coverage_fraction (S101)
# ---------------------------------------------------------------------------




# §1.39 — _compute_render_coverage (S103)


# §1.43 — _compute_adj_edge_coverage (S107)


# §1.44 — _compute_max_adjacent_gap (S108)


# §1.45 — _compute_canvas_width_ratio (S109)


# §1.47 — _compute_sign_inconsistency_rate (S111)


# §1.48 — _compute_adj_disp_cv (S112)


# §1.49 — _compute_adj_min_weight (S113)


# §1.50 — _compute_ba_max_residual (S114)


# §1.51 — _compute_min_adjacent_overlap (S115)


# §1.52 — _compute_ba_weighted_mean_residual (S116)


# §1.53 — _compute_canvas_memory_mb (S117)


# §1.54 — _compute_render_luma_std (S118)


# §9C — _apply_hires_keyframes (Sprint 8)
class TestApplyHiresKeyframes:
    """_apply_hires_keyframes replaces proxy frames with hires images and scales affines/masks."""

    @staticmethod
    def _make_frame(h: int, w: int, val: int = 128) -> np.ndarray:
        return np.full((h, w, 3), val, dtype=np.uint8)

    @staticmethod
    def _make_affine(tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
        M = np.eye(2, 3, dtype=np.float32)
        M[0, 2] = tx
        M[1, 2] = ty
        return M

    @staticmethod
    def _make_mask(h: int, w: int) -> np.ndarray:
        m = np.zeros((h, w), dtype=np.uint8)
        m[: h // 2, :] = 1
        return m

    def test_empty_dict_is_noop(self):
        """Empty hires_keyframes → returns original lists unchanged."""
        frames = [self._make_frame(100, 200)]
        affines = [self._make_affine(10.0, 5.0)]
        masks = [self._make_mask(100, 200)]

        n, f_out, a_out, m_out = _apply_hires_keyframes(frames, affines, masks, {})

        assert n == 0
        assert f_out is frames
        assert a_out is affines
        assert m_out is masks

    def test_scale_computation_and_frame_replacement(self, tmp_path):
        """Hires frame at 2× scale → affine tx/ty doubled, frame replaced."""
        proxy_h, proxy_w = 50, 100
        hires_h, hires_w = 100, 200  # 2× scale

        frames = [self._make_frame(proxy_h, proxy_w, val=50)]
        affines = [self._make_affine(tx=20.0, ty=10.0)]
        masks = [None]

        hires_img = self._make_frame(hires_h, hires_w, val=200)
        hires_path = tmp_path / "hires_0.png"
        cv2.imwrite(str(hires_path), hires_img)

        n, f_out, a_out, m_out = _apply_hires_keyframes(
            frames, affines, masks, {0: str(hires_path)}
        )

        assert n == 1
        assert f_out[0].shape == (hires_h, hires_w, 3)
        assert a_out[0][0, 2] == pytest.approx(40.0)  # tx * scale_x
        assert a_out[0][1, 2] == pytest.approx(20.0)  # ty * scale_y

    def test_proxy_upscale_fallback_for_non_hires_frames(self, tmp_path):
        """Frames without a hires path are upscaled to match the hires resolution."""
        proxy_h, proxy_w = 50, 100
        hires_h, hires_w = 100, 200

        frames = [
            self._make_frame(proxy_h, proxy_w, val=10),  # gets hires replacement
            self._make_frame(proxy_h, proxy_w, val=20),  # upscaled fallback
        ]
        affines = [self._make_affine(), self._make_affine()]
        masks = [None, None]

        hires_img = self._make_frame(hires_h, hires_w, val=200)
        hires_path = tmp_path / "hires_0.png"
        cv2.imwrite(str(hires_path), hires_img)

        n, f_out, a_out, m_out = _apply_hires_keyframes(
            frames, affines, masks, {0: str(hires_path)}
        )

        assert n == 1
        assert f_out[0].shape == (hires_h, hires_w, 3)
        assert f_out[1].shape == (hires_h, hires_w, 3)  # upscaled to match

    def test_bg_mask_resized_with_nearest_neighbor(self, tmp_path):
        """bg_masks are resized to hires resolution using INTER_NEAREST (binary-safe)."""
        proxy_h, proxy_w = 4, 8
        hires_h, hires_w = 8, 16

        mask = self._make_mask(proxy_h, proxy_w)  # binary, top-half=1
        frames = [self._make_frame(proxy_h, proxy_w)]
        affines = [self._make_affine()]
        masks = [mask]

        hires_img = self._make_frame(hires_h, hires_w)
        hires_path = tmp_path / "hires_0.png"
        cv2.imwrite(str(hires_path), hires_img)

        _, _, _, m_out = _apply_hires_keyframes(
            frames, affines, masks, {0: str(hires_path)}
        )

        assert m_out[0] is not None
        assert m_out[0].shape == (hires_h, hires_w)
        # binary values preserved (INTER_NEAREST)
        unique_vals = set(np.unique(m_out[0]))
        assert unique_vals <= {0, 1}

    def test_none_mask_stays_none(self, tmp_path):
        """None bg_mask entries pass through as None after hires substitution."""
        frames = [self._make_frame(50, 100)]
        affines = [self._make_affine()]
        masks = [None]

        hires_img = self._make_frame(100, 200)
        hires_path = tmp_path / "hires_0.png"
        cv2.imwrite(str(hires_path), hires_img)

        _, _, _, m_out = _apply_hires_keyframes(
            frames, affines, masks, {0: str(hires_path)}
        )

        assert m_out[0] is None

    def test_invalid_path_returns_zero_substitutions(self):
        """All paths unreadable → n=0, original lists returned unchanged."""
        frames = [self._make_frame(50, 100)]
        affines = [self._make_affine()]
        masks = [None]

        n, f_out, a_out, m_out = _apply_hires_keyframes(
            frames, affines, masks, {0: "/nonexistent/path/hires.png"}
        )

        assert n == 0
        assert f_out is frames
        assert a_out is affines
        assert m_out is masks

    def test_out_of_bounds_index_ignored(self, tmp_path):
        """Index outside frame range is silently skipped."""
        frames = [self._make_frame(50, 100)]
        affines = [self._make_affine()]
        masks = [None]

        hires_img = self._make_frame(100, 200)
        hires_path = tmp_path / "hires_99.png"
        cv2.imwrite(str(hires_path), hires_img)

        n, f_out, a_out, m_out = _apply_hires_keyframes(
            frames, affines, masks, {99: str(hires_path)}
        )

        assert n == 0

    def test_linear_submatrix_unchanged(self, tmp_path):
        """Affine rotation/shear components (2×2 sub-matrix) are NOT scaled."""
        proxy_h, proxy_w = 50, 100
        hires_h, hires_w = 100, 200

        # Affine with small rotation component
        M = np.array([[0.99, -0.14, 30.0], [0.14, 0.99, 15.0]], dtype=np.float32)
        frames = [self._make_frame(proxy_h, proxy_w)]
        affines = [M]
        masks = [None]

        hires_img = self._make_frame(hires_h, hires_w)
        hires_path = tmp_path / "hires_0.png"
        cv2.imwrite(str(hires_path), hires_img)

        _, _, a_out, _ = _apply_hires_keyframes(
            frames, affines, masks, {0: str(hires_path)}
        )

        # Linear sub-matrix must be exactly preserved
        np.testing.assert_array_almost_equal(a_out[0][:, :2], M[:, :2])
        # Translation scaled by 2×
        assert a_out[0][0, 2] == pytest.approx(60.0)
        assert a_out[0][1, 2] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# §1.55 — _compute_max_affine_rotation_deg (S120)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §3.16 — _smooth_affine_trajectory (S121)
# ---------------------------------------------------------------------------






class TestSortFramesByIndex:
    """§1.63 (S127): Numeric-suffix sort guard for frame paths."""

    def test_out_of_order_paths_sorted(self):
        paths = ["frame_003.png", "frame_001.png", "frame_002.png"]
        result = _sort_frames_by_index(paths)
        assert result == ["frame_001.png", "frame_002.png", "frame_003.png"]

    def test_already_sorted_unchanged(self):
        paths = ["img_01.jpg", "img_02.jpg", "img_03.jpg"]
        result = _sort_frames_by_index(paths)
        assert result == paths

    def test_zero_padded_sorted_numerically_not_lexicographically(self):
        paths = ["f_009.png", "f_010.png", "f_011.png"]
        result = _sort_frames_by_index(paths)
        assert result == ["f_009.png", "f_010.png", "f_011.png"]

    def test_paths_without_suffix_placed_last(self):
        paths = ["alpha.png", "frame_001.png", "frame_002.png"]
        result = _sort_frames_by_index(paths)
        assert result[0] == "frame_001.png"
        assert result[1] == "frame_002.png"
        assert result[2] == "alpha.png"

    def test_empty_list(self):
        assert _sort_frames_by_index([]) == []


# ---------------------------------------------------------------------------
# §1.67 — Frame canvas spread validation (S131)
# ---------------------------------------------------------------------------


def _make_spread_edge(i: int, j: int, ty: float, tx: float = 0.0) -> dict:
    return {"i": i, "j": j, "ty": ty, "tx": tx, "weight": 0.9}














# ===========================================================================
# Merged from test_pipeline_s160.py
# ===========================================================================







# ---------------------------------------------------------------------------
# TestSpatialDedupFramesBatchWiring — C++ batch dispatch (S46)
# ---------------------------------------------------------------------------


class TestSpatialDedupFramesBatchWiring:
    """Phase 5: _spatial_dedup_frames C++ dispatch via batch.frame_selection.

    These tests verify the M-affine → dx/dy conversion layer and the
    reconstruction of the 6-tuple return value from C++ keep indices.
    The C++ function uses cumulative-displacement greedy dedup; the
    tests exercise the same edge cases as TestSpatialDedupFrames.
    """

    def test_no_drop_returns_original_lists(self):
        """When no frames are dropped keep_idx covers all frames → return originals."""
        frames = [_make_frame(val=v) for v in [10, 20, 30]]
        scans = [_make_frame(val=v) for v in [10, 20, 30]]
        masks = [_make_mask() for _ in range(3)]
        paths = ["x0.png", "x1.png", "x2.png"]
        edges = [_make_edge(0, 1, dy=100.0), _make_edge(1, 2, dy=100.0)]

        out_f, out_s, _, _, out_e, n = _spatial_dedup_frames(
            frames, scans, masks, paths, edges, min_displacement_px=25.0
        )

        assert n == 0
        assert len(out_f) == 3
        assert len(out_s) == 3
        assert len(out_e) == 2

    def test_near_static_frame_dropped(self):
        """Frame 1 with dy=3 < 25 threshold must be dropped."""
        frames = [_make_frame(val=v) for v in [10, 20, 30]]
        scans = [_make_frame(val=v + 1) for v in [10, 20, 30]]
        masks = [_make_mask() for _ in range(3)]
        paths = ["y0.png", "y1.png", "y2.png"]
        edges = [_make_edge(0, 1, dy=3.0), _make_edge(1, 2, dy=80.0)]

        out_f, out_s, _, out_p, _, n = _spatial_dedup_frames(
            frames, scans, masks, paths, edges, min_displacement_px=25.0
        )

        assert n == 1
        assert len(out_f) == 2
        assert out_f[0][0, 0, 0] == 10  # frame 0 kept
        assert out_f[1][0, 0, 0] == 30  # frame 2 kept

    def test_scans_sync_preserved_after_drop(self):
        """scans_frames must be subset-synced with frames (§1.9A S28)."""
        frames = [_make_frame(val=v) for v in [10, 20, 30]]
        scans = [_make_frame(val=v + 5) for v in [10, 20, 30]]
        masks = [_make_mask() for _ in range(3)]
        paths = ["z0.png", "z1.png", "z2.png"]
        edges = [_make_edge(0, 1, dy=3.0), _make_edge(1, 2, dy=80.0)]

        _, out_s, _, _, _, n = _spatial_dedup_frames(
            frames, scans, masks, paths, edges, min_displacement_px=25.0
        )

        assert n == 1
        assert len(out_s) == 2
        assert out_s[0][0, 0, 0] == 15   # scans[0] = val+5=15
        assert out_s[1][0, 0, 0] == 35   # scans[2] = val+5=35

    def test_edge_reindex_correct_after_drop(self):
        """Surviving edges must have i/j remapped to the post-drop index space."""
        frames = [_make_frame() for _ in range(4)]
        scans = [_make_frame() for _ in range(4)]
        masks = [_make_mask() for _ in range(4)]
        paths = [f"w{i}.png" for i in range(4)]
        edges = [
            _make_edge(0, 1, dy=2.0),   # frame 1 near-static → dropped
            _make_edge(1, 2, dy=80.0),  # involves dropped frame → excluded
            _make_edge(2, 3, dy=80.0),  # both survive → remapped (1→2)
        ]

        _, _, _, _, out_e, n = _spatial_dedup_frames(
            frames, scans, masks, paths, edges, min_displacement_px=25.0
        )

        assert n == 1
        assert len(out_e) == 1
        assert out_e[0]["i"] == 1
        assert out_e[0]["j"] == 2

    def test_horizontal_displacement_respected(self):
        """dx-based displacement (horizontal scroll) is also respected."""
        frames = [_make_frame(val=v) for v in [10, 20, 30]]
        scans = [_make_frame(val=v) for v in [10, 20, 30]]
        masks = [_make_mask() for _ in range(3)]
        paths = ["h0.png", "h1.png", "h2.png"]
        # dx=3 horizontal, dy=0 — below threshold on both axes
        edges = [_make_edge(0, 1, dy=0.0, dx=3.0), _make_edge(1, 2, dy=0.0, dx=80.0)]

        _, _, _, _, _, n = _spatial_dedup_frames(
            frames, scans, masks, paths, edges, min_displacement_px=25.0
        )

        assert n == 1  # frame 1 dropped (ddx=3 < 25, ddy=0 < 25)


# ===========================================================================
# §4.7 — _compute_dy_cv
# ===========================================================================


class TestComputeDyCv:
    """§4.7: Coefficient of variation of adjacent vertical frame steps."""

    def test_zero_below_two_affines(self):
        # N < 2 → gate must not fire; return 0.0
        affines = [_make_affine(dy=0.0)]
        assert _compute_dy_cv(affines) == pytest.approx(0.0)

    def test_uniform_steps_zero_cv(self):
        # Identical steps → std=0 → CV=0.0
        affines = [_make_affine(dy=float(i * 100)) for i in range(5)]
        assert _compute_dy_cv(affines) == pytest.approx(0.0)

    def test_highly_irregular_steps_high_cv(self):
        # 9 small steps (5 px) + 1 large step (305 px) → CV ≈ 2.57 > 1.5 gate threshold
        tys = [5.0 * i for i in range(10)] + [350.0]
        affines = [_make_affine(dy=ty) for ty in tys]
        cv = _compute_dy_cv(affines)
        assert cv > 1.5

    def test_regular_steps_below_threshold(self):
        # Small random jitter on a regular scroll: CV ≈ 0.05 < 1.5 threshold
        base_step = 80.0
        tys = [base_step * i + (2.0 * (i % 2)) for i in range(6)]
        affines = [_make_affine(dy=ty) for ty in tys]
        cv = _compute_dy_cv(affines)
        assert cv < 1.5

    def test_near_zero_mean_returns_zero(self):
        # All steps ≈ 0 (static frames) — mean < 1.0 → returns 0.0 (guard)
        affines = [_make_affine(dy=0.1 * i) for i in range(4)]
        assert _compute_dy_cv(affines) == pytest.approx(0.0)


# ===========================================================================
# §5.8 — _compute_adaptive_dy_cv_max
# ===========================================================================


class TestAdaptiveDyCvMax:
    """§5.8: Adaptive dy_cv ceiling for large-N sequences."""

    def test_small_n_unchanged(self):
        # N < 8 → base_max returned unchanged (no scaling applied)
        assert _compute_adaptive_dy_cv_max(4, 1.5) == pytest.approx(1.5)
        assert _compute_adaptive_dy_cv_max(7, 1.5) == pytest.approx(1.5)

    def test_n_equals_8_unchanged(self):
        # N = 8 → scale factor = 8/8 = 1.0 → base_max unchanged
        assert _compute_adaptive_dy_cv_max(8, 1.5) == pytest.approx(1.5)

    def test_n_16_halved(self):
        # N = 16 → scale = 8/16 = 0.5 → 1.5 * 0.5 = 0.75, but floor=0.8 applies
        result = _compute_adaptive_dy_cv_max(16, 1.5)
        assert result == pytest.approx(0.8)

    def test_floor_enforced(self):
        # Very large N (100) → 1.5 * 8/100 = 0.12, clamped to floor 0.8
        result = _compute_adaptive_dy_cv_max(100, 1.5)
        assert result == pytest.approx(0.8)

    def test_custom_base_max(self):
        # N = 8, base = 2.0 → scale = 8/8 = 1.0 → returns 2.0 (above floor)
        assert _compute_adaptive_dy_cv_max(8, 2.0) == pytest.approx(2.0)


# ===========================================================================
# §5.38 Strip Saturation CV Gate — pipeline flag tests
# ===========================================================================



# ===========================================================================
# §5.39 — Pipeline Canvas Valid-Area Ratio Gate (Stage 11.35)
# ===========================================================================












# ===========================================================================
# §5.49: Stage 11.40 Strip Luma MAD Gate
# ===========================================================================




# ===========================================================================
# §5.50: Stage 11.41 Strip Sharpness CV Gate
# ===========================================================================




# ===========================================================================
# §5.53: Stage 11.42 Strip Contrast CV Gate
# ===========================================================================

























































