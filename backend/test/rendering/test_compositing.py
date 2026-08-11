"""
Tests for the Stage 11 Laplacian-blend composite (compositing.py).

Issue categories covered:
  A — Seam blending: feather zone, boundary search, Laplacian pyramid blend.
  §4.5 — Canvas-space DP seam composite (_canvas_dp_seam_composite).

All tests run without GPU — no BiRefNet or LoFTR dependencies.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest
from asp_backend.core import config
from asp_backend.rendering import compositing
from asp_backend.rendering.compositing import (  # noqa: E402
    _GC_FEATHER_PX,
    _GLOBAL_GAIN_COMP,
    _blocks_gain_compensate,
    _blocks_lum_compensate,
    _equalize_warped_gains,
    _overlap_graph_connected,
    _feather_gc_boundaries,
)

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

from asp_backend.rendering.compositing import (  # noqa: E402
    _adaptive_gain_clamp,
    _bg_gain_unclamped,
    _build_seam_cost_map,
    _coherence_skip_mask,
    _composite_foreground,
    _diff_to_feather,
    _gain_to_min_feather,
    _get_seam_cost_flags,
    _has_sufficient_bg,
    _make_seam_cache_key,
    _seam_color_match,
    _seam_cut,
    _single_pose_soft_edge,
    _soft_seam_weight,
)
from backend.src.constants import (  # noqa: E402
    FEATHER_MAX as _FEATHER_MAX,
)
from backend.src.constants import (  # noqa: E402
    FEATHER_MIN as _FEATHER_MIN,
)
from backend.src.constants import (  # noqa: E402
    FEATHER_TABLE as _FEATHER_TABLE,  # noqa: F401
)
from conftest import make_frame, make_translation_affine  # noqa: E402


class TestJointGainSafety:
    def test_filtered_overlap_graph_must_remain_connected(self):
        connected = [(0, 1, 10.0, 11.0, 200), (1, 2, 11.0, 12.0, 200), (2, 3, 12.0, 13.0, 200)]
        disconnected = connected[:1] + [(2, 3, 12.0, 13.0, 200)]
        assert _overlap_graph_connected(connected, 4)
        assert not _overlap_graph_connected(disconnected, 4)

    def test_single_frame_overlap_graph_is_safe(self):
        assert _overlap_graph_connected([], 1)

# ---------------------------------------------------------------------------
# 1. _diff_to_feather lookup table (Bug 1 regression)
# ---------------------------------------------------------------------------


class TestDiffToFeather:
    """
    The feather table maps frame-difference scores to blend half-widths.
    Low-diff boundaries get wide feathers (300px); high-diff boundaries get narrow (80px).
    """

    def test_identical_frames_get_max_feather(self):
        """diff ≤ 5.0 → FEATHER_MAX (300)."""
        assert _diff_to_feather(0.0) == 300
        assert _diff_to_feather(5.0) == 300

    def test_low_diff_gets_wide_feather(self):
        """diff = 7.5 (between 5 and 10) → 250."""
        assert _diff_to_feather(7.5) == 250

    def test_medium_diff_gets_medium_feather(self):
        """diff = 15.0 → 200."""
        assert _diff_to_feather(15.0) == 200

    def test_high_diff_gets_narrow_feather(self):
        """diff = 35.0 → 150; diff = 50.0 → 100."""
        assert _diff_to_feather(35.0) == 150
        assert _diff_to_feather(50.0) == 100

    def test_very_high_diff_gets_min_feather(self):
        """diff > 50 → FEATHER_MIN (80)."""
        assert _diff_to_feather(100.0) == _FEATHER_MIN
        assert _diff_to_feather(float("inf")) == _FEATHER_MIN

    def test_all_feather_values_in_valid_range(self):
        """All feather outputs must be ≥ FEATHER_MIN and ≤ FEATHER_MAX."""
        test_diffs = [0, 1, 5, 7, 10, 15, 20, 35, 50, 100, 1000]
        for d in test_diffs:
            f = _diff_to_feather(d)
            assert _FEATHER_MIN <= f <= _FEATHER_MAX, (
                f"_diff_to_feather({d}) = {f} outside [{_FEATHER_MIN}, {_FEATHER_MAX}]"
            )

    def test_feather_table_is_monotonically_decreasing(self):
        """Larger diff score should never produce a wider feather."""
        diffs = [0, 5, 10, 20, 35, 50, 100, 1000]
        feathers = [_diff_to_feather(d) for d in diffs]
        for i in range(len(feathers) - 1):
            assert feathers[i] >= feathers[i + 1], (
                f"Feather table not monotone at diff={diffs[i + 1]}: "
                f"{feathers[i]} → {feathers[i + 1]}"
            )


# ---------------------------------------------------------------------------
# 2. _composite_foreground output shape and stability
# ---------------------------------------------------------------------------


@pytest.mark.gc_heavy
class TestCompositeForeground:
    def _run_composite(self, n: int, H: int, W: int):
        frame_h = H // 2
        frames = [
            make_frame(frame_h, W, color=(int(c), int(c), int(c)))
            for c in np.linspace(60, 200, n, dtype=int)
        ]
        affines = [
            make_translation_affine(ty=i * float(frame_h) * 0.8) for i in range(n)
        ]
        canvas_h = int((n - 1) * frame_h * 0.8 + frame_h)
        canvas = np.zeros((canvas_h, W, 3), dtype=np.uint8)
        for _i, (f, aff) in enumerate(zip(frames, affines, strict=False)):
            wf = cv2.warpAffine(
                f,
                aff,
                (W, canvas_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            mask = wf.max(axis=2) > 0
            canvas[mask] = wf[mask]
        bg_masks = [None] * n
        return _composite_foreground(
            [], [], canvas, canvas_h, W, frames, affines, bg_masks
        )

    def test_output_shape_preserved(self):
        """Output canvas must retain its spatial dimensions and channel count."""
        result = self._run_composite(n=3, H=300, W=200)
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8

    def test_output_is_uint8(self):
        result = self._run_composite(n=4, H=400, W=150)
        assert result.dtype == np.uint8

    def test_output_values_in_range(self):
        result = self._run_composite(n=3, H=300, W=200)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_two_frames_minimal_case(self):
        """Minimal case: 2 frames → should run without error."""
        result = self._run_composite(n=2, H=200, W=100)
        assert result is not None
        assert result.ndim == 3

    def test_no_all_black_output(self):
        """Output should not be all zeros (compositing should place some pixels)."""
        result = self._run_composite(n=4, H=400, W=200)
        assert result.max() > 0, "Composite output is all black — no pixels were placed"

    def test_identity_no_crash_single_frame_height(self):
        """Frames whose height exactly divides the canvas should not crash."""
        frame_h, W = 100, 150
        frames = [
            make_frame(frame_h, W, color=(100, 100, 100)),
            make_frame(frame_h, W, color=(150, 150, 150)),
        ]
        affines = [
            make_translation_affine(ty=0.0),
            make_translation_affine(ty=float(frame_h)),
        ]
        canvas_h = frame_h * 2
        canvas = np.zeros((canvas_h, W, 3), dtype=np.uint8)
        for i, (f, aff) in enumerate(zip(frames, affines, strict=False)):
            wf = cv2.warpAffine(
                f,
                aff,
                (W, canvas_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            row_start = frame_h * i
            canvas[row_start : row_start + frame_h] = wf[
                row_start : row_start + frame_h
            ]
        bg_masks = [None, None]
        result = _composite_foreground(
            [], [], canvas, canvas_h, W, frames, affines, bg_masks
        )
        assert result is not None


# ---------------------------------------------------------------------------
# 3. _seam_cut — DP seam path shape and validity (S12 regression)
# ---------------------------------------------------------------------------


class TestSeamCutDP:
    """
    _seam_cut() takes two (h × W × 3) zone slices and returns a horizontal
    cut path of shape (W,) where each value is a y-offset in [0, h-1].
    """

    def _make_zone(self, h: int, W: int, base: int, noise: int = 10) -> np.ndarray:
        rng = np.random.default_rng(42)
        z = np.full((h, W, 3), base, dtype=np.uint8)
        z = np.clip(
            z.astype(np.int32) + rng.integers(-noise, noise + 1, z.shape), 0, 255
        )
        return z.astype(np.uint8)

    def test_output_shape_is_width(self):
        h, W = 40, 80
        fa = self._make_zone(h, W, 100)
        fb = self._make_zone(h, W, 150)
        import sys
        print("DEBUG: sys.path =", sys.path)
        print("DEBUG: BATCH_AVAILABLE =", compositing.BATCH_AVAILABLE)
        print("DEBUG: batch module =", compositing.batch)
        if compositing.batch is not None:
            print("DEBUG: batch file =", getattr(compositing.batch, "__file__", None))
            print("DEBUG: batch path =", getattr(compositing.batch, "__path__", None))
            print("DEBUG: batch dir =", dir(compositing.batch))
        path = _seam_cut(fa, fb)
        assert path.shape == (W,), f"Expected path shape ({W},), got {path.shape}"

    def test_path_values_in_valid_range(self):
        h, W = 30, 60
        fa = self._make_zone(h, W, 80)
        fb = self._make_zone(h, W, 180)
        path = _seam_cut(fa, fb)
        assert path.dtype == np.int32
        assert int(path.min()) >= 0
        assert int(path.max()) < h

    def test_identical_images_return_valid_path(self):
        """Path must be valid even when both frames are identical (zero energy everywhere)."""
        h, W = 20, 50
        fa = self._make_zone(h, W, 128)
        path = _seam_cut(fa, fa.copy())
        assert path.shape == (W,)
        assert int(path.min()) >= 0
        assert int(path.max()) < h

    def test_with_sem_cost_no_crash(self):
        """Providing a sem_cost map should not crash and must return valid path."""
        h, W = 24, 48
        fa = self._make_zone(h, W, 90)
        fb = self._make_zone(h, W, 170)
        sem = np.random.default_rng(7).random((h, W)).astype(np.float32)
        path = _seam_cut(fa, fb, sem_cost=sem)
        assert path.shape == (W,)
        assert int(path.min()) >= 0
        assert int(path.max()) < h

    def test_path_connectivity(self):
        """Adjacent path values must differ by at most 1 (DP 3-connectivity constraint)."""
        h, W = 40, 100
        fa = self._make_zone(h, W, 60)
        fb = self._make_zone(h, W, 200)
        path = _seam_cut(fa, fb)
        diffs = np.abs(np.diff(path.astype(np.int32)))
        assert int(diffs.max()) <= 1, (
            f"Path not 3-connected: max step={int(diffs.max())}"
        )


# ---------------------------------------------------------------------------
# 3b. _seam_cut — §2.11A Intelligent Scissors waypoint injection
# ---------------------------------------------------------------------------


class TestSeamCutWaypoints:
    """
    §2.11A: waypoints force the seam to pass through specific (x, y) pixels.
    The DP rows other than y_wp are set to +inf in column x_wp so the seam
    is guaranteed to land there; 3-connectivity is preserved end-to-end.
    """

    def _make_zone(self, h: int, W: int, base: int, noise: int = 5) -> np.ndarray:
        rng = np.random.default_rng(99)
        z = np.full((h, W, 3), base, dtype=np.uint8)
        z = np.clip(
            z.astype(np.int32) + rng.integers(-noise, noise + 1, z.shape), 0, 255
        )
        return z.astype(np.uint8)

    def test_single_waypoint_respected(self):
        """Seam must pass through the specified (x, y) waypoint exactly."""
        h, W = 40, 80
        fa = self._make_zone(h, W, 100)
        fb = self._make_zone(h, W, 150)
        x_wp, y_wp = 40, 10  # force through top-third row at mid-column
        path = _seam_cut(fa, fb, waypoints=[(x_wp, y_wp)])
        assert path[x_wp] == y_wp, (
            f"Seam did not pass through waypoint ({x_wp}, {y_wp}); got path[{x_wp}]={path[x_wp]}"
        )

    def test_multiple_waypoints_all_respected(self):
        """All waypoints must be honoured simultaneously."""
        h, W = 60, 100
        fa = self._make_zone(h, W, 80)
        fb = self._make_zone(h, W, 180)
        waypoints = [(20, 5), (50, 30), (80, 55)]
        path = _seam_cut(fa, fb, waypoints=waypoints)
        for x_wp, y_wp in waypoints:
            assert path[x_wp] == y_wp, (
                f"Waypoint ({x_wp}, {y_wp}) not respected; path[{x_wp}]={path[x_wp]}"
            )

    def test_waypoint_preserves_3connectivity(self):
        """Path with waypoints must still be 3-connected (|Δy| ≤ 1 per column)."""
        h, W = 50, 90
        fa = self._make_zone(h, W, 70)
        fb = self._make_zone(h, W, 190)
        waypoints = [(10, 5), (80, 40)]
        path = _seam_cut(fa, fb, waypoints=waypoints)
        diffs = np.abs(np.diff(path.astype(np.int32)))
        assert int(diffs.max()) <= 1, (
            f"3-connectivity broken with waypoints: max step={int(diffs.max())}"
        )

    def test_out_of_bounds_waypoint_ignored(self):
        """Waypoints with coordinates outside the zone must not crash and must be silently ignored."""
        h, W = 30, 60
        fa = self._make_zone(h, W, 120)
        fb = self._make_zone(h, W, 160)
        path = _seam_cut(fa, fb, waypoints=[(-1, 5), (W + 5, 10), (10, h + 2)])
        assert path.shape == (W,)
        assert int(path.min()) >= 0 and int(path.max()) < h

    def test_no_waypoints_matches_baseline(self):
        """Passing waypoints=None must produce the same path as omitting the argument."""
        h, W = 35, 70
        rng = np.random.default_rng(1)
        fa = rng.integers(50, 150, (h, W, 3), dtype=np.uint8)
        fb = rng.integers(100, 200, (h, W, 3), dtype=np.uint8)
        path_base = _seam_cut(fa, fb)
        path_none = _seam_cut(fa, fb, waypoints=None)
        path_empty = _seam_cut(fa, fb, waypoints=[])
        np.testing.assert_array_equal(path_base, path_none)
        np.testing.assert_array_equal(path_base, path_empty)


# ---------------------------------------------------------------------------
# 4. Parallel seam pre-computation (_precomp_paths) — S12 integration
# ---------------------------------------------------------------------------


@pytest.mark.gc_heavy
class TestParallelSeamPrecompute:
    """
    With ≥ 2 seam boundaries the composite should enter the ThreadPoolExecutor
    pre-computation path.  The result must be identical in shape and type to the
    sequential path.
    """

    def _run_composite_n(self, n: int, H: int = 500, W: int = 120) -> np.ndarray:
        frame_h = H // n
        frames = [
            make_frame(frame_h, W, color=(int(c), int(c), int(c)))
            for c in np.linspace(60, 200, n, dtype=int)
        ]
        affines = [
            make_translation_affine(ty=i * float(frame_h) * 0.9) for i in range(n)
        ]
        canvas_h = int((n - 1) * frame_h * 0.9 + frame_h)
        canvas = np.zeros((canvas_h, W, 3), dtype=np.uint8)
        for _i, (f, aff) in enumerate(zip(frames, affines, strict=False)):
            wf = cv2.warpAffine(
                f,
                aff,
                (W, canvas_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            mask = wf.max(axis=2) > 0
            canvas[mask] = wf[mask]
        return _composite_foreground(
            [], [], canvas, canvas_h, W, frames, affines, [None] * n
        )

    def test_five_frames_parallel_path_completes(self):
        """5 frames → 4 boundaries → parallel ThreadPoolExecutor path; must not raise."""
        result = self._run_composite_n(5)
        assert result is not None
        assert result.ndim == 3
        assert result.dtype == np.uint8

    def test_six_frames_output_shape_matches_canvas(self):
        """Output spatial dimensions must match the input canvas for 6-frame parallel run."""
        H, W, n = 600, 100, 6
        frame_h = H // n
        frames = [make_frame(frame_h, W, color=(100, 100, 100))] * n
        affines = [
            make_translation_affine(ty=i * float(frame_h) * 0.9) for i in range(n)
        ]
        canvas_h = int((n - 1) * frame_h * 0.9 + frame_h)
        canvas = np.zeros((canvas_h, W, 3), dtype=np.uint8)
        result = _composite_foreground(
            [], [], canvas, canvas_h, W, frames, affines, [None] * n
        )
        assert result.shape[1] == W
        assert result.shape[2] == 3

    def test_two_frames_single_seam_no_parallel(self):
        """2 frames → 1 boundary → single-seam fallback path (not ThreadPoolExecutor)."""
        result = self._run_composite_n(2)
        assert result is not None
        assert result.ndim == 3
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# 5. Single-pose soft-edge blend (_single_pose_soft_edge) — S15
# ---------------------------------------------------------------------------


class TestSinglePoseSoftEdge:
    """
    _single_pose_soft_edge applies a narrow ±sp_soft_px linear feather at the
    seam path to smooth the hard color step of single-pose seams.
    """

    def _zone(self, H: int, W: int, lum: int) -> np.ndarray:
        return np.full((H, W, 3), lum, dtype=np.uint8)

    def _path(self, W: int, y: int) -> np.ndarray:
        return np.full(W, y, dtype=np.int32)

    def _apply(self, H: int, W: int) -> np.ndarray:
        return np.ones((H, W), dtype=bool)

    def test_output_shape_and_dtype(self):
        """Output must match dom_zone shape and be uint8."""
        H, W = 40, 60
        dom = self._zone(H, W, 100)
        oth = self._zone(H, W, 200)
        path = self._path(W, H // 2)
        out = _single_pose_soft_edge(dom, oth, path, self._apply(H, W), 6)
        assert out.shape == dom.shape
        assert out.dtype == np.uint8

    def test_disabled_when_sp_soft_px_zero(self):
        """sp_soft_px=0 must return an unmodified copy of dom_zone."""
        H, W = 30, 50
        dom = self._zone(H, W, 80)
        oth = self._zone(H, W, 180)
        path = self._path(W, H // 2)
        out = _single_pose_soft_edge(dom, oth, path, self._apply(H, W), 0)
        np.testing.assert_array_equal(out, dom)

    def test_seam_row_is_blended_50_50(self):
        """Row exactly at the seam path gets a 50/50 blend (w_oth=0.5)."""
        H, W, seam_y = 40, 60, 20
        dom = self._zone(H, W, 100)
        oth = self._zone(H, W, 200)
        path = self._path(W, seam_y)
        out = _single_pose_soft_edge(dom, oth, path, self._apply(H, W), 8)
        expected = np.uint8(0.5 * 100 + 0.5 * 200)  # =150
        assert int(out[seam_y, 0, 0]) == expected

    def test_outside_band_pixels_are_unchanged(self):
        """Rows more than sp_soft_px away from the seam must equal dom_zone."""
        H, W, seam_y, soft_px = 80, 60, 40, 6
        dom = self._zone(H, W, 80)
        oth = self._zone(H, W, 180)
        path = self._path(W, seam_y)
        out = _single_pose_soft_edge(dom, oth, path, self._apply(H, W), soft_px)
        for r in [0, seam_y - soft_px, seam_y + soft_px, H - 1]:
            if 0 <= r < H:
                np.testing.assert_array_equal(out[r], dom[r])

    def test_in_band_pixels_are_strictly_between_dom_and_oth(self):
        """Within the blend band the output must be strictly between dom and oth."""
        H, W, seam_y, soft_px = 60, 50, 30, 6
        dom_lum, oth_lum = 60, 180
        dom = self._zone(H, W, dom_lum)
        oth = self._zone(H, W, oth_lum)
        path = self._path(W, seam_y)
        out = _single_pose_soft_edge(dom, oth, path, self._apply(H, W), soft_px)
        for r in range(seam_y - soft_px + 1, seam_y + soft_px):
            if 0 <= r < H:
                val = int(out[r, 0, 0])
                assert dom_lum <= val <= oth_lum, (
                    f"Row {r}: expected [{dom_lum}, {oth_lum}], got {val}"
                )

    def test_no_modification_where_apply_mask_is_false(self):
        """Pixels where apply_mask is False must not be modified."""
        H, W, seam_y = 40, 60, 20
        dom = self._zone(H, W, 100)
        oth = self._zone(H, W, 200)
        path = self._path(W, seam_y)
        apply = np.zeros((H, W), dtype=bool)  # nothing applied
        out = _single_pose_soft_edge(dom, oth, path, apply, 6)
        np.testing.assert_array_equal(out, dom)

    def test_no_modification_where_oth_has_no_content(self):
        """Blend only fires where oth_zone is non-zero; zero oth → dom unchanged."""
        H, W, seam_y = 40, 60, 20
        dom = self._zone(H, W, 100)
        oth = np.zeros((H, W, 3), dtype=np.uint8)  # all black (no content)
        path = self._path(W, seam_y)
        out = _single_pose_soft_edge(dom, oth, path, self._apply(H, W), 6)
        np.testing.assert_array_equal(out, dom)


# ---------------------------------------------------------------------------
# 6. Seam color match (_seam_color_match) — S16
# ---------------------------------------------------------------------------


class TestSeamColorMatch:
    """
    _seam_color_match shifts oth_zone channel means to match dom_zone in the
    seam band, reducing the color step before the S15 linear blend is applied.
    """

    def _zone(self, H: int, W: int, lum: int) -> np.ndarray:
        return np.full((H, W, 3), lum, dtype=np.uint8)

    def _path(self, W: int, y: int) -> np.ndarray:
        return np.full(W, y, dtype=np.int32)

    def test_output_shape_and_dtype(self):
        """Output must match oth_zone shape and be uint8."""
        H, W = 40, 60
        dom = self._zone(H, W, 100)
        oth = self._zone(H, W, 200)
        path = self._path(W, H // 2)
        out = _seam_color_match(dom, oth, path, 10)
        assert out.shape == oth.shape
        assert out.dtype == np.uint8

    def test_zero_band_returns_unchanged_copy(self):
        """band_px=0 must return an unmodified copy of oth_zone."""
        H, W = 30, 50
        dom = self._zone(H, W, 80)
        oth = self._zone(H, W, 180)
        path = self._path(W, H // 2)
        out = _seam_color_match(dom, oth, path, 0)
        np.testing.assert_array_equal(out, oth)

    def test_band_pixels_shifted_toward_dom_mean(self):
        """Band pixels in oth_zone must move toward dom_zone's mean."""
        H, W, seam_y, band_px = 60, 50, 30, 10
        dom_lum, oth_lum = 100, 200
        dom = self._zone(H, W, dom_lum)
        oth = self._zone(H, W, oth_lum)
        path = self._path(W, seam_y)
        out = _seam_color_match(dom, oth, path, band_px)
        # In the band, oth mean (200) shifts to dom mean (100) → delta=-100.
        # Resulting band pixels should be clipped to dom_lum (or close).
        seam_row = out[seam_y, 0, 0]
        assert int(seam_row) == dom_lum, (
            f"Band pixel should equal dom_lum={dom_lum}, got {seam_row}"
        )

    def test_outside_band_pixels_are_unchanged(self):
        """Rows beyond band_px from the seam must not be modified."""
        H, W, seam_y, band_px = 80, 60, 40, 8
        dom = self._zone(H, W, 80)
        oth = self._zone(H, W, 180)
        path = self._path(W, seam_y)
        out = _seam_color_match(dom, oth, path, band_px)
        for r in [0, seam_y - band_px, seam_y + band_px, H - 1]:
            if 0 <= r < H:
                np.testing.assert_array_equal(out[r], oth[r])

    def test_identical_zones_produce_no_shift(self):
        """When dom and oth have the same mean, delta=0 and output equals oth."""
        H, W = 40, 60
        lum = 128
        zone = self._zone(H, W, lum)
        path = self._path(W, H // 2)
        out = _seam_color_match(zone, zone.copy(), path, 10)
        np.testing.assert_array_equal(out, zone)

    def test_degenerate_zone_returns_unchanged(self):
        """When oth_zone is all-black in the band (no content), return copy unchanged."""
        H, W, seam_y = 40, 60, 20
        dom = self._zone(H, W, 100)
        oth = np.zeros((H, W, 3), dtype=np.uint8)
        path = self._path(W, seam_y)
        out = _seam_color_match(dom, oth, path, 10)
        np.testing.assert_array_equal(out, oth)

    def test_per_channel_delta_applied_independently(self):
        """Each BGR channel gets its own shift — channels are not mixed."""
        H, W, seam_y, band_px = 40, 60, 20, 8
        dom = np.zeros((H, W, 3), dtype=np.uint8)
        dom[:, :, 0] = 50  # B=50, G=0, R=0
        dom[:, :, 1] = 100  # B=50, G=100, R=0
        dom[:, :, 2] = 150  # B=50, G=100, R=150
        oth = np.full((H, W, 3), 200, dtype=np.uint8)  # B=G=R=200
        path = self._path(W, seam_y)
        out = _seam_color_match(dom, oth, path, band_px)
        # delta per channel: B=50-200=-150, G=100-200=-100, R=150-200=-50
        # band pixel after shift: B=clip(200-150,0,255)=50, G=100, R=150
        band_px_val = out[seam_y, 0]
        assert int(band_px_val[0]) == 50, (
            f"B channel: expected 50, got {band_px_val[0]}"
        )
        assert int(band_px_val[1]) == 100, (
            f"G channel: expected 100, got {band_px_val[1]}"
        )
        assert int(band_px_val[2]) == 150, (
            f"R channel: expected 150, got {band_px_val[2]}"
        )


# ---------------------------------------------------------------------------
# 7. Per-pixel DSFN seam weight (_soft_seam_weight) — S17
# ---------------------------------------------------------------------------


class TestSoftSeamWeight:
    """
    _soft_seam_weight (S17) returns a per-pixel blend weight in [0,1]:
      1.0 → fa_zone, 0.0 → fb_zone.
    The ramp radius is now per-pixel (driven by local similarity) rather than
    per-column mean, giving background pixels wide transitions and foreground
    pixels narrow ones automatically.
    """

    def _uniform(self, H: int, W: int, lum: int) -> np.ndarray:
        return np.full((H, W, 3), lum, dtype=np.uint8)

    def _path(self, W: int, y: int) -> np.ndarray:
        return np.full(W, y, dtype=np.int32)

    def test_output_shape_and_dtype(self):
        """Output must be (zone_h, W) float32."""
        H, W, seam_y = 60, 80, 30
        fa = self._uniform(H, W, 100)
        fb = self._uniform(H, W, 200)
        path = self._path(W, seam_y)
        out = _soft_seam_weight(fa, fb, path, H, W)
        assert out.shape == (H, W)
        assert out.dtype == np.float32

    def test_weight_in_unit_range(self):
        """All output values must be in [0, 1]."""
        H, W, seam_y = 50, 60, 25
        fa = self._uniform(H, W, 80)
        fb = self._uniform(H, W, 180)
        path = self._path(W, seam_y)
        out = _soft_seam_weight(fa, fb, path, H, W)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0 + 1e-5

    def test_weight_half_at_seam_for_identical_frames(self):
        """When frames are identical (zero diff), weight at seam row ≈ 0.5."""
        H, W, seam_y = 60, 80, 30
        fa = self._uniform(H, W, 128)
        path = self._path(W, seam_y)
        out = _soft_seam_weight(fa, fa.copy(), path, H, W)
        seam_weights = out[seam_y, :]
        assert np.allclose(seam_weights, 0.5, atol=0.05), (
            f"Expected ~0.5 at seam for identical frames, got {seam_weights.mean():.3f}"
        )

    def test_weight_one_far_above_seam(self):
        """Far above the seam, weight should be ~1.0 (all fa)."""
        H, W, seam_y = 100, 80, 50
        fa = self._uniform(H, W, 128)
        fb = self._uniform(H, W, 50)
        path = self._path(W, seam_y)
        out = _soft_seam_weight(fa, fb, path, H, W)
        # Row 0 is far above seam_y=50 → weight should be 1.0
        assert float(out[0].mean()) > 0.95, (
            f"Top row weight: {float(out[0].mean()):.3f}"
        )

    def test_weight_zero_far_below_seam(self):
        """Far below the seam, weight should be ~0.0 (all fb)."""
        H, W, seam_y = 100, 80, 50
        fa = self._uniform(H, W, 128)
        fb = self._uniform(H, W, 50)
        path = self._path(W, seam_y)
        out = _soft_seam_weight(fa, fb, path, H, W)
        # Last row is far below seam_y=50 → weight should be ~0.0
        assert float(out[-1].mean()) < 0.05, (
            f"Bottom row weight: {float(out[-1].mean()):.3f}"
        )

    def test_high_similarity_gives_wider_blend_than_low(self):
        """Identical frames (high sim) must produce a wider blend than very different frames (low sim)."""
        H, W, seam_y = 80, 60, 40
        fa_similar = self._uniform(H, W, 128)
        fb_similar = self._uniform(H, W, 130)  # Δ=2 → very similar

        fa_diff = self._uniform(H, W, 50)
        fb_diff = self._uniform(H, W, 200)  # Δ=150 → very different

        path = self._path(W, seam_y)

        out_sim = _soft_seam_weight(fa_similar, fb_similar, path, H, W)
        out_diff = _soft_seam_weight(fa_diff, fb_diff, path, H, W)

        # Count rows where weight is strictly between 0.05 and 0.95 (the transition band).
        trans_sim = int(((out_sim > 0.05) & (out_sim < 0.95)).all(axis=1).sum())
        trans_diff = int(((out_diff > 0.05) & (out_diff < 0.95)).all(axis=1).sum())

        assert trans_sim > trans_diff, (
            f"Similar frames should have wider blend zone: sim={trans_sim} rows, diff={trans_diff} rows"
        )

    def test_bg_mask_fg_fg_narrows_blend(self):
        """S20: all-fg bg_masks must narrow the blend compared to no-mask (same frames)."""
        H, W, seam_y = 80, 60, 40
        # Very similar frames → wide blend without mask
        fa = self._uniform(H, W, 128)
        fb = self._uniform(H, W, 130)
        path = self._path(W, seam_y)

        bg_all_fg = np.zeros((H, W), dtype=bool)  # all pixels = foreground (bg=False)
        out_no_mask = _soft_seam_weight(fa, fb, path, H, W)
        out_fg_mask = _soft_seam_weight(
            fa, fb, path, H, W, bg_mask_a=bg_all_fg, bg_mask_b=bg_all_fg
        )

        trans_no = int(((out_no_mask > 0.05) & (out_no_mask < 0.95)).all(axis=1).sum())
        trans_fg = int(((out_fg_mask > 0.05) & (out_fg_mask < 0.95)).all(axis=1).sum())
        assert trans_fg < trans_no, (
            f"fg mask should narrow blend: no_mask={trans_no}, fg_mask={trans_fg} transition rows"
        )

    def test_bg_mask_none_result_unchanged(self):
        """Passing None bg_masks must give the same result as calling without them."""
        H, W, seam_y = 60, 40, 30
        fa = self._uniform(H, W, 100)
        fb = self._uniform(H, W, 120)
        path = self._path(W, seam_y)
        out_default = _soft_seam_weight(fa, fb, path, H, W)
        out_none = _soft_seam_weight(fa, fb, path, H, W, bg_mask_a=None, bg_mask_b=None)
        np.testing.assert_array_equal(out_default, out_none)


# ---------------------------------------------------------------------------
# 8. _adaptive_gain_clamp  (S18 §1.4A)
# ---------------------------------------------------------------------------


class TestAdaptiveGainClamp:
    """_adaptive_gain_clamp: §1.4B continuous formula (S24)."""

    @staticmethod
    def _lo(ref: float) -> float:
        return 1.0 - (0.26 - 0.12 * (ref / 255.0))

    @staticmethod
    def _hi(ref: float) -> float:
        return 1.0 + (0.26 - 0.12 * (ref / 255.0))

    def test_normal_scene_clamps_large_downward_gain(self):
        # ref=100, frame=200 → raw gain=0.5, below continuous lo(100)
        result = _adaptive_gain_clamp(100.0, 200.0)
        assert result == pytest.approx(self._lo(100.0), abs=1e-5)

    def test_dark_scene_uses_wider_lo_clamp(self):
        # ref=50, frame=200 → raw gain=0.25, below continuous lo(50)
        result = _adaptive_gain_clamp(50.0, 200.0)
        assert result == pytest.approx(self._lo(50.0), abs=1e-5)

    def test_small_correction_passes_unclamped(self):
        # ref=100, frame=95 → raw gain≈1.053 — within continuous [lo, hi] → unclamped
        result = _adaptive_gain_clamp(100.0, 95.0)
        assert result == pytest.approx(100.0 / 95.0, rel=1e-5)

    def test_continuous_no_jump_at_ref_80(self):
        # §1.4B removes the S18 discontinuity at ref=80; neighboring values differ by < 0.001
        f_below = _adaptive_gain_clamp(79.9, 300.0)
        f_above = _adaptive_gain_clamp(80.0, 300.0)
        assert abs(f_below - f_above) < 0.001

    def test_zero_frame_lum_clamped_to_hi(self):
        # frame_lum=0 → gain=100/1=100 → clamped to continuous hi(100)
        result = _adaptive_gain_clamp(100.0, 0.0)
        assert result == pytest.approx(self._hi(100.0), abs=1e-5)

    def test_bright_ref_hi_matches_anchor(self):
        # At ref=255 the formula gives exactly hi=1.14 (the bright anchor)
        result = _adaptive_gain_clamp(255.0, 0.0)
        assert result == pytest.approx(1.14, abs=1e-6)

    def test_clamp_width_monotone_decreasing(self):
        # Darker scenes allow wider correction: lo(50) < lo(200)
        lo_dark = _adaptive_gain_clamp(50.0, 10000.0)  # forced to lo
        lo_bright = _adaptive_gain_clamp(200.0, 10000.0)
        assert lo_dark < lo_bright

    def test_mid_ref_continuous_formula(self):
        # ref=128 → exact continuous lo value
        result = _adaptive_gain_clamp(128.0, 10000.0)
        assert result == pytest.approx(self._lo(128.0), abs=1e-5)


# ---------------------------------------------------------------------------
# 9. _coherence_skip_mask  (S18 per-pair coherence gate)
# ---------------------------------------------------------------------------


class TestCoherenceSkipMask:
    """_coherence_skip_mask marks only the frames in bad adjacent pairs."""

    def test_all_small_diffs_none_skipped(self):
        order = np.array([0, 1, 2])
        lums = [100.0, 105.0, 110.0]
        result = _coherence_skip_mask(order, lums, 20.0)
        assert result == [False, False, False]

    def test_bad_pair_both_frames_skipped(self):
        # Pair (0,1) has diff=30 > 20; pair (1,2) is fine.
        order = np.array([0, 1, 2])
        lums = [100.0, 130.0, 135.0]
        result = _coherence_skip_mask(order, lums, 20.0)
        assert result[0] is True  # frame 0 in bad pair
        assert result[1] is True  # frame 1 in bad pair
        assert result[2] is False  # frame 2 unaffected

    def test_good_frames_after_bad_pair_not_skipped(self):
        # Pair (0,1) bad; pairs (1,2) and (2,3) fine.
        order = np.array([0, 1, 2, 3])
        lums = [100.0, 130.0, 128.0, 132.0]
        result = _coherence_skip_mask(order, lums, 20.0)
        assert result[0] is True
        assert result[1] is True
        assert result[2] is False
        assert result[3] is False

    def test_none_lum_pair_ignored(self):
        order = np.array([0, 1, 2])
        lums: list[float | None] = [100.0, None, 110.0]
        result = _coherence_skip_mask(order, lums, 20.0)
        assert result == [False, False, False]

    def test_exactly_at_limit_not_skipped(self):
        # diff == limit is NOT > limit → no skip
        order = np.array([0, 1])
        lums = [100.0, 120.0]
        result = _coherence_skip_mask(order, lums, 20.0)
        assert result == [False, False]

    def test_non_identity_order_maps_correctly(self):
        # Canvas order: frame2, frame0, frame1.  All pairs OK.
        order = np.array([2, 0, 1])
        lums = [105.0, 110.0, 100.0]  # lum[0]=105, lum[1]=110, lum[2]=100
        # lum_by_order = [100, 105, 110] → diffs 5,5 → all OK
        result = _coherence_skip_mask(order, lums, 20.0)
        assert result == [False, False, False]


# ---------------------------------------------------------------------------
# 10. _build_seam_cost_map  (S19 §1.6A tiered cost)
# ---------------------------------------------------------------------------


class TestSeamCostMap:
    """_build_seam_cost_map returns a tiered cost: interior=1.0, edge buffer=0.5, background=0.0."""

    # canvas_zone is only used for shape — pass a zero array.
    def _canvas(self, H: int, W: int) -> np.ndarray:
        return np.zeros((H, W, 3), dtype=np.uint8)

    def _bg_mask(self, H: int, W: int, fg_rows: int) -> np.ndarray:
        """255=background, 0=foreground for first fg_rows rows."""
        m = np.full((H, W), 255, dtype=np.uint8)
        m[:fg_rows] = 0
        return m

    def test_all_background_is_zero(self):
        H, W = 60, 40
        bm = np.full((H, W), 255, dtype=np.uint8)
        cost = _build_seam_cost_map(self._canvas(H, W), bm, None, dilate_px=5)
        assert cost.shape == (H, W)
        assert float(cost.max()) == pytest.approx(0.0)

    def test_fg_interior_is_1(self):
        # All-foreground mask → every pixel costs 1.0.
        H, W = 40, 20
        bm = np.zeros((H, W), dtype=np.uint8)  # bm=0 everywhere → all fg
        cost = _build_seam_cost_map(self._canvas(H, W), bm, None, dilate_px=5)
        assert float(cost.min()) == pytest.approx(1.0)

    def test_edge_buffer_is_0_5(self):
        # Rows 0–19 fg, rows 20–49 bg.  With dilate_px=5 the edge buffer
        # extends ~5px beyond the fg boundary → row 22 should be 0.5.
        H, W, fg_rows, dilate_px = 50, 20, 20, 5
        bm = self._bg_mask(H, W, fg_rows)
        cost = _build_seam_cost_map(self._canvas(H, W), bm, None, dilate_px=dilate_px)
        assert float(cost[22, 10]) == pytest.approx(0.5, abs=0.1), (
            f"row just past fg boundary should be edge-buffer=0.5, got {cost[22, 10]:.3f}"
        )

    def test_pure_background_far_from_fg_is_zero(self):
        H, W, fg_rows, dilate_px = 50, 20, 20, 5
        bm = self._bg_mask(H, W, fg_rows)
        cost = _build_seam_cost_map(self._canvas(H, W), bm, None, dilate_px=dilate_px)
        # Row 45 is 25px past the fg boundary — well beyond the 5px edge buffer.
        assert float(cost[45, 10]) == pytest.approx(0.0, abs=0.01)

    def test_fg_interior_not_lowered_by_edge_buffer(self):
        # A pixel deep inside the fg body should stay at 1.0, not be lowered to 0.5.
        H, W, fg_rows, dilate_px = 50, 20, 40, 5
        bm = self._bg_mask(H, W, fg_rows)
        cost = _build_seam_cost_map(self._canvas(H, W), bm, None, dilate_px=dilate_px)
        # Row 10 is 30px from the fg boundary at row 40 — deep interior.
        assert float(cost[10, 10]) == pytest.approx(1.0, abs=0.01)

    def test_none_masks_returns_zero(self):
        H, W = 30, 20
        cost = _build_seam_cost_map(self._canvas(H, W), None, None)
        assert float(cost.max()) == pytest.approx(0.0)

    def test_union_of_two_fg_masks(self):
        # Frame A has fg in left half, frame B in right half → both cost=1.0.
        H, W, dilate_px = 40, 40, 0
        bm_a = np.full((H, W), 255, dtype=np.uint8)
        bm_a[:, :20] = 0
        bm_b = np.full((H, W), 255, dtype=np.uint8)
        bm_b[:, 20:] = 0
        cost = _build_seam_cost_map(self._canvas(H, W), bm_a, bm_b, dilate_px=dilate_px)
        assert float(cost[20, 5]) == pytest.approx(1.0), "frame-A fg region"
        assert float(cost[20, 35]) == pytest.approx(1.0), "frame-B fg region"


# ---------------------------------------------------------------------------
# 10b. _build_seam_cost_map — §3.15A SemanticStitch column-level fg barrier (S33)
# ---------------------------------------------------------------------------


class TestSeamCostColumnFilter:
    """§3.15A: fg-dominated columns (>50% fg interior) raised to cost=2.0 when a
    background-corridor column exists; fallback to per-pixel costs when none does."""

    def _canvas(self, H: int, W: int) -> np.ndarray:
        return np.zeros((H, W, 3), dtype=np.uint8)

    def _fg_cols_mask(self, H: int, W: int, fg_cols: int) -> np.ndarray:
        """255=background, 0=foreground for first fg_cols columns."""
        m = np.full((H, W), 255, dtype=np.uint8)
        m[:, :fg_cols] = 0
        return m

    def test_fg_dominated_columns_raised_to_2(self):
        """Columns that are 100% fg interior → raised to 2.0 (column barrier)."""
        H, W = 40, 30
        # fg in cols 0–9, bg in cols 10–29; no edge buffer (dilate_px=0).
        bm = self._fg_cols_mask(H, W, fg_cols=10)
        cost = _build_seam_cost_map(self._canvas(H, W), bm, None, dilate_px=0)
        # All-fg columns should now be at 2.0 (barrier level).
        assert float(cost[:, :10].min()) == pytest.approx(2.0, abs=0.01), (
            "fg-dominated columns should be raised to 2.0"
        )

    def test_bg_corridor_unchanged_after_column_filter(self):
        """Background-corridor columns stay at 0.0 after the column barrier fires."""
        H, W = 40, 30
        bm = self._fg_cols_mask(H, W, fg_cols=10)
        cost = _build_seam_cost_map(self._canvas(H, W), bm, None, dilate_px=0)
        # Pure-background columns (10–29) should remain 0.0.
        assert float(cost[:, 15].max()) == pytest.approx(0.0, abs=0.01), (
            "background corridor columns should stay at 0.0"
        )

    def test_column_filter_skipped_when_all_columns_fg(self):
        """All columns fg → filter skipped; max cost stays at 1.0, not 2.0."""
        H, W = 40, 20
        bm = np.zeros((H, W), dtype=np.uint8)  # all foreground
        cost = _build_seam_cost_map(self._canvas(H, W), bm, None, dilate_px=0)
        assert float(cost.max()) == pytest.approx(1.0, abs=0.01), (
            "all-fg case: no corridor to exploit, filter must be skipped"
        )

    def test_column_filter_not_applied_when_no_fg(self):
        """No foreground → no column exceeds threshold; cost stays 0.0."""
        H, W = 40, 20
        bm = np.full((H, W), 255, dtype=np.uint8)  # all background
        cost = _build_seam_cost_map(self._canvas(H, W), bm, None, dilate_px=0)
        assert float(cost.max()) == pytest.approx(0.0, abs=0.01)

    def test_partial_fg_coverage_over_threshold_raises_column(self):
        """A column with 60% fg-interior pixels (>50% threshold) is raised to 2.0."""
        H, W = 10, 20
        bm = np.full((H, W), 255, dtype=np.uint8)
        # Make column 0 have 6/10 = 60% fg interior.
        bm[:6, 0] = 0
        # Columns 1–19 are fully background → corridor exists.
        cost = _build_seam_cost_map(self._canvas(H, W), bm, None, dilate_px=0)
        assert float(cost[:, 0].max()) == pytest.approx(2.0, abs=0.01), (
            "column with >50% fg interior should be raised to 2.0"
        )
        # Sanity: the corridor columns must stay below 2.0.
        assert float(cost[:, 5].max()) < 1.0, "corridor column must stay below 1.0"


# ---------------------------------------------------------------------------
# 10. _poisson_seam_blend — §1.6C gradient-domain seam blend (S21)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 11. _gain_to_min_feather — §1.6B gain-adaptive feather floor (S22)
# ---------------------------------------------------------------------------
class TestGainToMinFeather:
    """Tests for _gain_to_min_feather (§1.6B, S22).

    Formula: min(120, max(40, int(gain_diff × 300))).
    """

    def test_zero_diff_returns_floor(self):
        assert _gain_to_min_feather(0.0) == 40

    def test_small_diff_returns_floor(self):
        # 0.1 × 300 = 30 < 40 → floor
        assert _gain_to_min_feather(0.1) == 40

    def test_mid_diff_scales_linearly(self):
        # 0.2 × 300 = 60 → in (40, 120) → 60
        assert _gain_to_min_feather(0.2) == 60

    def test_large_diff_capped_at_120(self):
        # 0.5 × 300 = 150 → capped at 120
        assert _gain_to_min_feather(0.5) == 120

    def test_at_floor_boundary(self):
        # int(40/300 × 300) = int(40.0) = 40 → max(40, 40) = 40
        assert _gain_to_min_feather(40.0 / 300.0) == 40

    def test_just_above_floor_boundary(self):
        # int(0.14 × 300) = int(42.0) = 42 > 40
        assert _gain_to_min_feather(0.14) == 42


# ---------------------------------------------------------------------------
# TestBgGainUnclamped — §1.4C background gain clamp override (S40)
# ---------------------------------------------------------------------------


class TestBgGainUnclamped:
    """
    _bg_gain_unclamped returns the raw ideal gain (ref_lum / frame_lum) when
    the adaptive clamp would cut the correction by more than override_threshold
    (default 20 %).  Otherwise returns the clamped value unchanged.

    Background pixels tolerate aggressive correction because they are large
    uniform regions — character skin tones are excluded from the bg-only
    application site, so clipping artefacts are less visible.
    """

    def test_large_correction_returns_ideal(self):
        """Clamp cuts > 20 % → ideal gain returned for bg pixels."""
        # ref=128, frame=40 → ideal = 128/40 = 3.2
        # clamp_width for ref=128: 0.26 - 0.12*(128/255) ≈ 0.20 → hi≈1.20
        # clamped = 1.20, cut = (3.2-1.2)/3.2 = 0.625 > 0.20 → ideal
        result = _bg_gain_unclamped(128.0, 40.0)
        assert pytest.approx(result, rel=1e-4) == 128.0 / 40.0

    def test_small_correction_returns_clamped(self):
        """Clamp cuts ≤ 20 % → clamped gain returned unchanged."""
        # ref=100, frame=96 → ideal ≈ 1.042
        # clamp_width for ref=100: 0.26 - 0.12*(100/255) ≈ 0.213 → hi≈1.213
        # ideal=1.042 < hi → clamped=1.042, cut = 0 → same as clamped
        result = _bg_gain_unclamped(100.0, 96.0)
        assert result == pytest.approx(_adaptive_gain_clamp(100.0, 96.0), rel=1e-6)

    def test_zero_frame_lum_returns_one(self):
        """Guard: zero frame luminance → 1.0 (no correction)."""
        assert _bg_gain_unclamped(128.0, 0.0) == 1.0

    def test_threshold_boundary_at_exactly_20_percent(self):
        """When the cut is exactly 20 %, clamped value is returned (> not >=)."""
        # Find (ref, frame) such that cut == exactly 0.20.
        # For ref=100: clamp_width≈0.213, hi=1.213.
        # We want ideal such that (ideal - 1.213)/ideal = 0.20 → ideal = 1.213/0.80 ≈ 1.516
        # frame = ref / ideal = 100 / 1.516 ≈ 65.96
        ref = 100.0
        ideal_target = _adaptive_gain_clamp(ref, 1.0) / 0.80  # not exact but near 20%
        frame = ref / ideal_target
        clamped = _adaptive_gain_clamp(ref, frame)
        result = _bg_gain_unclamped(ref, frame, override_threshold=0.20)
        # At exactly 20%, result should equal clamped (strict >)
        ideal = ref / max(frame, 1e-9)
        cut = abs(ideal - clamped) / abs(ideal)
        if cut > 0.20:
            assert result == pytest.approx(ideal, rel=1e-4)
        else:
            assert result == pytest.approx(clamped, rel=1e-4)

    def test_darkening_case_returns_ideal_when_cut_exceeds_threshold(self):
        """Works symmetrically: frame brighter than ref → large darkening also overrides."""
        # ref=50, frame=200 → ideal = 50/200 = 0.25
        # clamp_width for ref=50: 0.26 - 0.12*(50/255) ≈ 0.237 → lo≈0.763
        # clamped = 0.763, cut = |0.25-0.763|/0.25 = 2.05 > 0.20 → ideal=0.25
        result = _bg_gain_unclamped(50.0, 200.0)
        assert pytest.approx(result, rel=1e-4) == 50.0 / 200.0


# ---------------------------------------------------------------------------
# §1.5D  Seam path cache (key helpers)
# ---------------------------------------------------------------------------


class TestSeamPathCache:
    """Unit tests for §1.5D seam-path caching helpers.

    _make_seam_cache_key() and _get_seam_cost_flags() are pure functions
    tested in isolation — no need to exercise the full _composite_foreground
    pipeline.
    """

    def test_make_seam_cache_key_returns_hashable(self):
        """Key can be stored as a dict key (i.e. is hashable)."""
        key = _make_seam_cache_key(("a.png", "b.png"), 0, (False, False))
        assert key is not None
        d: dict = {}
        d[key] = "sentinel"
        assert d[key] == "sentinel"

    def test_same_inputs_produce_equal_keys(self):
        """Identical arguments always yield the same key (deterministic)."""
        fk = ("frame_001.png", "frame_002.png", "frame_003.png")
        flags = (False, True)
        k1 = _make_seam_cache_key(fk, 1, flags)
        k2 = _make_seam_cache_key(fk, 1, flags)
        assert k1 == k2

    def test_different_boundary_index_produces_different_key(self):
        """Boundaries k=0 and k=1 must map to separate cache entries."""
        fk = ("f0.png", "f1.png", "f2.png")
        flags = _get_seam_cost_flags()
        assert _make_seam_cache_key(fk, 0, flags) != _make_seam_cache_key(fk, 1, flags)

    def test_different_frame_keys_produce_different_key(self):
        """Different frame sets must not collide in the cache."""
        flags = _get_seam_cost_flags()
        key_a = _make_seam_cache_key(("seq_a/f0.png", "seq_a/f1.png"), 0, flags)
        key_b = _make_seam_cache_key(("seq_b/f0.png", "seq_b/f1.png"), 0, flags)
        assert key_a != key_b

    def test_none_frame_keys_disables_cache(self):
        """frame_keys=None signals no-cache mode — key is None."""
        key = _make_seam_cache_key(None, 0, (False, False))
        assert key is None


# ---------------------------------------------------------------------------
# §1.4D  Multi-scale gain map
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §1.4E — _bg_histogram_lut (S49)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §1.4F — _reject_exposure_outliers (S50)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §1.14B — _seam_color_similarity and _check_seam_color_gate
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §1.14C — _seam_color_similarity_bgr (per-channel BGR Bhattacharyya)
# ---------------------------------------------------------------------------
















# ---------------------------------------------------------------------------
# §1.25 — TestSmoothSeamPath
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §1.26 — TestClampSeamPath
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §1.27 — TestHasSufficientBg
# ---------------------------------------------------------------------------


class TestHasSufficientBg:
    """Tests for _has_sufficient_bg() — §1.27 background coverage gate."""

    def test_sufficient_bg_returns_true(self):
        """Mask with more pixels than min_px returns True."""
        mask = np.ones((50, 50), dtype=bool)  # 2500 bg pixels
        assert _has_sufficient_bg(mask, min_px=200) is True

    def test_insufficient_bg_returns_false(self):
        """Mask with fewer pixels than min_px returns False."""
        mask = np.zeros((50, 50), dtype=bool)
        mask[:5, :5] = True  # only 25 bg pixels
        assert _has_sufficient_bg(mask, min_px=200) is False

    def test_exactly_at_threshold_returns_true(self):
        """Mask with exactly min_px pixels passes the floor."""
        mask = np.zeros((20, 20), dtype=bool)
        mask[:10, :10] = True  # exactly 100 pixels
        assert _has_sufficient_bg(mask, min_px=100) is True

    def test_none_mask_returns_false(self):
        """None mask (no BiRefNet output available) returns False."""
        assert _has_sufficient_bg(None, min_px=1) is False

    def test_all_fg_returns_false(self):
        """All-False mask (full-frame character, zero bg pixels) returns False."""
        mask = np.zeros((100, 100), dtype=bool)
        assert _has_sufficient_bg(mask, min_px=1) is False


# ---------------------------------------------------------------------------
# §1.28 — TestSeamPathStd
# ---------------------------------------------------------------------------




# ── TestZoneIsDegenerate — §1.30 minimum zone height guard (S74) ─────────────




# ── TestSeamFgPenetration — §1.31 seam fg penetration metric (S75) ───────────




# ── TestExclusionMasks — Issue 10A3 NL seam routing ──────────────────────────


class TestExclusionMasks:
    """_build_seam_cost_map: exclusion_masks param forces cost=1e6 in masked regions."""

    def _canvas(self, H=64, W=64, val=128):
        return np.full((H, W, 3), val, dtype=np.uint8)

    def _bg_mask(self, H=64, W=64):
        m = np.zeros((H, W), dtype=np.uint8)
        m[:, W // 4 : 3 * W // 4] = 255  # centre-column bg strip
        return m

    def test_no_exclusion_baseline(self):
        """Without exclusion_masks, cost stays in [0, 2] range."""
        H, W = 32, 32
        cost = _build_seam_cost_map(
            self._canvas(H, W), self._bg_mask(H, W), None, dilate_px=0
        )
        assert cost.max() <= 2.01  # tier-2 column barrier at most 2.0

    def test_exclusion_mask_raises_cost_to_barrier(self):
        """Pixels under the exclusion mask must have cost = 1e6."""
        H, W = 32, 32
        em = np.zeros((H, W), dtype=np.uint8)
        em[10:20, 8:24] = 255  # rectangle to exclude
        cost = _build_seam_cost_map(
            self._canvas(H, W),
            self._bg_mask(H, W),
            None,
            dilate_px=0,
            exclusion_masks=[em],
        )
        assert cost[15, 16] == pytest.approx(1e6)

    def test_non_excluded_pixels_unchanged(self):
        """Pixels outside the exclusion mask must not be affected."""
        H, W = 32, 32
        em = np.zeros((H, W), dtype=np.uint8)
        em[0:8, 0:8] = 255  # top-left corner only
        cost = _build_seam_cost_map(
            self._canvas(H, W),
            self._bg_mask(H, W),
            None,
            dilate_px=0,
            exclusion_masks=[em],
        )
        # Centre of bg strip (W//4..3W//4 = 8..24), well below excluded rows
        # Background area → cost should be low (0.0–1.0), not 1e6
        assert cost[28, 16] < 1.5

    def test_exclusion_mask_auto_resized(self):
        """An exclusion mask with wrong shape must be resized before application."""
        H, W = 32, 32
        em_large = np.full((64, 64), 255, dtype=np.uint8)  # full exclusion, double size
        cost = _build_seam_cost_map(
            self._canvas(H, W),
            self._bg_mask(H, W),
            None,
            dilate_px=0,
            exclusion_masks=[em_large],
        )
        assert cost.max() == pytest.approx(1e6)

    def test_none_in_exclusion_list_skipped(self):
        """A None entry in the exclusion_masks list must not raise an exception."""
        H, W = 32, 32
        cost = _build_seam_cost_map(
            self._canvas(H, W),
            self._bg_mask(H, W),
            None,
            dilate_px=0,
            exclusion_masks=[None, None],
        )
        assert cost.max() <= 2.01  # same as no-exclusion baseline

    def test_multiple_exclusion_masks_combine(self):
        """Multiple exclusion masks are cumulative — all covered pixels blocked."""
        H, W = 32, 32
        em1 = np.zeros((H, W), dtype=np.uint8)
        em1[4:8, 4:28] = 255
        em2 = np.zeros((H, W), dtype=np.uint8)
        em2[24:28, 4:28] = 255
        cost = _build_seam_cost_map(
            self._canvas(H, W),
            self._bg_mask(H, W),
            None,
            dilate_px=0,
            exclusion_masks=[em1, em2],
        )
        assert cost[6, 16] == pytest.approx(1e6)
        assert cost[26, 16] == pytest.approx(1e6)


# ── TestComputeInitialBoundaries ───────────────────────────────────────────────


class TestComputeInitialBoundaries:
    """Tests for _compute_initial_boundaries (S85 — HITL boundary editor helper)."""

    def _affine(self, ty: float) -> np.ndarray:
        a = np.eye(2, 3, dtype=np.float64)
        a[1, 2] = ty
        return a

    def _frame(self, h: int = 40, w: int = 30) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_two_frames_single_boundary_at_midpoint(self):
        """Two frames: single boundary at midpoint between strip centres."""
        from asp_backend.rendering.compositing import _compute_initial_boundaries

        frames = [self._frame(40), self._frame(40)]
        affines = [self._affine(0.0), self._affine(40.0)]
        bnd = _compute_initial_boundaries(affines, frames)
        assert len(bnd) == 1
        # Strip centres: 20.0 and 60.0 → midpoint = 40.0
        assert abs(bnd[0] - 40.0) < 1e-6

    def test_n_minus_one_boundaries(self):
        """N frames produce exactly N-1 boundaries."""
        from asp_backend.rendering.compositing import _compute_initial_boundaries

        N = 5
        frames = [self._frame() for _ in range(N)]
        affines = [self._affine(i * 40.0) for i in range(N)]
        bnd = _compute_initial_boundaries(affines, frames)
        assert len(bnd) == N - 1

    def test_boundaries_monotonically_increasing(self):
        """Boundaries should be monotonically increasing for top-to-bottom scroll."""
        from asp_backend.rendering.compositing import _compute_initial_boundaries

        frames = [self._frame() for _ in range(4)]
        affines = [self._affine(i * 50.0) for i in range(4)]
        bnd = _compute_initial_boundaries(affines, frames)
        for i in range(len(bnd) - 1):
            assert bnd[i] < bnd[i + 1]

    def test_preset_boundaries_accepted_by_composite_foreground(self):
        """_composite_foreground respects preset_boundaries of correct length."""
        from asp_backend.rendering.compositing import (
            _composite_foreground,
            _compute_initial_boundaries,
        )

        H, W = 40, 30
        frames = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(2)]
        frames[0][:, :] = 50  # gray
        frames[1][:, :] = 100  # brighter gray
        affines = [self._affine(0.0), self._affine(float(H))]
        canvas_h, canvas_w = H * 2, W
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        bg_masks = [None, None]
        bnd = _compute_initial_boundaries(affines, frames)
        # Should run without exception
        result = _composite_foreground(
            [],
            [],
            canvas,
            canvas_h,
            canvas_w,
            frames,
            affines,
            bg_masks,
            preset_boundaries=bnd,
        )
        assert result.shape == (canvas_h, canvas_w, 3)

    def test_wrong_length_preset_boundaries_ignored(self):
        """preset_boundaries with wrong length falls back to midpoint computation."""
        from asp_backend.rendering.compositing import _composite_foreground

        H, W = 40, 30
        frames = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(2)]
        affines = [self._affine(0.0), self._affine(float(H))]
        canvas_h, canvas_w = H * 2, W
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        bg_masks = [None, None]
        # Pass wrong-length preset_boundaries — should not crash
        result = _composite_foreground(
            [],
            [],
            canvas,
            canvas_h,
            canvas_w,
            frames,
            affines,
            bg_masks,
            preset_boundaries=np.array([10.0, 20.0, 30.0]),  # 3 boundaries for 2 frames
        )
        assert result.shape == (canvas_h, canvas_w, 3)


# ── TestPaintMask ──────────────────────────────────────────────────────────────


class TestPaintMask:
    """Tests for paint_mask param in _composite_foreground (S86 — seam painter)."""

    def _frame(self, h: int = 40, w: int = 30, val: int = 80) -> np.ndarray:
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:] = val
        return f

    def _affine(self, ty: float) -> np.ndarray:
        a = np.eye(2, 3, dtype=np.float64)
        a[1, 2] = ty
        return a

    def test_paint_mask_none_runs_without_error(self):
        """Passing paint_mask=None is equivalent to the default (no mask)."""
        from asp_backend.rendering.compositing import _composite_foreground

        H, W = 40, 30
        frames = [self._frame(H, W), self._frame(H, W, 100)]
        affines = [self._affine(0.0), self._affine(float(H))]
        canvas = np.zeros((H * 2, W, 3), dtype=np.uint8)
        result = _composite_foreground(
            [],
            [],
            canvas,
            H * 2,
            W,
            frames,
            affines,
            [None, None],
            paint_mask=None,
        )
        assert result.shape == (H * 2, W, 3)

    def test_paint_mask_correct_shape_accepted(self):
        """A paint_mask matching (canvas_h, canvas_w) is accepted without error."""
        from asp_backend.rendering.compositing import _composite_foreground

        H, W = 40, 30
        frames = [self._frame(H, W), self._frame(H, W, 100)]
        affines = [self._affine(0.0), self._affine(float(H))]
        canvas_h, canvas_w = H * 2, W
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        mask[35:45, :] = 255  # paint a band around the seam
        result = _composite_foreground(
            [],
            [],
            canvas,
            canvas_h,
            canvas_w,
            frames,
            affines,
            [None, None],
            paint_mask=mask,
        )
        assert result.shape == (canvas_h, canvas_w, 3)

    def test_paint_mask_wrong_shape_ignored(self):
        """A paint_mask with wrong canvas dimensions is silently ignored."""
        from asp_backend.rendering.compositing import _composite_foreground

        H, W = 40, 30
        frames = [self._frame(H, W), self._frame(H, W)]
        affines = [self._affine(0.0), self._affine(float(H))]
        canvas_h, canvas_w = H * 2, W
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        bad_mask = np.ones((10, 10), dtype=np.uint8) * 255  # wrong size
        result = _composite_foreground(
            [],
            [],
            canvas,
            canvas_h,
            canvas_w,
            frames,
            affines,
            [None, None],
            paint_mask=bad_mask,
        )
        assert result.shape == (canvas_h, canvas_w, 3)

    def test_paint_mask_combined_with_exclusion_masks(self):
        """paint_mask + existing exclusion_masks are both applied (no conflict)."""
        from asp_backend.rendering.compositing import _composite_foreground

        H, W = 40, 30
        frames = [self._frame(H, W, 60), self._frame(H, W, 120)]
        affines = [self._affine(0.0), self._affine(float(H))]
        canvas_h, canvas_w = H * 2, W
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        excl = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        excl[10:20, 5:25] = 255
        paint = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        paint[35:45, :] = 255
        result = _composite_foreground(
            [],
            [],
            canvas,
            canvas_h,
            canvas_w,
            frames,
            affines,
            [None, None],
            exclusion_masks=[excl],
            paint_mask=paint,
        )
        assert result.shape == (canvas_h, canvas_w, 3)

    def test_paint_mask_all_black_behaves_like_no_mask(self):
        """An all-zero paint_mask has no effect on seam routing."""
        from asp_backend.rendering.compositing import _composite_foreground

        H, W = 40, 30
        frames = [self._frame(H, W), self._frame(H, W)]
        affines = [self._affine(0.0), self._affine(float(H))]
        canvas_h, canvas_w = H * 2, W
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        zero_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        result_no = _composite_foreground(
            [],
            [],
            canvas.copy(),
            canvas_h,
            canvas_w,
            frames,
            affines,
            [None, None],
        )
        result_zm = _composite_foreground(
            [],
            [],
            canvas.copy(),
            canvas_h,
            canvas_w,
            frames,
            affines,
            [None, None],
            paint_mask=zero_mask,
        )
        np.testing.assert_array_equal(result_no, result_zm)


# ---------------------------------------------------------------------------
# §2.4B — _annotate_seams (S94)
# ---------------------------------------------------------------------------




# ── TestSeamMetaOut ─────────────────────────────────────────────────────────


class TestSeamMetaOut:
    """Tests for seam_meta_out / seam_overrides params in _composite_foreground (S95 — §2.4A)."""

    def _frame(self, h: int = 40, w: int = 30, val: int = 80) -> np.ndarray:
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:] = val
        return f

    def _affine(self, ty: float) -> np.ndarray:
        a = np.eye(2, 3, dtype=np.float64)
        a[1, 2] = ty
        return a

    def test_seam_meta_out_populated_with_boundaries(self):
        """seam_meta_out receives boundaries, seam_post_diffs, seam_single_pose on return."""
        from asp_backend.rendering.compositing import _composite_foreground

        H, W = 40, 30
        frames = [self._frame(H, W, 80), self._frame(H, W, 100)]
        affines = [self._affine(0.0), self._affine(float(H))]
        canvas = np.zeros((H * 2, W, 3), dtype=np.uint8)
        meta: dict = {}
        _composite_foreground(
            [],
            [],
            canvas,
            H * 2,
            W,
            frames,
            affines,
            [None, None],
            seam_meta_out=meta,
        )
        assert "boundaries" in meta, "seam_meta_out must contain 'boundaries'"
        assert "seam_post_diffs" in meta, "seam_meta_out must contain 'seam_post_diffs'"
        assert "seam_single_pose" in meta, (
            "seam_meta_out must contain 'seam_single_pose'"
        )
        # Two frames produce exactly one boundary
        assert len(meta["boundaries"]) == 1, (
            f"expected 1 boundary, got {len(meta['boundaries'])}"
        )

    def test_seam_meta_out_single_frame_empty_boundaries(self):
        """With only one frame no seams are created; boundaries list is empty."""
        from asp_backend.rendering.compositing import _composite_foreground

        H, W = 40, 30
        frames = [self._frame(H, W, 80)]
        affines = [self._affine(0.0)]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        meta: dict = {}
        _composite_foreground(
            [],
            [],
            canvas,
            H,
            W,
            frames,
            affines,
            [None],
            seam_meta_out=meta,
        )
        assert meta.get("boundaries", []) == [], (
            "single-frame composite should produce no seam boundaries"
        )

    def _bg_mask(self, h: int, w: int) -> np.ndarray:
        """All-background uint8 mask (255 = background, so wm > 127 = all-True)."""
        return np.full((h, w), 255, dtype=np.uint8)

    def test_force_single_pose_override_escalates_seam(self):
        """force_single_pose override must result in seam k appearing in seam_single_pose."""
        from asp_backend.rendering.compositing import _composite_foreground

        H, W = 40, 30
        frames = [self._frame(H, W, 80), self._frame(H, W, 100)]
        affines = [self._affine(0.0), self._affine(float(H))]
        canvas = np.zeros((H * 2, W, 3), dtype=np.uint8)
        # Provide non-None bg_masks so warped_bg[i] is not None (avoids early continue
        # at line 1814 that would bypass the force_single_pose code path).
        bg_masks = [self._bg_mask(H, W), self._bg_mask(H, W)]
        meta: dict = {}
        overrides = {0: {"force_single_pose": True, "force_blend": False}}
        _composite_foreground(
            [],
            [],
            canvas,
            H * 2,
            W,
            frames,
            affines,
            bg_masks,
            seam_meta_out=meta,
            seam_overrides=overrides,
        )
        assert 0 in meta.get("seam_single_pose", {}), (
            "force_single_pose override must escalate seam 0 to single-pose"
        )
        assert meta["seam_post_diffs"].get(0) == 99.0, (
            "force_single_pose sentinel diff must be 99.0"
        )

    def test_force_blend_override_removes_seam_from_single_pose(self):
        """force_blend override must remove seam k from seam_single_pose even if escalated.

        Uses overlapping zero-value frames so that:
          - zone_h > ASP_ZONE_MIN_HEIGHT (avoids degenerate-zone guard)
          - _seam_fg_penetration = 0 (avoids ASP_SEAM_FG_PENETRATION_MAX guard)
          - _seam_path_std = 0 (avoids ASP_SEAM_INSTABILITY_THRESH guard)
        This ensures that force_blend's post-loop removal is the only influence
        on whether seam 0 appears in seam_single_pose.
        """
        from asp_backend.rendering.compositing import _composite_foreground

        H, W = 60, 30
        # Overlap frames by 20px → max_feather ≥ 10 → zone_h ≥ 21 > ASP_ZONE_MIN_HEIGHT=20
        ty1 = H - 20  # frame 1 starts 20px below frame 0's top → 20px overlap
        # All-zero frames → FG penetration = 0.0 (no foreground pixels at all)
        frames = [
            np.zeros((H, W, 3), dtype=np.uint8),
            np.zeros((H, W, 3), dtype=np.uint8),
        ]
        a0 = np.eye(2, 3, dtype=np.float64)
        a1 = np.eye(2, 3, dtype=np.float64)
        a1[1, 2] = float(ty1)
        affines = [a0, a1]
        canvas_h = ty1 + H  # = 100px
        canvas = np.zeros((canvas_h, W, 3), dtype=np.uint8)
        # Non-None bg_masks required so the FG registration loop is entered.
        bg_masks = [self._bg_mask(H, W), self._bg_mask(H, W)]
        # First run: escalate via force_single_pose (even with zero frames)
        meta1: dict = {}
        _composite_foreground(
            [],
            [],
            canvas.copy(),
            canvas_h,
            W,
            frames,
            affines,
            bg_masks,
            seam_meta_out=meta1,
            seam_overrides={0: {"force_single_pose": True}},
        )
        assert 0 in meta1.get("seam_single_pose", {}), (
            "setup: seam 0 should be escalated"
        )
        # Second run: force_blend — seam 0 must NOT appear in seam_single_pose.
        # With zero-value frames the degenerate-zone, instability, and FG-penetration
        # blend-loop guards all remain inactive, so force_blend's removal is definitive.
        meta2: dict = {}
        _composite_foreground(
            [],
            [],
            canvas.copy(),
            canvas_h,
            W,
            frames,
            affines,
            bg_masks,
            seam_meta_out=meta2,
            seam_overrides={0: {"force_blend": True}},
        )
        assert 0 not in meta2.get("seam_single_pose", {}), (
            "force_blend override must remove seam 0 from seam_single_pose"
        )

    def test_seam_meta_out_none_does_not_raise(self):
        """Passing seam_meta_out=None (the default) must not raise any error."""
        from asp_backend.rendering.compositing import _composite_foreground

        H, W = 40, 30
        frames = [self._frame(H, W, 80), self._frame(H, W, 100)]
        affines = [self._affine(0.0), self._affine(float(H))]
        canvas = np.zeros((H * 2, W, 3), dtype=np.uint8)
        result = _composite_foreground(
            [],
            [],
            canvas,
            H * 2,
            W,
            frames,
            affines,
            [None, None],
            seam_meta_out=None,
        )
        assert result.shape == (H * 2, W, 3), "result shape must match canvas"


class TestSeamCropExtraction:
    """Tests for _extract_seam_crops (S96 — §2.4C)."""

    def _canvas(self, H: int = 100, W: int = 30, val: int = 128) -> np.ndarray:
        c = np.full((H, W, 3), val, dtype=np.uint8)
        return c

    def test_crops_extracted_for_all_seams(self):
        from asp_backend.rendering.compositing import _extract_seam_crops

        canvas = self._canvas(100, 30)
        boundaries = np.array([40, 70], dtype=np.float64)
        crops = _extract_seam_crops(canvas, boundaries, band_px=10)
        assert set(crops.keys()) == {0, 1}, "one crop per seam"

    def test_crop_height_is_twice_band(self):
        from asp_backend.rendering.compositing import _extract_seam_crops

        canvas = self._canvas(100, 30)
        boundaries = np.array([50.0])
        crops = _extract_seam_crops(canvas, boundaries, band_px=15)
        assert crops[0].shape == (30, 30, 3), "crop height = 2*band_px, full width"

    def test_crop_clamped_at_top_boundary(self):
        from asp_backend.rendering.compositing import _extract_seam_crops

        canvas = self._canvas(100, 30)
        # boundary at row 5 with band=20 → crop [0:25], height=25 (not 40)
        boundaries = np.array([5.0])
        crops = _extract_seam_crops(canvas, boundaries, band_px=20)
        assert crops[0].shape[0] == 25, "top clamp gives narrower crop (5+20=25)"

    def test_empty_boundaries_returns_empty_dict(self):
        from asp_backend.rendering.compositing import _extract_seam_crops

        canvas = self._canvas(100, 30)
        crops = _extract_seam_crops(canvas, np.array([]), band_px=10)
        assert crops == {}

    def test_seam_crops_in_meta_out(self):
        """_composite_foreground populates seam_meta_out['seam_crops']."""
        from asp_backend.rendering.compositing import _composite_foreground

        H, W = 40, 30
        frames = [
            np.full((H, W, 3), 80, dtype=np.uint8),
            np.full((H, W, 3), 100, dtype=np.uint8),
        ]
        a0 = np.eye(2, 3, dtype=np.float64)
        a1 = np.eye(2, 3, dtype=np.float64)
        a1[1, 2] = float(H)
        affines = [a0, a1]
        canvas = np.zeros((H * 2, W, 3), dtype=np.uint8)
        meta: dict = {}
        _composite_foreground(
            [], [], canvas, H * 2, W, frames, affines, [None, None], seam_meta_out=meta
        )
        assert "seam_crops" in meta, "seam_crops key must be in seam_meta_out"
        assert isinstance(meta["seam_crops"], dict)
        assert 0 in meta["seam_crops"], "seam 0 crop must be present"
        crop = meta["seam_crops"][0]
        assert crop.ndim == 3 and crop.shape[1] == W


# ---------------------------------------------------------------------------
# §1.90 — Bilateral seam smoothing
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §3.17 — High-frequency column seam cost
# ---------------------------------------------------------------------------




# ===========================================================================
# Merged from test_compositing_s148.py
# ===========================================================================




# ---------------------------------------------------------------------------
# §1.92 — Gaussian feather smoothing
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §3.18 — CQAS aggregate quality score
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §1.94 — Background plate consistency score
# ---------------------------------------------------------------------------




# ===========================================================================
# Merged from test_compositing_s149.py
# ===========================================================================




# ---------------------------------------------------------------------------
# §3.19 — Per-zone pre-blend chroma alignment
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §1.97 — Seam zone entropy asymmetry gate
# ---------------------------------------------------------------------------




# ===========================================================================
# Merged from test_compositing_s150.py
# ===========================================================================




# ---------------------------------------------------------------------------
# §3.20 — Extra fg-boundary dilation cost ring
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §1.99 — Seam endpoint bg-preference
# ---------------------------------------------------------------------------




# ===========================================================================
# Merged from test_compositing_s151.py
# ===========================================================================




# ---------------------------------------------------------------------------
# §1.102 — Warp residual momentum damping
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §1.103 — Reference-proximity dominant frame selection
# ---------------------------------------------------------------------------




# ===========================================================================
# Merged from test_compositing_s152.py
# ===========================================================================




# ---------------------------------------------------------------------------
# §1.105 — Fg-overlap blend weight cap
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# §1.106 — Post-composite seam luminance step audit
# ---------------------------------------------------------------------------


class TestAuditSeamLumSteps:
    def test_flat_canvas_zero_steps(self):
        from asp_backend.rendering.compositing import _audit_seam_lum_steps

        canvas = np.full((100, 80, 3), 128, dtype=np.uint8)
        steps = _audit_seam_lum_steps(canvas, [30.0, 60.0], band_px=5, warn_thresh=8.0)
        assert len(steps) == 2
        assert steps[0] < 1.0
        assert steps[1] < 1.0

    def test_sharp_step_detected(self):
        from asp_backend.rendering.compositing import _audit_seam_lum_steps

        canvas = np.zeros((100, 80, 3), dtype=np.uint8)
        canvas[:50] = 200
        canvas[50:] = 50
        steps = _audit_seam_lum_steps(canvas, [50.0], band_px=5, warn_thresh=8.0)
        assert steps[0] > 50.0

    def test_returns_dict_keyed_by_index(self):
        from asp_backend.rendering.compositing import _audit_seam_lum_steps

        canvas = np.full((80, 60, 3), 100, dtype=np.uint8)
        steps = _audit_seam_lum_steps(canvas, [20.0, 40.0, 60.0])
        assert isinstance(steps, dict)
        assert set(steps.keys()) == {0, 1, 2}

    def test_empty_boundaries_empty_dict(self):
        from asp_backend.rendering.compositing import _audit_seam_lum_steps

        canvas = np.full((60, 40, 3), 128, dtype=np.uint8)
        steps = _audit_seam_lum_steps(canvas, [])
        assert steps == {}

    def test_function_in_all(self):
        import asp_backend.rendering.compositing as comp

        assert "_audit_seam_lum_steps" in comp.__all__
        assert "_POST_SEAM_WARN_THRESH" in comp.__all__


# ---------------------------------------------------------------------------
# §1.127 — Zone hue equalization
# ---------------------------------------------------------------------------




# ===========================================================================
# Merged from test_compositing_s160.py
# ===========================================================================


class TestBlocksGainCompensate:
    """§4.1: Spatial blocks BGR gain compensation."""

    def test_flag_in_module(self):
        assert hasattr(compositing, "_BLOCKS_GAIN_COMP")

    def test_uniform_zone_no_change(self):
        # fa == fb == solid gray → gain ≈ 1.0 → output ≈ input
        zone = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = _blocks_gain_compensate(zone, zone.copy(), block_size=32)
        assert result.shape == zone.shape
        assert result.dtype == np.uint8
        diff = np.abs(result.astype(np.int32) - zone.astype(np.int32)).max()
        assert diff <= 2  # rounding tolerance

    def test_applies_gain_correction(self):
        # fa is brighter than fb → output should be brighter than fb
        fa = np.full((64, 64, 3), 200, dtype=np.uint8)
        fb = np.full((64, 64, 3), 100, dtype=np.uint8)
        result = _blocks_gain_compensate(fa, fb, block_size=32)
        assert float(result.mean()) > float(fb.mean())

    def test_clamps_extreme_gain(self):
        # fb ≈ 0 (very dark) — gain would be huge; clamped to 2.0 → no crash
        fa = np.full((64, 64, 3), 128, dtype=np.uint8)
        fb = np.full((64, 64, 3), 0, dtype=np.uint8)
        result = _blocks_gain_compensate(fa, fb, block_size=32)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_constant_added_to_schema(self):
        assert "ASP_BLOCKS_GAIN_COMP" in config._CONFIG_SCHEMA


class TestBlocksLumCompensate:
    """§4.4: LAB L-channel blocks gain compensation."""

    def test_flag_in_module(self):
        assert hasattr(compositing, "_BLOCKS_LUM_COMP")

    def test_uniform_zone_no_change(self):
        zone = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = _blocks_lum_compensate(zone, zone.copy(), block_size=32)
        assert result.shape == zone.shape
        assert result.dtype == np.uint8
        diff = np.abs(result.astype(np.int32) - zone.astype(np.int32)).max()
        assert diff <= 2

    def test_brightens_dark_fb(self):
        fa = np.full((64, 64, 3), 150, dtype=np.uint8)
        fb = np.full((64, 64, 3), 100, dtype=np.uint8)
        result = _blocks_lum_compensate(fa, fb, block_size=32)
        assert float(result.mean()) > float(fb.mean())

    def test_output_shape_preserved(self):
        fa = np.random.randint(50, 200, (80, 60, 3), dtype=np.uint8)
        fb = np.random.randint(50, 200, (80, 60, 3), dtype=np.uint8)
        result = _blocks_lum_compensate(fa, fb, block_size=32)
        assert result.shape == fb.shape
        assert result.dtype == np.uint8

    def test_schema_entry(self):
        assert "ASP_BLOCKS_LUM_COMP" in config._CONFIG_SCHEMA


# ---------------------------------------------------------------------------
# Phase 5b: blocks_gain_compensate_pair + blocks_lum_compensate_pair wiring
# ---------------------------------------------------------------------------


class TestBlocksGainCompensateBatchWiring:
    """Verify _blocks_gain_compensate produces correct output without C++ batch."""

    def test_identical_zones_identity(self):
        zone = np.full((64, 80, 3), 120, dtype=np.uint8)
        result = _blocks_gain_compensate(zone, zone.copy(), block_size=32)
        assert result.dtype == np.uint8
        assert result.shape == zone.shape
        diff = np.abs(result.astype(np.int32) - zone.astype(np.int32)).max()
        assert diff <= 2

    def test_brighter_fa_increases_fb(self):
        fa = np.full((64, 80, 3), 200, dtype=np.uint8)
        fb = np.full((64, 80, 3), 100, dtype=np.uint8)
        result = _blocks_gain_compensate(fa, fb, block_size=32)
        assert float(result.mean()) > float(fb.mean())

    def test_empty_zones_return_copy(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        fb = np.full((32, 32, 3), 50, dtype=np.uint8)
        assert _blocks_gain_compensate(empty, fb).shape == (32, 32, 3)

    def test_gain_clamp_upper(self):
        fa = np.full((32, 32, 3), 250, dtype=np.uint8)
        fb = np.full((32, 32, 3), 10, dtype=np.uint8)
        result = _blocks_gain_compensate(fa, fb, block_size=32)
        assert int(result.max()) <= 22  # 10 × 2.0 clamp + rounding

    def test_output_in_valid_range(self):
        rng = np.random.default_rng(7)
        fa = rng.integers(0, 256, (80, 64, 3), dtype=np.uint8)
        fb = rng.integers(0, 256, (80, 64, 3), dtype=np.uint8)
        result = _blocks_gain_compensate(fa, fb, block_size=32)
        assert result.min() >= 0
        assert result.max() <= 255


class TestBlocksLumCompensateBatchWiring:
    """Verify _blocks_lum_compensate produces correct output without C++ batch."""

    def test_identical_zones_identity(self):
        zone = np.full((64, 80, 3), 120, dtype=np.uint8)
        result = _blocks_lum_compensate(zone, zone.copy(), block_size=32)
        assert result.dtype == np.uint8
        assert result.shape == zone.shape
        diff = np.abs(result.astype(np.int32) - zone.astype(np.int32)).max()
        assert diff <= 2

    def test_brighter_fa_increases_fb(self):
        fa = np.full((64, 80, 3), 200, dtype=np.uint8)
        fb = np.full((64, 80, 3), 100, dtype=np.uint8)
        result = _blocks_lum_compensate(fa, fb, block_size=32)
        assert float(result.mean()) > float(fb.mean())

    def test_empty_zones_return_copy(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        fb = np.full((32, 32, 3), 50, dtype=np.uint8)
        assert _blocks_lum_compensate(empty, fb).shape == (32, 32, 3)

    def test_output_in_valid_range(self):
        rng = np.random.default_rng(11)
        fa = rng.integers(0, 256, (80, 64, 3), dtype=np.uint8)
        fb = rng.integers(0, 256, (80, 64, 3), dtype=np.uint8)
        result = _blocks_lum_compensate(fa, fb, block_size=32)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_lum_output_avoids_channel_cast(self):
        # Uniform color zone: lum gain should preserve hue (all channels scaled equally)
        fa = np.full((32, 32, 3), 160, dtype=np.uint8)
        fb_arr = np.zeros((32, 32, 3), dtype=np.uint8)
        fb_arr[:, :, 0] = 80   # B
        fb_arr[:, :, 1] = 40   # G
        fb_arr[:, :, 2] = 20   # R
        result = _blocks_lum_compensate(fa, fb_arr, block_size=32)
        # All channels scaled by same scalar → ratios preserved (within rounding)
        r_b = float(result[:, :, 0].mean())
        r_g = float(result[:, :, 1].mean())
        float(result[:, :, 2].mean())
        # B/G ratio ≈ 2.0 as in input
        assert abs(r_b / max(r_g, 0.5) - 2.0) < 0.15


# ---------------------------------------------------------------------------
# Phase 5d: find_optimal_boundaries batch wiring
# ---------------------------------------------------------------------------


def _call_find_optimal_boundaries_cpp(warped_list, order, init_bounds, H, W, **kw):
    """Call batch.compositing.find_optimal_boundaries; skip if stub or not compiled."""
    from asp_backend.rendering import compositing as comp_mod
    if not comp_mod.BATCH_AVAILABLE:
        pytest.skip("batch not available")
    if not hasattr(comp_mod.batch.compositing, "find_optimal_boundaries"):
        pytest.skip("find_optimal_boundaries not in batch.compositing — rebuild needed")
    try:
        return comp_mod.batch.compositing.find_optimal_boundaries(
            [np.ascontiguousarray(f) for f in warped_list],
            np.asarray(order, dtype=np.int64),
            np.asarray(init_bounds, dtype=np.float64),
            H, W,
            kw.get("search_range", 250),
            kw.get("search_slab", 20),
            kw.get("bg_masks"),
            kw.get("affines"),
        )
    except RuntimeError as e:
        if "not yet implemented" in str(e).lower() or "stub" in str(e).lower():
            pytest.skip(f"find_optimal_boundaries is a stub — {e}")
        raise


def _make_canvas_frame(H, W, val, dtype=np.uint8):
    return np.full((H, W, 3), val, dtype=dtype)


class TestFindOptimalBoundariesBatchWiring:
    """batch.compositing.find_optimal_boundaries — Phase 5d C++ wiring."""

    def test_output_shape_matches_n_minus_1(self):
        H, W = 200, 120
        frames = [_make_canvas_frame(H, W, v) for v in [80, 120, 160]]
        order = np.array([0, 1, 2], dtype=np.int64)
        init_bounds = np.array([H // 3, 2 * H // 3], dtype=np.float64)
        bounds, diffs = _call_find_optimal_boundaries_cpp(
            frames, order, init_bounds, H, W
        )
        assert bounds.shape == (2,)
        assert diffs.shape == (2,)

    def test_boundary_stays_within_canvas(self):
        H, W = 300, 80
        f0 = _make_canvas_frame(H, W, 100)
        f1 = _make_canvas_frame(H, W, 140)
        order = np.array([0, 1], dtype=np.int64)
        init_bounds = np.array([H // 2], dtype=np.float64)
        bounds, _ = _call_find_optimal_boundaries_cpp(
            [f0, f1], order, init_bounds, H, W
        )
        assert 0 <= float(bounds[0]) <= H

    def test_identical_frames_returns_a_finite_diff(self):
        H, W = 200, 100
        f = _make_canvas_frame(H, W, 128)
        order = np.array([0, 1], dtype=np.int64)
        init_bounds = np.array([H // 2], dtype=np.float64)
        bounds, diffs = _call_find_optimal_boundaries_cpp(
            [f.copy(), f.copy()], order, init_bounds, H, W
        )
        assert np.isfinite(diffs[0])
        assert diffs[0] == pytest.approx(0.0, abs=1.0)

    def test_gradient_frames_move_boundary_toward_low_diff(self):
        H, W = 200, 60
        # f0: bright top half, dark bottom half; f1: dark everywhere
        # The boundary between f0 and f1 should prefer the dark-bottom region of f0
        f0 = np.zeros((H, W, 3), dtype=np.uint8)
        f0[: H // 2, :] = 200
        f0[H // 2 :, :] = 40
        f1 = np.full((H, W, 3), 40, dtype=np.uint8)
        order = np.array([0, 1], dtype=np.int64)
        # Start boundary in the bright region (top third) — C++ should move it down
        init_bounds = np.array([H // 3], dtype=np.float64)
        bounds, diffs = _call_find_optimal_boundaries_cpp(
            [f0, f1], order, init_bounds, H, W, search_range=250
        )
        # Optimal is in the dark bottom half of f0 (low diff with f1)
        assert float(bounds[0]) > H // 3

    def test_zero_search_range_returns_initial_bound(self):
        H, W = 200, 80
        f0 = _make_canvas_frame(H, W, 90)
        f1 = _make_canvas_frame(H, W, 150)
        order = np.array([0, 1], dtype=np.int64)
        init_bounds = np.array([H // 2], dtype=np.float64)
        bounds, _ = _call_find_optimal_boundaries_cpp(
            [f0, f1], order, init_bounds, H, W, search_range=0
        )
        # With 0 search range the loop finds no candidates → returns initial position
        assert float(bounds[0]) == pytest.approx(float(init_bounds[0]), abs=20)


# ===========================================================================
# §4.5 — _canvas_dp_seam_composite
# ===========================================================================


def _solid_frame(h: int, w: int, val: int) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)




# ===========================================================================
# §3.33 — _feather_gc_boundaries
# ===========================================================================


def _solid(h: int, w: int, val: int) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def _make_ownership(h: int, w: int, split_row: int) -> list:
    """Two ownership masks split at split_row (frame 0 owns [0..split_row), frame 1 owns the rest)."""
    m0 = np.zeros((h, w), dtype=np.uint8)
    m0[:split_row, :] = 255
    m1 = np.zeros((h, w), dtype=np.uint8)
    m1[split_row:, :] = 255
    return [m0, m1]


class TestFeatherGcBoundaries:
    """§3.33: Feathered blend at GraphCut ownership transitions."""

    def test_returns_same_shape(self):
        H, W = 60, 80
        result = _solid(H, W, 100)
        ownership = _make_ownership(H, W, 30)
        frames = [_solid(H, W, 80), _solid(H, W, 160)]
        out = _feather_gc_boundaries(result, ownership, frames, feather_px=8)
        assert out.shape == (H, W, 3)
        assert out.dtype == np.uint8

    def test_feather_smooths_step_at_boundary(self):
        H, W = 60, 80
        split = 30
        fa = _solid(H, W, 50)
        fb = _solid(H, W, 200)
        ownership = _make_ownership(H, W, split)
        result = np.zeros((H, W, 3), dtype=np.uint8)
        result[:split, :] = 50
        result[split:, :] = 200
        out = _feather_gc_boundaries(result, ownership, [fa, fb], feather_px=8)
        # Within the ±8px band the std should be > 0 (values are transitioning)
        band = out[split - 8 : split + 8, :, 0].astype(float)
        assert band.std() > 0.0

    def test_zero_feather_returns_copy(self):
        H, W = 40, 50
        result = _solid(H, W, 128)
        ownership = _make_ownership(H, W, 20)
        frames = [_solid(H, W, 50), _solid(H, W, 200)]
        out = _feather_gc_boundaries(result, ownership, frames, feather_px=0)
        np.testing.assert_array_equal(out, result)

    def test_all_black_frame_skips_blend(self):
        H, W = 40, 50
        split = 20
        fa = _solid(H, W, 100)
        fb = np.zeros((H, W, 3), dtype=np.uint8)  # all black — no content
        ownership = _make_ownership(H, W, split)
        result = _solid(H, W, 100)
        out = _feather_gc_boundaries(result, ownership, [fa, fb], feather_px=8)
        # fb has no content → blend_here is False everywhere → out equals result
        np.testing.assert_array_equal(out, result)

    def test_gc_feather_px_flag_is_nonneg_int(self):
        assert isinstance(_GC_FEATHER_PX, int)
        assert _GC_FEATHER_PX >= 0


class TestEqualizeWarpedGains:
    """§4.10: Pre-seam global frame gain equalization."""

    def test_single_frame_returns_copy(self):
        frame = np.full((32, 40, 3), 100, dtype=np.uint8)
        result = _equalize_warped_gains([frame])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], frame)
        assert result[0] is not frame

    def test_luminance_step_reduced(self):
        H, W = 32, 40
        f0 = np.full((H, W, 3), 100, dtype=np.uint8)
        f1 = np.full((H, W, 3), 150, dtype=np.uint8)
        result = _equalize_warped_gains([f0, f1])
        np.testing.assert_array_equal(result[0], f0)
        mean_orig = float(f1.mean())
        mean_corr = float(result[1].mean())
        mean_ref = float(f0.mean())
        assert abs(mean_corr - mean_ref) < abs(mean_orig - mean_ref)

    def test_black_frame_not_corrected(self):
        H, W = 16, 20
        f0 = np.full((H, W, 3), 120, dtype=np.uint8)
        f1 = np.zeros((H, W, 3), dtype=np.uint8)
        result = _equalize_warped_gains([f0, f1])
        np.testing.assert_array_equal(result[1], f1)

    def test_output_length_matches_input(self):
        frames = [np.full((16, 20, 3), 80 + i * 20, dtype=np.uint8) for i in range(5)]
        result = _equalize_warped_gains(frames)
        assert len(result) == 5

    def test_global_gain_comp_flag_is_bool(self):
        assert isinstance(_GLOBAL_GAIN_COMP, bool)



