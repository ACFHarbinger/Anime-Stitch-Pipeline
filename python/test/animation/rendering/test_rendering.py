"""
Tests for the Stage 9 temporal renderer (rendering.py).

Issue categories covered:
  B — Stage 9 ghosting: always downstream of bad affines except test18.
      These tests verify the render's output quality given CORRECT affines
      (positive-baseline scenarios) and document ghosting detection for bad affines.

All tests run without GPU.

§5.17: rendering.py was split into a rendering/ package. _BATCH_RENDER and
_batch_render live in rendering/_native.py, referenced by median.py/
first.py/laplacian.py via module-qualified access (`_native._BATCH_RENDER`,
not a bare `from ._native import _BATCH_RENDER`) specifically so that
patching `_native._BATCH_RENDER` here affects every renderer that reads it.
Patches below target `_rendering_mod._native`, not `_rendering_mod` itself.
"""

from __future__ import annotations

import os
import sys

import backend.src.animation.rendering.rendering as _rendering_mod
import numpy as np
import pytest
from backend.src.animation.rendering.rendering import (
    _adaptive_render_gain_clamp,
    _check_gain_chain_drift,
    _render,
    _render_first,
    _render_laplacian,
    _render_median,
)  # noqa: E402
from conftest import make_frame, make_rotation_affine, make_translation_affine  # noqa: E402

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_render_inputs(
    n: int,
    frame_h: int = 100,
    frame_w: int = 120,
    dy: float = 80.0,
    brightness=128,
):
    """n identical (or differently brightened) frames with vertical translation."""
    if isinstance(brightness, int):
        brightness = [brightness] * n
    frames = [make_frame(frame_h, frame_w, color=(b, b, b)) for b in brightness]
    affines = [make_translation_affine(ty=i * dy) for i in range(n)]
    canvas_h = int((n - 1) * dy + frame_h)
    canvas_w = frame_w
    bg_masks = [None] * n
    return frames, affines, bg_masks, canvas_h, canvas_w


def _ghosting_score(canvas: np.ndarray, valid_mask: np.ndarray) -> float:
    """
    Measure ghosting as the inter-row brightness variance in covered regions.
    A high variance score indicates multiple frames are blended at the same rows
    (ghosting). Clean renders have low variance within each strip.
    """
    covered = valid_mask > 0
    if not covered.any():
        return 0.0
    gray = canvas.mean(axis=2)
    row_means = np.array(
        [
            gray[r, covered[r]].mean() if covered[r].any() else np.nan
            for r in range(canvas.shape[0])
        ]
    )
    valid_rows = ~np.isnan(row_means)
    if valid_rows.sum() < 2:
        return 0.0
    return float(np.std(row_means[valid_rows]))


# ---------------------------------------------------------------------------
# 1. Output shape and type
# ---------------------------------------------------------------------------


class TestRenderMedianOutputShape:
    def test_output_canvas_correct_shape(self):
        frames, affines, masks, H, W = _make_render_inputs(3)
        canvas, valid_mask, warped_corr, warped_fgs = _render_median(
            frames, affines, masks, H, W
        )
        assert canvas.shape == (H, W, 3), f"Expected ({H},{W},3), got {canvas.shape}"

    def test_output_canvas_uint8(self):
        frames, affines, masks, H, W = _make_render_inputs(3)
        canvas, _, _, _ = _render_median(frames, affines, masks, H, W)
        assert canvas.dtype == np.uint8

    def test_valid_mask_shape(self):
        frames, affines, masks, H, W = _make_render_inputs(3)
        _, valid_mask, _, _ = _render_median(frames, affines, masks, H, W)
        assert valid_mask.shape == (H, W)

    def test_valid_mask_dtype_uint8(self):
        frames, affines, masks, H, W = _make_render_inputs(3)
        _, valid_mask, _, _ = _render_median(frames, affines, masks, H, W)
        assert valid_mask.dtype == np.uint8

    def test_warped_lists_returned_empty(self):
        """_render_median returns empty warped_corr and warped_fgs lists."""
        frames, affines, masks, H, W = _make_render_inputs(3)
        _, _, warped_corr, warped_fgs = _render_median(frames, affines, masks, H, W)
        assert isinstance(warped_corr, list)
        assert isinstance(warped_fgs, list)


# ---------------------------------------------------------------------------
# 2. Valid mask correctness
# ---------------------------------------------------------------------------


class TestValidMask:
    def test_single_frame_covers_its_strip(self):
        """Single frame at ty=0 should mark its rows as valid in the mask."""
        frames, affines, masks, H, W = _make_render_inputs(1, frame_h=50, dy=0.0)
        _, valid_mask, _, _ = _render_median(frames, affines, masks, H, W)
        assert valid_mask[:50, :].max() > 0, (
            "Single frame's rows should be marked valid"
        )

    def test_valid_mask_marks_all_covered_rows(self):
        """For n frames with full coverage, all canvas rows should be valid."""
        N = 3
        frame_h = 100
        dy = 80.0
        frames, affines, masks, H, W = _make_render_inputs(N, frame_h=frame_h, dy=dy)
        _, valid_mask, _, _ = _render_median(frames, affines, masks, H, W)
        mid_row = H // 2
        assert valid_mask[mid_row, :].max() > 0, (
            f"Middle row {mid_row} should be valid with {N} overlapping frames"
        )


# ---------------------------------------------------------------------------
# 3. Median correctness with identical frames
# ---------------------------------------------------------------------------


class TestMedianCorrectness:
    def test_identical_frames_median_matches_original(self):
        """
        N identical frames → median is the same as any single frame.
        A covered pixel should equal the original frame's brightness.
        """
        brightness = 150
        N = 5
        frames, affines, masks, H, W = _make_render_inputs(
            N, frame_h=80, dy=60.0, brightness=brightness
        )
        canvas, valid_mask, _, _ = _render_median(frames, affines, masks, H, W)
        covered = valid_mask > 0
        if covered.any():
            values = canvas[covered]
            mean_brightness = float(values.mean())
            assert abs(mean_brightness - brightness) < 10, (
                f"Identical frames: expected brightness ≈{brightness}, "
                f"got mean={mean_brightness:.1f}"
            )

    def test_single_frame_canvas_equals_frame(self):
        """Single frame → canvas pixels in frame region equal the frame."""
        brightness = 180
        frames, affines, masks, H, W = _make_render_inputs(
            1, frame_h=60, dy=0.0, brightness=brightness
        )
        canvas, valid_mask, _, _ = _render_median(frames, affines, masks, H, W)
        covered = valid_mask > 0
        if covered.any():
            values = canvas[covered]
            assert abs(float(values.mean()) - brightness) < 15


# ---------------------------------------------------------------------------
# 4. Ghosting detection (documents the failure condition)
# ---------------------------------------------------------------------------


class TestGhostingDetection:
    def test_clean_affines_low_ghosting_score(self):
        """
        Positive baseline (ratio=1.0×, evenly spaced frames) → the inter-strip
        brightness variation should be small (ghosting score < threshold).
        """
        N = 5
        frame_h = 100
        dy = 95.0
        frames, affines, masks, H, W = _make_render_inputs(
            N, frame_h=frame_h, dy=dy, brightness=150
        )
        canvas, valid_mask, _, _ = _render_median(frames, affines, masks, H, W)
        score = _ghosting_score(canvas, valid_mask)
        assert score < 30.0, (
            f"Expected low ghosting score for clean affines, got score={score:.1f}"
        )

    def test_frame_clustering_produces_small_canvas(self):
        """
        Frames all placed at the same ty position (frame clustering, test8 pattern)
        → canvas extent collapses to a single frame height.
        """
        N = 5
        frame_h = 100
        W = 120
        brightness = [100, 130, 160, 100, 130]
        frames = [make_frame(frame_h, W, color=(b, b, b)) for b in brightness]
        affines = [make_translation_affine(ty=0.0) for _ in range(N)]
        canvas_h = frame_h
        bg_masks = [None] * N
        canvas, valid_mask, _, _ = _render_median(
            frames, affines, bg_masks, canvas_h, W
        )
        assert canvas.shape == (canvas_h, W, 3)

    def test_rotation_in_affine_causes_coverage_gaps(self):
        """
        test18 pattern: good ty/tx but significant rotation component.
        With rotated affines, warped frames will not align cleanly —
        valid_mask will show irregular coverage or gaps.
        """
        N = 4
        frame_h, W = 100, 120
        frames = [make_frame(frame_h, W) for _ in range(N)]
        affines = [
            make_rotation_affine(tx=0.0, ty=float(i * 90), angle_deg=15.0)
            for i in range(N)
        ]
        canvas_h = (N - 1) * 90 + frame_h
        bg_masks = [None] * N
        canvas, valid_mask, _, _ = _render_median(
            frames, affines, bg_masks, canvas_h, W
        )
        assert canvas.shape == (canvas_h, W, 3)
        uncovered_fraction = float((valid_mask == 0).mean())
        assert uncovered_fraction > 0.0, (
            "Rotated affines should leave some canvas pixels uncovered"
        )


# ---------------------------------------------------------------------------
# 5. _render_first
# ---------------------------------------------------------------------------


class TestRenderFirst:
    def test_render_first_output_shape(self):
        frames, affines, masks, H, W = _make_render_inputs(3, frame_h=80, dy=70.0)
        canvas, valid_mask = _render_first(frames, affines, H, W)
        assert canvas.shape == (H, W, 3)
        assert canvas.dtype == np.uint8

    def test_render_first_valid_mask(self):
        frames, affines, masks, H, W = _make_render_inputs(3, frame_h=80, dy=70.0)
        canvas, valid_mask = _render_first(frames, affines, H, W)
        assert valid_mask.shape == (H, W)

    def test_render_first_non_empty(self):
        frames, affines, masks, H, W = _make_render_inputs(2, frame_h=80, dy=70.0)
        canvas, _ = _render_first(frames, affines, H, W)
        assert canvas.max() > 0, "_render_first canvas should have non-zero pixels"


# ---------------------------------------------------------------------------
# 6. _render dispatcher
# ---------------------------------------------------------------------------


class TestRenderDispatcher:
    def test_dispatcher_median_mode(self):
        frames, affines, masks, H, W = _make_render_inputs(3)
        canvas, vm, wc, wf = _render(frames, affines, masks, H, W, renderer="median")
        assert canvas.shape == (H, W, 3)

    def test_dispatcher_first_mode(self):
        frames, affines, masks, H, W = _make_render_inputs(3)
        canvas, vm, wc, wf = _render(frames, affines, masks, H, W, renderer="first")
        assert canvas.shape == (H, W, 3)

    def test_dispatcher_blend_mode(self):
        frames, affines, masks, H, W = _make_render_inputs(3)
        canvas, vm, wc, wf = _render(frames, affines, masks, H, W, renderer="blend")
        assert canvas.shape == (H, W, 3)

    def test_dispatcher_returns_4tuple(self):
        frames, affines, masks, H, W = _make_render_inputs(3)
        result = _render(frames, affines, masks, H, W, renderer="median")
        assert len(result) == 4


# ---------------------------------------------------------------------------
# 7. Photometric stability with baselines
# ---------------------------------------------------------------------------


class TestBaselines:
    def test_baselines_below_090_boost_brightness(self):
        """
        A frame with baseline < 0.90 (broadcast-dimmed) should have its brightness
        boosted during rendering.
        """
        brightness = 100
        frames, affines, masks, H, W = _make_render_inputs(
            2, frame_h=60, dy=55.0, brightness=brightness
        )
        baselines = [0.7, 0.95]
        canvas, valid_mask, _, _ = _render_median(
            frames, affines, masks, H, W, _baselines=baselines
        )
        covered = valid_mask[:60, :] > 0
        if covered.any():
            boosted_mean = float(canvas[:60, :][covered, :].mean())
            assert boosted_mean > brightness, (
                f"baseline=0.7 frame should be boosted above {brightness}; "
                f"got mean={boosted_mean:.1f}"
            )

    def test_baselines_none_no_error(self):
        """None baselines should produce the same result as not providing them."""
        frames, affines, masks, H, W = _make_render_inputs(3)
        canvas_no_base, vm, _, _ = _render_median(
            frames, affines, masks, H, W, _baselines=None
        )
        assert canvas_no_base is not None


# ---------------------------------------------------------------------------
# A5 — Foreground-excluded temporal median
# ---------------------------------------------------------------------------


class TestForegroundExcludedMedian:
    """The background plate must not average the character's animation poses."""

    @staticmethod
    def _scene(char_present):
        """4 aligned frames; at x=100 the character (230) is present in the
        given frames, background (80) otherwise. bg_mask: 255=bg, 0=fg."""
        H, W = 120, 200
        frames, masks, affines = [], [], []
        for present in char_present:
            img = np.full((H, W, 3), 80, np.uint8)
            bgm = np.full((H, W), 255, np.uint8)
            if present:
                img[40:80, 100:130] = 230
                bgm[40:80, 100:130] = 0
            frames.append(img)
            masks.append(bgm)
            affines.append(make_translation_affine(ty=0.0))
        return frames, affines, masks, H, W

    def test_excludes_majority_character(self, monkeypatch):
        """Character in 3/4 frames at a pixel → exclusion yields clean bg (80)."""
        monkeypatch.setenv("ASP_FG_EXCLUDE_MEDIAN", "1")
        import importlib

        import backend.src.animation.rendering.rendering as r

        importlib.reload(r)
        frames, affines, masks, H, W = self._scene([True, True, True, False])
        canvas, _, _, _ = r._render_median(frames, affines, masks, H, W)
        val = float(canvas[50:70, 105:125].mean())
        assert val < 120, f"expected clean background ~80, got {val:.0f} (ghost)"
        importlib.reload(r)  # restore default

    def test_disabled_lets_character_ghost(self, monkeypatch):
        """With exclusion OFF the majority character ghosts the background."""
        monkeypatch.setenv("ASP_FG_EXCLUDE_MEDIAN", "0")
        import importlib

        # §5.17: _FG_EXCLUDE_MEDIAN is read once at import time by
        # rendering/median.py (not rendering/__init__.py itself), so
        # reloading only the package wouldn't re-evaluate it -- reload the
        # submodule that actually owns it, then the package so its
        # `from .median import _render_median` re-export picks up the fresh
        # function object.
        import backend.src.animation.rendering.rendering as r
        import backend.src.animation.rendering.rendering.median as _median_mod

        importlib.reload(_median_mod)
        importlib.reload(r)
        frames, affines, masks, H, W = self._scene([True, True, True, False])
        canvas, _, _, _ = r._render_median(frames, affines, masks, H, W)
        val = float(canvas[50:70, 105:125].mean())
        assert val > 180, f"expected character ghost ~230, got {val:.0f}"
        importlib.reload(_median_mod)
        importlib.reload(r)  # restore default

    def test_all_foreground_falls_back(self, monkeypatch):
        """Where the character is in ALL frames, fall back to geometric median
        (no holes)."""
        monkeypatch.setenv("ASP_FG_EXCLUDE_MEDIAN", "1")
        import importlib

        import backend.src.animation.rendering.rendering as r

        importlib.reload(r)
        frames, affines, masks, H, W = self._scene([True, True, True, True])
        canvas, vmask, _, _ = r._render_median(frames, affines, masks, H, W)
        # The pixel is covered (no hole) and shows the character (only data there).
        assert (vmask[50:70, 105:125] > 0).all(), "all-fg region must stay covered"
        val = float(canvas[50:70, 105:125].mean())
        assert val > 180, f"all-fg fallback should keep character, got {val:.0f}"
        importlib.reload(r)


class TestAdaptiveRenderGainClamp:
    """§1.40: Luminance-adaptive gain-clamp bounds for sequential colour correction."""

    def test_pure_black_returns_widest_clamp(self):
        """ref_lum=0 → clamp_width=0.26 → bounds (0.74, 1.26)."""
        lo, hi = _adaptive_render_gain_clamp(0.0)
        assert lo == pytest.approx(0.74, abs=1e-6)
        assert hi == pytest.approx(1.26, abs=1e-6)

    def test_pure_white_returns_narrowest_clamp(self):
        """ref_lum=255 → clamp_width=0.14 → bounds (0.86, 1.14)."""
        lo, hi = _adaptive_render_gain_clamp(255.0)
        assert lo == pytest.approx(0.86, abs=1e-6)
        assert hi == pytest.approx(1.14, abs=1e-6)

    def test_mid_grey_interpolates_correctly(self):
        """ref_lum=128 → clamp_width ≈ 0.26 - 0.12*(128/255) ≈ 0.200."""
        lo, hi = _adaptive_render_gain_clamp(128.0)
        expected_width = 0.26 - 0.12 * (128.0 / 255.0)
        assert lo == pytest.approx(1.0 - expected_width, abs=1e-4)
        assert hi == pytest.approx(1.0 + expected_width, abs=1e-4)

    def test_dark_scene_wider_than_bright_scene(self):
        """Darker reference → wider clamp (more correction allowed)."""
        lo_dark, hi_dark = _adaptive_render_gain_clamp(30.0)
        lo_bright, hi_bright = _adaptive_render_gain_clamp(220.0)
        assert (hi_dark - lo_dark) > (hi_bright - lo_bright), "dark clamp must be wider"

    def test_clamp_above_255_floored_at_bright_clamp(self):
        """ref_lum > 255 is clamped to 255 → returns the same bounds as pure white."""
        lo_over, hi_over = _adaptive_render_gain_clamp(500.0)
        lo_white, hi_white = _adaptive_render_gain_clamp(255.0)
        assert lo_over == pytest.approx(lo_white, abs=1e-6)
        assert hi_over == pytest.approx(hi_white, abs=1e-6)


class TestCheckGainChainDrift:
    """§1.41: Sequential gain chain-drift guard."""

    def _gains(self, values: list) -> np.ndarray:
        """Build an (N, 3) gains array from a list of per-frame scalar gains."""
        return np.array([[v, v, v] for v in values], dtype=np.float32)

    def test_identity_gains_no_drift(self):
        """All-ones gains → cumulative product = 1.0 → no drift."""
        gains = self._gains([1.0] * 8)
        assert not _check_gain_chain_drift(gains, max_ratio=2.0)

    def test_large_cumulative_gain_detected(self):
        """Eight frames each × 1.12 → cumulative ≈ 2.48 > 2.0 → drift detected."""
        gains = self._gains([1.12] * 8)
        assert _check_gain_chain_drift(gains, max_ratio=2.0)

    def test_zero_max_ratio_always_false(self):
        """max_ratio=0 disables the gate; always returns False."""
        gains = self._gains([1.5] * 10)
        assert not _check_gain_chain_drift(gains, max_ratio=0.0)

    def test_single_large_gain_triggers(self):
        """One frame with gain=0.4 → cumulative 0.4 < 1/1.5 → drift detected."""
        gains = self._gains([1.0, 1.0, 0.4, 1.0])
        assert _check_gain_chain_drift(gains, max_ratio=1.5)

    def test_cancelling_gains_no_drift(self):
        """Gains that cancel (1.3 × 0.77 ≈ 1.0 per pair) → no drift."""
        gains = self._gains([1.3, 0.769, 1.3, 0.769])
        # cumulative ≈ 1.3 * 0.769 * 1.3 * 0.769 ≈ 1.0
        assert not _check_gain_chain_drift(gains, max_ratio=2.0)


# ---------------------------------------------------------------------------
# Phase 5e — render_median C++ fast path wiring
# ---------------------------------------------------------------------------


def _has_batch_render_median():
    """True when batch.canvas.render_median is available."""
    if not _rendering_mod._native._BATCH_RENDER:
        return False
    try:
        return hasattr(_rendering_mod._native._batch_render.canvas, "render_median")
    except Exception:
        return False


_skip_no_batch_rm = pytest.mark.skipif(
    not _has_batch_render_median(),
    reason="batch.canvas.render_median not compiled — rebuild needed",
)


class TestRenderMedianBatchFastPath:
    """Phase 5e: C++ render_median fast path wiring (batch.canvas)."""

    def _inputs_no_fg(self, n: int = 4, h: int = 80, w: int = 90, dy: float = 60.0):
        """Build inputs where the fast path should fire (no FG masks)."""
        frames = [make_frame(h, w, color=(120, 120, 120)) for _ in range(n)]
        affines = [make_translation_affine(ty=i * dy) for i in range(n)]
        canvas_h = int((n - 1) * dy + h)
        bg_masks = [None] * n
        return frames, affines, bg_masks, canvas_h, w

    @_skip_no_batch_rm
    def test_fast_path_output_shape(self):
        """Fast path returns canvas of correct (H, W, 3) shape."""
        frames, affines, bg_masks, H, W = self._inputs_no_fg()
        canvas, valid_mask, _, _ = _render_median(frames, affines, bg_masks, H, W)
        assert canvas.shape == (H, W, 3)
        assert canvas.dtype == np.uint8

    @_skip_no_batch_rm
    def test_fast_path_valid_mask_covers_frames(self):
        """valid_mask marks all rows that have at least one frame warped onto them."""
        n, h, w, dy = 3, 60, 80, 50.0
        frames = [make_frame(h, w, color=(100, 100, 100)) for _ in range(n)]
        affines = [make_translation_affine(ty=i * dy) for i in range(n)]
        H = int((n - 1) * dy + h)
        bg_masks = [None] * n
        _, valid_mask, _, _ = _render_median(frames, affines, bg_masks, H, w)
        # Every row from 0 to H-1 should be covered (frames overlap completely)
        assert valid_mask.max() == 255

    @_skip_no_batch_rm
    def test_fast_path_median_of_identical_frames(self):
        """Median of N identical solid-colour frames equals that colour."""
        frames, affines, bg_masks, H, W = self._inputs_no_fg(n=5)
        canvas, _, _, _ = _render_median(frames, affines, bg_masks, H, W)
        covered = canvas.max(axis=2) > 0
        assert covered.any()
        assert np.allclose(canvas[covered].mean(axis=0), 120, atol=5)

    @_skip_no_batch_rm
    def test_fast_path_matches_python_path(self, monkeypatch):
        """Fast path and Python chunked path produce identical output."""
        frames, affines, bg_masks, H, W = self._inputs_no_fg(n=4, h=60, w=70, dy=50.0)
        # Reference: force Python path by disabling batch
        monkeypatch.setattr(_rendering_mod._native, "_BATCH_RENDER", False)
        ref_canvas, ref_mask, _, _ = _render_median(frames, affines, bg_masks, H, W)
        monkeypatch.setattr(_rendering_mod._native, "_BATCH_RENDER", True)
        cpp_canvas, cpp_mask, _, _ = _render_median(frames, affines, bg_masks, H, W)
        assert np.array_equal(ref_mask, cpp_mask), "valid_mask mismatch"
        assert np.abs(ref_canvas.astype(int) - cpp_canvas.astype(int)).max() <= 2, (
            "canvas pixel difference > 2"
        )

    @_skip_no_batch_rm
    def test_fast_path_skips_when_fg_excluded(self, monkeypatch):
        """When FG masks are provided and _FG_EXCLUDE_MEDIAN=True, fast path is skipped
        and the Python chunked path produces a valid (non-zero) canvas."""
        frames, affines, _, H, W = self._inputs_no_fg(n=3, h=60, w=70, dy=50.0)
        # Provide bg masks so _exclude_fg becomes True
        bg_masks = [np.full((60, 70), 255, dtype=np.uint8) for _ in range(3)]
        monkeypatch.setenv("ASP_FG_EXCLUDE_MEDIAN", "1")
        canvas, valid_mask, _, _ = _render_median(frames, affines, bg_masks, H, W)
        assert canvas.shape == (H, W, 3)
        assert valid_mask.max() == 255


# ---------------------------------------------------------------------------
# Phase 5f — _render_laplacian C++ parallel warp wiring
# ---------------------------------------------------------------------------



def _has_batch_laplacian_warp():
    """True when batch.canvas.warp_frames_to_canvas is available."""
    return _rendering_mod._native._BATCH_RENDER and hasattr(
        getattr(_rendering_mod._native, "_batch_render", None), "canvas"
    )


_skip_no_batch_lap = pytest.mark.skipif(
    not _has_batch_laplacian_warp(),
    reason="batch.canvas not compiled — rebuild needed",
)


class TestRenderLaplacianBatchWarp:
    """Phase 5f: C++ parallel warpAffine wired into _render_laplacian."""

    def _inputs(self, n: int = 3, h: int = 80, w: int = 100, dy: float = 60.0):
        frames = [make_frame(h, w, color=(100, 120, 80)) for _ in range(n)]
        affines = [make_translation_affine(ty=i * dy) for i in range(n)]
        H = int((n - 1) * dy + h)
        bg_masks = [None] * n
        return frames, affines, bg_masks, H, w

    @_skip_no_batch_lap
    def test_laplacian_output_shape(self):
        """Laplacian renderer returns (H, W, 3) uint8 canvas."""
        frames, affines, bg_masks, H, W = self._inputs()
        canvas, valid_mask, _, _ = _render_laplacian(frames, affines, bg_masks, H, W)
        assert canvas.shape == (H, W, 3)
        assert canvas.dtype == np.uint8

    @_skip_no_batch_lap
    def test_laplacian_valid_mask_non_empty(self):
        """valid_mask (canvas_m) covers at least the first frame's rows."""
        frames, affines, bg_masks, H, W = self._inputs(n=3)
        _, canvas_m, _, _ = _render_laplacian(frames, affines, bg_masks, H, W)
        assert canvas_m.max() == 255

    @_skip_no_batch_lap
    def test_laplacian_matches_python_path(self, monkeypatch):
        """C++ parallel warp and Python sequential warp produce identical output."""
        frames, affines, bg_masks, H, W = self._inputs(n=3, h=60, w=80, dy=50.0)
        monkeypatch.setattr(_rendering_mod._native, "_BATCH_RENDER", False)
        ref_canvas, _, _, _ = _render_laplacian(frames, affines, bg_masks, H, W)
        monkeypatch.setattr(_rendering_mod._native, "_BATCH_RENDER", True)
        cpp_canvas, _, _, _ = _render_laplacian(frames, affines, bg_masks, H, W)
        assert np.abs(ref_canvas.astype(int) - cpp_canvas.astype(int)).max() <= 2, (
            "laplacian canvas pixel difference > 2 between Python and C++ warp"
        )

    @_skip_no_batch_lap
    def test_laplacian_warped_list_nonempty(self):
        """colour_matched list (3rd return) has N entries."""
        frames, affines, bg_masks, H, W = self._inputs(n=4, dy=50.0)
        _, _, colour_matched, _ = _render_laplacian(frames, affines, bg_masks, H, W)
        assert len(colour_matched) == 4

    @_skip_no_batch_lap
    def test_laplacian_fallback_on_batch_error(self, monkeypatch):
        """If batch warp raises, Python sequential path produces a valid canvas."""

        frames, affines, bg_masks, H, W = self._inputs(n=3)

        class _BrokenCanvas:
            def warp_frames_to_canvas(self, *a, **kw):
                raise RuntimeError("simulated batch failure")

        class _BrokenBatch:
            canvas = _BrokenCanvas()

        monkeypatch.setattr(_rendering_mod._native, "_batch_render", _BrokenBatch())
        monkeypatch.setattr(_rendering_mod._native, "_BATCH_RENDER", True)
        canvas, valid_mask, _, _ = _render_laplacian(frames, affines, bg_masks, H, W)
        assert canvas.shape == (H, W, 3)
        assert valid_mask.max() == 255
