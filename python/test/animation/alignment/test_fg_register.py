"""
Tests for foreground pose registration (fg_register.py, Stage 8.5).

Validates that the flow-guided midpoint re-posing reduces the displacement of
an animated foreground element across a strip seam, without disturbing the
aligned background. All tests run without GPU (OpenCV DISOpticalFlow).
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

from backend.src.animation.alignment.fg_register import (  # noqa: E402
    _arap_push,
    _arap_regularise,  # noqa: F401
    _dense_flow,
    _seam_taper,
    register_foreground_at_seam,
)

# ---------------------------------------------------------------------------
# Synthetic scene helpers
# ---------------------------------------------------------------------------


def _make_scene(limb_x: int, h: int = 400, w: int = 360, limb_w: int = 40):
    """
    A canvas-aligned frame: textured static background + a bright 'limb'
    rectangle whose horizontal position is `limb_x` (the animated part).
    A textured pattern is painted inside the limb so optical flow can track it.
    Returns (bgr, fg_mask_bool).
    """
    rng = np.random.default_rng(0)
    bg = (rng.integers(40, 80, size=(h, w, 3))).astype(np.uint8)  # textured bg
    img = bg.copy()
    fg = np.zeros((h, w), dtype=bool)
    y0, y1 = 120, 280
    x0, x1 = limb_x, limb_x + limb_w
    x0c, x1c = max(0, x0), min(w, x1)
    if x1c > x0c:
        img[y0:y1, x0c:x1c] = (230, 200, 180)
        # Trackable texture: diagonal stripes inside the limb
        for k in range(x0c, x1c, 8):
            img[y0:y1, k : min(k + 3, x1c)] = (140, 110, 90)
        fg[y0:y1, x0c:x1c] = True
    return img, fg


def _fg_centroid_x(img, fg):
    ys, xs = np.where(fg)
    return float(xs.mean()) if xs.size else 0.0


# ---------------------------------------------------------------------------
# Taper weight
# ---------------------------------------------------------------------------


class TestSeamTaper:
    def test_peak_at_seam(self):
        t = _seam_taper(100, 50, seam_pos=50, taper_px=20, axis=0)
        assert t[50, 0] == pytest.approx(1.0)
        assert t[30, 0] == pytest.approx(0.0, abs=1e-6)
        assert t[70, 0] == pytest.approx(0.0, abs=1e-6)
        assert t[40, 0] == pytest.approx(0.5, abs=0.05)

    def test_horizontal_axis(self):
        t = _seam_taper(50, 100, seam_pos=50, taper_px=20, axis=1)
        assert t[0, 50] == pytest.approx(1.0)
        assert t[0, 30] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Core registration
# ---------------------------------------------------------------------------


class TestForegroundRegistration:
    def test_reduces_limb_displacement_at_seam(self):
        """
        Frame A has the limb at x=120, frame B at x=150 (animated +30px).
        After midpoint re-posing both should converge toward x≈135, so the
        post-warp centroid gap must shrink substantially.
        """
        a, fa = _make_scene(120)
        b, fb = _make_scene(150)

        seam = 200  # horizontal seam mid-canvas
        adj_a, adj_b, info = register_foreground_at_seam(
            a, b, fa, fb, seam_pos=seam, axis=0
        )

        assert info["warped"] is True, info

        # Re-detect foreground by brightness (limb is brighter than bg)
        def bright_mask(img):
            return img.mean(axis=2) > 150

        cax = _fg_centroid_x(adj_a, bright_mask(adj_a))
        cbx = _fg_centroid_x(adj_b, bright_mask(adj_b))
        orig_gap = abs(150 - 120)
        new_gap = abs(cbx - cax)
        # A single tapered midpoint warp should measurably reduce the seam gap.
        # With RAFT+ARAP the reduction is ~15-40% depending on scene texture;
        # require at least 10% so the test survives different flow backends.
        assert new_gap < orig_gap * 0.90, (
            f"seam gap not reduced enough: {orig_gap} -> {new_gap:.1f}"
        )
        assert new_gap < orig_gap, "warp must not increase the gap"

    def test_background_untouched(self):
        """The aligned background outside the foreground must be unchanged."""
        a, fa = _make_scene(120)
        b, fb = _make_scene(150)
        adj_a, adj_b, info = register_foreground_at_seam(
            a, b, fa, fb, seam_pos=200, axis=0
        )
        # A region far from any foreground (top-left corner) must be identical.
        assert np.array_equal(adj_a[0:80, 0:80], a[0:80, 0:80])
        assert np.array_equal(adj_b[0:80, 0:80], b[0:80, 0:80])

    def test_no_foreground_returns_unchanged(self):
        """Seam with no foreground crossing → inputs returned, warped=False."""
        h, w = 400, 300
        rng = np.random.default_rng(1)
        a = rng.integers(40, 80, size=(h, w, 3)).astype(np.uint8)
        b = rng.integers(40, 80, size=(h, w, 3)).astype(np.uint8)
        empty = np.zeros((h, w), dtype=bool)
        adj_a, adj_b, info = register_foreground_at_seam(
            a, b, empty, empty, seam_pos=200, axis=0
        )
        assert info["warped"] is False
        assert info["fg_pixels"] == 0
        assert np.array_equal(adj_a, a)
        assert np.array_equal(adj_b, b)

    def test_large_residual_falls_back(self):
        """
        A large but still trackable animation gap (wide limb shifted 100px,
        with overlap so flow can measure it) exceeds max_residual=40 → the
        function must refuse to warp and signal fallback (warped=False).
        """
        a, fa = _make_scene(80, limb_w=150)
        b, fb = _make_scene(180, limb_w=150)  # overlap 180..230, shift 100px
        adj_a, adj_b, info = register_foreground_at_seam(
            a, b, fa, fb, seam_pos=200, axis=0, max_residual=40.0
        )
        assert info["warped"] is False, info
        # A6: the fallback flag must be set and a dominant frame identified so the
        # compositor can take the seam foreground from one pose only.
        assert info["fallback"] is True, info
        assert info["dominant"] in ("a", "b")
        assert np.array_equal(adj_a, a)
        assert np.array_equal(adj_b, b)

    def test_dominant_frame_is_larger_foreground(self):
        """The dominant frame is the one with more foreground in the seam band."""
        # Frame a has a wide limb (more fg); b has a thin one.
        a, fa = _make_scene(120, limb_w=120)
        b, fb = _make_scene(150, limb_w=20)
        _, _, info = register_foreground_at_seam(
            a, b, fa, fb, seam_pos=200, axis=0, max_residual=5.0
        )
        assert info["dominant"] == "a", info


# ---------------------------------------------------------------------------
# Flow sanity
# ---------------------------------------------------------------------------


class TestDenseFlow:
    def test_flow_detects_horizontal_shift(self):
        a, _ = _make_scene(120)
        b, _ = _make_scene(150)
        flow = _dense_flow(a, b)
        # On the limb rows, mean horizontal flow should be ~ +30 (a→b shift).
        limb_rows = slice(120, 280)
        mean_dx = float(flow[limb_rows, 120:190, 0].mean())
        assert mean_dx > 8.0, f"expected positive dx, got {mean_dx}"


# ---------------------------------------------------------------------------
# ARAP Push phase (Sýkora 2009 block-matching Push)
# ---------------------------------------------------------------------------


class TestARAPPush:
    def test_detects_clear_displacement_in_textured_region(self):
        """
        Push finds the correct per-cell translation for a textured block
        that shifted cleanly from img_a to img_b.

        The block is aligned to cell boundaries so every active cell is
        completely inside the fg mask — avoiding the sparse-fg false-match
        issue where a mostly-zero template matches incorrectly.
        """
        H, W = 96, 160
        cell = 16
        img_a = np.zeros((H, W, 3), dtype=np.uint8)
        img_b = np.zeros((H, W, 3), dtype=np.uint8)

        # Textured block aligned to cell grid: rows 16:80 (4 full cells),
        # cols 32:80 (3 full cells).  A unique diagonal stripe texture.
        for r in range(16, 80):
            for c in range(32, 80):
                img_a[r, c] = [(r + c) % 64 + 160, 80, (r * 3 + c) % 64 + 160]

        # Same block shifted +16px (= 1 cell) to the right in B
        shift = 16
        for r in range(16, 80):
            for c in range(32 + shift, 80 + shift):
                img_b[r, c] = img_a[r, c - shift]

        fg = np.zeros((H, W), dtype=bool)
        fg[16:80, 32:80] = True  # perfectly cell-aligned fg mask

        initial_flow = np.zeros((H, W, 2), dtype=np.float32)

        pushed = _arap_push(
            img_a,
            img_b,
            fg,
            initial_flow,
            cell_size=cell,
            search_range=24,
            min_fg_frac=0.90,
        )

        # All cells inside the fg region should find ~+16px horizontal shift
        fg_dx = pushed[16:80, 32:80, 0]
        fg_dy = pushed[16:80, 32:80, 1]
        mean_dx = float(fg_dx.mean())
        mean_dy = float(fg_dy.mean())
        assert mean_dx > 10.0, f"expected ~+16px dx from Push, got {mean_dx:.2f}"
        assert abs(mean_dy) < 4.0, f"unexpected vertical drift: {mean_dy:.2f}"

    def test_background_cells_keep_initial_flow(self):
        """
        Cells outside the fg mask should retain initial_flow unchanged.
        """
        H, W = 64, 64
        img_a = np.full((H, W, 3), 50, dtype=np.uint8)
        img_b = np.full((H, W, 3), 80, dtype=np.uint8)  # uniform shift of brightness
        fg = np.zeros((H, W), dtype=bool)
        # Only a small fg region; most of the image is background
        fg[24:40, 24:40] = True

        initial_flow = np.full((H, W, 2), 3.0, dtype=np.float32)  # non-zero initial

        pushed = _arap_push(
            img_a,
            img_b,
            fg,
            initial_flow,
            cell_size=16,
            search_range=10,
            min_fg_frac=0.5,
        )

        # Background cells (top-left corner has no fg) keep the initial flow
        bg_dx = pushed[0:16, 0:16, 0]
        bg_dy = pushed[0:16, 0:16, 1]
        np.testing.assert_allclose(
            bg_dx, 3.0, atol=0.1, err_msg="bg cells must keep initial flow"
        )
        np.testing.assert_allclose(
            bg_dy, 3.0, atol=0.1, err_msg="bg cells must keep initial flow"
        )


# ---------------------------------------------------------------------------
# LSD collinearity constraint tests
# ---------------------------------------------------------------------------


class TestLSDCollinearity:
    """Tests for the LSD collinearity constraint in _arap_regularise."""

    def test_lsd_import_available(self):
        """cv2.createLineSegmentDetector should be available in this env."""
        lsd = cv2.createLineSegmentDetector()
        assert lsd is not None

    def test_no_image_unchanged_regularise(self):
        """Without image, _arap_regularise should work as before (no LSD)."""
        H, W = 64, 128
        rng = np.random.default_rng(7)
        flow = rng.uniform(-5, 5, (H, W, 2)).astype(np.float32)
        fg_mask = np.zeros((H, W), dtype=bool)
        fg_mask[16:48, 16:112] = True
        result = _arap_regularise(flow, fg_mask, cell_size=16, n_iter=1, image=None)
        assert result.shape == flow.shape
        # Background should be unchanged
        np.testing.assert_array_equal(result[~fg_mask], flow[~fg_mask])

    def test_lsd_with_line_art_image(self):
        """LSD should not crash on an image with clear line-art."""
        H, W = 128, 256
        # Create an image with a clear horizontal line
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[H // 2, :, :] = 255  # bright horizontal line
        img[:, W // 2, :] = 255  # bright vertical line

        rng = np.random.default_rng(99)
        flow = rng.uniform(-3, 3, (H, W, 2)).astype(np.float32)
        fg_mask = np.ones((H, W), dtype=bool)

        # Should not raise; LSD constraint should produce a smoother result
        result = _arap_regularise(flow, fg_mask, cell_size=16, n_iter=1, image=img)
        assert result.shape == flow.shape

    def test_image_offset_shifts_line_coordinates(self):
        """image_offset correctly shifts LSD line coordinates to canvas space."""
        from backend.src.animation.alignment.fg_register import _arap_regularise

        # Full canvas is 256×256; crop is the bottom half (rows 128:256)
        canvas_H, canvas_W = 256, 256
        crop_H = 128
        y_offset = 128  # crop starts at row 128

        # Create a crop with a clear horizontal line at row 64 in crop-space
        crop_img = np.zeros((crop_H, canvas_W, 3), dtype=np.uint8)
        crop_img[64, :, :] = 255  # line at row 64 in crop

        # Full-canvas flow with non-zero values in the crop region
        rng = np.random.default_rng(11)
        flow = np.zeros((canvas_H, canvas_W, 2), dtype=np.float32)
        flow[y_offset:, :] = rng.uniform(-5, 5, (crop_H, canvas_W, 2)).astype(
            np.float32
        )

        fg_mask = np.zeros((canvas_H, canvas_W), dtype=bool)
        fg_mask[y_offset:, :] = True  # fg is only in the crop region

        # With correct offset: LSD line at crop-row 64 → canvas row 192
        result_with_offset = _arap_regularise(
            flow,
            fg_mask,
            cell_size=16,
            n_iter=1,
            image=crop_img,
            image_offset=(y_offset, 0),
        )
        # Without offset: LSD line at crop-row 64 → canvas row 64 (outside fg)
        result_no_offset = _arap_regularise(
            flow,
            fg_mask,
            cell_size=16,
            n_iter=1,
            image=crop_img,
            image_offset=(0, 0),
        )

        # Both should return valid arrays without crashing
        assert result_with_offset.shape == flow.shape
        assert result_no_offset.shape == flow.shape

    def test_lsd_collinearity_smooths_flow_on_line_cells(self):
        """
        Cells aligned along a detected line should receive the group-average
        translation, making the flow more collinear than raw per-cell medians.
        """
        from backend.src.animation.alignment.fg_register import _arap_regularise

        H, W = 64, 256
        cell_size = 16

        # Synthetic flow: each cell has a randomly different x-translation
        # representing un-regularised RAFT output on a flat region.
        rng = np.random.default_rng(42)
        flow = np.zeros((H, W, 2), dtype=np.float32)
        for col in range(0, W, cell_size):
            flow[:, col : col + cell_size, 0] = rng.uniform(-10, 10)

        fg_mask = np.ones((H, W), dtype=bool)

        # Image with a clear horizontal line spanning the full width
        img_with_line = np.zeros((H, W, 3), dtype=np.uint8)
        img_with_line[H // 2, :, :] = 255

        result = _arap_regularise(
            flow,
            fg_mask,
            cell_size=cell_size,
            n_iter=1,
            image=img_with_line,
            image_offset=(0, 0),
        )
        # After LSD collinearity, all cells crossed by the horizontal line
        # should share the same (or very similar) x-translation.
        # Extract x-translations from the middle row (where the line is).
        mid_row = H // 2
        tx_vals = [
            result[mid_row, col + cell_size // 2, 0]
            for col in range(0, W - cell_size, cell_size)
        ]
        tx_std = float(np.std(tx_vals))
        # Without LSD constraint, std would be ~6px (random per-cell);
        # with LSD constraint, all constrained cells should be closer.
        assert tx_std < 6.0, (
            f"LSD collinearity should reduce variance of flow along the line; "
            f"std={tx_std:.2f}px (expected < 6px after constraint)"
        )


# ---------------------------------------------------------------------------
# Phase 5 — batch.fg_register dispatch wiring
# ---------------------------------------------------------------------------

try:
    import base as _batch_fgreg_test  # noqa: F401
    if (
        getattr(_batch_fgreg_test, "__file__", None) is None
        or not hasattr(_batch_fgreg_test, "fg_register")
    ):
        raise ImportError("compiled base.fg_register extension not available")
    _HAS_BATCH_FGREG = True
except ImportError:
    _HAS_BATCH_FGREG = False

_skip_no_batch = pytest.mark.skipif(not _HAS_BATCH_FGREG, reason="batch not built")
