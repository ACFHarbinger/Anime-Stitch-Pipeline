"""Unit tests for P1 canvas-aligned background plate + single foreground pose compositor."""

from __future__ import annotations

import numpy as np
from asp_backend.rendering.compositing._gain_compensation import _apply_joint_gain_solve
from asp_backend.rendering.compositing._normalization import _warp_inputs
from asp_backend.rendering.compositing._plate_compositor import (
    _build_aligned_background_plate,
    composite_plate_single_pose,
    plate_single_pose_enabled,
    plate_single_pose_safe_for_phases,
)


def test_plate_compositor_enabled_flag(monkeypatch):
    monkeypatch.delenv("ASP_PLATE_SINGLE_POSE", raising=False)
    assert not plate_single_pose_enabled()
    monkeypatch.setenv("ASP_PLATE_SINGLE_POSE", "1")
    assert plate_single_pose_enabled()


def test_plate_single_pose_rejects_multiple_phases():
    assert plate_single_pose_safe_for_phases(None, 3)
    assert plate_single_pose_safe_for_phases([0, 0, 0], 3)
    assert not plate_single_pose_safe_for_phases([0, 1, 1], 3)
    assert not plate_single_pose_safe_for_phases([0, 0, 0], 2)
    assert not plate_single_pose_safe_for_phases([0, 0, 0], 3, True)


def test_build_aligned_background_plate_reconstruction():
    """Verify plate builder recovers static background from masked inputs."""
    H, W = 100, 100
    # Background is a gradient
    bg_base = np.zeros((H, W, 3), dtype=np.uint8)
    for r in range(H):
        bg_base[r, :, :] = (r * 2) % 255

    # Frame 0: Character box at top (y: 10-30, x: 20-40)
    f0 = bg_base.copy()
    f0[10:30, 20:40] = [255, 0, 0]
    bg0 = np.ones((H, W), dtype=bool)
    bg0[10:30, 20:40] = False

    # Frame 1: Character box at bottom (y: 60-80, x: 20-40)
    f1 = bg_base.copy()
    f1[60:80, 20:40] = [0, 255, 0]
    bg1 = np.ones((H, W), dtype=bool)
    bg1[60:80, 20:40] = False

    plate, valid = _build_aligned_background_plate([f0, f1], [bg0, bg1], H, W)
    assert valid.all()
    # Plate should have cleanly reconstructed bg_base without the character boxes
    diff = np.abs(plate.astype(np.int16) - bg_base.astype(np.int16))
    assert diff.max() <= 1


def test_composite_plate_single_pose_one_character():
    """Verify that composite takes exactly ONE character pose per region."""
    H, W = 120, 120
    bg_base = np.full((H, W, 3), 100, dtype=np.uint8)

    # Frame 0: small character box (area 100)
    f0 = bg_base.copy()
    f0[40:50, 40:50] = 200
    bg0 = np.ones((H, W), dtype=bool)
    bg0[40:50, 40:50] = False

    # Frame 1: complete character box (area 400, strictly superior)
    f1 = bg_base.copy()
    f1[35:55, 35:55] = 250
    bg1 = np.ones((H, W), dtype=bool)
    bg1[35:55, 35:55] = False

    canvas = bg_base.copy()
    result, claimed, meta = composite_plate_single_pose(
        [f0, f1], [bg0, bg1], canvas, soft_edge_px=0
    )

    assert meta["n_claimed_pixels"] > 0
    assert len(meta["zones"]) == 1
    # Frame 1 has larger coverage so it must be chosen
    assert meta["zones"][0]["chosen_frame"] == 1
    assert np.all(result[35:55, 35:55] == 250)


def test_composite_preserves_canvas_where_warp_padding_has_no_content():
    """Zero-filled warp padding must not claim background plate ownership."""
    H, W = 80, 80
    canvas = np.full((H, W, 3), 41, dtype=np.uint8)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[20:60, 20:60] = 120
    bg = np.ones((H, W), dtype=bool)
    valid = np.zeros((H, W), dtype=bool)
    valid[20:60, 20:60] = True

    result, _claimed, _meta = composite_plate_single_pose(
        [frame], [bg], canvas, warped_valid=[valid]
    )

    assert np.all(result[:20] == 41)
    assert np.all(result[20:60, 20:60] == 120)


def test_composite_uses_canvas_for_single_sample_background():
    """P1 must not turn a one-frame plate contribution into a visible strip."""
    H, W = 40, 40
    canvas = np.full((H, W, 3), 41, dtype=np.uint8)
    frame0 = np.full((H, W, 3), 100, dtype=np.uint8)
    frame1 = frame0.copy()
    bg0 = np.ones((H, W), dtype=bool)
    bg1 = np.ones((H, W), dtype=bool)
    valid1 = np.ones((H, W), dtype=bool)
    valid1[10:30, :] = False

    result, _claimed, _meta = composite_plate_single_pose(
        [frame0, frame1], [bg0, bg1], canvas, warped_valid=[np.ones((H, W), bool), valid1],
        soft_edge_px=0,
    )

    assert np.all(result[:10] == 100)
    assert np.all(result[10:30] == 41)


def test_plate_compositor_p2_multiband_and_edge_preserve():
    """Verify P2 multiband blending runs cleanly over the background plate."""
    H, W = 128, 128
    bg0 = np.full((H, W, 3), 50, dtype=np.uint8)
    bg1 = np.full((H, W, 3), 150, dtype=np.uint8)
    mask0 = np.ones((H, W), dtype=bool)
    mask1 = np.ones((H, W), dtype=bool)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    result, claimed, meta = composite_plate_single_pose(
        [bg0, bg1],
        [mask0, mask1],
        canvas,
        edge_preserve=True,
        multiband=True,
        multiband_levels=3,
    )
    assert meta.get("multiband_applied") is True
    # Background should be non-zero and blend smoothly
    assert result.mean() > 0


def test_plate_compositor_p4_ternary_uncertainty_exclusion():
    """Verify that uncertain pixels (value 128) are excluded from background plate generation."""
    H, W = 80, 80
    bg_clean = np.full((H, W, 3), 100, dtype=np.uint8)

    # Frame 0: Clean background everywhere
    f0 = bg_clean.copy()
    m0 = np.full((H, W), 255, dtype=np.uint8)

    # Frame 1: Contaminated leak in patch (20:40, 20:40)
    f1 = bg_clean.copy()
    f1[20:40, 20:40] = 255  # Contaminating color
    # Ternary mask: marks the contaminated patch as uncertain (128)
    m1 = np.full((H, W), 255, dtype=np.uint8)
    m1[20:40, 20:40] = 128

    plate, valid = _build_aligned_background_plate([f0, f1], [m0, m1], H, W)
    assert valid.all()
    # Frame 1 is excluded from the uncertain patch, so the plate matches Frame 0.
    assert np.all(plate[20:40, 20:40] == 100)


def test_warp_inputs_preserves_p4_uncertainty_for_plate_path():
    """The P1/P2 path must retain 128 rather than treating it as background."""
    frame = np.full((8, 8, 3), 100, dtype=np.uint8)
    mask = np.full((8, 8), 255, dtype=np.uint8)
    mask[2:6, 2:6] = 128
    affine = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    _frames, warped_bg, _valid = _warp_inputs(
        [frame], [affine], [mask], 8, 8, 1, include_valid=True, preserve_ternary=True
    )

    assert warped_bg[0].dtype == np.uint8
    assert np.all(warped_bg[0][2:6, 2:6] == 128)


def test_joint_gain_uses_boolean_selector_for_p4_masks():
    """Ternary P4 masks must not become uint8 advanced-index selectors."""
    frames = [
        np.full((8, 8, 3), 100, dtype=np.uint8),
        np.full((8, 8, 3), 120, dtype=np.uint8),
    ]
    masks = [np.full((8, 8), 255, dtype=np.uint8) for _ in frames]
    masks[0][2:6, 2:6] = 128

    out = _apply_joint_gain_solve(frames, masks)

    assert len(out) == 2
    assert out[0].shape == frames[0].shape
