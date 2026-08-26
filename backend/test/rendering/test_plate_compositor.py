"""Unit tests for P1 canvas-aligned background plate + single foreground pose compositor."""

from __future__ import annotations

import numpy as np
import pytest

from asp_backend.rendering.compositing._plate_compositor import (
    _build_aligned_background_plate,
    composite_plate_single_pose,
    plate_single_pose_enabled,
)


def test_plate_compositor_enabled_flag(monkeypatch):
    monkeypatch.delenv("ASP_PLATE_SINGLE_POSE", raising=False)
    assert not plate_single_pose_enabled()
    monkeypatch.setenv("ASP_PLATE_SINGLE_POSE", "1")
    assert plate_single_pose_enabled()


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
    result, claimed, meta = composite_plate_single_pose([f0, f1], [bg0, bg1], canvas, soft_edge_px=0)

    assert meta["n_claimed_pixels"] > 0
    assert len(meta["zones"]) == 1
    # Frame 1 has larger coverage so it must be chosen
    assert meta["zones"][0]["chosen_frame"] == 1
    assert np.all(result[35:55, 35:55] == 250)


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
    # In the patch (20:40, 20:40), Frame 1 was excluded, so the plate must match Frame 0 (100) exactly
    assert np.all(plate[20:40, 20:40] == 100)


