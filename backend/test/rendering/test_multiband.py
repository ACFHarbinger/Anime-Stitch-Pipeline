"""Unit tests for M5 Multi-band / Laplacian pyramid blending in Stage 11 compositing."""

from __future__ import annotations

import numpy as np
import pytest

from asp_backend.rendering.compositing._multiband import multiband_blend_background


def test_multiband_reconstruction_identical_inputs():
    """Invariant: blending identical inputs across any mask must yield identity."""
    H, W = 128, 128
    img = np.random.randint(20, 230, (H, W, 3), dtype=np.uint8)
    mask = np.random.rand(H, W).astype(np.float32)

    blended = multiband_blend_background(img, img, mask, levels=4, hf_lock=True)
    # Numerical reconstruction must match within 1 LSB
    diff = np.abs(blended.astype(np.int16) - img.astype(np.int16))
    assert diff.max() <= 1


def test_multiband_smooth_gradient_transition():
    """Verify that multi-band blending smoothly transitions low frequencies."""
    H, W = 64, 128
    # Side A: Dark background (val=50), Side B: Bright background (val=150)
    img_a = np.full((H, W, 3), 50, dtype=np.uint8)
    img_b = np.full((H, W, 3), 150, dtype=np.uint8)

    # Vertical split at W=64
    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, :64] = 1.0

    blended = multiband_blend_background(img_a, img_b, mask, levels=4, hf_lock=False)
    # Check that middle pixels (near seam x=64) smoothly bridge between 50 and 150
    mid_vals = blended[H // 2, 50:78, 0]
    assert 50 < mid_vals[14] < 150  # Right at x=64
    assert np.all(np.diff(mid_vals) >= 0)  # Monotonic transition


def test_multiband_hf_lock_preserves_sharp_boundary():
    """Verify that hf_lock=True preserves high-frequency detail on each side."""
    H, W = 64, 64
    img_a = np.full((H, W, 3), 40, dtype=np.uint8)
    img_b = np.full((H, W, 3), 200, dtype=np.uint8)

    # Add single high-frequency black line on side A away from extreme boundary
    img_a[:, 16] = [0, 0, 0]

    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, :32] = 1.0

    blended = multiband_blend_background(img_a, img_b, mask, levels=3, hf_lock=True)
    # Sharp line at x=16 on side A should be sharply preserved
    assert blended[H // 2, 16, 0] <= 5
    # Pixels on either side of the line should retain their side A value (~40)
    assert 35 <= blended[H // 2, 14, 0] <= 45


def test_multiband_bg_mask_bypass():
    """Verify that non-background (character) pixels bypass multi-band smoothing."""
    H, W = 64, 64
    img_a = np.full((H, W, 3), 40, dtype=np.uint8)
    img_b = np.full((H, W, 3), 180, dtype=np.uint8)

    # Character patch at center on side A (val=255)
    img_a[20:44, 20:44] = 255

    bg_mask_a = np.ones((H, W), dtype=bool)
    bg_mask_a[20:44, 20:44] = False  # FG character cel
    bg_mask_b = np.ones((H, W), dtype=bool)

    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, :32] = 1.0

    blended = multiband_blend_background(
        img_a, img_b, mask, levels=3, hf_lock=True, bg_mask_a=bg_mask_a, bg_mask_b=bg_mask_b
    )
    # Character area should take exact value 255 without being smeared into side B
    assert np.all(blended[22:42, 22:30] == 255)
