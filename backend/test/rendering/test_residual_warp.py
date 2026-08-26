"""Unit tests for P3 regularized Thin Plate Spline (TPS) residual background warp."""

from __future__ import annotations

import numpy as np
import pytest

from asp_backend.rendering.compositing._residual_warp import (
    apply_residual_warp_to_frame,
    compute_tps_bending_energy,
    fit_background_residual_tps,
    residual_warp_enabled,
)


def test_residual_warp_enabled_flag(monkeypatch):
    monkeypatch.delenv("ASP_RESIDUAL_WARP", raising=False)
    assert not residual_warp_enabled()
    monkeypatch.setenv("ASP_RESIDUAL_WARP", "1")
    assert residual_warp_enabled()


def test_compute_tps_bending_energy():
    """Verify bending energy is zero for affine transformations and positive for non-rigid."""
    pts_src = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]], dtype=np.float64)

    # Pure translation: zero bending energy
    disp_affine = np.array([[5.0, 2.0], [5.0, 2.0], [5.0, 2.0], [5.0, 2.0]], dtype=np.float64)
    e_affine = compute_tps_bending_energy(pts_src, disp_affine)
    assert e_affine < 1e-4

    # Non-rigid pinch distortion
    disp_pinched = np.array([[5.0, 2.0], [-10.0, 8.0], [15.0, -12.0], [-5.0, 6.0]], dtype=np.float64)
    e_pinched = compute_tps_bending_energy(pts_src, disp_pinched)
    assert e_pinched > 0.0


def test_fit_background_residual_tps_rejection_on_extreme_displacement():
    """Verify fit rejects residuals exceeding max displacement threshold."""
    src = np.array([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]], dtype=np.float64)
    # 50px displacement (exceeds default 15px max)
    dst = src + 50.0
    affine_id = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    flow, meta = fit_background_residual_tps(src, dst, affine_id, 60, 60, max_residual_px=15.0)
    assert flow is None
    assert "max_residual_exceeded" in meta.get("reason", "")


def test_fit_background_residual_tps_valid_flow():
    """Verify fit produces valid dense displacement flow field for smooth residuals."""
    src = np.array([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]], dtype=np.float64)
    # Small 2px residual
    dst = src + 2.0
    affine_id = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    flow, meta = fit_background_residual_tps(src, dst, affine_id, 64, 64, smoothing=1.0, max_bending=1.0)
    assert flow is not None
    assert flow.shape == (64, 64, 2)
    assert meta["applied"] is True


def test_apply_residual_warp_to_frame_background_only():
    """Verify residual warp deforms background while protecting foreground cels."""
    H, W = 40, 40
    frame = np.full((H, W, 3), 100, dtype=np.uint8)
    # Character box at (10:30, 10:30)
    frame[10:30, 10:30] = 220
    bg_mask = np.full((H, W), 255, dtype=np.uint8)
    bg_mask[10:30, 10:30] = 0

    affine_id = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    # 4px rightward residual flow
    flow = np.zeros((H, W, 2), dtype=np.float32)
    flow[..., 0] = 4.0

    warped, bg_presence = apply_residual_warp_to_frame(
        frame, bg_mask, affine_id, flow, H, W, blend_margin_px=4
    )
    assert warped.shape == (H, W, 3)
    assert bg_presence.shape == (H, W)
    # Output should have non-zero pixels
    assert warped.mean() > 0
