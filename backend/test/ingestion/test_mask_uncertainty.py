"""Unit tests for P4 temporal mask uncertainty and Trapped-Ball disagreement refinement."""

from __future__ import annotations

import numpy as np
import pytest

from asp_backend.ingestion.mask_uncertainty import (
    compute_pairwise_mask_disagreement,
    compute_temporal_mask_uncertainty,
    mask_uncertainty_enabled,
    resolve_disputed_mask_region,
)


def test_mask_uncertainty_enabled_flag(monkeypatch):
    monkeypatch.delenv("ASP_MASK_UNCERTAINTY", raising=False)
    assert not mask_uncertainty_enabled()
    monkeypatch.setenv("ASP_MASK_UNCERTAINTY", "1")
    assert mask_uncertainty_enabled()


def test_compute_pairwise_mask_disagreement():
    """Verify disagreement detector correctly identifies mismatched classifications."""
    H, W = 60, 60
    # Frame A has background everywhere except a 20x20 box at (20, 20)
    mask_a = np.full((H, W), 255, dtype=np.uint8)
    mask_a[20:40, 20:40] = 0

    # Frame B has background everywhere except a 20x20 box at (30, 20)
    mask_b = np.full((H, W), 255, dtype=np.uint8)
    mask_b[30:50, 20:40] = 0

    # Identity alignment
    affine_id = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    disagree, overlap = compute_pairwise_mask_disagreement(mask_a, mask_b, affine_id)
    assert overlap.all()
    # Disagreement should be present in the non-overlapping parts of the boxes
    # Box A (20:30, 20:40) and Box B (40:50, 20:40)
    assert disagree[20:30, 20:40].all()
    assert disagree[40:50, 20:40].all()
    # Where both agree on foreground (30:40, 20:40), there should be NO disagreement
    assert not disagree[30:40, 20:40].any()


def test_resolve_disputed_mask_region():
    """Verify Trapped-Ball refinement labels unresolved disputes as uncertain (128)."""
    H, W = 50, 50
    # Flat background connected to border
    frame_bgr = np.full((H, W, 3), 180, dtype=np.uint8)

    # Simulated BiRefNet error: falsely marked an open background patch (20:30, 20:30) as foreground (0)
    birefnet_mask = np.full((H, W), 255, dtype=np.uint8)
    birefnet_mask[20:30, 20:30] = 0

    # Temporal disagreement flagged over that patch
    disagreement = np.zeros((H, W), dtype=bool)
    disagreement[20:30, 20:30] = True

    refined = resolve_disputed_mask_region(frame_bgr, birefnet_mask, disagreement, ball_radius=3)
    # Trapped-Ball sees open space -> background (255) vs BiRefNet (0) -> marked uncertain (128)
    assert (refined[20:30, 20:30] == 128).all()
    # Undisputed background remains 255
    assert (refined[0:10, 0:10] == 255).all()


def test_compute_temporal_mask_uncertainty_sequence():
    """Verify batch sequence processing computes ternary masks across frames."""
    H, W = 40, 40
    f0 = np.full((H, W, 3), 200, dtype=np.uint8)
    f1 = np.full((H, W, 3), 200, dtype=np.uint8)

    m0 = np.full((H, W), 255, dtype=np.uint8)
    m0[10:20, 10:20] = 0  # Character at (10, 10)

    m1 = np.full((H, W), 255, dtype=np.uint8)
    m1[20:30, 10:20] = 0  # Character moved to (20, 10)

    affines = [
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    ]

    refined = compute_temporal_mask_uncertainty([f0, f1], [m0, m1], affines, ball_radius=2)
    assert len(refined) == 2
    assert refined[0] is not None
    assert refined[1] is not None
    # Ternary values must only be 0, 128, or 255
    unique_vals_0 = set(np.unique(refined[0]))
    assert unique_vals_0.issubset({0, 128, 255})
