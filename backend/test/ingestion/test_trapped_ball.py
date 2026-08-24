"""Unit tests for Trapped-Ball anime line-art and cel segmentation (Zhang et al. 2009)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from asp_backend.ingestion.trapped_ball import (
    compute_trapped_ball_masks,
    get_trapped_ball_radius,
    trapped_ball_enabled,
    trapped_ball_segmentation,
)


def test_trapped_ball_enabled_flag(monkeypatch):
    monkeypatch.delenv("ASP_TRAPPED_BALL", raising=False)
    assert not trapped_ball_enabled()
    monkeypatch.setenv("ASP_TRAPPED_BALL", "1")
    assert trapped_ball_enabled()


def test_trapped_ball_radius_env(monkeypatch):
    monkeypatch.delenv("ASP_TRAPPED_BALL_RADIUS", raising=False)
    assert get_trapped_ball_radius() == 4
    monkeypatch.setenv("ASP_TRAPPED_BALL_RADIUS", "6")
    assert get_trapped_ball_radius() == 6


def test_trapped_ball_closed_contour_isolation():
    """Verify trapped-ball treats the interior of a closed contour as foreground."""
    H, W = 100, 100
    # Clean white background
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    # Draw a solid black closed box (character silhouette boundary) in the middle
    cv2.rectangle(img, (30, 30), (70, 70), (0, 0, 0), thickness=2)

    bg_mask = trapped_ball_segmentation(img, ball_radius=3)

    # Outside borders must be background (255)
    assert bg_mask[5, 5] == 255
    assert bg_mask[95, 95] == 255
    # Inside the box must be foreground (0) because the line art is closed
    assert bg_mask[50, 50] == 0


def test_trapped_ball_gapped_line_art_closure():
    """Verify trapped-ball closes small line-art gaps (< 2r) and isolates the interior."""
    H, W = 100, 100
    img = np.full((H, W, 3), 255, dtype=np.uint8)

    # Draw a box with a 4px gap on the right side
    # Top, bottom, left
    cv2.line(img, (30, 30), (70, 30), (0, 0, 0), thickness=2)
    cv2.line(img, (30, 70), (70, 70), (0, 0, 0), thickness=2)
    cv2.line(img, (30, 30), (30, 70), (0, 0, 0), thickness=2)
    # Right line with gap from y=48 to y=52 (4px gap)
    cv2.line(img, (70, 30), (70, 48), (0, 0, 0), thickness=2)
    cv2.line(img, (70, 52), (70, 70), (0, 0, 0), thickness=2)

    # Ball radius = 4 (diameter = 9) easily bridges a 4px gap
    bg_mask = trapped_ball_segmentation(img, ball_radius=4)

    # Outer border is background
    assert bg_mask[5, 5] == 255
    # Interior remains foreground (0) despite the 4px gap in line art
    assert bg_mask[50, 50] == 0


def test_compute_trapped_ball_masks_batch():
    H, W = 64, 64
    img1 = np.full((H, W, 3), 200, dtype=np.uint8)
    img2 = np.full((H, W, 3), 100, dtype=np.uint8)

    masks = compute_trapped_ball_masks([img1, img2], ball_radius=3)
    assert len(masks) == 2
    assert masks[0].shape == (H, W)
    assert masks[1].shape == (H, W)
