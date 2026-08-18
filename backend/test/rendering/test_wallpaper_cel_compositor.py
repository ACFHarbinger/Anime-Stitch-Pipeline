"""Unit tests for hero cel compositing (#428)."""

from __future__ import annotations

import numpy as np
import pytest

from asp_backend.rendering.wallpaper import CelCompositeResult, composite_hero_cel, select_hero_cel
from .wallpaper._synthetic_scene import H, W, IDENTITY, add_char, bg_mask_for, make_background


def test_composite_hero_cel_rigid_placement():
    bg = make_background(level=150)
    char_frame = add_char(bg, x=100, y=60, w=50, h=100, color=(10, 20, 30))
    fg_mask = (~(bg_mask_for(100, y=60, w=50, h=100) == 255)).astype(np.uint8) * 255

    hero = select_hero_cel([char_frame], [fg_mask])
    plate = make_background(level=200)

    res = composite_hero_cel(plate, hero, IDENTITY, blend_mode="feather")

    assert isinstance(res, CelCompositeResult)
    assert res.composite.shape == (H, W, 3)
    assert res.hero_bbox_canvas == (100, 60, 150, 160)
    assert res.cel_canvas_mask.shape == (H, W)

    # Check that character pixels in composite reflect the hero cel color
    center_val = res.composite[110, 125]
    assert np.allclose(center_val, (10, 20, 30), atol=5)

    # Check background outside the cel is unchanged plate
    plate_val = res.composite[10, 10]
    assert np.allclose(plate_val, plate[10, 10], atol=1)


def test_composite_hero_cel_affine_translation():
    bg = make_background(level=150)
    char_frame = add_char(bg, x=10, y=10, w=30, h=40, color=(200, 50, 50))
    fg_mask = (~(bg_mask_for(10, y=10, w=30, h=40) == 255)).astype(np.uint8) * 255
    hero = select_hero_cel([char_frame], [fg_mask])

    plate = make_background(level=100)

    # Translate +50 in x, +30 in y
    affine = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 30.0]], dtype=np.float32)

    res = composite_hero_cel(plate, hero, affine, blend_mode="feather")
    assert res.hero_bbox_canvas == (60, 40, 90, 80)
    assert np.allclose(res.composite[55, 75], (200, 50, 50), atol=5)
