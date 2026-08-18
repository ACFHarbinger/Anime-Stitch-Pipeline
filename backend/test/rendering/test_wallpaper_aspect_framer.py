"""Unit tests for aspect framer and background extension solver (#429)."""

from __future__ import annotations

import numpy as np
import pytest

from asp_backend.rendering.wallpaper import (
    FramedWallpaper,
    composite_hero_cel,
    frame_wallpaper,
    select_hero_cel,
)
from .wallpaper._synthetic_scene import H, W, IDENTITY, add_char, bg_mask_for, make_background


def test_frame_wallpaper_standard_16_9():
    bg = make_background(level=120)
    char_frame = add_char(bg, x=140, y=80, w=40, h=80, color=(10, 10, 10))
    fg_mask = (~(bg_mask_for(140, y=80, w=40, h=80) == 255)).astype(np.uint8) * 255

    hero = select_hero_cel([char_frame], [fg_mask])
    comp_res = composite_hero_cel(bg, hero, IDENTITY)

    valid_mask = np.ones((H, W), dtype=bool)

    framed = frame_wallpaper(
        comp_res.composite,
        valid_mask,
        comp_res.hero_bbox_canvas,
        aspect="16:9",
    )

    assert isinstance(framed, FramedWallpaper)
    assert framed.target_aspect == "16:9"
    out_h, out_w = framed.wallpaper.shape[:2]
    # Check 16:9 ratio within integer rounding
    assert abs((out_w / out_h) - (16.0 / 9.0)) < 0.05
    assert not framed.needs_generative_outpaint


def test_frame_wallpaper_overflow_triggers_outpaint_flag():
    # Small canvas with tall/wide aspect forcing overflow
    small_h, small_w = 100, 100
    comp = np.full((small_h, small_w, 3), 128, dtype=np.uint8)
    valid_mask = np.ones((small_h, small_w), dtype=bool)
    hero_bbox = (20, 20, 80, 80)  # 60x60 square

    # 21:9 on a 60x60 box requires width ~140 > 100
    framed = frame_wallpaper(
        comp,
        valid_mask,
        hero_bbox,
        aspect="21:9",
        allow_outpaint=True,
    )

    assert framed.void_ratio > 0.10
    assert framed.needs_generative_outpaint
    assert framed.wallpaper.shape[1] >= 140
