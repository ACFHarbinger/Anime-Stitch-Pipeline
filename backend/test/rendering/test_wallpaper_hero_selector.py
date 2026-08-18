"""Unit tests for hero frame scoring and cel extraction (#426)."""

from __future__ import annotations

import numpy as np
import pytest

from asp_backend.rendering.wallpaper import HeroCel, score_candidate_frame, select_hero_cel
from .wallpaper._synthetic_scene import H, W, add_char, bg_mask_for, make_background


def test_score_candidate_frame_penalizes_border_intersection():
    bg = make_background()
    # Centered character (no border intersection)
    centered_frame = add_char(bg, x=100, y=60, w=40, h=80)
    centered_mask = (~(bg_mask_for(100, y=60, w=40, h=80) == 255)).astype(np.uint8) * 255

    # Truncated character touching frame boundary
    truncated_frame = add_char(bg, x=0, y=0, w=40, h=80)
    truncated_mask = (~(bg_mask_for(0, y=0, w=40, h=80) == 255)).astype(np.uint8) * 255

    score_c, bd_c = score_candidate_frame(centered_frame, centered_mask)
    score_t, bd_t = score_candidate_frame(truncated_frame, truncated_mask)

    assert bd_c["border_intersection"] == 0.0
    assert bd_t["border_intersection"] > 0.0
    assert score_c > score_t


def test_select_hero_cel_picks_highest_scoring_frame():
    bg = make_background()
    # Frame 0: small/truncated figure
    f0 = add_char(bg, x=0, y=0, w=30, h=50)
    m0 = (~(bg_mask_for(0, y=0, w=30, h=50) == 255)).astype(np.uint8) * 255

    # Frame 1: large, centered, unclipped figure
    f1 = add_char(bg, x=120, y=40, w=80, h=150)
    m1 = (~(bg_mask_for(120, y=40, w=80, h=150) == 255)).astype(np.uint8) * 255

    # Frame 2: turned/low symmetry figure
    f2 = add_char(bg, x=10, y=100, w=40, h=60)
    m2 = (~(bg_mask_for(10, y=100, w=40, h=60) == 255)).astype(np.uint8) * 255

    hero = select_hero_cel([f0, f1, f2], [m0, m1, m2])

    assert isinstance(hero, HeroCel)
    assert hero.frame_idx == 1
    assert hero.cel_rgba.shape == (H, W, 4)
    assert hero.bbox == (120, 40, 200, 190)
    assert len(hero.all_scores) == 3
    assert hero.all_scores[0][0] == 1


def test_select_hero_cel_honors_manual_override():
    bg = make_background()
    f0 = add_char(bg, x=120, y=40, w=80, h=150)
    m0 = (~(bg_mask_for(120, y=40, w=80, h=150) == 255)).astype(np.uint8) * 255
    f1 = add_char(bg, x=50, y=50, w=40, h=80)
    m1 = (~(bg_mask_for(50, y=50, w=40, h=80) == 255)).astype(np.uint8) * 255

    # Override to frame 1 even though frame 0 has higher score
    hero = select_hero_cel([f0, f1], [m0, m1], override_frame_idx=1)
    assert hero.frame_idx == 1
    assert hero.bbox == (50, 50, 90, 130)


def test_select_hero_cel_multi_subject_detection():
    bg = make_background()
    # Frame with two distinct large characters
    f = add_char(bg, x=50, y=60, w=50, h=100)
    f = add_char(f, x=180, y=60, w=50, h=100)

    m = np.zeros((H, W), dtype=np.uint8)
    m[60:160, 50:100] = 255
    m[60:160, 180:230] = 255

    hero = select_hero_cel([f], [m])
    assert hero.multi_subject_count == 2
    assert hero.bbox == (50, 60, 230, 160)
