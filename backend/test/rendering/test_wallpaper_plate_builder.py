"""Tests for the wallpaper background plate builder (#427).

Synthetic scenes only — no GPU, no fixtures.  Geometry mirrors asp_test97:
a static camera (identity affines) and a character walking horizontally across
a known static background.
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from asp_backend.rendering.wallpaper import BackgroundPlate, build_background_plate
from asp_backend.rendering.wallpaper._plate_builder import _generative_outpaint

H, W = 240, 320
IDENTITY = np.eye(3)[:2]


def make_background(level: int = 130) -> np.ndarray:
    bg = np.full((H, W, 3), level, dtype=np.uint8)
    grad = np.linspace(0, 40, H)[:, None, None].astype(np.uint8)
    return np.clip(bg.astype(np.int32) + grad, 0, 255).astype(np.uint8)


def add_char(
    frame: np.ndarray,
    x: int,
    y: int = 60,
    w: int = 40,
    h: int = 120,
    color: tuple[int, int, int] = (30, 30, 30),
) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), color, -1)
    return out


def char_rect(x: int, y: int = 60, w: int = 40, h: int = 120) -> tuple[int, int, int, int]:
    return (x, y, w, h)


def bg_mask_for(x: int, y: int = 60, w: int = 40, h: int = 120) -> np.ndarray:
    # Pipeline convention (see _warp_inputs): uint8 0/255, True=background.
    mask = np.full((H, W), 255, dtype=np.uint8)
    mask[y : y + h, x : x + w] = 0
    return mask


def rect_footprint(x: int, y: int = 60, w: int = 40, h: int = 120) -> np.ndarray:
    return (~bg_mask_for(x, y, w, h)).astype(bool)


def test_plate_recovers_background_hero_footprint_excluded() -> None:
    bg = make_background()
    xs = [20, 60, 100, 140, 180, 220]
    frames = [add_char(bg, x) for x in xs]
    affines = [IDENTITY.copy()] * len(frames)
    bg_masks = [bg_mask_for(x) for x in xs]

    hero_x = xs[2]
    footprint = rect_footprint(hero_x)
    plate = build_background_plate(
        frames, affines, bg_masks, footprint, (H, W), hero_frame_idx=2
    )

    assert isinstance(plate, BackgroundPlate)
    assert plate.plate.shape == (H, W, 3)
    assert plate.gains.shape == (len(frames),)
    np.testing.assert_allclose(plate.gains, 1.0, atol=1e-3)

    # Every canvas pixel got a real background sample (char always elsewhere).
    assert plate.valid_mask.all()
    assert plate.sample_count.min() >= 1
    assert plate.void_ratio == 0.0
    assert not plate.inpainted_mask.any()

    # Hero footprint region is filled from OTHER frames' backgrounds, so the
    # plate must match the static background there, not the character.
    x, y, w, h = char_rect(hero_x)
    err = np.abs(plate.plate[y : y + h, x : x + w].astype(int) - bg[y : y + h, x : x + w].astype(int))
    assert err.max() <= 3
    assert plate.sample_count[y : y + h, x : x + w].min() == len(frames) - 1


def test_stationary_char_creates_hole_and_inpaint_fills() -> None:
    bg = make_background()
    x, y, w, h = char_rect(140)
    frames = [add_char(bg, x) for _ in range(6)]
    affines = [IDENTITY.copy()] * 6
    bg_masks = [bg_mask_for(x)] * 6
    footprint = rect_footprint(x)  # hero cel sits where the char always is
    plate = build_background_plate(frames, affines, bg_masks, footprint, (H, W), hero_frame_idx=0)

    # No frame ever shows background inside the char rect -> hole.
    hole = plate.inpainted_mask
    assert hole[y : y + h, x : x + w].mean() >= 0.95
    assert plate.sample_count[y : y + h, x : x + w].max() == 0
    assert plate.void_ratio == pytest.approx((w * h) / (H * W), abs=1e-3)
    assert not plate.valid_mask[y : y + h, x : x + w].any()

    # Telea fills the hole from the surrounding background; the character's
    # dark pixels must be gone from the plate.
    filled = plate.plate[y : y + h, x : x + w].astype(int)
    assert filled.max() > 90


def test_joint_gain_compensation_applied() -> None:
    bg0 = add_char(make_background(100), x=40)
    bg1 = add_char(make_background(80), x=200)
    frames = [bg0, bg1]
    affines = [IDENTITY.copy(), IDENTITY.copy()]
    bg_masks = [bg_mask_for(40), bg_mask_for(200)]
    footprint = np.zeros((H, W), dtype=bool)

    plate = build_background_plate(frames, affines, bg_masks, footprint, (H, W))

    assert plate.gains.shape == (2,)
    # The solve regularizes toward 1.0 but must pull the two frames together.
    overlap = (bg_mask_for(40) & bg_mask_for(200)).astype(bool)
    lum = lambda f: f[overlap].astype(np.float64).dot(np.array([0.114, 0.587, 0.299])).mean()
    d_raw = abs(lum(bg0) - lum(bg1))
    d_corr = abs(lum(bg0) * plate.gains[0] - lum(bg1) * plate.gains[1])
    assert d_corr < 0.5 * d_raw


def test_none_bg_masks_fall_back_to_presence() -> None:
    bg = make_background()
    xs = [20, 120, 220]
    frames = [add_char(bg, x) for x in xs]
    affines = [IDENTITY.copy()] * 3
    footprint = rect_footprint(xs[1])

    plate = build_background_plate(frames, affines, [None, None, None], footprint, (H, W))

    assert plate.valid_mask.all()
    assert plate.void_ratio == 0.0
    x, y, w, h = char_rect(xs[1])
    err = np.abs(plate.plate[y : y + h, x : x + w].astype(int) - bg[y : y + h, x : x + w].astype(int))
    assert err.max() <= 3


def test_generative_outpaint_noop_without_generator() -> None:
    plate_arr = np.zeros((8, 8, 3), dtype=np.uint8)
    void = np.zeros((8, 8), dtype=bool)
    void[0, 0] = True
    out, mask = _generative_outpaint(plate_arr, void)
    assert out is plate_arr
    assert not mask.any()


def test_generative_outpaint_calls_only_above_threshold() -> None:
    calls = []

    def generator(plate, void_mask):
        calls.append(1)
        return plate, void_mask

    small = np.zeros((8, 8), dtype=bool)
    small[0, 0] = True  # 1/64 ≈ 1.6% < 10%
    out, mask = _generative_outpaint(np.zeros((8, 8, 3), dtype=np.uint8), small, generator=generator)
    assert not calls
    assert out.shape == (8, 8, 3)

    large = np.ones((8, 8), dtype=bool)
    out, mask = _generative_outpaint(np.zeros((8, 8, 3), dtype=np.uint8), large, generator=generator)
    assert len(calls) == 1
    assert mask.all()


def test_temporal_decimation_caps_sample_count() -> None:
    bg = make_background()
    xs = [10 + 20 * i for i in range(12)]
    frames = [add_char(bg, x) for x in xs]
    affines = [IDENTITY.copy()] * len(frames)
    bg_masks = [bg_mask_for(x) for x in xs]

    plate = build_background_plate(
        frames, affines, bg_masks, np.zeros((H, W), dtype=bool), (H, W), max_samples=6
    )
    assert plate.valid_mask.all()
    assert plate.sample_count.max() <= 6
    assert plate.sample_count.min() >= 1


def test_api_validates_inputs() -> None:
    bg = make_background()
    foot = np.zeros((H, W), dtype=bool)
    with pytest.raises(ValueError, match="non-empty"):
        build_background_plate([], [], [], foot, (H, W))
    with pytest.raises(ValueError, match="same length"):
        build_background_plate([bg], [IDENTITY.copy()], [None, None], foot, (H, W))
    with pytest.raises(ValueError, match="shape"):
        build_background_plate([bg], [IDENTITY.copy()], [None], np.zeros((10, 10), dtype=bool), (H, W))