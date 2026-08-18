"""Shared synthetic-scene fixtures for the wallpaper plate-builder tests.

Synthetic scenes only — no GPU, no fixtures.  Geometry mirrors asp_test97:
a static camera (identity affines) and a character walking horizontally across
a known static background.
"""

from __future__ import annotations

import cv2
import numpy as np

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
