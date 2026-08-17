"""M3 red-set crop-loss helper tests (no dump, no GPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SRC = (
    Path(__file__).resolve().parents[3]
    / "backend"
    / "benchmark"
    / "screen_coherence_v2.py"
)
_spec = importlib.util.spec_from_file_location("screen_coherence_v2", _SRC)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _box(h: int, w: int, y0: int, y1: int, x0: int, x1: int, val: int = 80):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[y0:y1, x0:x1] = val
    return img


def test_crop_loss_detects_smaller_content():
    big = _box(40, 40, 2, 38, 2, 38)
    small = _box(40, 40, 10, 20, 10, 20)
    assert mod.crop_loss_increased(big, small) is True
    assert mod.crop_loss_increased(big, big) is False


def test_even_sample_is_deterministic():
    paths = [f"f{i:02d}.png" for i in range(10)]
    assert mod._even_sample(paths, 4) == ["f00.png", "f03.png", "f06.png", "f09.png"]
    assert mod._even_sample(paths, 20) == paths


def test_median_dy_uses_alignment_steps():
    ds = {"alignment": {"dy_steps": [80.0, 90.0, 100.0]}, "frames": {"source_h": 1080}}
    assert abs(mod._median_dy(ds, 0.5) - 45.0) < 1e-6
