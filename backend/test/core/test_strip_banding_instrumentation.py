"""M2: strip_banding_score is a first-class benchmark metric again."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from asp_backend.core.pipeline.safety_metrics import strip_banding_score

_BENCH = (
    Path(__file__).resolve().parents[3]
    / "backend"
    / "benchmark"
    / "bench_anime_stitch.py"
)


def _stacked(top: int, bot: int, h: int = 100, w: int = 80) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2] = top
    img[h // 2 :] = bot
    return img


def _affine(ty: float) -> np.ndarray:
    m = np.eye(2, 3, dtype=np.float32)
    m[1, 2] = ty
    return m


def test_strip_banding_zero_without_affines():
    assert strip_banding_score(_stacked(20, 235)) == 0.0


def test_strip_banding_detects_adjacent_strip_jump():
    assert strip_banding_score(_stacked(20, 235), [_affine(0.0), _affine(50.0)]) > 50.0


def test_compute_all_metrics_emits_strip_banding():
    spec = importlib.util.spec_from_file_location("bench_anime_stitch_m2", _BENCH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    img = _stacked(20, 235)
    affines = [_affine(0.0), _affine(50.0)]
    metrics = mod._compute_all_metrics(img, affines)
    assert "strip_banding_score" in metrics
    assert metrics["strip_banding_score"] == round(
        strip_banding_score(img, affines), 2
    )
    assert mod._compute_all_metrics(img)["strip_banding_score"] == 0.0
