"""Benchmark suite for compositing component isolation (roadmap
analytics_and_interpretability.md §12.6, issue #70).

Micro-benchmarks the seam-DP building blocks in
`backend/src/animation/rendering/compositing/` directly, at the input
scales the roadmap item names:

- `_seam_cut()` — 96 seams (the item's own count), at three canvas heights.
- `_soft_seam_weight()` — canvas widths 500px / 2000px / 5000px.
- `_build_seam_cost_map()` — foreground-mask fraction 10% / 50% / 90%.

`_poisson_seam_blend()`, the item's fourth bullet, is not implemented here
for the same reason `bench_asp_stages.py` (§12.2) doesn't benchmark a
Poisson-blend composite path: the function was removed from the active
compositing module in the 2026-07-09 "great trim" (S200) along with
GraphCut, which that trim measured worse than the DP seam path. It survives
only in `backend/src/core/image_merger/_legacy_compositing.py`, a different
(non-ASP) feature — nothing left in the ASP pipeline to benchmark "with".

Run standalone:
    python backend/benchmark/bench_compositing_components.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from asp_backend.rendering.compositing import (  # noqa: E402
    _build_seam_cost_map,
    _seam_cut,
    _soft_seam_weight,
)
from backend.benchmark.managers import BenchmarkManager, measure_memory  # noqa: E402


def _make_zone(h: int, w: int, base: int, noise: int = 10, seed: int = 42) -> np.ndarray:
    """Mirrors TestSeamCutDP._make_zone in
    backend/test/animation/rendering/test_compositing.py."""
    rng = np.random.default_rng(seed)
    z = np.full((h, w, 3), base, dtype=np.uint8)
    z = np.clip(z.astype(np.int32) + rng.integers(-noise, noise + 1, z.shape), 0, 255)
    return z.astype(np.uint8)


# ── §12.6 _seam_cut — 96 seams at three canvas heights ─────────────────────

_SEAM_COUNT = 96
_SEAM_ZONES_H100 = [
    (_make_zone(100, 300, 80, seed=i), _make_zone(100, 300, 180, seed=i + 1000))
    for i in range(_SEAM_COUNT)
]
_SEAM_ZONES_H500 = [
    (_make_zone(500, 300, 80, seed=i), _make_zone(500, 300, 180, seed=i + 1000))
    for i in range(_SEAM_COUNT)
]
_SEAM_ZONES_H2000 = [
    (_make_zone(2000, 300, 80, seed=i), _make_zone(2000, 300, 180, seed=i + 1000))
    for i in range(_SEAM_COUNT)
]

runner = BenchmarkManager("Compositing Component Isolation")


@runner.benchmark("seam_cut_96_seams_h100", iterations=3, warmup=1)
@measure_memory
def bench_seam_cut_h100():
    for fa, fb in _SEAM_ZONES_H100:
        _seam_cut(fa, fb)


@runner.benchmark("seam_cut_96_seams_h500", iterations=3, warmup=1)
@measure_memory
def bench_seam_cut_h500():
    for fa, fb in _SEAM_ZONES_H500:
        _seam_cut(fa, fb)


@runner.benchmark("seam_cut_96_seams_h2000", iterations=2, warmup=1)
@measure_memory
def bench_seam_cut_h2000():
    for fa, fb in _SEAM_ZONES_H2000:
        _seam_cut(fa, fb)


# ── §12.6 _soft_seam_weight — canvas widths 500 / 2000 / 5000px ───────────


def _soft_seam_weight_fixture(zone_h: int, w: int):
    fa = _make_zone(zone_h, w, 80)
    fb = _make_zone(zone_h, w, 180)
    path_local = _seam_cut(fa, fb)
    return fa, fb, path_local, zone_h, w


_SSW_500 = _soft_seam_weight_fixture(120, 500)
_SSW_2000 = _soft_seam_weight_fixture(120, 2000)
_SSW_5000 = _soft_seam_weight_fixture(120, 5000)


@runner.benchmark("soft_seam_weight_w500", iterations=5, warmup=1)
@measure_memory
def bench_soft_seam_weight_500():
    fa, fb, path_local, zone_h, w = _SSW_500
    return _soft_seam_weight(fa, fb, path_local, zone_h, w)


@runner.benchmark("soft_seam_weight_w2000", iterations=5, warmup=1)
@measure_memory
def bench_soft_seam_weight_2000():
    fa, fb, path_local, zone_h, w = _SSW_2000
    return _soft_seam_weight(fa, fb, path_local, zone_h, w)


@runner.benchmark("soft_seam_weight_w5000", iterations=3, warmup=1)
@measure_memory
def bench_soft_seam_weight_5000():
    fa, fb, path_local, zone_h, w = _SSW_5000
    return _soft_seam_weight(fa, fb, path_local, zone_h, w)


# ── §12.6 _build_seam_cost_map — fg fraction 10% / 50% / 90% ──────────────


def _cost_map_fixture(h: int, w: int, fg_fraction: float):
    """bg_mask is True where BACKGROUND; a foreground band of height
    `fg_fraction * h` (True->False split) at canvas top gives an exact,
    controllable fg fraction rather than a noisy random mask."""
    canvas_zone = _make_zone(h, w, 120)
    bg_mask = np.ones((h, w), dtype=bool)
    fg_rows = int(round(h * fg_fraction))
    bg_mask[:fg_rows, :] = False
    return canvas_zone, bg_mask


_COST_H, _COST_W = 300, 400
_COST_FG10 = _cost_map_fixture(_COST_H, _COST_W, 0.10)
_COST_FG50 = _cost_map_fixture(_COST_H, _COST_W, 0.50)
_COST_FG90 = _cost_map_fixture(_COST_H, _COST_W, 0.90)


@runner.benchmark("build_seam_cost_map_fg10pct", iterations=5, warmup=1)
@measure_memory
def bench_cost_map_fg10():
    canvas_zone, bg_mask = _COST_FG10
    return _build_seam_cost_map(canvas_zone, bg_mask, None, dilate_px=15)


@runner.benchmark("build_seam_cost_map_fg50pct", iterations=5, warmup=1)
@measure_memory
def bench_cost_map_fg50():
    canvas_zone, bg_mask = _COST_FG50
    return _build_seam_cost_map(canvas_zone, bg_mask, None, dilate_px=15)


@runner.benchmark("build_seam_cost_map_fg90pct", iterations=5, warmup=1)
@measure_memory
def bench_cost_map_fg90():
    canvas_zone, bg_mask = _COST_FG90
    return _build_seam_cost_map(canvas_zone, bg_mask, None, dilate_px=15)


if __name__ == "__main__":
    runner.run()
    runner.print_results()
    runner.save_json()
