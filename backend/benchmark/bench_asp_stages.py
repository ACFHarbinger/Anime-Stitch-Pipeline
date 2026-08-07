"""Benchmark suite for ASP pipeline stage isolation (roadmap
analytics_and_interpretability.md §12.2, issue #70).

Measures four ASP stages independently, on synthetic panning frames, rather
than only end-to-end via `bench_anime_stitch.py` -- the goal is to quantify
the compute cost of each S-coded feature toggle to guide future
optimisations, per the roadmap item:

- `_pairwise_match`: classical (phase-correlation/template) fallback chain
  vs the chain with a real LoFTR wrapper attached, when one is importable.
- `_bundle_adjust_affine`: with vs without the `_spanning_tree_inlier_filter`
  consensus pre-filter (§1.1B).
- `_composite_foreground`: with vs without a warmed `seam_path_cache` dict
  (a cold vs hot seam-DP cache). The roadmap item also asks for "with/without
  Poisson seam blend" -- that feature was removed from the active compositing
  module in the 2026-07-09 "great trim" (S200; see
  `docs/moon/roadmaps/asp_trim_2026-07.md`) along with GraphCut, which the same
  trim measured worse than the DP seam path. `_poisson_seam_blend` now only
  exists in `backend/src/core/image_merger/_legacy_compositing.py`, a
  different (non-ASP) feature -- there is nothing left to benchmark "with"
  in the current pipeline, so this half of the bullet is not implemented.
- `_ecc_refine`: at ECC_MAX_ITER = 20 / 50 / 80 (default) per-pyramid-level
  iteration caps, via a module-attribute monkeypatch (the constant is read
  fresh from `asp_backend.alignment.ecc` on every call, not
  captured at import time, so this is a legitimate way to vary it without
  touching the pipeline itself).

Run standalone:
    python backend/benchmark/bench_asp_stages.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from asp_backend.alignment import ecc as _ecc_module  # noqa: E402
from asp_backend.alignment.bundle_adjust import (  # noqa: E402
    _bundle_adjust_affine,
    _spanning_tree_inlier_filter,
)
from asp_backend.alignment.ecc import _ecc_refine  # noqa: E402
from asp_backend.alignment.matching import _pairwise_match  # noqa: E402
from asp_backend.rendering.compositing import _composite_foreground  # noqa: E402
from backend.benchmark.managers import BenchmarkManager, measure_memory  # noqa: E402

try:
    from backend.src.models.wrappers.loftr_wrapper import LoFTRWrapper  # type: ignore[import]

    _LOFTR_AVAILABLE = True
except Exception:
    LoFTRWrapper = None  # type: ignore[assignment]
    _LOFTR_AVAILABLE = False


# ── Shared fixtures (built once, reused across benchmarks) ────────────────────


def _make_panning_frames(n: int = 6, h: int = 480, w: int = 640, step: int = 80, seed: int = 7):
    """Textured (Gaussian-blurred noise) vertical-pan frames sliced from one
    tall canvas at a fixed step -- gives classical matchers (phase
    correlation, template match) real, matchable texture instead of the flat
    solid colours the compositing/bundle-adjust tests use (those stages
    don't need texture; the matching stage does)."""
    rng = np.random.default_rng(seed)
    canvas_h = h + step * (n - 1)
    base = rng.integers(0, 255, (canvas_h, w, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    frames = [base[i * step : i * step + h, :].copy() for i in range(n)]
    bg_masks = [None] * n
    return frames, bg_masks


def _make_synthetic_edges(n: int, dy: float = 300.0):
    """Synthetic pairwise-match edges -- mirrors `make_edge()` in
    backend/test/conftest.py, duplicated locally so this file has no test
    on its import path (benchmarks must run standalone in prod checkouts)."""
    edges = []
    for i in range(n - 1):
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, dy]], dtype=np.float32)
        rng = np.random.default_rng(i * 1000 + (i + 1))
        pts_i = rng.uniform(50, 400, (50, 2)).astype(np.float32)
        pts_j = pts_i + np.array([0.0, dy], dtype=np.float32)
        edges.append({"i": i, "j": i + 1, "M": M, "pts_i": pts_i, "pts_j": pts_j, "weight": 1.0})
    return edges


def _make_composite_fixture(n: int, h: int, w: int):
    """Mirrors TestCompositeForeground._run_composite in
    backend/test/animation/rendering/test_compositing.py: n solid frames
    warped onto a shared canvas by fixed-step translation affines."""
    frame_h = h // 2
    frames = [
        np.full((frame_h, w, 3), int(c), dtype=np.uint8) for c in np.linspace(60, 200, n, dtype=int)
    ]
    affines = []
    for i in range(n):
        M = np.eye(2, 3, dtype=np.float32)
        M[1, 2] = i * float(frame_h) * 0.8
        affines.append(M)
    canvas_h = int((n - 1) * frame_h * 0.8 + frame_h)
    canvas = np.zeros((canvas_h, w, 3), dtype=np.uint8)
    for f, aff in zip(frames, affines, strict=False):
        wf = cv2.warpAffine(
            f, aff, (w, canvas_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        mask = wf.max(axis=2) > 0
        canvas[mask] = wf[mask]
    bg_masks = [None] * n
    return frames, affines, canvas, canvas_h, bg_masks


_FRAMES, _BG_MASKS = _make_panning_frames(n=6, h=480, w=640, step=80)
_EDGES_SMALL = _make_synthetic_edges(8)
_EDGES_LARGE = _make_synthetic_edges(24)
_COMP_FRAMES, _COMP_AFFINES, _COMP_CANVAS, _COMP_CANVAS_H, _COMP_BG_MASKS = (
    _make_composite_fixture(n=6, h=1200, w=400)
)
_ECC_FRAMES, _ECC_BG_MASKS = _make_panning_frames(n=4, h=480, w=640, step=60, seed=11)
_ECC_AFFINES = [
    np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -60.0 * i]], dtype=np.float32)
    for i in range(len(_ECC_FRAMES))
]

runner = BenchmarkManager("ASP Stage Isolation")


# ── §12.2 _pairwise_match ──────────────────────────────────────────────────


@runner.benchmark("pairwise_match_classical", iterations=3, warmup=1)
@measure_memory
def bench_pairwise_match_classical():
    return _pairwise_match(_FRAMES, _BG_MASKS, loftr_wrapper=None, use_loftr=False)


@runner.benchmark("pairwise_match_with_loftr", iterations=2, warmup=1)
@measure_memory
def bench_pairwise_match_with_loftr():
    if not _LOFTR_AVAILABLE:
        return None
    try:
        wrapper = LoFTRWrapper()
    except Exception:
        return None
    return _pairwise_match(_FRAMES, _BG_MASKS, loftr_wrapper=wrapper, use_loftr=True)


# ── §12.2 _bundle_adjust_affine (with/without spanning-tree filter) ───────


@runner.benchmark("bundle_adjust_no_spanning_tree", iterations=5, warmup=1)
@measure_memory
def bench_bundle_adjust_without_spanning_tree():
    return _bundle_adjust_affine(_EDGES_LARGE, num_frames=24)


@runner.benchmark("bundle_adjust_with_spanning_tree", iterations=5, warmup=1)
@measure_memory
def bench_bundle_adjust_with_spanning_tree():
    filtered = _spanning_tree_inlier_filter(_EDGES_LARGE, num_frames=24)
    return _bundle_adjust_affine(filtered, num_frames=24)


# ── §12.2 _composite_foreground (cold vs hot seam cache) ──────────────────


@runner.benchmark("composite_foreground_cold_seam_cache", iterations=3, warmup=1)
@measure_memory
def bench_composite_cold_cache():
    return _composite_foreground(
        [], [], _COMP_CANVAS.copy(), _COMP_CANVAS_H, _COMP_CANVAS.shape[1],
        _COMP_FRAMES, _COMP_AFFINES, _COMP_BG_MASKS,
        seam_path_cache={},
    )


_WARM_SEAM_CACHE: dict = {}


@runner.benchmark("composite_foreground_hot_seam_cache", iterations=3, warmup=1)
@measure_memory
def bench_composite_hot_cache():
    # First call (outside the timed decorator machinery would be ideal, but
    # BenchmarkManager times the whole wrapped call) populates the cache;
    # subsequent iterations within the same process reuse the same dict, so
    # only the very first of the `iterations` timed calls pays the cold cost
    # -- the reported avg/min still shows the hot-path improvement once
    # iterations > 1, and min_sec in particular reflects the pure hot path.
    return _composite_foreground(
        [], [], _COMP_CANVAS.copy(), _COMP_CANVAS_H, _COMP_CANVAS.shape[1],
        _COMP_FRAMES, _COMP_AFFINES, _COMP_BG_MASKS,
        seam_path_cache=_WARM_SEAM_CACHE,
    )


# ── §12.2 _ecc_refine at different iteration caps ──────────────────────────


def _bench_ecc_at(max_iter: int):
    original = _ecc_module.ECC_MAX_ITER
    _ecc_module.ECC_MAX_ITER = max_iter
    try:
        return _ecc_refine(_ECC_FRAMES, _ECC_AFFINES, _ECC_BG_MASKS)
    finally:
        _ecc_module.ECC_MAX_ITER = original


@runner.benchmark("ecc_refine_iter_20", iterations=3, warmup=1)
@measure_memory
def bench_ecc_iter_20():
    return _bench_ecc_at(20)


@runner.benchmark("ecc_refine_iter_50", iterations=3, warmup=1)
@measure_memory
def bench_ecc_iter_50():
    return _bench_ecc_at(50)


@runner.benchmark("ecc_refine_iter_80_default", iterations=3, warmup=1)
@measure_memory
def bench_ecc_iter_80():
    return _bench_ecc_at(80)


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _LOFTR_AVAILABLE:
        print(
            "WARNING: LoftrWrapper not importable (no GPU/model weights?) — "
            "pairwise_match_with_loftr will no-op."
        )
    runner.run()
    runner.print_results()
    runner.save_json()
