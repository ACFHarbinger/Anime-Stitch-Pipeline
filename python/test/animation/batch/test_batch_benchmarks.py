"""
backend/test/animation/batch/test_batch_benchmarks.py
=======================================================
Timing benchmarks: verify C++ batch functions achieve target speedups.

These tests MUST live in Python because:
  - They import Python reference implementations for comparison.
  - They measure wall-clock time using time.perf_counter() across
    Python↔C++ boundary calls (which require the GIL context).

Expected speedups (from asp_cpp_migration.md §3):
  seam_cut      > 5× (Phase 2 target: 27×)
  bundle_adjust > 5× (Phase 3 target: 25×)

All tests:
  - Are xfail until the C++ implementation ships.
  - Are marked @pytest.mark.slow so standard CI skips them with -m "not slow".
  - Can be run explicitly with: just test-cpp or pytest -m slow

Usage:
  pytest backend/test/animation/batch/test_batch_benchmarks.py -v -m slow
"""

from __future__ import annotations

import time

import numpy as np
import pytest

try:
    import base as batch
    if (
        getattr(batch, "__file__", None) is None
        or not hasattr(batch, "bundle_adjust")
    ):
        raise ImportError("compiled base ASP extension not available")
    HAS_BATCH = True
except ImportError:
    HAS_BATCH = False

pytestmark = [
    pytest.mark.skipif(not HAS_BATCH, reason="batch not built"),
    pytest.mark.slow,
]

MIN_SPEEDUP = 5.0  # Minimum acceptable speedup; Phase targets are 20–27×


def _mean_wall_time(fn, *args, n: int = 10, **kwargs) -> float:
    """Return mean wall-clock time per call (seconds)."""
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args, **kwargs)
    return (time.perf_counter() - t0) / n


def _rand_bgr(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# bundle_adjust speedup
# ---------------------------------------------------------------------------


class TestBundleAdjustSpeedup:
    @staticmethod
    def _chain_edges(N: int, step_px: float = 100.0) -> list:
        rng = np.random.default_rng(0)
        return [
            {
                "i": i, "j": i + 1,
                "dx": float(rng.normal(0, 1)),
                "dy": float(step_px + rng.normal(0, 2)),
                "weight": 0.9,
                "type": "adjacent",
            }
            for i in range(N - 1)
        ]

    @pytest.mark.xfail(
        reason="batch.bundle_adjust.bundle_adjust_affine not implemented yet (Phase 3)",
        raises=RuntimeError,
        strict=False,
    )
    def test_bundle_adjust_exceeds_5x_speedup(self):
        try:
            from backend.src.animation.alignment.bundle_adjust import _bundle_adjust_affine_python  # type: ignore
        except ImportError:
            pytest.skip("Python reference _bundle_adjust_affine_python not exposed")

        N = 14
        edges = self._chain_edges(N)

        py_time  = _mean_wall_time(_bundle_adjust_affine_python, edges, N, n=20)
        cpp_time = _mean_wall_time(batch.bundle_adjust.bundle_adjust_affine, edges, N, n=20)
        speedup  = py_time / max(cpp_time, 1e-9)

        print(
            f"\nbundle_adjust: py={py_time*1000:.2f}ms  "
            f"cpp={cpp_time*1000:.2f}ms  speedup={speedup:.1f}×"
        )
        assert speedup >= MIN_SPEEDUP, (
            f"bundle_adjust speedup {speedup:.1f}× < {MIN_SPEEDUP}×."
        )
