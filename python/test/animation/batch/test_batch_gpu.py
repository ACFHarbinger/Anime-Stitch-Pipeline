"""
backend/test/animation/batch/test_batch_gpu.py
===============================================
Phase 6 GPU dispatch smoke tests.

Validates that Phase 6 try_gpu parameter additions are accepted by the
pybind11 bindings and that the GPU path either succeeds or gracefully falls
back to the CPU implementation (no crash, correct output shape).

All tests are skipped automatically when ``batch`` has not been built.
"""

import numpy as np
import pytest

try:
    import base as batch
    if (
        getattr(batch, "__file__", None) is None
        or not hasattr(batch, "canvas")
    ):
        raise ImportError("compiled base ASP extension not available")
    HAS_BATCH = True
except ImportError:
    HAS_BATCH = False

# Phase 6 symbols live in the same .so but may not be in the currently compiled binary.
HAS_PHASE6 = HAS_BATCH and hasattr(batch, "canvas") and hasattr(batch.canvas, "gpu_device_count")

pytestmark = pytest.mark.skipif(not HAS_BATCH, reason="batch not built")


def _small_frame(h: int = 32, w: int = 48, c: int = 3) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(10, 230, (h, w, c), dtype=np.uint8)


def _identity_affine() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


skip_phase6 = pytest.mark.skipif(not HAS_PHASE6, reason="Phase 6 symbols not in compiled .so — rebuild batch first")


# ---------------------------------------------------------------------------
# gpu_device_count
# ---------------------------------------------------------------------------


@skip_phase6
class TestGpuDeviceCount:
    def test_returns_non_negative_int(self):
        count = batch.canvas.gpu_device_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_callable_multiple_times(self):
        a = batch.canvas.gpu_device_count()
        b = batch.canvas.gpu_device_count()
        assert a == b


# ---------------------------------------------------------------------------
# warp_frames_to_canvas with try_gpu
# ---------------------------------------------------------------------------


@skip_phase6
class TestWarpFramesGpu:
    def test_try_gpu_false_returns_correct_shape(self):
        frames = [_small_frame()]
        affines = [_identity_affine()]
        result = batch.canvas.warp_frames_to_canvas(frames, affines, 32, 48, try_gpu=False)
        assert len(result) == 1
        assert result[0].shape == (32, 48, 3)

    def test_try_gpu_true_does_not_crash(self):
        frames = [_small_frame(), _small_frame()]
        affines = [_identity_affine(), _identity_affine()]
        # GPU path may fall back to CPU if OpenCL is unavailable — both outcomes are valid.
        result = batch.canvas.warp_frames_to_canvas(frames, affines, 32, 48, try_gpu=True)
        assert len(result) == 2
        assert result[0].shape == (32, 48, 3)

    def test_cpu_and_gpu_same_output_for_identity(self):
        frame = _small_frame()
        affine = _identity_affine()
        cpu = batch.canvas.warp_frames_to_canvas([frame], [affine], 32, 48, try_gpu=False)
        gpu = batch.canvas.warp_frames_to_canvas([frame], [affine], 32, 48, try_gpu=True)
        np.testing.assert_array_equal(cpu[0], gpu[0])


# ---------------------------------------------------------------------------
# render_median with try_gpu
# ---------------------------------------------------------------------------


@skip_phase6
class TestRenderMedianGpu:
    def _warped(self, n: int = 3) -> list:
        return [_small_frame() for _ in range(n)]

    def test_try_gpu_false_correct_shape(self):
        result = batch.canvas.render_median(self._warped(), try_gpu=False)
        assert result.shape == (32, 48, 3)
        assert result.dtype == np.uint8

    def test_try_gpu_true_does_not_crash(self):
        result = batch.canvas.render_median(self._warped(), try_gpu=True)
        assert result.shape == (32, 48, 3)
