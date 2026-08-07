"""§3.11A — GPU-accelerated nanmedian (Option A)."""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# §3.11A — GPU temporal median (Option A).
# When enabled, each chunk's nanmedian is computed on the GPU via torch.nanmedian,
# then copied back to CPU.  Falls back to numpy silently if CUDA is unavailable
# or if torch raises an exception.  Worth enabling on RTX 3090 Ti.
# Enable: ASP_GPU_MEDIAN=1.
_GPU_MEDIAN: bool = os.environ.get("ASP_GPU_MEDIAN", "0") != "0"
_cuda_available: bool | None = None  # lazily initialised on first call


def _gpu_nanmedian(arr: np.ndarray) -> np.ndarray:
    """Compute nanmedian(arr, axis=0) on GPU when _GPU_MEDIAN is set and CUDA is present.

    arr : float32 (N, P, 3) where NaN marks missing samples.
    Returns float32 (P, 3).  Falls back to numpy on any failure.
    """
    global _cuda_available
    if not _GPU_MEDIAN:
        return np.nanmedian(arr, axis=0)
    if _cuda_available is None:
        try:
            import torch as _t

            _cuda_available = _t.cuda.is_available()
        except ImportError:
            _cuda_available = False
    if not _cuda_available:
        return np.nanmedian(arr, axis=0)
    try:
        import torch as _t

        t = _t.from_numpy(arr).cuda()
        result = _t.nanmedian(t, dim=0).values.cpu().numpy()
        return result
    except Exception as exc:
        logger.debug("GPU median failed (%s), falling back to numpy", exc)
        return np.nanmedian(arr, axis=0)


__all__ = ["_gpu_nanmedian", "_GPU_MEDIAN"]
