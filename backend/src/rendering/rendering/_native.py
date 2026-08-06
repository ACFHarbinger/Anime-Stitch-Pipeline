"""Shared probe for the optional ``base.canvas`` C++ extension + GPU flag.

Referenced via ``from . import _native; _native._BATCH_RENDER`` (module-
qualified, not ``from ._native import _BATCH_RENDER``) everywhere it's used,
so that tests patching ``_native._BATCH_RENDER``/``_native._batch_render``
correctly affect every renderer (median/first/laplacian) that reads it --
patching a name that was copied into another module's namespace via a bare
``from`` import would not propagate.
"""

from __future__ import annotations

import os

try:
    import base as _batch_render
    if (
        getattr(_batch_render, "__file__", None) is None
        or not hasattr(_batch_render, "canvas")
    ):
        raise ImportError("compiled base.canvas extension not available")
    _BATCH_RENDER = True
except ImportError:
    _batch_render = None  # type: ignore[assignment]
    _BATCH_RENDER = False

# Phase 6: OpenCL/CUDA GPU acceleration for warp and blend.
# Set ASP_BATCH_GPU=1 to enable UMat paths in C++ (requires rebuilt .so).
_BATCH_GPU = os.environ.get("ASP_BATCH_GPU", "0") != "0"


__all__ = ["_batch_render", "_BATCH_RENDER", "_BATCH_GPU"]
