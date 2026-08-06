"""Shared probe for the optional ``base.compositing``/``base.seam`` C++ extension."""

from __future__ import annotations

try:
    import base as batch
    if (
        getattr(batch, "__file__", None) is None
        or not hasattr(batch, "compositing")
    ):
        raise ImportError("compiled base.compositing extension not available")
    BATCH_AVAILABLE = True
except ImportError:
    batch = None
    BATCH_AVAILABLE = False


__all__ = ["batch", "BATCH_AVAILABLE"]
