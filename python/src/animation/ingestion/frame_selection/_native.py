"""Shared probe for the optional ``base.frame_selection`` C++ extension.

Several frame-selection filters have a native fast-path (hold detection,
temporal variance, near-dup luma); each falls back to pure NumPy when the
extension is unavailable or the native call raises.
"""

from __future__ import annotations

try:
    import base as _batch
    if (
        getattr(_batch, "__file__", None) is None
        or not hasattr(_batch, "frame_selection")
    ):
        raise ImportError("compiled base.frame_selection extension not available")
    _BATCH_FSEL = True
except ImportError:
    _batch = None  # type: ignore[assignment]
    _BATCH_FSEL = False


__all__ = ["_batch", "_BATCH_FSEL"]
