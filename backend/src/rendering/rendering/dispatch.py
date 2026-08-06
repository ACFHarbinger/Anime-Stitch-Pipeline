"""Dispatcher for different rendering modes."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .first import _render_first
from .laplacian import _render_laplacian
from .median import _render_median


def _render(
    frames: List[np.ndarray],
    affines: List[np.ndarray],
    bg_masks: List[Optional[np.ndarray]],
    canvas_h: int,
    canvas_w: int,
    renderer: str = "median",
    baselines: Optional[List[float]] = None,
    confidence_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Dispatcher for different rendering modes."""
    if renderer == "median":
        return _render_median(
            frames,
            affines,
            bg_masks,
            canvas_h,
            canvas_w,
            _baselines=baselines,
            confidence_weights=confidence_weights,
        )
    elif renderer == "first":
        c, v = _render_first(frames, affines, canvas_h, canvas_w)
        return c, v, [], []
    else:
        return _render_laplacian(frames, affines, bg_masks, canvas_h, canvas_w)


__all__ = ["_render"]
