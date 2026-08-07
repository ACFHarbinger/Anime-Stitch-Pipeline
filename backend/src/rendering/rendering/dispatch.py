"""Dispatcher for different rendering modes."""

from __future__ import annotations

import numpy as np

from .first import _render_first
from .laplacian import _render_laplacian
from .median import _render_median


def _render(
    frames: list[np.ndarray],
    affines: list[np.ndarray],
    bg_masks: list[np.ndarray | None],
    canvas_h: int,
    canvas_w: int,
    renderer: str = "median",
    baselines: list[float] | None = None,
    confidence_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
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
