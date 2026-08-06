"""Animation-phase re-render pass, shared by ``_render_median``'s GPU fast
path and its chunked Python path.

The pre-split ``rendering.py`` had this exact logic duplicated almost
verbatim in both places (one difference: the fast path never propagates
``_baselines``/``confidence_weights`` into the recursive call -- harmless,
since the fast path only runs when both are already None per its own gate
condition). Deduplicated here per architecture.md §5.17 Option C ("apply
the relevant existing section [DRY] first... rather than mechanically
relocating duplication into more files").
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ._phase_clustering import _cluster_animation_phases

logger = logging.getLogger(__name__)


def _render_animation_repaint(
    canvas: np.ndarray,
    frames: List[np.ndarray],
    affines: List[np.ndarray],
    bg_masks: List[Optional[np.ndarray]],
    H: int,
    W: int,
    _baselines: Optional[List[float]] = None,
    confidence_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Detect cyclic animation pixels and re-render just the majority phase
    group over ``canvas``, in place. Returns ``canvas`` unchanged when there
    are too few frames, the vertical span is too large for reliable FFT
    detection, or no animation phases are found.
    """
    N = len(frames)
    if N < 4:
        return canvas

    # Skip animation detection for pan shots: large vertical span means each
    # canvas pixel is covered by few frames → FFT gives spurious AC signal.
    ty_vals = [float(a[1, 2]) for a in affines]
    ty_span = max(ty_vals) - min(ty_vals)
    if ty_span > 0.25 * H:
        anim_mask, phase_groups = None, None
    else:
        anim_mask, phase_groups = _cluster_animation_phases(frames, affines, H, W)

    if anim_mask is None or phase_groups is None:
        return canvas

    logger.info(
        "[Stitch]   Animation detected: %d phases — re-rendering animation pixels...",
        len(phase_groups),
    )
    majority_group = max(phase_groups, key=len)
    sub_frames = [frames[idx] for idx in majority_group]
    sub_affines = [affines[idx] for idx in majority_group]
    sub_masks = [bg_masks[idx] for idx in majority_group]
    sub_bl = (
        [_baselines[idx] for idx in majority_group] if _baselines is not None else None
    )
    sub_cw = (
        confidence_weights[majority_group] if confidence_weights is not None else None
    )

    # Deferred import: _render_median (median.py) calls this helper, so a
    # module-level import here would be circular.
    from .median import _render_median

    anim_canvas, _, _, _ = _render_median(
        sub_frames,
        sub_affines,
        sub_masks,
        H,
        W,
        _baselines=sub_bl,
        _skip_anim=True,
        confidence_weights=sub_cw,
    )
    anim_has_content = anim_canvas.max(axis=2) > 0
    overwrite_px = (anim_mask > 0) & anim_has_content
    canvas[overwrite_px] = anim_canvas[overwrite_px]
    return canvas


__all__ = ["_render_animation_repaint"]
