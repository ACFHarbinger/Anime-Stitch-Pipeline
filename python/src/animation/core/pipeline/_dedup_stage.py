"""Pre-Stage-5: deduplicate near-static consecutive frames.

Extracted from ``AnimeStitchPipeline.run()`` as a standalone function -- pure
code motion, no logic change (see _photometric_stage.py's docstring). This
block has exactly one early-return (falls back to SCANS when dedup leaves
fewer than 2 frames); the sentinel-return pattern below preserves that
without introducing new control-flow machinery -- the caller checks
`early_result is not None` and returns it directly, exactly mirroring the
original inline `if N < 2: return ...`.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from backend.src.animation.alignment.canvas import _scan_stitch_fallback
from backend.src.constants import NEAR_DUP_LUMA_THRESH

from ._frame_utils import _reload_scans_frames

logger = logging.getLogger(__name__)


def _dedup_near_static_frames(
    frames: List[np.ndarray],
    scans_frames: List[np.ndarray],
    bg_masks: List[Optional[np.ndarray]],
    image_paths: List[str],
    N: int,
    output_path: str,
) -> "tuple[Optional[Image.Image], List[np.ndarray], List[np.ndarray], List[Optional[np.ndarray]], List[str], int]":
    """Drop consecutive frames whose luma is near-identical to the last kept one.

    Returns ``(early_result, frames, scans_frames, bg_masks, image_paths, N)``.
    ``early_result`` is the SCANS-fallback Image when dedup leaves fewer than
    2 frames (caller must ``return early_result`` immediately in that case);
    otherwise it's None and the other five values are the (possibly
    unchanged) updated state to continue the pipeline with.
    """
    if N < 3:
        return None, frames, scans_frames, bg_masks, image_paths, N

    _luma_cache = [
        cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames
    ]
    keep = [True] * N
    _prev_kept = 0
    for _fi in range(1, N):
        _la, _lb = _luma_cache[_fi], _luma_cache[_prev_kept]
        if _la.shape != _lb.shape:
            # Different heights — cannot be duplicates; keep both
            _prev_kept = _fi
            continue
        diff = float(np.abs(_la - _lb).mean())
        if diff < NEAR_DUP_LUMA_THRESH:
            keep[_fi] = False
            logger.debug(
                f"[Stitch]   Dedup: frame {_fi} ≈ frame {_prev_kept} "
                f"(luma_diff={diff:.2f}) — dropped."
            )
        else:
            _prev_kept = _fi
    if not all(keep):
        keep_idx = [i for i, k in enumerate(keep) if k]
        frames = [frames[i] for i in keep_idx]
        scans_frames = (
            [scans_frames[i] for i in keep_idx] if scans_frames else []
        )  # §1.9C
        bg_masks = [bg_masks[i] for i in keep_idx]
        image_paths = [image_paths[i] for i in keep_idx]
        N = len(frames)
        logger.debug(
            f"[Stitch]   Dedup complete: {sum(not k for k in keep)} "
            f"removed, {N} remain."
        )
        if N < 2:
            _sf = scans_frames or _reload_scans_frames(image_paths)
            return (
                _scan_stitch_fallback(_sf, output_path),
                frames,
                scans_frames,
                bg_masks,
                image_paths,
                N,
            )

    return None, frames, scans_frames, bg_masks, image_paths, N


__all__ = ["_dedup_near_static_frames"]
