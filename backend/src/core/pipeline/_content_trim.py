"""Stage 12.5: scroll-axis-aware content crop (§2.6).

Extracted from ``AnimeStitchPipeline.run()`` as a standalone function -- pure
code motion, no logic change (see _photometric_stage.py's docstring for why
this particular block was safe to extract while most of run() was not).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _trim_content_crop(
    canvas: np.ndarray,
    valid_mask: np.ndarray,
    affines: List[np.ndarray],
    bg_masks: List[Optional[np.ndarray]],
    N: int,
    canvas_h: int,
    canvas_w: int,
) -> "Tuple[np.ndarray, np.ndarray]":
    """Trim leading/trailing pure-background canvas strips (§2.6).

    After compositing, the canvas may have leading/trailing strips of pure
    background that contain zero foreground character pixels across all
    frames. These pure-bg rows/columns inflate the scale factor relative to
    GT (GT's panorama starts/ends with the first/last character-containing
    frame). Trim them to reduce GT-framing bias.

    Only trims rows/columns where ALL warped frames have bg-only content
    (i.e., no character pixels from any frame reach that canvas position).
    Positions where even one frame has fg content are kept -- they contain
    mid-scroll character data even if the median/composite shows bg there.

    Cap: never trims more than 15% of canvas height/width per side, to avoid
    over-cropping datasets where the first/last frame is entirely background
    (static camera opening shot).

    Returns ``(canvas, valid_mask)``, both possibly cropped. On any failure,
    returns the inputs unchanged (best-effort, matching the pre-split
    try/except-and-skip behaviour).
    """
    try:
        _trim_cap_frac = 0.15
        # Determine dominant scroll axis from affine translations
        _tys_trim = [float(affines[k][1, 2]) for k in range(N)]
        _txs_trim = [float(affines[k][0, 2]) for k in range(N)]
        _ty_span = max(_tys_trim) - min(_tys_trim)
        _tx_span = max(_txs_trim) - min(_txs_trim)
        _is_vert_scroll = _ty_span >= _tx_span

        if bg_masks and any(m is not None for m in bg_masks):
            # Build a union fg map across all warped frames:
            # any pixel that is foreground in AT LEAST ONE warped frame
            # is protected from trimming.
            _union_fg = np.zeros((canvas_h, canvas_w), dtype=bool)
            for _idx_trim in range(N):
                if bg_masks[_idx_trim] is None:
                    continue
                _wfg = cv2.warpAffine(
                    (bg_masks[_idx_trim] < 127).astype(np.uint8),
                    affines[_idx_trim],
                    (canvas_w, canvas_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                _union_fg |= _wfg > 0

            if _is_vert_scroll:
                # Find row range with any fg content
                _row_has_fg = _union_fg.any(axis=1)  # (canvas_h,)
                _fg_rows = np.where(_row_has_fg)[0]
                if len(_fg_rows) > 0:
                    _cap_px = int(canvas_h * _trim_cap_frac)
                    _new_top = max(0, min(int(_fg_rows[0]), _cap_px))
                    _new_bot = min(
                        canvas_h, max(int(_fg_rows[-1]) + 1, canvas_h - _cap_px)
                    )
                    if _new_top > 0 or _new_bot < canvas_h:
                        canvas = canvas[_new_top:_new_bot]
                        valid_mask = valid_mask[_new_top:_new_bot]
                        logger.info(
                            f"[Stitch] Stage 12.5: vertical scroll content trim "
                            f"rows [{_new_top}:{_new_bot}] / {canvas_h} "
                            f"(−{_new_top}top, −{canvas_h - _new_bot}bot)"
                        )
            else:
                # Horizontal scroll: trim pure-bg columns at left/right
                _col_has_fg = _union_fg.any(axis=0)  # (canvas_w,)
                _fg_cols = np.where(_col_has_fg)[0]
                if len(_fg_cols) > 0:
                    _cap_px = int(canvas_w * _trim_cap_frac)
                    _new_lft = max(0, min(int(_fg_cols[0]), _cap_px))
                    _new_rgt = min(
                        canvas_w, max(int(_fg_cols[-1]) + 1, canvas_w - _cap_px)
                    )
                    if _new_lft > 0 or _new_rgt < canvas_w:
                        canvas = canvas[:, _new_lft:_new_rgt]
                        valid_mask = valid_mask[:, _new_lft:_new_rgt]
                        logger.info(
                            f"[Stitch] Stage 12.5: horizontal scroll content trim "
                            f"cols [{_new_lft}:{_new_rgt}] / {canvas_w} "
                            f"(−{_new_lft}left, −{canvas_w - _new_rgt}right)"
                        )
    except Exception as _trim_e:
        logger.debug(f"[Stitch] Stage 12.5 content trim skipped ({_trim_e}).")

    return canvas, valid_mask


__all__ = ["_trim_content_crop"]
