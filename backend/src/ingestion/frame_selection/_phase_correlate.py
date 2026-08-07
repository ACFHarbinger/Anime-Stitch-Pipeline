"""Pairwise thumbnail phase-correlation for ``smart_select_frames``.

Extracted from ``smart_select_frames`` step 3 as a standalone, individually
testable function -- pure code motion, no logic change.
"""

from __future__ import annotations

import cv2
import numpy as np

from ._thumbs import _otsu_bg_mask_pair


def _pairwise_phase_correlate(
    N: int,
    thumbs: list[np.ndarray],
    hold_ids: list[int],
    hold_threshold: float,
    otsu_bg_corr: bool,
    bg_thumb_mask: np.ndarray | None,
    scale_x: float,
    scale_y: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Phase-correlate every consecutive thumbnail pair.

    Within the same hold block (``hold_ids[i] == hold_ids[i + 1]``), skips
    ``phaseCorrelate`` entirely (§1.11 speedup) since the frames are the same
    character cel with negligible camera drift -- the displacement
    contribution is zeroed instead of computed, and treated as a perfect
    correlation (response=1.0, MAD=0.0) so the caller's high-animation gate
    never misfires on held frames.

    Returns ``(raw_dx, raw_dy, responses, frame_mads)``, each length N-1,
    with dx/dy already scaled to full-resolution canvas pixels.
    """
    raw_dx: list[float] = []
    raw_dy: list[float] = []
    responses: list[float] = []
    frame_mads: list[float] = []

    for i in range(N - 1):
        a = thumbs[i]
        b = thumbs[i + 1]
        th = max(a.shape[0], b.shape[0])
        tw = max(a.shape[1], b.shape[1])
        if a.shape != (th, tw):
            a = np.pad(a, ((0, th - a.shape[0]), (0, tw - a.shape[1])))
        if b.shape != (th, tw):
            b = np.pad(b, ((0, th - b.shape[0]), (0, tw - b.shape[1])))

        # Hold-block skip (§1.11 speedup): within the same hold block,
        # consecutive frames have the same character cel and negligible
        # camera drift.  We zero-out the displacement contribution instead of
        # running phaseCorrelate, reducing correlation pairs from N-1 to K-1
        # (K hold blocks) for typical anime with ~3-frame holds. Blocks larger
        # than _MAX_SKIPPABLE_HOLD_SIZE were already split back to singletons
        # by the caller, so this never fires on a false-positive giant "hold".
        # The MAD is set to 0.0 (identical frames → camera step dominates)
        # so the high_anim_mad gate never misfires on held frames.
        if hold_threshold > 0.0 and hold_ids[i] == hold_ids[i + 1]:
            raw_dx.append(0.0)
            raw_dy.append(0.0)
            responses.append(1.0)  # treat as perfect correlation (same cel)
            frame_mads.append(0.0)
            continue

        if bg_thumb_mask is not None and bg_thumb_mask.shape == a.shape:
            _m = bg_thumb_mask
            (dx_t, dy_t), response = cv2.phaseCorrelate(a * _m, b * _m)
            _fg = 1.0 - _m
            frame_mads.append(float(np.sum(np.abs(b - a) * _fg) / max(_fg.sum(), 1.0)))
        elif otsu_bg_corr:
            # §1A: per-pair Otsu bg mask — faster than BiRefNet, per-frame accurate.
            _m = _otsu_bg_mask_pair(a, b)
            if _m is not None and _m.shape == a.shape:
                (dx_t, dy_t), response = cv2.phaseCorrelate(a * _m, b * _m)
                _fg = 1.0 - _m
                frame_mads.append(
                    float(np.sum(np.abs(b - a) * _fg) / max(_fg.sum(), 1.0))
                )
            else:
                (dx_t, dy_t), response = cv2.phaseCorrelate(a, b)
                frame_mads.append(float(np.mean(np.abs(b - a))))
        else:
            (dx_t, dy_t), response = cv2.phaseCorrelate(a, b)
            frame_mads.append(float(np.mean(np.abs(b - a))))
        raw_dx.append(dx_t * scale_x)
        raw_dy.append(dy_t * scale_y)
        responses.append(response)

    return raw_dx, raw_dy, responses, frame_mads


__all__ = ["_pairwise_phase_correlate"]
