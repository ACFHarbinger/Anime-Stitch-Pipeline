"""Sequential per-frame photometric gain/bias correction (§1.40, §1.41)."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# §1.40 — Adaptive gain clamp for sequential colour correction.
# When ON, replaces the fixed [0.88, 1.12] gain clamp in _compute_sequential_color_gains
# with a luminance-adaptive variant using the same formula as §1.4B in compositing.py:
#   clamp_width = 0.26 - 0.12 * (ref_lum / 255)  →  ±26% at black, ±14% at white.
# The fixed ±12% clamp is too tight for dark-scene overlap zones (where a small
# absolute brightness difference produces a large ratio) and too wide for bright
# scenes where 12% corrections would overshoot.
# Default OFF.  Enable: ASP_ADAPTIVE_RENDER_GAIN=1.
_ADAPTIVE_RENDER_GAIN: bool = os.environ.get("ASP_ADAPTIVE_RENDER_GAIN", "0") != "0"

# §1.41 — Sequential gain chain-drift guard.
# After all per-pair corrections are chained, the cumulative product of gains
# (frame 0 → frame N-1) can stray far from 1.0 if each pair nudges in the same
# direction (e.g., each frame consistently under-exposes its successor).  When the
# cumulative product exceeds *max_ratio* fold in any channel, the correction chain
# is clearly wrong — reset the whole batch to identity rather than apply a
# systematically drifted correction.
# Default 0.0 = off.  Recommend ASP_GAIN_DRIFT_MAX=2.0.
_GAIN_DRIFT_MAX: float = float(os.environ.get("ASP_GAIN_DRIFT_MAX", "0.0"))


def _adaptive_render_gain_clamp(ref_lum: float) -> "tuple[float, float]":
    """§1.40: Luminance-adaptive gain-clamp bounds for sequential colour correction.

    Uses the same continuous formula as §1.4B in ``compositing.py``:
    ``clamp_width = 0.26 − 0.12 × (ref_lum / 255)``, yielding ±26 % at pure
    black and ±14 % at pure white.  *ref_lum* is clamped to [0, 255] before use.

    Returns
    -------
    (lo, hi) : lower and upper gain bounds, both positive floats.
    """
    lum = max(0.0, min(255.0, ref_lum))
    clamp_width = max(0.14, 0.26 - 0.12 * (lum / 255.0))
    return 1.0 - clamp_width, 1.0 + clamp_width


def _check_gain_chain_drift(gains: np.ndarray, max_ratio: float) -> bool:
    """§1.41: True when the cumulative gain chain exceeds *max_ratio* in any channel.

    *gains* is an (N, 3) float32 array where ``gains[0]`` is always 1.0 (the
    anchor frame) and ``gains[i]`` corrects frame i relative to frame i-1.
    The cumulative product ``prod(gains[:, c])`` represents the total photometric
    shift applied from frame 0 to the last frame.  When this exceeds *max_ratio*
    (or falls below its reciprocal), the chain has accumulated beyond a plausible
    scene-brightness variation and something went wrong.

    Parameters
    ----------
    gains     : (N, 3) float32 gain array from ``_compute_sequential_color_gains``.
    max_ratio : upper bound on the cumulative fold-change (e.g., 2.0 = two-fold).
                Values ≤ 0 are treated as "disabled" and always return False.

    Returns
    -------
    bool — True when drift is detected and caller should reset to identity.
    """
    if max_ratio <= 0.0 or gains.size == 0:
        return False
    cum = np.prod(gains, axis=0)  # (3,) per-channel cumulative product
    log_limit = np.log(max(max_ratio, 1.0 + 1e-9))
    return bool(np.any(np.abs(np.log(np.maximum(cum, 1e-9))) > log_limit))


def _compute_sequential_color_gains(
    frames: List[np.ndarray],
    affines: List[np.ndarray],
    bg_masks: Optional[List[Optional[np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sequential per-frame color gain/bias via overlap-zone photometric matching.

    Frame 0 is the photometric anchor. Each subsequent frame is corrected to
    match its predecessor's color in their shared canvas overlap zone, then the
    correction is chained so all frames end up photometrically consistent.

    Samples multiple horizontal stripes spread across the overlap zone and uses
    the per-channel median across stripes. When bg_masks are provided, only
    background pixels (mask > 127) are included, preventing foreground
    character movement from corrupting the photometric reference.
    """
    N = len(frames)
    gains = np.ones((N, 3), dtype=np.float32)
    biases = np.zeros((N, 3), dtype=np.float32)

    N_BLOCKS_Y = 4
    N_BLOCKS_X = 4
    MIN_BG_PX = 200

    for i in range(1, N):
        H_i, W_i = frames[i].shape[:2]
        H_p, W_p = frames[i - 1].shape[:2]
        ty_i = float(affines[i][1, 2])
        ty_p = float(affines[i - 1][1, 2])
        tx_i = float(affines[i][0, 2])
        tx_p = float(affines[i - 1][0, 2])

        # Canvas overlap bounding box
        ov_top = max(ty_i, ty_p)
        ov_bot = min(ty_i + H_i, ty_p + H_p)
        ov_left = max(tx_i, tx_p)
        ov_right = min(tx_i + W_i, tx_p + W_p)

        if ov_bot - ov_top < 40 or ov_right - ov_left < 40:
            continue

        # Source-frame row/col bounds for the overlap zone
        r0_i = max(0, round(ov_top - ty_i))
        r1_i = min(H_i, int(round(ov_bot - ty_i)))
        c0_i = max(0, round(ov_left - tx_i))
        c1_i = min(W_i, int(round(ov_right - tx_i)))

        r0_p = max(0, round(ov_top - ty_p))
        r1_p = min(H_p, int(round(ov_bot - ty_p)))
        c0_p = max(0, round(ov_left - tx_p))
        c1_p = min(W_p, int(round(ov_right - tx_p)))

        # Background masks for foreground-exclusion
        bm_i = (
            bg_masks[i] if (bg_masks is not None and bg_masks[i] is not None) else None
        )
        bm_p = (
            bg_masks[i - 1]
            if (bg_masks is not None and bg_masks[i - 1] is not None)
            else None
        )

        stripe_means_i = [[] for _ in range(3)]
        stripe_means_p = [[] for _ in range(3)]

        bh = max(10, (r1_i - r0_i) // N_BLOCKS_Y)
        bw = max(10, (c1_i - c0_i) // N_BLOCKS_X)

        for s_r in range(N_BLOCKS_Y):
            for s_c in range(N_BLOCKS_X):
                row_i = r0_i + s_r * bh
                col_i = c0_i + s_c * bw

                # Corresponding predecessor coords (same canvas X, Y)
                canvas_y = ty_i + row_i
                canvas_x = tx_i + col_i
                row_p = round(canvas_y - ty_p)
                col_p = round(canvas_x - tx_p)

                # Safe bounds
                row_p = max(r0_p, min(r1_p - bh, row_p))
                col_p = max(c0_p, min(c1_p - bw, col_p))
                row_i = max(r0_i, min(r1_i - bh, row_i))
                col_i = max(c0_i, min(c1_i - bw, col_i))

                slab_i = frames[i][row_i : row_i + bh, col_i : col_i + bw].astype(
                    np.float32
                )
                slab_p = frames[i - 1][row_p : row_p + bh, col_p : col_p + bw].astype(
                    np.float32
                )

                # Background mask for this block
                valid = np.ones(slab_i.shape[:2], dtype=bool)
                if bm_i is not None:
                    valid &= bm_i[row_i : row_i + bh, col_i : col_i + bw] > 127
                if bm_p is not None:
                    valid_p = bm_p[row_p : row_p + bh, col_p : col_p + bw] > 127
                    valid &= valid_p

                if valid.sum() < MIN_BG_PX:
                    valid = np.ones(slab_i.shape[:2], dtype=bool)
                    if valid.sum() < MIN_BG_PX:
                        continue

                slab_p_corr = np.clip(slab_p * gains[i - 1] + biases[i - 1], 0, 255)

                for c in range(3):
                    mu_i = float(slab_i[valid, c].mean())
                    mu_p = float(slab_p_corr[valid, c].mean())
                    if mu_i > 5.0:
                        stripe_means_i[c].append(mu_i)
                        stripe_means_p[c].append(mu_p)

        for c in range(3):
            if len(stripe_means_i[c]) < 3:
                continue
            arr_i = np.array(stripe_means_i[c])
            arr_p = np.array(stripe_means_p[c])
            # Use median ratio across stripes → robust to outlier stripes
            ratios = arr_p / np.maximum(arr_i, 1.0)
            # §1.40: adaptive clamp scales with scene luminance; fixed ±12% otherwise
            if _ADAPTIVE_RENDER_GAIN:
                _g_lo, _g_hi = _adaptive_render_gain_clamp(float(np.mean(arr_i)))
            else:
                _g_lo, _g_hi = 0.88, 1.12
            g = float(np.clip(np.median(ratios), _g_lo, _g_hi))
            b = float(np.clip(float(np.median(arr_p - arr_i * g)), -20.0, 20.0))
            gains[i, c] = g
            biases[i, c] = b

    # §1.41: Chain-drift guard — reset to identity when cumulative gain is implausible.
    if _GAIN_DRIFT_MAX > 0.0 and _check_gain_chain_drift(gains, _GAIN_DRIFT_MAX):
        logger.warning(
            "[Render] §1.41: sequential gain chain drifted beyond %.2f× — "
            "resetting to identity gains.",
            _GAIN_DRIFT_MAX,
        )
        gains = np.ones((len(gains), 3), dtype=np.float32)
        biases = np.zeros((len(biases), 3), dtype=np.float32)

    return gains, biases


__all__ = [
    "_adaptive_render_gain_clamp",
    "_check_gain_chain_drift",
    "_compute_sequential_color_gains",
]
