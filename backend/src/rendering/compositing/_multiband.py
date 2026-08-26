"""M5 Multi-band / Laplacian pyramid blending for background plate regions.

Provides frequency-split multi-band blending (Burt & Adelson 1983 analog)
specifically optimized for cel animation:
- Blends low-frequency illumination gradients across wide spatial radii on static
  background regions.
- Tightly locks high-frequency details (Level 0 / line art) to the discrete DP
  seam path to prevent blurring of ink lines and fine boundaries.
- Character / foreground regions bypass multi-band blending to prevent ghosting.
"""

from __future__ import annotations

import cv2
import numpy as np


def multiband_blend_background(
    img_a: np.ndarray,
    img_b: np.ndarray,
    mask_float: np.ndarray,
    *,
    levels: int = 5,
    hf_lock: bool = True,
    bg_mask_a: np.ndarray | None = None,
    bg_mask_b: np.ndarray | None = None,
) -> np.ndarray:
    """Perform multi-band Laplacian pyramid blending between img_a and img_b.

    Parameters
    ----------
    img_a : np.ndarray
        (H, W, 3) uint8 BGR image for side A.
    img_b : np.ndarray
        (H, W, 3) uint8 BGR image for side B.
    mask_float : np.ndarray
        (H, W) float32 in [0, 1] specifying initial weight (1.0 = side A, 0.0 = side B).
    levels : int, default 5
        Number of Gaussian / Laplacian pyramid levels.
    hf_lock : bool, default True
        If True, Level 0 (highest frequency / line art) is thresholded strictly
        at mask_float >= 0.5 to prevent ink-line softening.
    bg_mask_a : np.ndarray | None
        Optional (H, W) boolean/uint8 background mask for image A.
    bg_mask_b : np.ndarray | None
        Optional (H, W) boolean/uint8 background mask for image B.

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 BGR blended image.
    """
    H, W = img_a.shape[:2]
    if H < 8 or W < 8 or levels < 2:
        # Fallback to direct alpha blend on trivial canvas sizes
        m3 = mask_float[:, :, None]
        return np.clip(img_a * m3 + img_b * (1.0 - m3), 0, 255).astype(np.uint8)

    # Determine maximum valid pyramid depth based on canvas dimensions
    max_levels = 1
    min_dim = min(H, W)
    while min_dim >= 8 and max_levels < levels:
        min_dim //= 2
        max_levels += 1
    levels = max_levels

    # Build Gaussian Pyramids for img_a, img_b, and mask_float
    # Level 0 is full resolution, level K-1 is smallest
    ga = [img_a.astype(np.float32)]
    gb = [img_b.astype(np.float32)]
    gm = [mask_float.astype(np.float32)]

    for _ in range(levels - 1):
        ga.append(cv2.pyrDown(ga[-1]))
        gb.append(cv2.pyrDown(gb[-1]))
        gm.append(cv2.pyrDown(gm[-1]))

    # Build Laplacian Pyramids (Level 0 is finest detail, Level K-1 is coarsest base)
    la = []
    lb = []
    for k in range(levels - 1):
        dst_size = (ga[k].shape[1], ga[k].shape[0])
        la.append(ga[k] - cv2.pyrUp(ga[k + 1], dstsize=dst_size))
        lb.append(gb[k] - cv2.pyrUp(gb[k + 1], dstsize=dst_size))
    # Coarsest base level
    la.append(ga[-1])
    lb.append(gb[-1])

    # Blend each pyramid band
    blended_pyr = []
    for k in range(levels):
        h_k, w_k = la[k].shape[:2]
        m = gm[k]
        if m.shape[:2] != (h_k, w_k):
            m = cv2.resize(m, (w_k, h_k), interpolation=cv2.INTER_LINEAR)

        # High-frequency lock: lock finest octaves (e.g. k=0) tightly to the sharp seam
        if hf_lock and k == 0:
            m = (m >= 0.5).astype(np.float32)

        m3 = m[:, :, None] if m.ndim == 2 else m
        blended_band = la[k] * m3 + lb[k] * (1.0 - m3)
        blended_pyr.append(blended_band)

    # Reconstruct: start from coarsest base (levels-1) up to finest (0)
    reconstructed = blended_pyr[-1]
    for k in range(levels - 2, -1, -1):
        dst_size = (blended_pyr[k].shape[1], blended_pyr[k].shape[0])
        reconstructed = cv2.pyrUp(reconstructed, dstsize=dst_size) + blended_pyr[k]

    result = np.clip(reconstructed, 0, 255).astype(np.uint8)

    # If background masks are supplied, enforce that pure foreground pixels
    # in either source are untouched (bypass multi-band smoothing)
    if bg_mask_a is not None and bg_mask_b is not None:
        bg_a_bool = bg_mask_a > 127 if bg_mask_a.dtype == np.uint8 else bg_mask_a.astype(bool)
        bg_b_bool = bg_mask_b > 127 if bg_mask_b.dtype == np.uint8 else bg_mask_b.astype(bool)
        pure_bg = bg_a_bool & bg_b_bool

        # Where not pure background, use sharp cut from mask_float
        not_bg = ~pure_bg
        if not_bg.any():
            take_a = not_bg & (mask_float >= 0.5)
            take_b = not_bg & (mask_float < 0.5)
            result[take_a] = img_a[take_a]
            result[take_b] = img_b[take_b]

    return result


__all__ = ["multiband_blend_background"]
