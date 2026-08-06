"""
Standalone helper utilities for the anime stitching pipeline.

These functions hold no state and are shared by every stage of the pipeline.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from backend.src.constants import LAPLACIAN_BANDS


def _luma(bgr: np.ndarray) -> np.ndarray:
    """Return Y' channel (uint8 2-D) from a BGR uint8 image."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0]


def _highpass(gray: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Subtract Gaussian-blurred version to isolate high-frequency content."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    hp = gray.astype(np.float32) - blurred.astype(np.float32)
    # Shift to [0,255] for phase correlation
    hp = hp - hp.min()
    hp = (hp / (hp.max() + 1e-6) * 255.0).astype(np.float32)
    return hp


def _trim_dark_border(arr: np.ndarray, pct: float = 0.35) -> np.ndarray:
    """
    Remove broadcast-safe dark bars (common in anime BDs cropped for TV).
    Trims rows/columns whose mean brightness is below `pct` of the overall median.
    Uses a higher default threshold (0.35) so that letterbox bars and dim
    player-chrome rows are removed while keeping actual scene content.
    """
    if arr.shape[0] < 8 or arr.shape[1] < 8:
        return arr
    gray = arr.mean(axis=2)
    row_m = gray.mean(axis=1)
    col_m = gray.mean(axis=0)
    med_r = float(np.median(row_m)) or 1.0
    med_c = float(np.median(col_m)) or 1.0
    thr_r = max(med_r * pct, 4.0)
    thr_c = max(med_c * pct, 4.0)

    top = next((y for y in range(len(row_m)) if row_m[y] >= thr_r), 0)
    bot = (
        next(
            (y for y in range(len(row_m) - 1, -1, -1) if row_m[y] >= thr_r),
            len(row_m) - 1,
        )
        + 1
    )
    left = next((x for x in range(len(col_m)) if col_m[x] >= thr_c), 0)
    right = (
        next(
            (x for x in range(len(col_m) - 1, -1, -1) if col_m[x] >= thr_c),
            len(col_m) - 1,
        )
        + 1
    )

    trimmed = arr[top:bot, left:right]
    return trimmed if trimmed.size > 0 else arr


def _laplacian_blend(
    a: np.ndarray,
    b: np.ndarray,
    mask_float: np.ndarray,
    bands: int = LAPLACIAN_BANDS,
    alpha_schedule: bool = False,
) -> np.ndarray:
    """
    Multi-band (Laplacian pyramid) blending.

    `a` is taken where mask_float = 1, `b` where mask_float = 0.
    Low frequencies blended broadly; high frequencies blended narrowly at seam.
    Superior to Poisson blending for cel-shaded anime (avoids color bleeding
    across hard cel boundaries).

    Parameters
    ----------
    a, b : (H, W, 3) uint8 BGR images.
    mask_float : (H, W) float32 in [0, 1].
    bands : pyramid depth.
    """
    mask = mask_float[:, :, np.newaxis].astype(np.float32)
    ga = [a.astype(np.float32)]
    gb = [b.astype(np.float32)]
    gm = [mask]
    for _ in range(bands - 1):
        ga.append(cv2.pyrDown(ga[-1]))
        gb.append(cv2.pyrDown(gb[-1]))
        gm.append(cv2.pyrDown(gm[-1]))

    la = [ga[-1]]
    lb = [gb[-1]]
    for k in range(len(ga) - 1, 0, -1):
        la.append(ga[k - 1] - cv2.pyrUp(ga[k], dstsize=ga[k - 1].shape[1::-1]))
        lb.append(gb[k - 1] - cv2.pyrUp(gb[k], dstsize=gb[k - 1].shape[1::-1]))

    blended = []
    for k in range(bands):
        m = gm[bands - 1 - k]
        if m.shape[:2] != la[k].shape[:2]:
            m = cv2.resize(m, (la[k].shape[1], la[k].shape[0]))
        if m.ndim == 2:
            m = m[:, :, np.newaxis]
        blended.append(la[k] * m + lb[k] * (1.0 - m))

    result = blended[0]
    for k in range(1, bands):
        result = cv2.pyrUp(result, dstsize=blended[k].shape[1::-1]) + blended[k]

    # §1.108: Alpha schedule — mix a sharp-masked fine-frequency version (30%)
    # with the Laplacian result (70%) to reduce HF colour bleeding at edges.
    if alpha_schedule:
        sharp_mask = mask_float[:, :, np.newaxis].astype(np.float32) ** 2
        sharp_blend = (
            a.astype(np.float32) * sharp_mask
            + b.astype(np.float32) * (1.0 - sharp_mask)
        )
        result = np.clip(0.3 * sharp_blend + 0.7 * result, 0, 255)

    return np.clip(result, 0, 255).astype(np.uint8)


def _seam_dp(
    img1: np.ndarray,
    img2: np.ndarray,
    horizontal: bool = True,
    search_half: int | None = None,
) -> np.ndarray:
    """
    Dynamic-programming optimal seam between two images.
    Energy = colour diff + gradient diff.  Seam avoids high-contrast edges
    (anime line art), preferring flat cel-shaded regions.

    Parameters
    ----------
    horizontal    : if True, the seam is horizontal (separates top/bottom);
                    if False, the seam is vertical (separates left/right).
    search_half   : when provided, restricts the seam search to ±search_half
                    pixels around the image midpoint along the seam axis.
                    Use when the cross-axis displacement is small (≤5 px) so the
                    seam stays near centre rather than wandering to an edge.
                    Roadmap item 1.5: ±100 px when dx_cv < 5.

    Returns
    -------
    path : int array of length W (horizontal) or H (vertical),
           giving the seam row per column (or column per row).
    """
    diff = cv2.absdiff(img1, img2).astype(np.float32).mean(axis=2)
    gx = cv2.Sobel(diff, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(diff, cv2.CV_32F, 0, 1, ksize=3)
    energy = diff + 0.5 * (np.abs(gx) + np.abs(gy))

    if not horizontal:
        energy = energy.T
    h, w = energy.shape

    # ── Search-window restriction (item 1.5) ──────────────────────────────
    # When the cross-axis displacement is small, confine the seam to a band
    # around the image midpoint so it cannot wander to a border.
    if search_half is not None and search_half < w:
        mid = w // 2
        lo = max(0, mid - search_half)
        hi = min(w, mid + search_half + 1)
        mask = np.full((h, w), np.inf, dtype=np.float32)
        mask[:, lo:hi] = energy[:, lo:hi]
        energy = mask

    M = energy.copy()
    for i in range(1, h):
        left = np.empty_like(M[i - 1])
        left[0] = np.inf
        left[1:] = M[i - 1, :-1]
        right = np.empty_like(M[i - 1])
        right[-1] = np.inf
        right[:-1] = M[i - 1, 1:]
        M[i] += np.minimum(M[i - 1], np.minimum(left, right))

    path = np.zeros(h, np.int32)
    j = int(np.argmin(M[h - 1]))
    path[h - 1] = j
    for i in range(h - 2, -1, -1):
        nbrs = [j]
        if j > 0:
            nbrs.append(j - 1)
        if j < w - 1:
            nbrs.append(j + 1)
        j = nbrs[int(np.argmin([M[i, c] for c in nbrs]))]
        path[i] = j

    return path  # horizontal=False: path is column-per-row (transposed)


def _largest_valid_rect(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the largest axis-aligned rectangle of valid (non-zero) pixels.

    Strategy
    --------
    Vertical pans produce a canvas that is valid everywhere except a
    narrow black border.  We exploit this with a two-step approach:

    1. Fast path: project the mask onto rows and columns.
       - Find the widest contiguous column band where >=95% of rows are valid.
       - Find the tallest contiguous row band where >=95% of columns are valid.
       - The intersection is returned immediately if it covers >=40% of valid pixels.

    2. Histogram fallback on an aggressively downsampled (16x) mask.
       At 16x downsampling, a 3844x7372 canvas becomes 240x461 cells —
       110K iterations of the Python stack loop rather than 27M.

    Returns (x0, y0, x1, y1) — half-open column/row bounds.
    """
    h, w = mask.shape
    if h == 0 or w == 0:
        return (0, 0, w, h)

    binary = mask > 0  # bool (H, W)

    # -- Fast path -----------------------------------------------------------
    # row_valid_frac[r] = fraction of columns that are valid in row r
    row_frac = binary.mean(axis=1)  # (H,) float
    col_frac = binary.mean(axis=0)  # (W,) float

    def _longest_thresh_run(frac, thr=0.95):
        """Longest contiguous run where frac >= thr."""
        bools = frac >= thr
        best_s, best_l = 0, 0
        cs, cl = 0, 0
        for i, v in enumerate(bools):
            if v:
                if cl == 0:
                    cs = i
                cl += 1
                if cl > best_l:
                    best_l, best_s = cl, cs
            else:
                cl = 0
        return best_s, best_l

    r0, rlen = _longest_thresh_run(row_frac, 0.95)
    c0, clen = _longest_thresh_run(col_frac, 0.95)

    fast_area = rlen * clen
    valid_px = max(int(binary.sum()), 1)

    if fast_area >= 0.40 * valid_px:
        return (c0, r0, c0 + clen, r0 + rlen)

    # -- Histogram fallback at 16x downscaling -------------------------------
    DS = 16
    hs = max(h // DS, 1)
    ws = max(w // DS, 1)
    small = cv2.resize(
        binary.astype(np.uint8) * 255,
        (ws, hs),
        interpolation=cv2.INTER_NEAREST,
    )
    bin_s = (small > 0).astype(np.int32)
    heights = np.zeros(ws, np.int32)
    best = (0, 0, w, h)
    best_area = 0

    for row in range(hs):
        heights = np.where(bin_s[row], heights + 1, 0)
        stack: List[int] = []
        for col in range(ws + 1):
            cur_h = int(heights[col]) if col < ws else 0
            start = col
            while stack and int(heights[stack[-1]]) > cur_h:
                idx = stack.pop()
                hh = int(heights[idx])
                ww = col - (stack[-1] + 1 if stack else 0)
                area = hh * ww
                if area > best_area:
                    best_area = area
                    x0s = stack[-1] + 1 if stack else 0
                    y0s = row - hh + 1
                    best = (
                        min(x0s * DS, w),
                        min(y0s * DS, h),
                        min((x0s + ww) * DS, w),
                        min((y0s + hh) * DS, h),
                    )
                start = idx
            stack.append(start)
            if col < ws:
                heights[start] = cur_h

    return best  # (x0, y0, x1, y1)  half-open


__all__ = [
    "_luma",
    "_highpass",
    "_trim_dark_border",
    "_laplacian_blend",
    "_seam_dp",
    "_largest_valid_rect",
]
