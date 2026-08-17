"""No-reference scores used by the Safe ASP output gates (M1b).

Moved out of ``bench_anime_stitch.py`` without changing the formulae.
The benchmark re-exports the original ``_``-prefixed names.
"""

from __future__ import annotations

import cv2
import numpy as np


def ghosting_score_v2(img: np.ndarray) -> float:
    """§3.8A: Double-edge autocorrelation ghosting score.

    A ghost (double-image artifact) creates a pair of parallel edges separated
    by displacement D in the scroll direction.  This shows up as a secondary
    peak in the normalized autocorrelation of the column-mean gradient-magnitude
    profile at lag D.

    Score interpretation:
      0–10  : no detectable double-edge structure (clean output)
      10–30 : mild periodic gradient pattern (natural scene texture, low concern)
      30–60 : moderate secondary peak (ghost possible, inspect)
      60+   : strong secondary peak (ghost highly likely)

    Unlike ``_ghosting_score`` / ``edge_energy_score`` (double-Sobel sharpness
    proxy, §3.32), this metric is specifically sensitive to *repeated* edge
    patterns at a fixed displacement — the signature of a misaligned character
    copy — while being less sensitive to high-frequency texture that is NOT
    ghost-related.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = gray.astype(np.float32)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(gy)

    profile = mag.mean(axis=1)  # (H,) column-mean gradient profile
    H = len(profile)
    if H < 20:
        return 0.0

    p = profile - profile.mean()
    n = 2 * H  # zero-pad to avoid circular aliasing
    P = np.fft.rfft(p, n=n)
    acorr = np.fft.irfft(P * P.conj(), n=n)[:H]

    zero_lag = float(acorr[0])
    if zero_lag < 1e-6:
        return 0.0

    acorr /= zero_lag  # normalize: acorr[0] = 1.0

    lag_min = 5
    lag_max = max(lag_min + 1, H // 4)
    secondary = float(acorr[lag_min:lag_max].max()) if lag_max > lag_min else 0.0
    return float(np.clip(secondary, 0.0, 1.0) * 100.0)


def seam_coherence(img: np.ndarray) -> float:
    """
    Standard deviation of per-row mean luminance.

    A clean panorama produced by genuine camera panning has smoothly varying
    row means (std ≈ 5–20).  An image with severe horizontal color banding —
    caused by the composite stacking frames with different animation-state
    colors — has wildly different row means across the height (std > 30).

    This metric is a better quality indicator than sharpness for detecting the
    catastrophic strip-banding failures that corrupt the Laplacian-variance score.
    Lower = more coherent (better).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Only consider rows that have content (not pure black borders)
    content_rows = gray.mean(axis=1) > 5
    if content_rows.sum() < 10:
        return 0.0
    row_means = gray[content_rows].mean(axis=1)
    return float(np.std(row_means))


def strip_banding_score(
    render_img: np.ndarray,
    affines: list[np.ndarray] | None = None,
) -> float:
    """
    Maximum luminance jump between adjacent frame-strip zones.

    Samples the mean luminance in a ±25px band around each frame's canvas entry
    row (where the frame's affine ty places it).  If two adjacent strips differ
    by more than the returned value, severe color banding is present.

    Used as a fallback trigger: if max_strip_jump > 20.0 lum units, the Stage 11
    composite is likely to produce visible color bands and SCANS fallback is
    preferable.
    """
    if affines is None or len(affines) < 2:
        return 0.0
    gray = (
        cv2.cvtColor(render_img, cv2.COLOR_BGR2GRAY)
        if render_img.ndim == 3
        else render_img
    )
    H = gray.shape[0]
    strip_means = []
    for a in sorted(affines, key=lambda m: float(m[1, 2])):
        ty = int(float(a[1, 2]))
        y0 = max(0, ty)
        y1 = min(H, ty + 50)
        if y1 > y0:
            band = gray[y0:y1]
            # Skip near-black border regions
            if band.mean() > 5:
                strip_means.append(float(band.mean()))
    if len(strip_means) < 2:
        return 0.0
    diffs = [
        abs(strip_means[i + 1] - strip_means[i]) for i in range(len(strip_means) - 1)
    ]
    return max(diffs)


def seam_visibility_score(
    output_img: np.ndarray,
    affines: list[np.ndarray] | None = None,
) -> float:
    """
    Worst-case horizontal luminance discontinuity (no-reference).

    Computes the per-row mean absolute difference profile across the full
    output image, then reports the maximum peak value.  A perfectly blended
    seam contributes nothing to this profile; a hard single-pose seam cut
    produces a large spike exactly at the seam row.

    Lower = smoother output (better).  0 = no visible discontinuities.
    Typical clean outputs: < 6.  Single-pose seam cuts: 12–50+.

    Unlike ``seam_coherence`` (global row-mean variance), this detects
    localised hard cuts rather than broad brightness drift, making it
    complementary to the existing metrics.  Works for all 96 tests with
    no ground truth required.

    Parameters
    ----------
    output_img : (H, W) or (H, W, 3) uint8 panorama
    affines    : unused; kept for API compatibility with _compute_all_metrics
    """
    gray = (
        cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        if output_img.ndim == 3
        else output_img
    )
    g = gray.astype(np.float32)
    H, W = g.shape

    # Per-row mean luminance, excluding near-black border pixels.
    content = g > 5  # True where pixel is non-black
    row_content_count = content.sum(axis=1)
    row_valid = row_content_count > W * 0.1  # rows with ≥10% content
    if row_valid.sum() < 4:
        return 0.0

    # Compute mean only for content rows to avoid empty-slice warnings.
    valid_idx = np.where(row_valid)[0]
    row_sums = np.where(content[valid_idx], g[valid_idx], 0.0).sum(axis=1)
    row_mean_vals = row_sums / np.maximum(row_content_count[valid_idx], 1)

    # Adjacent-row absolute difference on content rows only.
    diffs = np.abs(np.diff(row_mean_vals))

    # The worst-case single-row jump is the seam visibility score.
    return round(float(np.nanmax(diffs)) if len(diffs) > 0 else 0.0, 2)


from .anime_metrics import (
    cel_flatness_variance,
    extract_flat_cel_mask,
    extract_line_art,
    flat_region_edge_leakage,
    line_art_fracture_score,
)

__all__ = [
    "ghosting_score_v2",
    "seam_coherence",
    "seam_visibility_score",
    "strip_banding_score",
    "extract_line_art",
    "extract_flat_cel_mask",
    "line_art_fracture_score",
    "cel_flatness_variance",
    "flat_region_edge_leakage",
]
