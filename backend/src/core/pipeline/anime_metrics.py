"""2D Anime-adapted computer vision quality metrics (Milestone §M2.5a, #32).

Provides specialized image metrics adapted for 2D animation / cel-shaded imagery
featuring piecewise-constant regions (skin, clothing, sky) and sharp ink line art.

Unlike photographic sharpness / edge metrics (e.g. Sobel, Laplacian, SIQE) which
suffer catastrophic inverse correlation (rho = -0.47 to -0.60) by mistaking torn
anatomy and broken seam cuts for high-frequency 'detail', these metrics explicitly
separate ink line art skeletons from flat cel-shaded regions:

1. ``line_art_fracture_score``: Measures line endpoint density and contour
   fragmentation. Correctly penalizes severed limbs, displaced strips, and broken
   anatomical lines (human Spearman rho = +0.320, p = 0.0014).
2. ``cel_flatness_variance``: Measures local luminance variance strictly within
   non-edge cel regions, penalizing color banding, exposure steps, and sensor noise.
3. ``flat_region_edge_leakage``: Measures edge energy leaking into flat cel fills.

Per roadmap governance, all metrics in this module are currently non-gating
diagnostic candidates.
"""

from __future__ import annotations

import cv2
import numpy as np


def extract_line_art(img: np.ndarray) -> np.ndarray:
    """Extract anime ink line-art contours via adaptive thresholding on grayscale.

    Parameters
    ----------
    img : (H, W) or (H, W, 3) uint8 image.

    Returns
    -------
    lines : (H, W) uint8 binary mask where 255 represents ink lines.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    lines = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 7
    )
    return lines


def extract_flat_cel_mask(img: np.ndarray, lines: np.ndarray | None = None) -> np.ndarray:
    """Extract mask of flat cel-shaded regions (excluding borders and line art).

    Parameters
    ----------
    img : (H, W) or (H, W, 3) uint8 image.
    lines : optional precomputed line art binary mask.

    Returns
    -------
    flat_mask : (H, W) boolean mask where True indicates interior cel regions.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if lines is None:
        lines = extract_line_art(gray)
    content = gray > 10  # Exclude black background borders
    dilated_lines = cv2.dilate(lines, np.ones((7, 7), np.uint8))
    return content & (dilated_lines == 0)


def _skeletonize(img_bin: np.ndarray) -> np.ndarray:
    """Morphological 1-pixel skeletonization of binary mask."""
    skel = np.zeros_like(img_bin, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = img_bin.copy()
    while cv2.countNonZero(img) > 0:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
    return skel


def line_art_fracture_score(img: np.ndarray) -> float:
    """Compute line art fragmentation and endpoint density.

    Torn anatomy, broken seams, and misordered frame slices fracture ink outlines
    and create high numbers of dangling line endpoints and micro-fragments.

    Score interpretation:
      Lower = smoother, continuous line art (better).
      0-15 : high-quality continuous outlines.
      15-40: mild line fragmentation or natural high-density cross-hatching.
      40+  : severe structural tearing or fragmented outlines.

    Empirical validation on 97 human-reviewed cases:
      Spearman rho vs human delta = +0.320 (p = 0.0014, statistically significant).

    Parameters
    ----------
    img : (H, W) or (H, W, 3) uint8 image.

    Returns
    -------
    score : float fragmentation index (endpoints + 2 * components per 1000 line px).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lines = extract_line_art(gray)
    total_px = float((lines > 0).sum())
    if total_px < 20.0:
        return 0.0

    # Skeletonize lines to 1px thickness for robust endpoint detection
    skel = _skeletonize(lines)
    skel_bin = (skel > 0).astype(np.float32)

    # 3x3 filter to count 8-neighbors: center gets weight 10, neighbors get 1.
    filt = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.float32)
    conv = cv2.filter2D(skel_bin, -1, filt)
    endpoints = float((conv == 11).sum())  # Center pixel (10) + exactly 1 neighbor (1)

    # Connected components to detect line fragmentation
    num_labels, _ = cv2.connectedComponents(lines)
    components = float(max(0, num_labels - 1))

    fracture_index = ((endpoints + components * 2.0) / total_px) * 1000.0
    return round(float(fracture_index), 2)


def cel_flatness_variance(img: np.ndarray) -> float:
    """Measure median local variance within flat cel-shaded regions.

    Clean anime fills (skin, cloth, sky) have piecewise-uniform luminance.
    Color banding, compression noise, and exposure steps increase local variance.

    Lower = cleaner cel fills (better).

    Parameters
    ----------
    img : (H, W) or (H, W, 3) uint8 image.

    Returns
    -------
    score : float median local standard deviation within flat regions.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lines = extract_line_art(gray)
    flat = extract_flat_cel_mask(img, lines)
    if flat.sum() < 100:
        return 0.0

    g32 = gray.astype(np.float32)
    local_mean = cv2.blur(g32, (15, 15))
    local_sq_mean = cv2.blur(g32**2, (15, 15))
    local_var = np.maximum(0.0, local_sq_mean - local_mean**2)
    local_std = np.sqrt(local_var)

    return round(float(np.median(local_std[flat])), 2)


def flat_region_edge_leakage(img: np.ndarray) -> float:
    """Measure high-frequency Laplacian edge energy leaking into flat cel regions.

    Parameters
    ----------
    img : (H, W) or (H, W, 3) uint8 image.

    Returns
    -------
    score : float mean absolute Laplacian within flat cel regions.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lines = extract_line_art(gray)
    flat = extract_flat_cel_mask(img, lines)
    if flat.sum() < 100:
        return 0.0

    lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3))
    return round(float(np.mean(lap[flat])), 3)


__all__ = [
    "extract_line_art",
    "extract_flat_cel_mask",
    "line_art_fracture_score",
    "cel_flatness_variance",
    "flat_region_edge_leakage",
]
