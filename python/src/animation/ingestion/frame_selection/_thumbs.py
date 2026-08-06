"""Thumbnail I/O and per-pair Otsu background masking."""

from __future__ import annotations

import concurrent.futures
import os
from typing import List, Optional

import cv2
import numpy as np

_SELECTOR_THUMB_LONG = 256  # thumbnail longest side for phase-correlation pass

# §1A: Per-pair Otsu background mask for phase correlation.
# Faster and more accurate than the 5-probe BiRefNet intersection (_TWO_CHANNEL_SELECT)
# because each pair gets its own mask rather than sharing a static 5-probe estimate.
# Falls back to plain phaseCorrelate when the combined bg coverage < 10%.
# Default OFF — enable with ASP_OTSU_BG_CORR=1 or in asp_config.toml.
_OTSU_BG_CORR: bool = os.environ.get("ASP_OTSU_BG_CORR", "0") != "0"


def _load_thumb_gray(path: str) -> np.ndarray:
    """Load a grayscale float32 thumbnail for phase correlation."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((_SELECTOR_THUMB_LONG, _SELECTOR_THUMB_LONG), dtype=np.float32)
    h, w = img.shape
    scale = _SELECTOR_THUMB_LONG / max(h, w, 1)
    tw = max(1, int(w * scale))
    th = max(1, int(h * scale))
    return cv2.resize(img, (tw, th)).astype(np.float32) / 255.0


def _load_thumbs_parallel(
    frames_paths: List[str], max_workers: int = 8
) -> List[np.ndarray]:
    """Load thumbnails in parallel (I/O-bound; GIL released in cv2.imread)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_load_thumb_gray, frames_paths))


def _otsu_bg_mask_pair(
    a: np.ndarray, b: np.ndarray, min_bg_frac: float = 0.10
) -> Optional[np.ndarray]:
    """§1A: per-pair Otsu background mask for bg-only phase correlation.

    Computes an Otsu threshold on each float32 grayscale thumbnail ([0,1]),
    treats pixels brighter than the threshold as background, then erodes
    both masks slightly to remove foreground-edge contamination.  Returns
    the pixel-wise minimum (intersection) so only pixels classified as
    background in BOTH frames are used for phase correlation.

    Returns None when the combined background coverage is below
    ``min_bg_frac`` (character fills most of the frame — no reliable signal).

    Parameters
    ----------
    a, b : (H, W) float32 thumbnails in [0, 1].
    min_bg_frac : minimum fraction of background pixels required to proceed.
    """
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masks = []
    for thumb in (a, b):
        u8 = (thumb * 255.0).clip(0, 255).astype(np.uint8)
        thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # pixels > Otsu threshold are "light" — typically background in anime
        bg_u8 = (u8 > thr).astype(np.uint8) * 255
        bg_u8 = cv2.erode(bg_u8, erode_k)
        masks.append(bg_u8.astype(np.float32) / 255.0)
    combined = np.minimum(masks[0], masks[1])
    if float(combined.mean()) < min_bg_frac:
        return None
    return combined


__all__ = [
    "_SELECTOR_THUMB_LONG",
    "_OTSU_BG_CORR",
    "_load_thumb_gray",
    "_load_thumbs_parallel",
    "_otsu_bg_mask_pair",
]
