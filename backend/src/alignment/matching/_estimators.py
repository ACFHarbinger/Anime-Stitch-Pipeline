"""RANSAC estimator method selection — the MAGSAC++ swap (deepseek vendor-mining
critique, 2026-08-23).

ASP's Stage 5-6 matching historically uses plain OpenCV RANSAC
(``cv2.RANSAC``) for the per-pair affine/homography estimates. MAGSAC++
(``cv2.USAC_MAGSAC``) marginalizes over noise scales and stays stable at the
high-outlier ratios caused by animated foreground contamination of background
correspondences. Behind ``ASP_USAC_MAGSAC`` (default off); the method is
resolved at each call so the flag can differ between processes without a
code change. All five Stage 5-6 estimator call sites route through here.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

_USAC_MAGSAC: bool = os.environ.get("ASP_USAC_MAGSAC", "0") == "1"


def _method() -> int:
    return cv2.USAC_MAGSAC if _USAC_MAGSAC else cv2.RANSAC


def estimate_affine2d(
    pts1: np.ndarray,
    pts2: np.ndarray,
    ransacReprojThreshold: float = 5.0,
    **kwargs,
):
    """``cv2.estimateAffine2D`` with the configured robust estimator."""
    return cv2.estimateAffine2D(
        pts1, pts2, method=_method(), ransacReprojThreshold=ransacReprojThreshold, **kwargs
    )


def estimate_affine_partial2d(
    pts1: np.ndarray,
    pts2: np.ndarray,
    ransacReprojThreshold: float = 2.0,
    confidence: float = 0.999,
    maxIters: int = 10_000,
):
    """``cv2.estimateAffinePartial2D`` — always RANSAC.

    ``cv2.USAC_MAGSAC`` is **not supported** by ``estimateAffinePartial2D`` in
    OpenCV 4.11 (only ``estimateAffine2D`` and ``findHomography`` accept the
    USAC methods), so the MAGSAC++ swap applies to those two and this one
    deliberately stays on RANSAC. Routed through here so the estimator policy
    lives in one place.
    """
    return cv2.estimateAffinePartial2D(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransacReprojThreshold,
        confidence=confidence,
        maxIters=maxIters,
    )


def find_homography(
    pts1: np.ndarray,
    pts2: np.ndarray,
    ransacReprojThreshold: float = 5.0,
):
    """``cv2.findHomography`` with the configured robust estimator."""
    return cv2.findHomography(
        pts1, pts2, _method(), ransacReprojThreshold=ransacReprojThreshold
    )


__all__ = [
    "estimate_affine2d",
    "estimate_affine_partial2d",
    "find_homography",
    "_USAC_MAGSAC",
]
