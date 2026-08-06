"""Cross-image visualizations: ORB/SIFT feature-match line maps and dense
optical flow in HSV space. These take two BGR images (typically ASP vs
Simple, or ASP vs ground truth) rather than one — kept in a separate module
from ``visualizations_basic`` since their signature and cost profile
(detector/matcher, dense flow) differ enough to not share a helper.
"""

from __future__ import annotations

import cv2
import numpy as np
from matplotlib.figure import Figure

from .figure_theme import themed_figure


def feature_match_figure(img_a: np.ndarray, img_b: np.ndarray, method: str = "orb") -> Figure:
    """Draws ORB (or SIFT) keypoint matches as connecting lines — chaotic
    crisscrossing lines are the visual signature of a low-texture anime
    region confusing the matcher."""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    if method == "sift" and hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=400)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=400)
        norm = cv2.NORM_HAMMING

    kp_a, des_a = detector.detectAndCompute(gray_a, None)
    kp_b, des_b = detector.detectAndCompute(gray_b, None)

    fig, ax = themed_figure(figsize=(9.0, 5.0))
    if des_a is None or des_b is None or len(kp_a) == 0 or len(kp_b) == 0:
        ax.set_title(f"{method.upper()} matches — no features detected")
        ax.axis("off")
        return fig

    matcher = cv2.BFMatcher(norm, crossCheck=True)
    matches = sorted(matcher.match(des_a, des_b), key=lambda m: m.distance)[:60]
    match_img = cv2.drawMatches(
        img_a, kp_a, img_b, kp_b, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    ax.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{method.upper()} Feature Matches ({len(matches)} shown)")
    ax.axis("off")
    return fig


def optical_flow_hsv_figure(img_a: np.ndarray, img_b: np.ndarray) -> Figure:
    """Dense Farneback optical flow rendered in HSV: hue = direction,
    value = magnitude — the standard way to make displacement fields
    legible at a glance."""
    h, w = img_a.shape[:2]
    b_resized = cv2.resize(img_b, (w, h)) if img_b.shape[:2] != (h, w) else img_b
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b_resized, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    fig, ax = themed_figure()
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("Dense Optical Flow (hue=direction, value=magnitude)")
    ax.axis("off")
    return fig
