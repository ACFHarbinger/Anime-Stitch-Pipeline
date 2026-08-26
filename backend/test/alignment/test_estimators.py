"""MAGSAC++ estimator swap (deepseek vendor-mining critique, 2026-08-23).

Verifies the flag selects cv2.USAC_MAGSAC for estimateAffine2D/findHomography
(the two estimators OpenCV 4.11 supports it for) and that the partial-affine
path stays on RANSAC (USAC_MAGSAC is unsupported there).
"""

from __future__ import annotations

import numpy as np
import pytest

from asp_backend.alignment.matching._estimators import (
    estimate_affine2d,
    estimate_affine_partial2d,
    find_homography,
)


def _pts():
    pts1 = np.float32(
        [[0, 0], [10, 0], [0, 10], [10, 10], [5, 5], [20, 20], [30, 30], [100, 100], [200, 200], [250, 250]]
    )
    pts2 = np.float32(
        [[1, 1], [11, 1], [1, 11], [11, 11], [6, 6], [21, 21], [31, 31], [0, 0], [200, 200], [300, 300]]
    )
    return pts1, pts2


def test_estimate_affine2d_runs(monkeypatch):
    monkeypatch.delenv("ASP_USAC_MAGSAC", raising=False)
    pts1, pts2 = _pts()
    M, inliers = estimate_affine2d(pts1, pts2)
    assert M is not None
    assert int(np.asarray(inliers).sum()) >= 7


def test_partial_affine_stays_ransac():
    import cv2

    from asp_backend.alignment.matching import _estimators as mod

    calls = []

    def fake_partial(pts1, pts2, **kw):
        calls.append(kw.get("method"))
        return np.eye(2, 3, dtype=np.float32), np.ones(len(pts1), dtype=np.uint8)

    real = mod.cv2.estimateAffinePartial2D
    mod.cv2.estimateAffinePartial2D = fake_partial
    try:
        estimate_affine_partial2d(*_pts())
    finally:
        mod.cv2.estimateAffinePartial2D = real
    assert calls and calls[0] == cv2.RANSAC


def test_flag_selects_magsac(monkeypatch):
    import cv2

    from asp_backend.alignment.matching import _estimators as mod

    monkeypatch.setattr(mod, "_USAC_MAGSAC", True)
    assert mod._method() == cv2.USAC_MAGSAC
    # estimateAffine2D + findHomography accept USAC_MAGSAC without error.
    pts1, pts2 = _pts()
    M, inliers = estimate_affine2d(pts1, pts2)
    assert M is not None
    H, status = find_homography(pts1, pts2)
    assert H is not None
