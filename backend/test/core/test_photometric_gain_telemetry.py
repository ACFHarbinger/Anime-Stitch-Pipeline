"""M2: background photometric stage reports gain residuals/clamps."""

from __future__ import annotations

import numpy as np
from asp_backend.core.pipeline._photometric_stage import (
    _apply_background_photometric_normalization,
)


def _frame(lum: int, h: int = 40, w: int = 40) -> np.ndarray:
    return np.full((h, w, 3), lum, dtype=np.uint8)


def _bg_mask(h: int = 40, w: int = 40) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def test_gain_telemetry_records_clamp_and_residual():
    # Three frames so the median-reference path runs; one is much darker.
    frames = [_frame(80), _frame(40), _frame(80)]
    masks = [_bg_mask(), _bg_mask(), _bg_mask()]
    telem: dict = {}
    out = _apply_background_photometric_normalization(frames, masks, 3, telemetry=telem)
    assert len(out) == 3
    assert telem["frames"]
    assert "n_clamped" in telem
    assert "mean_residual" in telem
    dark = next(row for row in telem["frames"] if row["frame"] == 1)
    assert dark["residual"] >= 0.0
    assert dark["clamp_lo"] < dark["clamp_hi"]
    assert len(dark["gain_bgr"]) == 3


def test_no_telemetry_arg_is_still_pixel_compatible():
    frames = [_frame(80), _frame(80), _frame(80)]
    out = _apply_background_photometric_normalization(
        frames, [_bg_mask()] * 3, 3
    )
    assert len(out) == 3
