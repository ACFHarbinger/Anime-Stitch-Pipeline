"""
Tests for backend.src.animation.ingestion.masking — Issue 10A2 S83.

Focus: _compute_fg_masks_sam2_stateful and _cleanup_sam2_state.
All tests pass whether or not SAM-2 is installed (graceful fallback paths).
"""

from __future__ import annotations

import os
import tempfile
import unittest.mock as mock

import numpy as np
from backend.src.animation.ingestion.masking import (
    _cleanup_sam2_state,
    _compute_fg_masks_sam2_stateful,
)


def _make_frames(n: int = 3, h: int = 16, w: int = 16) -> list:
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


# ── TestComputeFgMasksSam2Stateful ─────────────────────────────────────────────


class TestComputeFgMasksSam2Stateful:
    def test_returns_six_tuple(self):
        """Always returns a 6-tuple regardless of SAM-2 / BiRefNet availability."""
        frames = _make_frames(2)
        result = _compute_fg_masks_sam2_stateful(frames, None, use_birefnet=False)
        assert isinstance(result, tuple)
        assert len(result) == 6

    def test_no_birefnet_returns_none_predictor(self):
        """When use_birefnet=False, predictor/state/tmp_dir are all None."""
        frames = _make_frames(3)
        masks, pred, state, tmp, fh, fw = _compute_fg_masks_sam2_stateful(
            frames, None, use_birefnet=False
        )
        assert pred is None
        assert state is None
        assert tmp is None
        assert len(masks) == 3

    def test_frame_dimensions_returned(self):
        """frame_h and frame_w reflect the first frame's shape."""
        frames = _make_frames(2, h=48, w=64)
        _, _, _, _, fh, fw = _compute_fg_masks_sam2_stateful(
            frames, None, use_birefnet=False
        )
        assert fh == 48
        assert fw == 64

    def test_sam2_unavailable_falls_back_gracefully(self):
        """With birefnet=None (no mask for bbox), predictor is None and mask count correct."""
        frames = _make_frames(4, h=32, w=32)
        # birefnet_wrapper=None with use_birefnet=True triggers early-return
        masks, pred, state, tmp, fh, fw = _compute_fg_masks_sam2_stateful(
            frames, None, use_birefnet=True
        )
        assert pred is None
        assert state is None
        assert tmp is None
        assert len(masks) == 4

    def test_birefnet_exception_fallback_is_valid_tuple(self):
        """When BiRefNet throws on frame 0, falls back to per-frame path, still 6-tuple."""
        frames = _make_frames(2, h=8, w=8)
        bad_birefnet = mock.MagicMock()
        bad_birefnet.get_background_mask.side_effect = RuntimeError("birefnet offline")
        # get_mask also used as fallback API
        bad_birefnet.get_mask.side_effect = RuntimeError("birefnet offline")

        masks, pred, state, tmp, fh, fw = _compute_fg_masks_sam2_stateful(
            frames, bad_birefnet, use_birefnet=True
        )
        assert pred is None
        assert len(masks) == 2


# ── TestCleanupSam2State ───────────────────────────────────────────────────────


class TestCleanupSam2State:
    def test_cleanup_all_none_is_noop(self):
        """Passing all-None values to cleanup is safe (no exception)."""
        _cleanup_sam2_state(None, None, None)

    def test_cleanup_removes_tmp_dir(self):
        """cleanup removes the on-disk tmp directory."""
        tmp = tempfile.mkdtemp(prefix="asp_test_")
        assert os.path.isdir(tmp)
        _cleanup_sam2_state(None, None, tmp)
        assert not os.path.isdir(tmp)

    def test_cleanup_calls_reset_state(self):
        """cleanup calls predictor.reset_state(inference_state) when both are provided."""
        mock_predictor = mock.MagicMock()
        mock_state = {"num_frames": 3}
        _cleanup_sam2_state(mock_predictor, mock_state, None)
        mock_predictor.reset_state.assert_called_once_with(mock_state)

    def test_cleanup_tolerates_reset_exception(self):
        """cleanup does not propagate predictor.reset_state exceptions."""
        mock_predictor = mock.MagicMock()
        mock_predictor.reset_state.side_effect = RuntimeError("CUDA OOM")
        _cleanup_sam2_state(mock_predictor, {}, None)  # must not raise

    def test_cleanup_tolerates_missing_tmp_dir(self):
        """cleanup is safe when tmp_dir path no longer exists."""
        _cleanup_sam2_state(None, None, "/nonexistent/tmp/asp_test_xyz")
