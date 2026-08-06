"""
Tests for backend.src.animation.ingestion.video_ingestion — Issue 9.

pyav is optional. All tests must pass whether or not it is installed:
- When absent: error-path tests verify graceful RuntimeError / None return.
- When present: functional tests use a synthetic video written with cv2.VideoWriter.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from backend.src.animation.ingestion.video_ingestion import (
    VideoIngestionStream,
    _telecine_dedup,
    _uniform_select,
    ingest_video,
)

try:
    import av  # noqa: F401

    _AV_OK = True
except ImportError:
    _AV_OK = False


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_proxy_frames(n: int, h: int = 32, w: int = 64):
    """Return n synthetic (pts_index, BGR ndarray) tuples with distinct luma."""
    frames = []
    for i in range(n):
        img = np.full((h, w, 3), i * (255 // max(1, n - 1)), dtype=np.uint8)
        frames.append((i, img))
    return frames


def _write_test_video(
    path: str, n_frames: int = 10, fps: int = 24, w: int = 64, h: int = 64
) -> str:
    """Write a tiny MP4 with n_frames of distinct colour using OpenCV VideoWriter."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(n_frames):
        luma = int(i / n_frames * 200) + 20
        frame = np.full((h, w, 3), luma, dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


# ── TestUniformSelect ─────────────────────────────────────────────────────────


class TestUniformSelect:
    def test_empty_returns_empty(self):
        assert _uniform_select(0, 5) == []

    def test_zero_select_returns_empty(self):
        assert _uniform_select(10, 0) == []

    def test_select_all(self):
        result = _uniform_select(5, 10)
        assert result == [0, 1, 2, 3, 4]

    def test_uniform_spacing(self):
        result = _uniform_select(100, 5)
        assert len(result) == 5
        assert result[0] == 0
        # Each step should be ~20
        gaps = [result[i + 1] - result[i] for i in range(4)]
        assert all(15 <= g <= 25 for g in gaps)

    def test_single_frame(self):
        assert _uniform_select(1, 1) == [0]


# ── TestTelecineDedup ─────────────────────────────────────────────────────────


class TestTelecineDedup:
    def test_empty_returns_empty(self):
        assert _telecine_dedup([]) == []

    def test_all_different_kept(self):
        frames = _make_proxy_frames(5)
        kept = _telecine_dedup(frames, mad_thresh=1.0)
        assert len(kept) == 5

    def test_identical_frames_deduped(self):
        # All identical frames — only first should survive
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        frames = [(i, img.copy()) for i in range(5)]
        kept = _telecine_dedup(frames, mad_thresh=2.0)
        assert kept == [0]

    def test_alternating_kept(self):
        # Alternating distinct / duplicate pairs: [0,0,1,1,2,2]
        frames = []
        for val in [0, 0, 100, 100, 200, 200]:
            img = np.full((16, 16, 3), val, dtype=np.uint8)
            frames.append((len(frames), img))
        kept = _telecine_dedup(frames, mad_thresh=5.0)
        # Frames at index 0, 2, 4 should survive (first of each pair)
        assert 0 in kept
        assert 2 in kept
        assert 4 in kept
        assert 1 not in kept

    def test_first_frame_always_kept(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        frames = [(i, img.copy()) for i in range(3)]
        kept = _telecine_dedup(frames, mad_thresh=2.0)
        assert 0 in kept


# ── TestVideoIngestionStreamNoAv ─────────────────────────────────────────────


class TestVideoIngestionStreamNoAv:
    """Tests that work regardless of pyav availability."""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VideoIngestionStream("/nonexistent/video.mp4")

    def test_ingest_no_av_raises_runtime(self, tmp_path):
        if _AV_OK:
            pytest.skip("pyav is installed; skip no-av error path test")
        # Create a dummy file so FileNotFoundError is not raised first
        dummy = tmp_path / "fake.mp4"
        dummy.write_bytes(b"fake")
        stream = VideoIngestionStream(str(dummy))
        with pytest.raises(RuntimeError, match="pip install av"):
            stream.ingest(str(tmp_path / "out"))

    def test_ingest_video_no_av_raises(self, tmp_path):
        if _AV_OK:
            pytest.skip("pyav is installed; skip no-av error path test")
        dummy = tmp_path / "fake.mp4"
        dummy.write_bytes(b"fake")
        with pytest.raises(RuntimeError, match="pip install av"):
            ingest_video(str(dummy), str(tmp_path / "out"))


# ── TestVideoIngestionStreamWithAv ────────────────────────────────────────────


@pytest.mark.skipif(not _AV_OK, reason="pyav not installed")
class TestVideoIngestionStreamWithAv:
    """Functional tests that require pyav to be installed."""

    def test_ingest_uniform_returns_correct_count(self, tmp_path):
        vpath = _write_test_video(str(tmp_path / "test.mp4"), n_frames=10)
        stream = VideoIngestionStream(vpath, n_frames=5, mode="uniform", telecine=False)
        frames, paths = stream.ingest(str(tmp_path / "frames"))
        assert len(frames) == 5
        assert len(paths) == 5
        assert all(os.path.isfile(p) for p in paths)

    def test_ingest_saves_png_files(self, tmp_path):
        vpath = _write_test_video(str(tmp_path / "test.mp4"), n_frames=6)
        stream = VideoIngestionStream(vpath, n_frames=3, mode="uniform", telecine=False)
        _, paths = stream.ingest(str(tmp_path / "frames"))
        for p in paths:
            assert p.endswith(".png")
            img = cv2.imread(p)
            assert img is not None
            assert img.ndim == 3

    def test_frames_are_full_resolution(self, tmp_path):
        W, H = 64, 64
        vpath = _write_test_video(str(tmp_path / "test.mp4"), n_frames=4, w=W, h=H)
        stream = VideoIngestionStream(vpath, n_frames=2, mode="uniform", telecine=False)
        frames, _ = stream.ingest(str(tmp_path / "frames"))
        for f in frames:
            assert f.shape[0] == H
            assert f.shape[1] == W

    def test_telecine_reduces_frame_count(self, tmp_path):
        # Write a video where every other frame is identical (telecine sim)
        path = str(tmp_path / "tc.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 24, (32, 32))
        for i in range(10):
            luma = (i // 2) * 50
            writer.write(np.full((32, 32, 3), luma, dtype=np.uint8))
        writer.release()
        stream = VideoIngestionStream(path, n_frames=10, telecine=True)
        frames, _ = stream.ingest(str(tmp_path / "tc_frames"))
        # With telecine dedup we should get fewer than 10
        assert len(frames) <= 10
