"""SCANS fallback must not crash the batch when OpenCV returns status=1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from asp_backend.alignment.canvas import _reuse_simple_stitch, _scan_stitch_fallback
from backend.src.errors import CanvasError


def test_reuse_simple_stitch_copies_sibling(tmp_path: Path) -> None:
    from PIL import Image

    src = tmp_path / "opencv_stitch.png"
    dest = tmp_path / "panorama.png"
    Image.new("RGB", (6, 6), (1, 2, 3)).save(src)
    reused = _reuse_simple_stitch(str(dest))
    assert reused is not None
    assert dest.is_file()
    assert dest.stat().st_size == src.stat().st_size


def test_reuse_simple_stitch_missing_returns_none(tmp_path: Path) -> None:
    dest = tmp_path / "panorama.png"
    assert _reuse_simple_stitch(str(dest)) is None


def test_scan_fallback_reuses_simple_stitch_when_stitcher_fails(tmp_path: Path) -> None:
    """One tiny frame cannot stitch (status=1); sibling opencv_stitch is used."""
    from PIL import Image

    simple = tmp_path / "opencv_stitch.png"
    Image.new("RGB", (8, 8), (20, 40, 60)).save(simple)
    dest = tmp_path / "panorama.png"
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    out = _scan_stitch_fallback(frames, str(dest))
    assert dest.is_file()
    assert out.size == (8, 8)


def test_reuse_simple_stitch_finds_parent_output_dir(tmp_path: Path) -> None:
    """Bench writes run_output.png under panorama_stages/; stitch is one level up."""
    from PIL import Image

    out_dir = tmp_path / "output"
    stages = out_dir / "panorama_stages"
    stages.mkdir(parents=True)
    Image.new("RGB", (10, 10), (9, 8, 7)).save(out_dir / "opencv_stitch.png")
    dest = stages / "run_output.png"
    reused = _reuse_simple_stitch(str(dest))
    assert reused is not None
    assert dest.is_file()


def test_scan_fallback_raises_when_no_retry_and_no_sibling(tmp_path: Path) -> None:
    dest = tmp_path / "panorama.png"
    with pytest.raises(CanvasError, match="SCANS fallback failed"):
        _scan_stitch_fallback([], str(dest))
