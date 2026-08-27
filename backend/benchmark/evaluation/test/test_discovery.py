"""Dataset discovery: standard benchmark layout and renderer-export bundles."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ..constants.schema import (  # noqa: E402
    IMAGE_ASP,
    IMAGE_BASELINE,
    IMAGE_P1,
    IMAGE_P1P2,
    IMAGE_SIMPLE,
)
from ..other import discovery  # noqa: E402


def _png(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((8, 8, 3), 127, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Standard benchmark layout
# ---------------------------------------------------------------------------


def test_discover_standard_layout(tmp_path):
    for name in ("asp_test01", "asp_test07", "asp_test03"):
        _png(tmp_path / "output" / f"{name}_anime_stitch.png")

    assert discovery.discover_datasets(str(tmp_path)) == [
        "asp_test01",
        "asp_test03",
        "asp_test07",
    ]
    assert discovery.is_renderer_export_dir(str(tmp_path)) is False


# ---------------------------------------------------------------------------
# Renderer-export bundle layout (asp_testXX/ arm subdirectories)
# ---------------------------------------------------------------------------


def test_discover_renderer_export_bundle(tmp_path):
    for name in ("asp_test01", "asp_test17", "asp_test05"):
        for fname in ("baseline.png", "p1_single_pose.png", "p1p2_multiband.png"):
            _png(tmp_path / name / fname)
    # A stray non-matching dir and a dir with no arm images are both ignored.
    (tmp_path / "notes").mkdir()
    (tmp_path / "asp_test99").mkdir()

    names = discovery.discover_datasets(str(tmp_path))
    assert names == ["asp_test01", "asp_test05", "asp_test17"]
    assert discovery.is_renderer_export_dir(str(tmp_path)) is True


def test_renderer_export_partial_triplet_still_discovered(tmp_path):
    """A case with only some arms present is still offered for review."""
    _png(tmp_path / "asp_test42" / "baseline.png")
    assert discovery.discover_datasets(str(tmp_path)) == ["asp_test42"]


def test_load_test_assets_resolves_renderer_export_arms(tmp_path):
    for fname in ("baseline.png", "p1_single_pose.png", "p1p2_multiband.png"):
        _png(tmp_path / "asp_test01" / fname)

    assets = discovery.load_test_assets(str(tmp_path), "asp_test01", str(tmp_path))

    assert set(assets.available()) == {IMAGE_BASELINE, IMAGE_P1, IMAGE_P1P2}
    assert assets.paths[IMAGE_BASELINE].endswith("asp_test01/baseline.png")
    assert IMAGE_ASP not in assets.paths
    assert IMAGE_SIMPLE not in assets.paths
    assert assets.metrics == {}


def test_empty_dir_discovers_nothing(tmp_path):
    assert discovery.discover_datasets(str(tmp_path)) == []
    assert discovery.is_renderer_export_dir(str(tmp_path)) is False
