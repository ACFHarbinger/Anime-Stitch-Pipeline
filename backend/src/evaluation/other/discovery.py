"""Discovers ASP benchmark datasets and locates every on-disk artifact one
test can have, mirroring ``bench_anime_stitch.py``'s own directory layout so
the tool reads exactly what that script already writes without recomputing
anything:

- ``{base_dir}/output/{name}_anime_stitch.png``  — the ASP composite
- ``{base_dir}/output/{name}_opencv_stitch.png`` — the OpenCV SCANS baseline
  (``_simple_stitch.png`` before this tool's 2026-07-30 rename; still found as
  a fallback so a corpus generated under the old name keeps working)
- ``{base_dir}/{name}/output/overmix_stitch.png`` — the Overmix comparator
  (§0.3; present for all 97 tests as of 2026-07-28)
- ``{base_dir}/{name}/output/hugin_stitch.png``   — the Hugin comparator
  (§0.5; only the tests whose Hugin run actually produced a panorama)
- ``{base_dir}/ground_truth/{name}.{png,jpg,jpeg}``
- the ``paths``/``metrics_*`` fields of the newest ``anime_stitch_*.json``

The comparator paths come from the results JSON's own ``overmix_path`` /
``hugin_path`` fields when present, and fall back to the documented on-disk
convention when the JSON predates those fields — so a test's Overmix output is
found whether or not the benchmark run that produced the JSON knew about it.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..constants.schema import (
    COMPARATOR_KEYS,
    IMAGE_ASP,
    IMAGE_GROUND_TRUTH,
    IMAGE_HUGIN,
    IMAGE_OVERMIX,
    IMAGE_SIMPLE,
)


@dataclasses.dataclass
class TestAssets:
    name: str
    # image key -> absolute path; a key is absent when that comparator has no
    # output for this test, so callers never have to special-case per-key
    # attributes to find out what is available.
    paths: Dict[str, str]
    plots_dir: Optional[str]
    stage_dir: Optional[str]
    metrics: Dict

    def available(self) -> List[str]:
        """Comparator keys that exist on disk, in display order."""
        return [key for key in COMPARATOR_KEYS if key in self.paths]

    # Kept so existing callers/tests that predate the N-way rewrite keep
    # working against the three images the old dashboard knew about.
    @property
    def asp_path(self) -> Optional[str]:
        return self.paths.get(IMAGE_ASP)

    @property
    def simple_path(self) -> Optional[str]:
        return self.paths.get(IMAGE_SIMPLE)

    @property
    def gt_path(self) -> Optional[str]:
        return self.paths.get(IMAGE_GROUND_TRUTH)


def discover_datasets(base_dir: str) -> List[str]:
    out_dir = os.path.join(base_dir, "output")
    names = []
    for p in sorted(glob.glob(os.path.join(out_dir, "asp_test*_anime_stitch.png"))):
        names.append(os.path.basename(p)[: -len("_anime_stitch.png")])
    return names


def _find_gt_path(name: str, gt_dir: str) -> Optional[str]:
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(gt_dir, f"{name}{ext}")
        if os.path.exists(p):
            return p
    return None


_METRICS_CACHE: Dict[str, Dict[str, Dict]] = {}


def results_files(repo_root: str) -> List[str]:
    """Every ``anime_stitch_*.json`` results file, oldest first."""
    results_dir = os.path.join(repo_root, "backend", "benchmark", "output")
    return sorted(glob.glob(os.path.join(results_dir, "anime_stitch_*.json")))


def load_metrics(repo_root: str, results_path: Optional[str] = None) -> Dict[str, Dict]:
    """Index a ``bench_anime_stitch.py`` results JSON by dataset name.

    Defaults to the most recent run. Cached per resolved path rather than in a
    single module-level slot, so a caller comparing two runs (a cross-run
    regression read) doesn't have one evict the other.
    """
    if results_path is None:
        files = results_files(repo_root)
        if not files:
            return {}
        results_path = files[-1]
    if results_path in _METRICS_CACHE:
        return _METRICS_CACHE[results_path]
    try:
        with open(results_path) as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        _METRICS_CACHE[results_path] = {}
        return {}
    indexed = {d["name"]: d for d in doc.get("datasets", []) if "name" in d}
    # The run-level metadata/summary blocks are useful context in the UI
    # (which run am I looking at, how did the corpus do overall), so they ride
    # along under keys no dataset can collide with.
    indexed["__metadata__"] = doc.get("metadata", {})
    indexed["__summary__"] = doc.get("summary", {})
    indexed["__path__"] = {"path": results_path}
    _METRICS_CACHE[results_path] = indexed
    return indexed


def repo_root_from(file_path: str) -> str:
    """Walk up from ``file_path`` to the repo root (the directory holding the
    workspace-root ``pyproject.toml``, identified by ``[tool.uv.workspace]``).
    Depth-independent on purpose: an earlier version hardcoded a fixed
    ``dirname`` chain depth that resolved to ``backend/`` once this package
    was nested, silently splitting evaluations across two directories. A
    plain "any pyproject.toml" check has the same failure mode now that
    ``backend/`` and ``gui/`` each carry their own workspace-member
    ``pyproject.toml`` -- only the root one is the actual repo root."""
    d = os.path.dirname(os.path.abspath(file_path))
    while d != os.path.dirname(d):
        candidate = os.path.join(d, "pyproject.toml")
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                if "[tool.uv.workspace]" in f.read():
                    return d
        d = os.path.dirname(d)
    raise RuntimeError(f"Could not locate repo root (pyproject.toml) above {file_path}")


def _first_existing(*candidates: Optional[str]) -> Optional[str]:
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def load_test_assets(
    base_dir: str, name: str, repo_root: str, results_path: Optional[str] = None
) -> TestAssets:
    out_dir = os.path.join(base_dir, "output")
    test_out_dir = os.path.join(base_dir, name, "output")
    gt_dir = os.path.join(base_dir, "ground_truth")

    entry = load_metrics(repo_root, results_path).get(name, {})
    json_paths = entry.get("paths", {})

    resolved = {
        IMAGE_ASP: _first_existing(
            json_paths.get("anime_stitch"),
            os.path.join(out_dir, f"{name}_anime_stitch.png"),
        ),
        IMAGE_SIMPLE: _first_existing(
            json_paths.get("simple_stitch"),
            os.path.join(out_dir, f"{name}_opencv_stitch.png"),
            os.path.join(out_dir, f"{name}_simple_stitch.png"),  # pre-rename corpora
        ),
        IMAGE_OVERMIX: _first_existing(
            entry.get("overmix_path"),
            os.path.join(test_out_dir, "overmix_stitch.png"),
        ),
        IMAGE_HUGIN: _first_existing(
            entry.get("hugin_path"),
            os.path.join(test_out_dir, "hugin_stitch.png"),
        ),
        IMAGE_GROUND_TRUTH: _find_gt_path(name, gt_dir),
    }

    plots_dir = json_paths.get("plots_dir") or os.path.join(test_out_dir, "plots")
    stage_dir = json_paths.get("stage_dir") or os.path.join(test_out_dir, "panorama_stages")
    return TestAssets(
        name=name,
        paths={key: path for key, path in resolved.items() if path},
        plots_dir=plots_dir if os.path.isdir(plots_dir) else None,
        stage_dir=stage_dir if os.path.isdir(stage_dir) else None,
        metrics=entry,
    )


def imread_bgr(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    return cv2.imread(path)


def load_images(assets: TestAssets) -> Dict[str, np.ndarray]:
    """Decode every available comparator for one test, skipping any file that
    fails to decode (a truncated PNG from an interrupted run must not take the
    whole test down)."""
    images: Dict[str, np.ndarray] = {}
    for key in assets.available():
        img = imread_bgr(assets.paths[key])
        if img is not None:
            images[key] = img
    return images


def list_plot_images(plots_dir: Optional[str]) -> List[str]:
    if not plots_dir:
        return []
    return sorted(glob.glob(os.path.join(plots_dir, "*.png")))


def list_stage_images(stage_dir: Optional[str]) -> List[str]:
    """Per-stage debug renders, grouped by stage prefix.

    ``panorama_stages/`` holds 100+ files per test named
    ``stageNN_<label>_frameNN.png``; returning them flat would make an
    unusable list, so callers get them sorted and can group on the
    ``stageNN_<label>`` prefix.
    """
    if not stage_dir:
        return []
    return sorted(glob.glob(os.path.join(stage_dir, "*.png")))


def stage_groups(stage_dir: Optional[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for path in list_stage_images(stage_dir):
        stem = os.path.splitext(os.path.basename(path))[0]
        prefix = stem.rsplit("_frame", 1)[0] if "_frame" in stem else stem
        groups.setdefault(prefix, []).append(path)
    return groups
