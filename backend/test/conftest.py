"""Shared pytest fixtures for backend/test/.

Registers this repo's own backend/src and gui/src packages under the same
asp_backend / asp_gui aliases Image-Toolkit's own bootstrap uses (see
Image-Toolkit's _submodule_bootstrap.py), so tests here and tests run via
Image-Toolkit import identically regardless of which repo mounted them.

Also registers backend/benchmark/evaluation under asp_backend_evaluation,
for the same reason: this repo's and Image-Toolkit's top-level `backend`
directories share a name and neither has a package-qualifying `__init__.py`
distinguishing them from the other, so a raw `backend.benchmark.evaluation`
absolute import resolves inconsistently (or not at all) depending on which
repo's `backend` package Python's import system finds first -- see issue
#3. The alias sidesteps that entirely, the same way asp_backend/asp_gui do.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_package(alias: str, src_dir: Path) -> None:
    if alias in sys.modules or not src_dir.is_dir():
        return
    init_file = src_dir / "__init__.py"
    if not init_file.is_file():
        return
    spec = importlib.util.spec_from_file_location(
        alias, init_file, submodule_search_locations=[str(src_dir)]
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)


_load_package("asp_backend", _REPO_ROOT / "backend" / "src")
_load_package("asp_gui", _REPO_ROOT / "gui" / "src")
_load_package("asp_backend_benchmark", _REPO_ROOT / "backend" / "benchmark")
_load_package("asp_backend_evaluation", _REPO_ROOT / "backend" / "benchmark" / "evaluation")




# ----------------------------------------------------------------------
# Anime stitch pipeline helpers
# Shared builders for backend/test/ alignment, core, and rendering tests —
# pure NumPy/OpenCV so tests run without any GPU or model dependency.
# Recovered from Image-Toolkit's backend/test/conftest.py (this repo's
# tests were split out of that repo but the fixture helpers were dropped
# in the split / "S200 great trim").
# ----------------------------------------------------------------------


def make_frame(h: int = 480, w: int = 640, color=(128, 128, 128)) -> np.ndarray:
    """BGR uint8 frame filled with a solid colour."""
    return np.full((h, w, 3), color, dtype=np.uint8)


def make_gradient_frame(h: int = 480, w: int = 640, top=100, bottom=180) -> np.ndarray:
    """BGR frame with a vertical brightness gradient (mimics scene lighting)."""
    grad = np.linspace(top, bottom, h, dtype=np.float32)
    frame = np.stack([grad] * w, axis=1)[:, :, np.newaxis]
    return np.repeat(frame, 3, axis=2).astype(np.uint8)


def make_translation_affine(tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    """Identity 2x3 affine with specified translation."""
    M = np.eye(2, 3, dtype=np.float32)
    M[0, 2] = tx
    M[1, 2] = ty
    return M


def make_rotation_affine(tx: float, ty: float, angle_deg: float = 5.0) -> np.ndarray:
    """2x3 affine with translation AND a small rotation (off-diagonal elements)."""
    theta = np.deg2rad(angle_deg)
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), tx],
            [np.sin(theta), np.cos(theta), ty],
        ],
        dtype=np.float32,
    )


def make_edge(
    i: int,
    j: int,
    dx: float = 0.0,
    dy: float = 300.0,
    n_pts: int = 50,
    weight: float = 1.0,
) -> dict:
    """Synthetic edge dict matching the format produced by the matching stages."""
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    rng = np.random.default_rng(i * 1000 + j)
    pts_i = rng.uniform(50, 400, (n_pts, 2)).astype(np.float32)
    pts_j = pts_i + np.array([dx, dy], dtype=np.float32)
    return {"i": i, "j": j, "M": M, "pts_i": pts_i, "pts_j": pts_j, "weight": weight}


def compute_ty_gaps(affines: list) -> np.ndarray:
    """Extract sorted ty values and return consecutive gaps."""
    tys = sorted(float(a[1, 2]) for a in affines)
    return np.diff(tys)


@pytest.fixture
def single_frame():
    return make_frame(h=200, w=300)


@pytest.fixture
def three_frames():
    return [make_frame(h=200, w=300, color=(c, c, c)) for c in (80, 128, 180)]


@pytest.fixture
def chain_edges_300():
    """Perfect sequential chain: 4 frames, each 300 px below the previous."""
    return [make_edge(i, i + 1, dx=0.0, dy=300.0) for i in range(3)]


@pytest.fixture
def chain_edges_5frames():
    """5-frame sequential chain with dy=250."""
    return [make_edge(i, i + 1, dx=0.0, dy=250.0) for i in range(4)]
