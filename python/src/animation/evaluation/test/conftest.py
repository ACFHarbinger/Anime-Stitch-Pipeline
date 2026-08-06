"""
backend/benchmark/evaluation/test/conftest.py
==============================================
Shared fixtures for the Benchmark Evaluation Inspector test suite.

Qt widget tests run under the ``offscreen`` QPA so they work on headless CI
machines without an X server.  All tests in this directory use these fixtures
automatically (``autouse`` where appropriate).

Run:
    pytest backend/benchmark/evaluation/test/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Force the offscreen Qt platform adapter BEFORE any PySide6 import.
# This allows QWidget geometry/layout tests to run without a display.
# ---------------------------------------------------------------------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ---------------------------------------------------------------------------
# Make sure repo root is importable (mirrors bench_eval_dispatch's bootstrap).
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _THIS_DIR
for _ in range(8):
    if os.path.exists(os.path.join(_REPO_ROOT, "pyproject.toml")):
        break
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Pytest markers used by this sub-suite
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "inspector_ui: tests that construct PySide6 widgets under the offscreen QPA",
    )
    config.addinivalue_line(
        "markers",
        "regression: regression guards for bugs that previously shipped",
    )


# ---------------------------------------------------------------------------
# QApplication — one per session to avoid double-init errors
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Return a session-scoped QApplication running under the offscreen QPA."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    # Do not call app.quit() — let Python's atexit / the test runner handle it.


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------

_W, _H = 800, 1200  # tall image — exercises fit-scale < 1


@pytest.fixture
def solid_image() -> np.ndarray:
    """800x1200 solid grey BGR image."""
    return np.full((_H, _W, 3), 128, dtype=np.uint8)


@pytest.fixture
def wide_image() -> np.ndarray:
    """2000x600 wide BGR image — fit-scale constrained by width."""
    return np.full((600, 2000, 3), 200, dtype=np.uint8)


@pytest.fixture
def gradient_image() -> np.ndarray:
    """800x1200 vertical gradient BGR image (mimics real anime stitch output)."""
    img = np.zeros((_H, _W, 3), dtype=np.uint8)
    col = np.linspace(40, 220, _H, dtype=np.uint8)
    img[:, :, :] = col[:, np.newaxis, np.newaxis]
    return img


@pytest.fixture
def comparator_images(solid_image, wide_image, gradient_image) -> dict:
    """Dict of three named comparator images matching InspectorWindow's schema."""
    return {
        "asp": solid_image,
        "simple": wide_image,
        "gt": gradient_image,
    }
