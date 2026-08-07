"""Shared pytest fixtures for gui/test/.

Registers this repo's own backend/src and gui/src packages under the same
asp_backend / asp_gui aliases Image-Toolkit's own bootstrap uses (see
Image-Toolkit's _submodule_bootstrap.py), so tests here and tests run via
Image-Toolkit import identically regardless of which repo mounted them.

Also provides the `q_app` fixture every test under test/tabs/ and
test/dialogs/ expects: when run via Image-Toolkit's own test suite this
comes from its shared root conftest.py, but this repo's own conftest.py
never defined it standalone, so `gui/test/` could never collect on its own
(see docs/TESTING.md and issue #3) -- every test needing q_app failed at
setup with "fixture 'q_app' not found", not a real test failure.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _load_package(alias: str, src_dir: Path) -> None:
    if alias in sys.modules or not src_dir.is_dir():
        return
    spec = importlib.util.spec_from_file_location(
        alias, src_dir / "__init__.py", submodule_search_locations=[str(src_dir)]
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)


_load_package("asp_backend", _REPO_ROOT / "backend" / "src")
_load_package("asp_gui", _REPO_ROOT / "gui" / "src")


@pytest.fixture(scope="session")
def q_app():
    """Session-scoped QApplication, matching the offscreen-QPA convention
    already used by backend/benchmark/evaluation/test/conftest.py's `qapp`
    fixture. Reuses any already-running QApplication.instance() rather than
    constructing a second one, which Qt does not allow."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    # Do not call app.quit() -- let Python's atexit / the test runner handle it.
