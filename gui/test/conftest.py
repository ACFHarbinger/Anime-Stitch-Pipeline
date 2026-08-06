"""Shared pytest fixtures for gui/test/.

Registers this repo's own backend/src and gui/src packages under the same
asp_backend / asp_gui aliases Image-Toolkit's own bootstrap uses (see
Image-Toolkit's _submodule_bootstrap.py), so tests here and tests run via
Image-Toolkit import identically regardless of which repo mounted them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


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
