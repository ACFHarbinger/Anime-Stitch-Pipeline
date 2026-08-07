#!/usr/bin/env python3
"""Standalone launcher for the ASP stitch tab — a minimal QMainWindow
hosting just StitchTab, for isolated testing/demo without the full
Image-Toolkit app. Registers asp_backend/asp_gui the same way
Image-Toolkit's own _submodule_bootstrap.py does.

StitchTab's elements/ modules import Image-Toolkit's own gui.src.constants/
styles/utils/windows (shared UI constants, not vendored here — same
cross-repo coupling documented in base/CMakeLists.txt and cpp/README.md),
so this only runs when checked out at
Image-Toolkit/submodules/Anime-Stitch-Pipeline/, same as `just test::gui`.

Usage:
    cd gui && uv run python launch.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_IMAGE_TOOLKIT_ROOT = _REPO_ROOT.parent.parent


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


def main() -> int:
    if str(_IMAGE_TOOLKIT_ROOT) not in sys.path:
        sys.path.insert(0, str(_IMAGE_TOOLKIT_ROOT))
    _load_package("asp_backend", _REPO_ROOT / "backend" / "src")
    _load_package("asp_gui", _REPO_ROOT / "gui" / "src")

    from asp_gui.elements.manager import StitchTab
    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Anime-Stitch-Pipeline")
    window.setCentralWidget(StitchTab())
    window.resize(1400, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
