"""Sphinx configuration for the Python module's API reference.

Run: sphinx-build -b html docs/sphinx site/sphinx-api
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── sys.path for autodoc imports ─────────────────────────────────────────────
REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ── Project metadata ─────────────────────────────────────────────────────────
project = "Dev-Repo-Template — Python Reference"
author = "ACFHarbinger"
release = "0.1.0"
html_title = "Python API Reference"
copyright = "2026, ACFHarbinger"

# ── Extensions ───────────────────────────────────────────────────────────────
extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

# ── sphinx-autoapi ────────────────────────────────────────────────────────────
autoapi_dirs = [str(REPO_ROOT / "python" / "src")]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_ignore = ["**/test_*.py", "**/__pycache__/**", "**/conftest.py"]
autoapi_add_toctree_entry = True
autoapi_keep_files = False

# ── Napoleon ─────────────────────────────────────────────────────────────────
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_returns = True

# ── Intersphinx ──────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

# ── HTML (Furo theme) ────────────────────────────────────────────────────────
html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]

# ── Build behaviour ───────────────────────────────────────────────────────────
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nitpicky = False
