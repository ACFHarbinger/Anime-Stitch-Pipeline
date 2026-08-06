"""
Sphinx documentation configuration file.
"""

import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath("../../"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Dev-Repo-Template"
copyright = "2026, ACFHarbinger"
author = "ACFHarbinger"
release = "1.1.0"

sys.path.insert(0, os.path.abspath("../../.."))  # project root
sys.path.insert(0, os.path.abspath("../.."))  # logic directory

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autodoc2",  # Enhanced autodoc
    "myst_nb",  # Replaces myst_parser
    "sphinx_design",  # Cards, grids, and tabs
    "sphinx_copybutton",  # Copy buttons for code fences
    "sphinxcontrib.mermaid",
    "hoverxref.extension",
    "sphinx.ext.autodoc",  # Keep your existing API tools
    "sphinx.ext.napoleon",
    "sphinx_uml",
]

# The directory containing your python packages
autodoc2_packages = [
    "../../src",
]

# Optional: Where the generated files should go (defaults to 'autodoc2')
autodoc2_output_dir = "references"

autodoc2_render_plugin = "myst"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

# Enable specific MyST features
myst_enable_extensions = [
    "colon_fence",  # Use ::: for directives
    "dollarmath",  # LaTeX math
    "tasklist",  # - [ ] checkboxes
]
