# Project Documentation

Welcome to the documentation for **this project**. Replace this paragraph with a short description of what the project does.

```{toctree}
:maxdepth: 2
:caption: Getting Started
:maxdepth: 3
:caption: API Reference

references/index
```

## 1. Introduction

### Project Overview
- **Source Code**: [logic/src](../../src)
- **Status**: Development (PhD Research)

### Indices and tables
- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

---

## 2. Configure `conf.py` for `autodoc2`

Update your `logic/docs/source/conf.py` with the following settings. This configuration tells Sphinx to scan your source code and render the docstrings as Markdown files inside a virtual `reference/` directory.

```python
import os
import sys

# Ensure Sphinx can find your source code
sys.path.insert(0, os.path.abspath("../../"))

# 1. Required Extensions
extensions = [
    "autodoc2",
    "myst_nb",         # Handles Markdown and Jupyter Notebooks
    "sphinx_design",   # For grids, cards, and tabs
    "sphinx_copybutton",
]

# 2. Autodoc2 Configuration
autodoc2_packages = [
    "../../src",       # Path to your Python package root
]

# Render API as MyST Markdown instead of reStructuredText
autodoc2_render_plugin = "myst"

# The directory where the API reference will be "built"
autodoc2_output_dir = "reference"

# 3. General Settings
master_doc = "index"
html_theme = "sphinx_book_theme"

# Enable specific Markdown features
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "tasklist",
]
```

## 3. Architecture Overview

Below is the automated class hierarchy for the module. Click on any class to jump to its API documentation.

::: {uml}
:classes: ../../src
:members:
:exclude: InternalHelper
:::
