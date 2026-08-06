# python/validation

Dev-tooling scripts for the Python module: import-graph/circular-import checks, LOC counters, docstring linting, embedded-language detection, type-coverage reporting, and a Sphinx skeleton for API docs.

This directory is a **merge** of the `validation/` and `docs/` subdirectories found under the Python module of each source repository, since in every case those two directories hold dev tooling rather than product code or end-user documentation:

- `*.py` (root of this directory), `html/`, `target/` — from `Image-Toolkit/backend/validation/` (the superset: also includes `constant_checker.py`, `ast_scope_visitor.py`, and `dependency_grapher.py`, which `Build-Optimization/middleware/validation/` and `WSmart-Route/logic/validation/` don't have).
- `docs/add_docstrings_batch.py`, `docs/check_docstrings.py`, `docs/check_google_style.py`, `docs/__init__.py` — from `Image-Toolkit/backend/docs/` (Google-style docstring linting/batch-fixing).
- `docs/Makefile`, `docs/make.bat`, `docs/source/` — from `WSmart-Route/logic/docs/` (a Sphinx build skeleton for the module's own API docs, distinct from the repo-wide `docs/sphinx/` at the repo root).

`nglab/python/docs/` was **not** merged in here — unlike the above, it holds actual markdown project documentation (`ARCHITECTURE.md`, `TUTORIAL.md`, etc.), which belongs alongside the repo-root [`docs/`](../../docs/) rather than in dev tooling.

## Usage

```bash
cd python
uv run python validation/check_circular_imports.py src
uv run python validation/count_loc.py src
uv run python validation/docs/check_docstrings.py src
```
