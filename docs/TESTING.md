# Testing Guide

Each module owns its own test suite under `<module>/test/` (or `tests/` for
`base/`, the C++ core — see each module's README).

| Module | Framework | Command |
| --- | --- | --- |
| `backend/` | pytest | `uv run pytest test -v --cov=src` (run from `backend/`) — **`uv sync` now works on a standard (non-CUDA) machine** (fixed 2026-08-06, issue #5): the CUDA-only/research matcher plugins (`mamba_ssm`, `ptlflow`, `romatch`, `pycocotools`, `sam-2`) moved from hard `dependencies` to an optional `matchers` extra — every call site already wraps these in `try/except ImportError` (`core/pipeline/_probes.py`'s `_ROMA_OK`/`_SEA_RAFT_OK` flags, etc.) and degrades gracefully, so this was a packaging bug, not a real requirement. Install `uv sync --extra matchers` on a CUDA-capable machine to exercise those matcher paths. **Running the actual test suite still needs more than `uv sync`**: `backend/test/conftest.py`'s absolute imports (`from backend.src.constants import ...`) resolve to Image-Toolkit's own `backend/src/constants/` (deliberate cross-repo constant sharing, same pattern as `gui/`'s coupling — see issue #3), so tests only collect when run via Image-Toolkit's own root interpreter with its root on `PYTHONPATH`: `cd backend && PYTHONPATH=/path/to/Image-Toolkit /path/to/Image-Toolkit/.venv/bin/python -m pytest test -v`. Doing that surfaces two further **pre-existing, independent** issues, both filed rather than fixed here: 6 test files import test-fixture helpers (`make_frame`, `make_translation_affine`, etc.) that don't exist anywhere in the repo (issue #6), and a handful of others fail because ASP's and Image-Toolkit's top-level `backend`/`gui` packages share names and Python's namespace-package merging resolves them inconsistently depending on `sys.path` order (issue #3, expanded). |
| `gui/` | pytest | `uv run pytest test -v` (run from `gui/`) — runs in CI (`lint-test-gui`) but **currently fails at collection** (`continue-on-error: true`, non-blocking) for the same cross-repo coupling reason as `backend/` above (issue #3). |
| `base/` | GoogleTest via CTest | `ctest --test-dir build --output-on-failure` (after `cmake -S base -B base/build -DBUILD_TESTING=ON && cmake --build base/build`), or `just test-base-cpp` |

ASP-specific benchmark/regression suites (not unit tests) live under
`backend/benchmark/` — see `docs/moon/ROADMAP.md`'s Ground Rules for
`just asp-benchmark` / `just asp-benchmark-verify` and the one-change-one-
benchmark discipline that governs any change to `backend/src/animation/` or
`base/src/`.

## Coverage

CI uploads coverage to [Codecov](https://codecov.io/); thresholds are configured in [`git/codecov.yaml`](../git/codecov.yaml).

## Writing Tests

See [`.agent/rules/test_writing.md`](../.agent/rules/test_writing.md) and [`.agent/workflows/test_writing.md`](../.agent/workflows/test_writing.md).
