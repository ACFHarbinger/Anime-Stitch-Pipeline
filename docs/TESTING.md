# Testing Guide

**Sanctioned pytest invocation** (Image-Toolkit parent interpreter +
`PYTHONPATH` at the parent root). `from backend.src.constants import …`
in ASP source is **deliberate cross-repo sharing** (issue #3), not a
missing package. It does not resolve if you run
`just test::backend` / `uv run pytest` from `submodules/ASP` alone,
because that puts ASP's own `backend/` on `sys.path` first.

```bash
cd /path/to/Image-Toolkit
.venv/bin/python -m pytest submodules/ASP/backend/test -q
# or:
cd submodules/ASP/backend && \
  PYTHONPATH=/path/to/Image-Toolkit \
  /path/to/Image-Toolkit/.venv/bin/python -m pytest test -q
```

Each module owns its own test suite under `<module>/test/` (or `tests/` for
`base/`, the C++ core — see each module's README).

| Module | Framework | Command |
| --- | --- | --- |
| `backend/` | pytest | `uv run pytest test -v --cov=src` (run from `backend/`) — **`uv sync` now works on a standard (non-CUDA) machine** (fixed 2026-08-06, issue #5): the CUDA-only/research matcher plugins (`mamba_ssm`, `ptlflow`, `romatch`, `pycocotools`, `sam-2`) moved from hard `dependencies` to an optional `matchers` extra — every call site already wraps these in `try/except ImportError` (`core/pipeline/_probes.py`'s `_ROMA_OK`/`_SEA_RAFT_OK` flags, etc.) and degrades gracefully, so this was a packaging bug, not a real requirement. Install `uv sync --extra matchers` on a CUDA-capable machine to exercise those matcher paths. **Running the actual test suite still needs more than `uv sync`**: `backend/test/conftest.py`'s absolute imports (`from backend.src.constants import ...`) resolve to Image-Toolkit's own `backend/src/constants/` (deliberate cross-repo constant sharing, same pattern as `gui/`'s coupling — see issue #3), so tests only collect when run via Image-Toolkit's own root interpreter with its root on `PYTHONPATH`: `cd backend && PYTHONPATH=/path/to/Image-Toolkit /path/to/Image-Toolkit/.venv/bin/python -m pytest test -v`. Doing that used to surface two further pre-existing issues; one is fixed: 6 test files' missing fixture helpers (`make_frame`, `make_translation_affine`, etc.) were recovered from Image-Toolkit's own `backend/test/conftest.py` (issue #6, closed 2026-08-06) — 342 tests unblocked. **Still open**: a handful of files (`test/benchmarks/test_bench_dashboard.py`, `test/benchmarks/test_bench_metrics.py`, `test/core/test_pipeline_trace.py`) fail because ASP's and Image-Toolkit's top-level `backend`/`gui` packages share names and Python's namespace-package merging resolves them inconsistently depending on `sys.path` order (issue #3, expanded) — collection error count for `backend/test/` is now 3 (was 9). |
| `gui/` | pytest | `uv run pytest test -v` (run from `gui/`) — **`gui/test/`'s missing `q_app` fixture was fixed 2026-08-07** (added to `gui/test/conftest.py`, matching the `qapp` fixture pattern already used in `backend/benchmark/evaluation/test/conftest.py`): all 50 tests pass when run via Image-Toolkit's shared interpreter, `cd gui && PYTHONPATH=/path/to/Image-Toolkit /path/to/Image-Toolkit/.venv/bin/python -m pytest test -v` (was 17 passed / 33 collection errors). **CI's own invocation still fails at collection** (`continue-on-error: true`, non-blocking) for the same cross-repo coupling reason as `backend/` above (issue #3) — a standalone `uv sync`-only environment has no way to satisfy `backend.src.constants` without either checking out the parent Image-Toolkit repo in CI or reducing the coupling; that's the real remaining blocker, not the fixture gap. |
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
