# Testing Guide

Each module owns its own test suite under `<module>/test/` (or `tests/` for
`base/`, the C++ core — see each module's README).

| Module | Framework | Command |
| --- | --- | --- |
| `backend/` | pytest | `uv run pytest test -v --cov=src` (run from `backend/`) — **`uv sync` currently fails on a standard (non-CUDA) machine**: `mamba_ssm` (a matcher-plugin dependency) compiles a CUDA extension from source and needs `nvcc` at build time; this almost certainly also fails on GitHub's standard `ubuntu-latest` CI runners, which have no CUDA toolchain. Needs either a GPU-enabled CI runner, a prebuilt `mamba_ssm` wheel, or moving the CUDA-only matcher plugins (`mamba_ssm`, and possibly `ptlflow`/`romatch`/`sam-2`, already commented as "lazily imported per-matcher" in `backend/pyproject.toml`) into their own optional extra so a base `uv sync` doesn't require them. |
| `gui/` | pytest | `uv run pytest test -v` (run from `gui/`) — runs in CI (`lint-test-gui`) but **currently fails at collection** (`continue-on-error: true`, non-blocking): `gui/test/conftest.py` loads `backend/src` and its absolute `from backend.src... import` requires the repo root on `sys.path`, which a standalone `uv sync` in `gui/` doesn't provide. Needs either the repo root added to `PYTHONPATH` for this CI step or a rework of `conftest.py`'s dual import path. |
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
