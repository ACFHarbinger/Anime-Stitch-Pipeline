# python/

Python module template. Managed with [uv](https://github.com/astral-sh/uv).

```bash
uv sync --all-extras --dev
uv run pytest test -v
uv run ruff check .
```

| Directory | Purpose |
| --- | --- |
| `src/` | Application/library source |
| `test/` | Tests (pytest), with shared fixtures in `conftest.py` |
| `benchmark/` | Performance benchmarks (pytest-benchmark) |
| `config/` | Runtime/experiment configuration |
| `validation/` | Dev-tooling scripts (import graph, docstring, LOC checks) — see [`validation/README.md`](validation/README.md) |
