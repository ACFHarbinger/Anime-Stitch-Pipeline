# dev/ — Developer Assistant Tool

> **Template-meta.** This tool analyzes *this template's own polyglot
> layout* as a running example. Delete this directory (and
> `moon/roadmaps/developer_tools.md`) when adopting the template for a real
> project, unless you specifically want to keep building it. See
> [`moon/roadmaps/developer_tools.md`](../moon/roadmaps/developer_tools.md)
> for the full architecture plan and milestone tracker.

A polyglot code-analysis tool: dependency/import-graph visualization,
circular-dependency detection, and layer-boundary enforcement, with a
modular per-language `input/` layer and a plugin system for repo-specific
extras.

## What's implemented so far (D1–D5)

- `src/input/protobuf/codegraph.proto` — the schema every language parser
  emits (`Node`, `Edge`, `Diagnostic`, `CodeGraph`).
- `src/input/python/parser.py` — a real, AST-based Python import-graph
  parser (no external dependencies).
- `src/core/model.py` — a hand-mirrored Python model of the protobuf schema
  (no `protoc` available in this environment; swap in generated bindings
  later without changing call sites).
- `src/core/aggregate.py` — merges any number of per-language graphs.
- `src/core/layers.py` — layer classification + forbidden-cross-layer-edge
  detection, configured via `config/layers.yaml`.
- `src/core/cycles.py` — circular-dependency detection via Tarjan's SCC.
- `src/output/html/report.py` — a self-contained, zero-install HTML report
  (vis.js from a CDN + Jinja2 template), CI-usable via the `check` CLI verb.
- `src/cli.py` — ties the above together.

## Not yet implemented

Everything else in the roadmap (D6+): the Tauri/React interactive app,
the plugin registry, and real parsers for typescript/kotlin/java/rust/go/cpp
(each `input/<language>/` currently just has a placeholder README). See the
roadmap doc for the full milestone list and linked GitHub issues.

## Usage

```bash
cd dev
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Write a static HTML dependency report for a directory:
python src/cli.py report ../python/src -o report.html

# CI-usable gate — exits 1 on layer violations or circular dependencies:
python src/cli.py check ../python/src

# Run the test suite:
pytest
```

## Layout

| Directory | Purpose |
| --- | --- |
| `src/input/<language>/` | One independently-removable parser per language; only `python/` is implemented |
| `src/input/protobuf/` | The schema contract between parsers and `core/` |
| `src/core/` | Language-agnostic aggregation, layer rules, cycle detection |
| `src/output/html/` | Static report generator (implemented) |
| `src/output/app/` | Tauri/React interactive app (not yet started, D6/D7) |
| `src/plugins/` | Plugin extension points (not yet started, D8) |
| `test/` | pytest suite, mirrors `src/` layout, with a fixture project containing an intentional import cycle |
| `benchmark/` | Manual micro-benchmarks |
| `config/` | `layers.yaml` (layer rules), `plugins.yaml` (plugin registry, not yet consumed) |
