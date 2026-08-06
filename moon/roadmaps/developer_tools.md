# Developer Assistant Tool — Roadmap

> **⚠️ Template-meta content — delete on adoption.** This roadmap plans a tool for *this template repository's own tooling ecosystem*, not for a product built from the template. When you use **"Use this template"** to start a real project, delete this file (and, once built, the `dev/` directory it describes) unless your new project specifically wants to keep maintaining it. See the note in [`README.md`](../../README.md#developer-assistant-tool).

> **Version**: 0.2
> **Date**: 2026-07-31
> **Status**: 🚧 In Progress — `dev/` scaffolded, D1–D5 implemented and tested (13 passing pytest cases)

Status markers: ✅ Done · 🚧 In Progress · 📋 Pending

---

## 1. Goal

Build a **developer assistant tool** — living at `dev/` in the repo root — that helps a developer analyze *any* repository seeded from this template: visualize its dependency/import graph, detect circular imports and layering violations, trace a symbol's definition/usage, and surface other cross-cutting code-health signals. It must be:

1. **Polyglot** — able to analyze Python, TypeScript, Kotlin, Rust, Go, and C++ sources, matching the six language modules this template ships.
2. **Modular by removal** — a project that drops the `typescript/` module should be able to delete `dev/src/input/typescript/` and everything keeps working; nothing in the core should hard-depend on every language being present.
3. **Modular by extension** — a project with unusual needs (e.g. an image/video-heavy pipeline) should be able to drop in a **plugin** that adds analysis logic and/or UI without touching core code.
4. **Dual-output** — always able to produce a static, zero-install HTML report (headless/CI-friendly) *and* a full interactive desktop app for deep exploration. Both are first-class, not "pick one."

## 2. Prior art surveyed

This plan is a synthesis of tooling that already exists, independently, across the five source repos this template was built from. Nothing here is invented from scratch — it's consolidation + gap-filling.

| Source | What it contributes |
| --- | --- |
| `Build-Optimization` (`middleware/validation/visualize_module_graph.py`, `trace_dependencies.py`) | Python `ast`-based import-graph builder, layer classification (`DEFAULT_LAYERS`/`FORBIDDEN_DIRECTIONS`), `pyvis`/vis.js rendering, Jinja2 UML node/edge panel injected into the generated HTML (`_inject_uml_panel`) |
| `WSmart-Route` (`logic/validation/*`) | Same tool family, plus pattern-externalization (`target/pattern.{html,css,js,sql}` loaded at runtime rather than hardcoded regex) and a second, independent Sphinx-based UML generator (`logic/docs`) |
| `Image-Toolkit` (`backend/src/utils/validation/check_circular_imports.py`, `backend/benchmark/evaluation/`) | Working Tarjan's-SCC circular-dependency detection wired to `just check-circular-imports`, `import-linter` contracts enforcing layer boundaries in CI, **and** the strongest architectural reference: the `evaluation/` inspector app's `other/`→`logic/`→`ui/`→`plugin/` layering (data layer has zero UI dependency), lazy Qt import via module `__getattr__` (headless mode never pays for the UI import), and the `ToolTabBase` plugin-tab pattern (register `(name, callable)`, base class handles list/cache/render plumbing) |
| `Visual-Graph-Programming` (`moon/ROADMAP.md` Milestones A/B/D) | The scaling plan for graph rendering: React Flow for small/explicit graphs, **Cosmograph (cosmos.gl, WebGL2)** for 1M+-node graphs paired with DuckDB-WASM filtering, **sigma.js** as a mid-scale fallback, semantic zoom (module → file → class/function), tree-sitter-based code-property-graph extraction, and a config-driven automation-rules pattern (`github/config/automation_rules.yaml`) worth reusing for plugin/rule config |
| `nglab` (`tools/*/justfile`, `moon/roadmaps/code_quality.md`) | The "thin orchestration layer over per-language external tools" pattern already used throughout this template's own `justfile` + `tools/`, and a Developer Experience wishlist (recommended editor settings, `QUICKSTART.md`, extending pre-commit) worth folding into `dev/`'s own onboarding |

## 3. Architecture

```
dev/
├── README.md                  # what this tool is, how to run it, how to remove a module
├── src/
│   ├── input/                 # one independently-removable submodule per language
│   │   ├── python/            # ast-based parser — extends python/validation/'s existing scripts
│   │   ├── typescript/        # TS Compiler API / ts-morph based parser
│   │   ├── kotlin/            # Kotlin compiler (embeddable) / PSI based parser
│   │   ├── rust/               # syn / rust-analyzer based parser
│   │   ├── go/                # go/ast + go/packages based parser
│   │   ├── cpp/                # libclang / clang-tooling based parser
│   │   └── protobuf/           # shared .proto schema — the ONLY contract between input/<lang> and core
│   ├── core/                  # aggregator: collects CodeGraph messages from whichever input/<lang>
│   │                          #   modules are present, merges into one graph, applies layer rules,
│   │                          #   runs Tarjan's SCC for circular-dependency detection
│   ├── output/
│   │   ├── html/               # static pyvis/vis.js-style report generator (default, CI/headless)
│   │   └── app/                 # Tauri + React + React Flow (small graphs) / Cosmograph & sigma.js
│   │                            #   (large graphs) desktop app
│   └── plugins/                # optional feature/UI extensions (see §5)
├── test/                      # per-submodule tests (mirrors src/ layout)
├── benchmark/                 # perf harness for core aggregation + large-graph rendering
└── config/                    # layer definitions, forbidden-direction rules, plugin registry
                                #   (per-consuming-repo overridable, e.g. dev/config/layers.yaml)
```

### Why a `protobuf/` schema, not a shared library

Each `input/<language>` submodule is written in — and uses the native tooling of — its own language, because some analyses are only practical in-language (e.g. resolving Rust trait bounds needs `syn`/`rust-analyzer`, not a generic AST walk). A shared Rust or Python *library* would force every parser into one runtime. Instead, `input/protobuf/` defines the message contract (`CodeGraph`, `Node`, `Edge`, `Diagnostic`, `ParseRequest`/`ParseResponse`) that every language parser emits and `core/` consumes — each parser can run as its own CLI/subprocess and speak nothing but that schema. Deleting `input/go/` because a project has no Go code costs `core/` nothing beyond one less source of `CodeGraph` messages.

### Why two outputs, not one

`output/html/` is the always-available baseline: no install beyond the CLI itself, works in CI (exit code reflects violation count, same convention as the source repos' scripts), and is what `just dev report` produces by default. `output/app/` is the opt-in-per-session interactive experience for actually exploring a large graph — React Flow when the graph is small enough to lay out explicitly, Cosmograph/sigma.js when it isn't. A consuming project that never builds a desktop app can delete `output/app/` and keep `output/html/` working unchanged, since they don't share code, only the same upstream `CodeGraph` data.

## 4. Modularity rules

1. **`input/<language>/` is deletable.** `core/` discovers available parsers at runtime (or via `dev/config/`) and degrades gracefully — a repo with no `cpp/` module just never gets C++ nodes in the graph.
2. **`output/app/` is deletable.** `output/html/` has no dependency on it; removing the Tauri app loses interactivity, not correctness.
3. **`plugins/` are additive-only.** A plugin registers into `core/`'s tab/rule registry (see §5); core code never imports a specific plugin by name.
4. Ports the lazy-import discipline from Image-Toolkit's `evaluation/__init__.py` (`__getattr__`-based deferred import): importing `dev/src/core` for a headless CLI run must never pull in Tauri/React/Qt-equivalent UI dependencies.

## 5. Plugin architecture

Adapts Image-Toolkit's `ToolTabBase` pattern: a plugin registers `(name, callable)` pairs into a shared registry; the base class supplies the list UI, result caching, and render plumbing so a plugin author only writes the analysis + a render function. `dev/config/plugins.yaml` enables/disables plugins per consuming repo — this is where a project would, e.g., register an image/video diff panel (a stripped, generic-ized version of Image-Toolkit's `ImagePanel`/pixel-probe tooling) without that code shipping in the core template at all.

## 6. Milestones

Tracked live on [Project 16 — "Developer Assistant Application"](https://github.com/users/ACFHarbinger/projects/16/); each row links to its GitHub issue.

| # | Item | Effort | Status | Issue |
| --- | --- | --- | --- | --- |
| D1 | Define `input/protobuf/` schema: `CodeGraph`, `Node`, `Edge`, `Diagnostic`, `ParseRequest`/`ParseResponse` | M | ✅ Done | [#18](https://github.com/ACFHarbinger/Dev-Repo-Template/issues/18) |
| D2 | Scaffold `dev/` directory tree, per-language `input/` stubs emitting a minimal node list | M | ✅ Done | [#19](https://github.com/ACFHarbinger/Dev-Repo-Template/issues/19) |
| D3 | `core/`: multi-source graph merge, layer classification + forbidden-direction violations (ported from Build-Optimization/WSmart-Route `DEFAULT_LAYERS`/`FORBIDDEN_DIRECTIONS`) | M | ✅ Done | [#20](https://github.com/ACFHarbinger/Dev-Repo-Template/issues/20) |
| D4 | `core/`: circular-dependency detection via Tarjan's SCC (ported from Image-Toolkit `check_circular_imports.py`) | S | ✅ Done | [#21](https://github.com/ACFHarbinger/Dev-Repo-Template/issues/21) |
| D5 | `output/html/`: static pyvis/vis.js-style report + Jinja2 UML node/edge panels (ported from Build-Optimization/WSmart-Route `html/` templates), CI-usable exit codes | M | ✅ Done | [#22](https://github.com/ACFHarbinger/Dev-Repo-Template/issues/22) |
| D6 | `output/app/`: Tauri + React shell, React Flow rendering for small graphs | L | 📋 Pending | [#23](https://github.com/ACFHarbinger/Dev-Repo-Template/issues/23) |
| D7 | `output/app/`: Cosmograph (cosmos.gl) large-graph rendering + sigma.js fallback, semantic zoom levels (module → file → class/function), per Visual-Graph-Programming's roadmap | L | 📋 Pending | [#24](https://github.com/ACFHarbinger/Dev-Repo-Template/issues/24) |
| D8 | Plugin registry (`ToolTabBase`-equivalent) + lazy-import discipline for headless mode | M | 📋 Pending | [#25](https://github.com/ACFHarbinger/Dev-Repo-Template/issues/25) |
| D9 | `input/python/`, `input/typescript/`, `input/kotlin/`, `input/rust/`, `input/go/`, `input/cpp/`: real per-language parsers (start with import/dependency edges only; call graphs and dataflow are stretch) | XL | 📋 Pending | [#26](https://github.com/ACFHarbinger/Dev-Repo-Template/issues/26) |
| D10 | Reference plugin: generic-ized image/video diff panel ported from Image-Toolkit's `evaluation/` inspector, as a worked example of a repo-specific extension | M | 📋 Pending | [#27](https://github.com/ACFHarbinger/Dev-Repo-Template/issues/27) |
| D11 | Wire into root `justfile` (`just dev report`, `just dev app`, `just dev check` for CI) and `.github/workflows/` | S | 📋 Pending | [#28](https://github.com/ACFHarbinger/Dev-Repo-Template/issues/28) |
| D12 | `dev/README.md` onboarding + editor settings recommendation (from nglab's `code_quality.md` Developer Experience wishlist) | S | 📋 Pending | [#29](https://github.com/ACFHarbinger/Dev-Repo-Template/issues/29) |

## 7. Open questions for implementation time

- Exact IPC transport for `input/<lang>` → `core/` (subprocess + stdin/stdout protobuf framing vs. a local gRPC/UDS server) — deferred to D1/D2.
- Whether `core/` itself is Rust, Python, or Go — no language dependency is implied by this roadmap; pick based on `input/protobuf` tooling maturity at implementation time.
- Versioning/back-compat policy for the `input/protobuf/` schema once multiple language parsers depend on it.
