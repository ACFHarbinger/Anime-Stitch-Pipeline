# Architecture

> **Partial:** module boundaries below reflect the actual current layout.
> Per-module internals (stage breakdowns, data flow) still need a real
> write-up — see `docs/moon/ROADMAP.md` and `.agent/cache/asp_state_of_the_pipeline.md`
> for the most authoritative current description of the ASP pipeline itself
> in the meantime. This page should stay in sync with
> `docs/structurizr/workspace.dsl`.

## Overview

The Anime Stitch Pipeline (ASP) is a desktop application for editing and
stitching scrolling anime/manga capture frames into a single panoramic
image. It has two independent stitching paths: an automated ML pipeline
(`AnimeStitchPipeline`, matching → bundle adjustment → foreground/background
compositing, tuned for animated content) and a manual/interactive tool
(`HybridStitchPanel`) for human-directed control-point, seam, and warp
editing. A C++ core (`base/`) provides the performance-critical hot paths
(matching, bundle adjustment, canvas construction, seam finding, exposure,
compositing) called from the Python orchestration layer (`backend/`) that
also owns ingestion, model wrappers, and the benchmark/evaluation
subsystem. The desktop UI (`gui/`, PySide6) is the one working, shipped
surface today; a cross-platform Tauri frontend and mobile apps are
scaffolded but frozen/removed pending product validation (see
`docs/moon/ROADMAP.md` §0).

## Module Boundaries

| Module | Language | Responsibility |
| --- | --- | --- |
| `base/` | C++ | Performance-critical stitching kernels (matching, bundle adjustment, canvas, seam, exposure, compositing), exposed via pybind11. |
| `backend/` | Python | Pipeline orchestration (`AnimeStitchPipeline`), ingestion, model wrappers, alignment, rendering, and the benchmark/evaluation subsystem. |
| `gui/` | Python (PySide6) | The desktop UI: the automated-pipeline Stitch tab and the manual `HybridStitchPanel` interactive stitching tool. |
| `frontend/` | TypeScript/Rust (Tauri) | Scaffold only, frozen — see `frontend/README.md`. |
| `docs/website/` | Vue | Documentation site rendering `docs/**/*.md`. |

## C4 Diagrams

See [`docs/structurizr/`](structurizr/README.md) for the rendered C4 model (Context → Container → Component).

## Architecture Decision Records

Significant, hard-to-reverse decisions are recorded under [`docs/adr/`](adr/) using the [Michael Nygard ADR format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).
