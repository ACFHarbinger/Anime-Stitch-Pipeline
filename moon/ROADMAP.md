# Dev-Repo-Template Roadmap

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)

> **Version**: 1.0
> **Date**: 2026-07-30
> **Status**: Template

## Overview

This document tracks planned scaffolding work for `Dev-Repo-Template` itself. Once this template seeds a real project, replace this file's contents with that project's actual roadmap — per-module detail then lives in `moon/roadmaps/<module>.md`. Completed items move to [`moon/CHANGELOG.md`](CHANGELOG.md).

Status markers: ✅ Done · 🚧 In Progress · 📋 Pending

---

## Track: Template Scaffolding

| # | Item | Effort | Status |
| --- | --- | --- | --- |
| T1 | Root scaffolding: LICENSE, README, .env.example, git config, pre-commit | S | ✅ Done |
| T2 | `.github/` CI/CD: workflows, issue/PR templates, dependabot | M | ✅ Done |
| T3 | `docs/` documentation portal: MkDocs, Sphinx, Structurizr, ADRs | M | ✅ Done |
| T4 | `moon/` roadmap and changelog | S | 🚧 In Progress |
| T5 | `docker/` infrastructure: Dockerfile, Compose stack | S | 📋 Pending |
| T6 | `.agent/` LLM coding-agent scaffolding | M | ✅ Done |
| T7 | `justfile` + `tools/` command runner | M | 📋 Pending |
| T8 | `desktop/` per-OS packaging scripts | S | 📋 Pending |
| T9 | `.devcontainer/` Dev Container definition | S | 📋 Pending |
| T10 | `env/` Conda/pip environment definitions | S | 📋 Pending |
| T11 | Language module skeletons (python, typescript, kotlin, java, rust, go, cpp) | L | ✅ Done |
| T12 | `python/validation/` merged dev-tooling from source repos | M | 📋 Pending |
| T13 | Root workspace orchestrator files (`pyproject.toml`, `package.json`, `Cargo.toml`, `go.work`, `settings.gradle.kts`) | S | ✅ Done |
| T14 | `dev/` developer assistant tool — see [`moon/roadmaps/developer_tools.md`](roadmaps/developer_tools.md) (template-meta, delete on adoption) | XL | 📋 Pending |

## Track: Post-Template Adoption

> **TODO:** Once a real project is generated from this template, replace this section with that project's actual feature roadmap.

See per-module detail in [`moon/roadmaps/`](roadmaps/).
