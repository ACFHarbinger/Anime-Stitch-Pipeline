# Development Guide

## Prerequisites

- Git, [`just`](https://github.com/casey/just), `pre-commit`
- Per-module toolchains: `uv` (Python 3.11+), Node 22 (TypeScript), JDK 21 + Gradle wrapper (Kotlin), stable Rust (Cargo), Go 1.22+, CMake + a C++17 compiler

## Local Setup

```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>
cp .env.example .env
pip install pre-commit && pre-commit install
just --list
```

## Running a Single Module

```bash
just test python       # or typescript, kotlin, rust, go, cpp
just lint python
```

## Containerized Dev Environment

Open the repo in VS Code and choose "Reopen in Container" — see [`.devcontainer/devcontainer.json`](../.devcontainer/devcontainer.json). Or run the full stack with:

```bash
docker compose -f docker/docker-compose.yml up
```

## Common Tasks

| Task | Command |
| --- | --- |
| Run all tests | `just test` |
| Run all linters | `just lint` |
| Build docs | `just docs` |
| Build & start Docker stack | `just docker-up` |
