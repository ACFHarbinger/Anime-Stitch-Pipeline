# Anime-Stitch-Pipeline Task Automation — Root Justfile
# https://github.com/casey/just
#
# Recipes are organised into per-domain sub-modules under tools/. Invoke a
# sub-module recipe directly (e.g. `just build::rust`, `just test::python`),
# or use the root shorthands below.

set shell := ["bash", "-c"]
set unstable := true

# --- Sub-module declarations (imported from tools/) ---

mod helper     "tools/helper/justfile"
mod dev        "tools/dev/justfile"
mod build      "tools/build/justfile"
mod test       "tools/test/justfile"
mod validation "tools/validation/justfile"
mod docs       "tools/docs/justfile"
mod bench      "tools/bench/justfile"
mod ci         "tools/ci/justfile"

# --- Default target ---

default: help

# List all commands across every sub-module
help:
    @just helper::help

# --- Setup & maintenance (→ tools/dev) ---

# Set up the full development environment
setup:
    @just dev::setup

# Update all dependencies
update:
    @just dev::update

# Run pre-commit hooks
pre-commit:
    @just dev::pre-commit

# Clean build artifacts
clean:
    @just dev::clean

# --- Build (→ tools/build) ---
#
# Named build-all/test-all/docs-build/bench-all below, not build/test/docs/
# bench: `just` 1.46 forbids a recipe sharing a name with a declared `mod`,
# and the `mod` names above are what make `just build::rust`-style direct
# submodule access work (fixed 2026-08-06, issue #9 -- this collision made
# every `just` command in this repo a hard parse error until now).

# Build every language module
build-all:
    @just build::all

# --- Test (→ tools/test) ---

# Run every language module's test suite
test-all:
    @just test::all

# --- Validation (→ tools/validation) ---

# Run every language module's linters
lint:
    @just validation::all

# --- Docs (→ tools/docs) ---

# Build the documentation site (MkDocs + Sphinx)
docs-build:
    @just docs::build

# --- Benchmark (→ tools/bench) ---

bench-all:
    @just bench::all
