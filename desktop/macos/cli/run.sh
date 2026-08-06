#!/usr/bin/env bash
# Thin CLI wrapper: run the app from a terminal on macOS.
# Usage: ./run.sh [args...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

(cd python && uv run python -m src.main "$@")
