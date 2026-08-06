#!/usr/bin/env bash
# Thin CLI wrapper: run the app from a terminal / .desktop launcher.
# Usage: ./run.sh [args...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

(cd python && uv run python -m src.main "$@")
