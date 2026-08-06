#!/usr/bin/env bash
# Bootstrap the dev environment on macOS.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

command -v brew >/dev/null || { echo "Homebrew required: https://brew.sh"; exit 1; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
(cd python && uv sync)
(cd typescript && npm install)

echo "Setup complete. Activate with: source python/.venv/bin/activate"
