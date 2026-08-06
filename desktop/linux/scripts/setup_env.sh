#!/usr/bin/env bash
# Bootstrap the dev environment on Linux.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
(cd python && uv sync)
(cd typescript && npm install)

echo "Setup complete. Activate with: source python/.venv/bin/activate"
