#!/usr/bin/env bash
# Launch the app in dev mode on Linux.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../typescript"

npm run dev
