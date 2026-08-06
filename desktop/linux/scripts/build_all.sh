#!/usr/bin/env bash
# Build every language module on Linux.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

(cd rust && cargo build --release)
(cd go && go build ./...)
(cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/build)
(cd kotlin && ./gradlew build)
(cd typescript && npm run build)
