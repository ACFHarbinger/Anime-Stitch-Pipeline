# Testing Guide

Each language module owns its own test suite under `<module>/test/` (or `tests/`/`benches/` where the ecosystem convention differs — see each module's README).

| Module | Framework | Command |
| --- | --- | --- |
| `python/` | pytest | `uv run pytest test -v --cov=src` |
| `typescript/` | Vitest + Testing Library | `npm test` |
| `kotlin/` | JUnit 5 + Kotest | `./gradlew test` |
| `rust/` | built-in `cargo test` | `cargo test` |
| `go/` | built-in `testing` | `go test ./...` |
| `cpp/` | GoogleTest via CTest | `ctest --test-dir build --output-on-failure` |

## Coverage

CI uploads coverage to [Codecov](https://codecov.io/); thresholds are configured in [`git/codecov.yaml`](../git/codecov.yaml).

## Writing Tests

See [`.agent/rules/test_writing.md`](../.agent/rules/test_writing.md) and [`.agent/workflows/test_writing.md`](../.agent/workflows/test_writing.md).
