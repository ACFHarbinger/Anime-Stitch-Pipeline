# Rust Rules

- Format with `cargo fmt`; lint with `cargo clippy --all-targets -- -D warnings`. Both must pass before committing.
- Prefer `Result<T, E>` with a project-specific error enum (e.g. via `thiserror`) over `unwrap()`/`panic!()` outside of tests and `main`.
- Avoid `unsafe` unless crossing an FFI boundary; document every `unsafe` block with a `// SAFETY:` comment explaining the invariant.
- Tests live under `rust/tests/` for integration tests and `#[cfg(test)]` modules next to the code for unit tests.
- Benchmarks live under `rust/benches/`, run via `cargo bench` (criterion).
- Keep the public API surface (`pub`) minimal; default to private and widen visibility only when a consumer needs it.
