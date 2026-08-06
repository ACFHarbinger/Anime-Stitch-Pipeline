# Workflow: Rust Change

1. Sketch the types/error enum for the change before writing logic.
2. Write unit tests alongside the code (`#[cfg(test)]`) as you implement, not after.
3. Run `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, and `cargo test` before committing.
4. For public API changes, check downstream crates/bindings (FFI, `pyo3`, `napi`) that consume them.
5. Add a benchmark under `rust/benches/` if the change is performance-sensitive.
