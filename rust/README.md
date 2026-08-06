# rust/

Rust module template.

```bash
cargo build
cargo test
cargo bench
cargo clippy --all-targets -- -D warnings
```

| Directory | Purpose |
| --- | --- |
| `src/` | Library/binary source (unit tests live in `#[cfg(test)]` modules) |
| `tests/` | Integration tests |
| `benches/` | Criterion benchmarks |
| `config/` | Runtime configuration |
