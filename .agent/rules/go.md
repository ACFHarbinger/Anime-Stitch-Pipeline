# Go Rules

- Format with `gofmt`/`goimports`; vet with `go vet ./...`. Both must be clean before committing.
- Follow standard project layout: `go/cmd/<binary>/main.go` for entry points, `go/internal/` for private packages.
- Return errors, don't panic, outside of `main` and truly unrecoverable states; wrap errors with `fmt.Errorf("...: %w", err)` to preserve context.
- Keep interfaces small and defined at the consumer, not the implementer.
- Tests live alongside the code as `_test.go` files using the standard `testing` package (plus `testify/assert` where helpful); table-driven tests are the default style.
- Benchmarks use `func BenchmarkX(b *testing.B)` in the same `_test.go` files, run via `go test -bench=.`.
