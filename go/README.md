# go/

Go module template.

```bash
go build ./...
go test ./...
go test -bench=. ./...
gofmt -l . && go vet ./...
```

| Directory | Purpose |
| --- | --- |
| `cmd/app/` | Binary entry point |
| `internal/` | Private packages (tests live alongside as `_test.go` files) |
| `config/` | Runtime configuration |
