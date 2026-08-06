# java/

Java module template, built with Maven. Conventions adapted from a real Maven
multi-module reactor project, simplified to a single-module skeleton to match
this template's other language modules.

```bash
mvn compile
mvn test
mvn package
```

| Directory | Purpose |
| --- | --- |
| `src/main/java/` | Application source |
| `src/test/java/` | JUnit 5 tests |
| `benchmark/` | Manual micro-benchmarks (run directly with `java`, no harness dependency) |
| `config/` | Runtime configuration |
