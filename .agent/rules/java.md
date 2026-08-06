# Java Rules

- Target Java 21, built via Maven (`pom.xml`).
- Keep dependency versions centralized in `<dependencyManagement>` (BOM imports where available), not scattered as hardcoded versions per plugin/dependency.
- Prefer immutable types (`record`, `final` fields) over mutable state; avoid returning `null` where an `Optional<T>` or a checked result type communicates intent better.
- Tests live under `java/src/test/java`, using JUnit 5 (`org.junit.jupiter`); prefer AssertJ-style fluent assertions once a project needs more than `assertEquals`.
- Run `mvn -B test` (or `mvn -B compile` for the pre-commit hook) before committing; keep `maven-compiler-plugin`/`maven-surefire-plugin` versions pinned in `pom.xml`, not left to Maven defaults.
