# Kotlin Rules

- Target Kotlin 2.0 / JVM 21, built via Gradle Kotlin DSL (`build.gradle.kts`).
- Format/lint with `ktlint` (`./gradlew ktlintCheck`, `./gradlew ktlintFormat`).
- Prefer immutable data classes and `val` over `var`; avoid platform types leaking from Java interop without an explicit null-check.
- Use coroutines (`kotlinx.coroutines`) for async work instead of raw threads or callbacks.
- Tests live under `kotlin/src/test/kotlin`, using JUnit 5 + Kotest assertions.
- Keep Gradle build logic in `buildSrc`/version catalogs (`gradle/libs.versions.toml`), not hardcoded versions scattered across modules.
