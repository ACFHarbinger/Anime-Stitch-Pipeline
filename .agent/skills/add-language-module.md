# Skill: Add a New Language Module

Use when a project generated from this template needs a language module beyond the seven provided (python, typescript, kotlin, java, rust, go, cpp).

1. Create `<language>/` at the repo root with, at minimum: the module's dependency manifest, `src/`, `test/`, `benchmark/`, `config/`.
2. Add a matching rules file at `.agent/rules/<language>.md` and reference it from `.agent/AGENTS.md` §2/§5.
3. Add lint/test hooks to `.pre-commit-config.yaml` and a CI job to `.github/workflows/ci.yml`.
4. Add the language's tech badge to `README.md`.
5. Add a `tools/<language>/` directory if the `justfile` needs module-specific recipes.
