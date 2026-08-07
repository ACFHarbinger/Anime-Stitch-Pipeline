# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Continuity note (2026-08-06):** `docs/moon/ROADMAP.md` cites specific
> session numbers (e.g. "S214", "S216") as being documented "in
> `docs/moon/CHANGELOG.md`". That detailed, session-by-session history
> (~S1–S266+) was kept in the original standalone `Anime-Stitch-Pipeline`
> repository before it was imported into this repo's current
> template-generated layout; it was never migrated into this file, which
> otherwise only contains that template's own scaffolding history. Those
> roadmap citations
> are dead pending an explicit import of that history — treat any
> `docs/moon/CHANGELOG.md SNNN` reference in the roadmap as pointing at
> content that does not currently exist in this repo, not as a citation you
> can follow.

## [Unreleased]

### Added

- Measured `ASP_POSE_WINDOW_PX=80` (DINOv2 pose-consistent frame selection) via a real GPU A/B: 3/5 real composites (was 2/5), verdict counts 1 asp_better / 1 comparable / 3 simple_better (was 0/3/2) — the first `asp_better` verdict produced by any flag this session. GT-SSIM slightly worse on average (0.717 vs 0.7324) but sharpness/ghosting both improve. Genuinely mixed, not a clean win — flagged for a human coherence look, not flipped to default-on.
- Triaged `backend/`'s mypy errors (issue #7, 137 real current count — the earlier 345 figure predated the evaluation-dir move): fixed 21 verified-safe type-annotation errors (no logic changes, full test suite identical before/after). Categorized the rest: ~83 need case-by-case judgment (mostly mechanical, not bugs), 33 need an architectural call (mixin-composition attribute visibility, cv2 stub gaps). Found one real bug in the process (not a type-checker false positive) — `_pass2_pose_refine`'s internal `_pose_dist` helper always gets `None` DINOv2 features from a wrong-argument-type call; the function's primary scoring path is unaffected and confirmed working via a real benchmark run. Filed as issue #12.
- Applied safe `ruff check . --fix` auto-fixes across `backend/` (typing-syntax modernization, import sorting — no logic changes). Verified: full test suite gives the exact same 908 passed / 1 failed (pre-existing) / 3 skipped before and after. Error count 2264 → 280. Part of issue #7; mypy's 345 errors remain untouched.
- Fixed the SAM-2 mask/frame dimension mismatch that crashed 2/5 verify-subset tests (issue #11): `_compute_fg_masks_sam2_stateful` resized every mask against a single shared reference from `frames[0]` instead of each frame's own shape, which silently mismatched whenever per-frame heights differed by a few px (normal — `_normalise_widths` only normalises width). Also fixed an adjacent `cv2.resize()` call passing `cv2.INTER_LINEAR` as the wrong positional argument. Verified via a real GPU run: all 5/5 verify-subset tests now complete cleanly (was 3/5). Real result: GT-SSIM 0.7197 vs BiRefNet-only baseline's 0.7324 (worse), sharpness/ghosting both improve — a genuine mixed result. `ASP_USE_SAM2` stays default OFF (no human rating pass exists for either config yet), but is honestly measurable now. Closes issue #11.
- Fixed `bench_anime_stitch.py` to actually route through `AnimeStitchPipeline`'s `_USE_SAM2`-aware masking (was calling the raw BiRefNet-only function directly, see below). With that fixed and the missing SAM-2 checkpoint downloaded, found a real pipeline bug: `ASP_USE_SAM2=1` crashes 2/5 verify-subset tests with a mask/frame dimension mismatch inside `_compute_fg_masks_sam2_stateful`. Flag stays default OFF. Filed as issue #11.
- Found (not fixed): `ASP_USE_SAM2` has zero effect on `bench_anime_stitch.py`'s scored output — the benchmark script calls the raw non-SAM2-aware `_compute_fg_masks()` directly and constructs `AnimeStitchPipeline` with every stage toggle disabled, bypassing the pipeline's own env-flag-aware masking entirely. Confirmed via a real A/B run producing byte-identical results with the flag on vs off. Filed as issue #10 — may affect other masking-adjacent flags too, not independently audited.
- Ran a real 5-test GPU verify benchmark (`anime_stitch_20260807_002510.json`, RTX 3090 Ti, real corpus) for the first time since this session's packaging fixes — confirms no regression in pipeline behavior (results land at the post-trim baseline: GT-SSIM 0.7324 vs 0.7423, 0/3/2 asp_better/comparable/simple_better). Not a Phase-0.1 human rating pass.
- Fixed `just` failing to parse at all in this repo: the root `justfile` declared `mod build`/`mod test`/`mod docs`/`mod bench` submodules *and* top-level recipes with the same names, which `just` 1.46 rejects outright. Renamed the wrapper recipes to `build-all`/`test-all`/`docs-build`/`bench-all`. Wired up `tools/bench/justfile`'s `asp-benchmark`, `asp-benchmark-verify`, `asp-benchmark-assess`, `asp-triage`, `asp-triage-db`, `asp-run-overmix`, `asp-run-hugin`, `test-base-cpp` — referenced throughout this roadmap for two months but never actually implemented. Fixed `eval_dispatch.py`, `bench_anime_stitch.py`, `run_hugin.py`, and `run_overmix.py` to self-bootstrap their `asp_backend` alias (previously only worked when invoked from within another bootstrap); verified all four run cleanly end-to-end via `--help` (`triage_fallback_classes.py` needed only the docstring fix — it doesn't import `asp_backend`). Closes issue #9.
- Moved `backend/src/evaluation/` back to `backend/benchmark/evaluation/`, matching its pre-ASP-extraction layout in Image-Toolkit. To avoid reintroducing the issue #3 namespace collision (ASP's and Image-Toolkit's top-level `backend` packages share a name), registered it under a new `asp_backend_evaluation` alias in `backend/test/conftest.py` rather than using raw `backend.benchmark.evaluation.X` imports — verified this preserves all 249 previously-passing `backend/test/evaluation/` tests (a naive literal-path version of this move silently broke them). Closes issue #8.
- Restored `backend/test/conftest.py`'s missing fixture helpers (`make_frame`, `make_translation_affine`, `make_rotation_affine`, `make_edge`, `compute_ty_gaps` + 4 fixtures), recovered verbatim from Image-Toolkit's own `backend/test/conftest.py` (where this repo's ASP test suite was originally split from). Unblocks 342 previously-uncollectible tests across 6 files. Closes issue #6.
- Fixed `backend/pyproject.toml` requiring a CUDA toolchain (`nvcc`) just to run `uv sync`: moved `mamba_ssm`/`ptlflow`/`romatch`/`pycocotools`/`sam-2` (already-lazy-imported research matcher plugins) into an optional `matchers` extra. Closes issue #5.
- Phase 6.1 (Tutorials & Onboarding): `gui/src/tabs/stencil/onboarding_wizard.py::HybridStitchOnboardingWizard` — a first-run, dismissible, re-invocable `QWizard` guided tour over `HybridStitchPanel`, paging through pages that switch the panel's live tool tabs to match. New "?" toolbar button re-invokes it. 10 new tests. Closes issue #1.
- Phase 6.2 (Tutorials & Onboarding): `docs/tutorials/getting-started-hybridstitch.md` and `docs/tutorials/pipeline-overview.md` — beginner-facing docs-site tutorials for `HybridStitchPanel` and the automated `AnimeStitchPipeline`, wired into `docs/mkdocs.yml` nav and `docs/index.md`. Closes issue #2.
- Moved `moon/`, `reports/`, and `research/` into `docs/` (`docs/moon/`, `docs/reports/`, `docs/research/`), consolidating all documentation under one directory; extended `docs/mkdocs.yml`'s nav with Roadmap/Changelog/Research/Reports sections at the new locations plus a link back to the parent Image-Toolkit project.
- Added `docs/website/` — a Vue 3 + Vite documentation site (same design as Image-Toolkit's own) rendering every `docs/**/*.md` directly, nav/search generated from this repo's own `docs/mkdocs.yml`. Includes a "Related Projects" sidebar section embedding Image-Toolkit's and Cel-Shaded-Generator's own docs sites via iframe. Deployed alongside the MkDocs portal in CI (`.github/workflows/docs.yml` + Forgejo/Gitea/GitLab mirrors) at `/app/`.
- Fixed `.gitlab/.gitlab-ci.yml`, `.gitlab/issue_templates/`, `.gitlab/merge_request_templates/`, and `.devcontainer/` — all still referenced the polyglot template's removed rust/typescript/kotlin/go/java dirs, missed when the `base`/`backend`/`gui` rename+flatten landed.
- Imported the Anime Stitch Pipeline (ASP) engine from Image-Toolkit: the C++ pipeline (`cpp/src/animation`, `cpp/test/animation`), the Python pipeline (`python/src/animation`), tests (`python/test/animation`, `python/test/gui`), the benchmark evaluation dashboard (`python/src/animation/evaluation`), benchmark scripts (`python/benchmark`), the stitch GUI tab/elements/helpers/dialogs (`python/src/animation/gui`), QML mockups (`qml/`), the roadmap (`moon/ROADMAP.md`), research reports, and ASP-specific `.agent/` workflow/rule/skill/prompt/cache files.
- Pruned the template to Python + C++ only (removed rust/java/kotlin/typescript/go scaffolding, the template-meta `dev/` tool, and desktop/infra scaffolding this project doesn't need).

### Template scaffolding

- Initial template scaffolding: root files (`LICENSE.md`, `README.md`, `.env.example`, `.pre-commit-config.yaml`, `.gitignore`/`.gitattributes`), `.github/` CI/CD, `git/` (`CONTRIBUTING.md`, `codecov.yaml`), `docs/` documentation portal (MkDocs + Sphinx + Structurizr + ADRs), `moon/` roadmap and changelog.
- `.agent/` LLM coding-agent scaffolding: `AGENTS.md` plus generic rules, workflows, prompts, and skills covering all six supported languages.
- Six language module skeletons (`python/`, `typescript/`, `kotlin/`, `rust/`, `go/`, `cpp/`), root workspace orchestrator files, and merged `python/validation/` dev-tooling.
- `java/` Maven module (7th language), wired into CI/pre-commit/justfile/docs alongside the existing six.
- Root Gradle wrapper and multi-project build files pairing with the existing `settings.gradle.kts`.
- `moon/roadmaps/developer_tools.md`: architecture plan for a polyglot `dev/` developer-assistant tool, synthesized from prior art across the org's other repos.
- GitHub Project (V2) backlog automation (`github/` + `.github/workflows/agent_sync.yml`), ported from Visual-Graph-Programming.
- `infra/{k8s,helm,terraform,ansible}/` infra-as-code scaffolding, alongside the relocated `infra/docker/`.
- `dev/` developer-assistant tool, milestones D1–D5 of `moon/roadmaps/developer_tools.md`: the `input/protobuf/codegraph.proto` schema, a hand-mirrored Python data model (`core/model.py`), a real AST-based Python import-graph parser (`input/python/parser.py`), multi-source graph aggregation (`core/aggregate.py`), layer classification + forbidden-direction violation detection (`core/layers.py`), Tarjan's-SCC circular-dependency detection (`core/cycles.py`), a self-contained vis.js/Jinja2 HTML report generator (`output/html/report.py`), and a `cli.py` tying it together (`report`/`check` subcommands). 13 passing pytest cases, including a fixture project with an intentional import cycle.

### Changed

- Moved `docker/` to `infra/docker/` to make room for other infra-as-code stacks; updated all referencing files.

## [0.1.0] — 2026-07-30

### Added

- Repository created from scratch as a GitHub template.
