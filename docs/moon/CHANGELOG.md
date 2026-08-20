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

## S365 — 2026-08-07 (Photometric Fix for Seam Rendering)

Implemented an HSV-based value scaling fix for seam-photometric diagnosis in both Python (`bench_anime_stitch.py`) and C++ (`compositing.cpp`). This resolves excessive color gradient shifts across seams on borderline datasets (e.g., test87, test10) by applying scalar gain only to the luminance/value channel rather than multiplicatively across all RGB channels.

## [Unreleased]

### Added

- **#34 M3 coherence_v2 ownership-factor slice (2026-08-20):**
  `_pick_owner` in `backend/src/rendering/compositing/coherence_v2.py`
  extended with a weighted tie-break stage covering the roadmap's
  remaining named factors — visibility (compactness of each candidate's
  own pose mask), boundary truncation, frame-quality sharpness, and
  temporal consistency (`composite_coherence_v2`'s multi-pair fold now
  passes forward the already-claimed map as `prior_owner`). Additive/
  optional; existing area+confidence-only call sites unaffected.
  Ownership decisions now propagate to `PipelineSession.observability()`
  via `seam_meta_out["coherence_ownership"]`. C++ parity check: no C++
  counterpart exists for this logic at all (Python-only by construction),
  so the roadmap's parity rule is trivially satisfied. Still default-off
  (`ASP_COHERENCE_V2`); structural red-set/smoke-set live rerun deferred
  this pass (elevated CPU temps from concurrent work) — needed before any
  promotion decision. 14/14 `test_coherence_v2.py` (+3 new), 277/277
  `rendering`+`core/pipeline` suites, no regressions. See
  `docs/moon/asp_change_roadmap_2026q3.md` §5 M3.

- **#33 M2.5b learned-proxy feasibility spike (2026-08-20):**
  `backend/benchmark/learned_proxy_feasibility.py` — leave-one-source-
  group-out ridge regression over the 9 existing `metrics_asp` signals
  against reviewed human coherence labels, on the frozen post-M1 ungated
  97-run (#30). Only 27 cases are true Raw ASP composites with reviewed
  labels (not 43 — M1's canonical adapter changed the fallback set).
  Verdict: **not yet feasible** — max per-feature Pearson r is +0.328, and
  the ridge model scores worse than the training-fold mean baseline (RMSE
  1.536 vs 1.058). Non-gating; no pipeline behavior changed. See
  `.agent/reports/claude/m2_5b_learned_proxy_feasibility_20260820.md` and
  `docs/moon/asp_change_roadmap_2026q3.md` §5 M2.5.

- **#31 M2 verification pass (2026-08-20):** re-confirmed (no code changes
  needed) that `ASP_HOLD_BG_SUB` registration, the `ghost_telemetry_only`
  default-promotion, and the ≤20-key default profile + Advanced-config
  surface (`gui/src/components/dialogs/asp_advanced_config_dialog.py` +
  `docs/website/src/components/config/AdvancedConfigDrawer.tsx` in the
  parent Image-Toolkit repo) are all already shipped and correct. Two
  items remain, both blocked on a Harbinger decision rather than further
  agent work: the CompositeGate empty-gate redesign, and the
  discriminating-fallback bar (§17 item 2) which is proven infeasible
  under the three existing gates on this corpus. See
  `docs/moon/asp_change_roadmap_2026q3.md` §5 M2 for detail.

- **#49 repeated-sequence benchmark stall fix (2026-08-20):** Model unloads,
  the pair loop, and the canonical matching stage no longer force synchronous
  CUDA allocator flushes or full Python garbage collections. Matcher outputs
  are detached and copied before release, ALIKED/LightGlue instances are reused,
  and affine BA bounds correspondence counts and uses a sparse solver. The
  final hidden stall was an unbounded OpenCV PANORAMA fallback after affine
  validation failed: sequences above 12 frames now use bounded SCANS directly
  (`ASP_PANORAMA_MAX_FRAMES` overrides the limit). A live `asp_test04 → 08 → 09`
  run completed with reports written and about 5.1 GB RSS at the former stall.
- **#49 third-stall diagnostics (2026-08-20):** Benchmark resource
  checkpoints no longer force a CUDA allocator flush at every stage. The
  expensive diagnostic behavior is opt-in with `ASP_RESOURCE_FLUSH_CUDA=1`.
  Post-matching bundle-adjustment, ECC, render, and foreground-composite
  phases now print start/finish timings so long CPU phases cannot appear hung.
- **#49 Stage 5–6 matcher stall diagnostics (2026-08-20):** Matcher loading
  and per-pair inference now emit flushed INFO-level lifecycle messages, and
  matcher exceptions are reported instead of being silently discarded. The
  pairwise loop no longer forces a CUDA synchronize/cache flush after every
  pair; the old behavior remains available with `ASP_MATCH_CUDA_SYNC=1`.
- **#49 Stage 4 hang (2026-08-20, Grok):** OpenCV OpenCL was claiming the
  same NVIDIA GPU as PyTorch CUDA (BaSiC then BiRefNet). Disabled OpenCL
  at benchmark start and `AnimeStitchPipeline.run()`. `_compute_fg_masks`
  now prefers `get_mask_batch` and no longer calls
  `empty_cache`/`synchronize`/`gc.collect` after every frame. Per-frame
  progress is flushed so a long mask pass cannot look like a dead hang.

- **#47 (M0c) layered synthetic pan/hold fixture generator (2026-08-20, Gemini):**
  Added `backend/src/alignment/synthetic.py` (`HeldCel`, `SyntheticPanSequence`,
  `generate_layered_pan_sequence`, `export_synthetic_sequence`). Generates
  high-frequency textured background plates with 2–3 held foreground character cels
  across known camera pan trajectories `(dx, dy)` and discrete hold spans. Exports
  complete ground-truth metadata manifests and panorama composites. Integrated into
  `backend/benchmark/generate_samples.py`. 5 unit tests in
  `backend/test/alignment/test_synthetic_fixture.py`.

- **#46 (M0b) experiment manifest + OTel telemetry (2026-08-20, Grok):**
  `TelemetrySink` on `PipelineSession` (Null / OTLP-JSONL / optional Rerun).
  Canonical `run()` still does not import `opentelemetry` or `rerun`.
  Manifest records git commit, profile, effective `ASP_*` env, model
  versions, input/output hashes, stage timings, peak RSS, and peak VRAM
  when CUDA is present (omitted on CPU CI). Canonical metrics:
  `asp.stage.duration_ms`, `asp.vram.peak_bytes`, `asp.gain.clamp_residual`,
  `asp.seam.cut_energy`. `compare_traces` treats duration-only deltas as
  timing noise and flags note/fallback divergence as nondeterministic.
  Opt-in extra: `desktop_quality` (`rerun-sdk`). Tests:
  `backend/test/core/test_experiment_manifest.py`.

- **#48 (M0d) versioned development slices (2026-08-20, Gemini):** Added `backend/benchmark/slices.py`
  defining canonical benchmark slices `smoke_v1` (5-case fast sanity: `asp_test04`, `08`, `09`, `27`, `57`)
  and `structural_red_v1` covering crop loss (`asp_test07`, `97`), torn anatomy (`asp_test04`, `06`, `12`, `15`),
  duplicated strips (`asp_test04`, `08`), misordered content (`asp_test12`, `41`), banding (`asp_test11`, `26`),
  known-good controls (`asp_test28`, `58`), and the test-14 oracle (`asp_test14`). Exported canonical manifest
  to `data/benchmarks/benchmark_slices_v1.json`. 8 unit tests in `backend/test/benchmarks/test_slices.py`.

- **#34 full-res red-set renders (2026-08-17):** `screen_coherence_v2.py
  --scale 1.0 --max-frames 8 --image-dir` writes default/v2 PNG pairs.
  Crop gate still 0/7. Recipe: `just bench::asp-coherence-v2-redset-full`.
- **#31 GhostGate telemetry-only promoted to product default (2026-08-17,
  Harbinger ACK):** `product_safe_asp_policy()` now returns
  `ghost_telemetry_only=True`. `ghosting_score_v2` is still recorded on every
  `GateDecision` (`status="telemetry_only_inverse_validated"`) but no longer
  drives Safe ASP selection — the promotion-ladder replay showed zero
  identity changes because the gate never actually rejected on this corpus.
  `CompositeGate`'s `sb`/`sc` telemetry-only candidates remain default-off;
  demoting `sb` was shown to change 26 historic identities with no
  raw-composite ground truth, so that promotion was not requested.
- **#34 exclusive-keep unblocks crop gate (2026-08-17):** contested
  overlap is single-pose; exclusive FG stays with its source. Red-set
  crop-loss re-screen **0/7** (test96 coverage held). Still default-off.
- **#34 M3 red-set compositor A/B (2026-08-17):**
  `screen_coherence_v2.py` runs default seam loop vs `ASP_COHERENCE_V2=1`
  on 04/06/07/12/14/15/96 (subsampled frames, median-dy affines). Crop-loss
  gate **fails** (6/7 including known-good test96). Not promoted. Sidecar:
  `docs/website/public/data/coherence_v2_redset.json`. Report:
  `.agent/reports/grok/m3_coherence_v2_redset_20260817.md`.
- **#34 M3 slice 2 — apply ownership to warped frames (2026-08-17):**
  `apply_coherence_v2` / `composite_coherence_v2` copy each owned region
  from exactly one pose (no blend). `_composite_foreground` takes this
  path only when `ASP_COHERENCE_V2=1`; default seam loop unchanged.
  Tests: 11 in `test_coherence_v2.py`.
- **#34 M3 first slice — `coherence_v2` assignment (2026-08-17):** new
  isolated module `rendering/compositing/coherence_v2.py` (Critical
  Evaluation §9.2 Stage 2). Each FG overlap region gets exactly one
  owner; no background corridor ⇒ explicit single-pose handoff. Not
  wired into the live seam loop. `ASP_COHERENCE_V2` registered
  default-off. Tests: `test_coherence_v2.py`.
- **#31 SeamVis threshold sweep is discriminating-infeasible
  (2026-08-17):** no `(floor, ratio)` pair catches catastrophes
  04/06/07/12/14/15 and keeps known-good test96. test96's
  `seam_visibility` (32.2) is *higher* than every catastrophe (max 29.58;
  binding case test15 at 12.55). Retuning SeamVis cannot pass the M2
  exit. Tool: `screen_seamvis_threshold.py`. Report:
  `.agent/reports/grok/m2_seamvis_threshold_infeasible_20260817.md`.
- **#32 M2.5a Per-Defect Category & Stage-Attributed Correlation Audit
  (2026-08-17, Gemini):** New statistical CLI and analysis engine
  `backend/benchmark/audit_defect_correlation.py` plus unit tests
  (`test_audit_defect_correlation.py`). Analyzes the 97 human-reviewed cases
  across 10 distinct defect classes (`torn_anatomy`, `banding`, `color_shift`,
  `seam_line`, `ghosting`, `misordered_content`, `duplicated_strip`, etc.)
  and maps them to pipeline stages.
  - Confirmed `seam_visibility` ($\rho = +0.425$) and `seam_gradient`
    ($\rho = +0.473$) are human-aligned across photometric ($\rho = +0.76$)
    and structural ($\rho = +0.42$ to $+0.64$) defect modes.
  - Demonstrated that Sobel `sharpness` ($\rho = -0.471$), `edge_energy`
    ($\rho = -0.531$), and `ghosting_siqe` ($\rho = -0.600$) are strongly
    inverted due to high-frequency edge inflation on torn seams and
    duplicated geometry.
  - Emits `public/data/defect_correlation_matrix.json` for web dashboard.
  - Added interactive Per-Defect &times; Per-Metric Heatmap Matrix and
    Pipeline Stage Attribution section to `RatingsDashboard.tsx` and
    `RatingsDashboard.css`.
  - Analytical report: `.agent/reports/gemini/m2_5a_defect_category_correlation_20260817.md`.

- **#31 CompositeGate `sb` telemetry-only candidate (2026-08-17):**
  `ASP_COMPOSITE_SB_TELEMETRY_ONLY=1` records `would_reject_sb` and never
  rejects on strip-banding. Default unchanged. Offline 97-case replay:
  **26 identity changes** (all historic `composite_gate_sb`; none also
  fail SeamVis/`sc`). Discriminating-policy exit **fails**: known-good
  `asp_test96` stays Raw ASP, but catastrophes 04/06/07/12/14/15 are
  already Raw ASP under current gates (SeamVis misses them too).
  CompositeGate after `sb` demotion has only no-signal `sc` left — not
  worth keeping as a structural gate. Report:
  `.agent/reports/grok/m2_composite_sb_and_discriminating_20260817.md`.
- **M2 observability slice 2 (2026-08-17):** pose rows now label
  `bundle_adjust` vs `affine_recovery` and `refined_by` (`ecc` /
  `sea_raft` / `none`). Frame provenance tracks `near_static` then
  `spatial_dedup` drop reasons. Geometry also recorded at load/save.
  Canonical bench JSON / adapter extra persist `observability`.
- **M2 session observability envelope (2026-08-17):** `PipelineSession`
  publishes `artifacts["observability"]` on `finish()` with per-stage
  geometry, frame provenance, pose provenance, gain residuals/clamps,
  seam feasibility, and `fallback_reason`. Stage 4.5 writes gain telemetry
  without changing pixels; Stage 11 records seam corridor metadata
  (no `seam_crops`). Tests: `test_pipeline_session`,
  `test_photometric_gain_telemetry`; entry-parity still green.
- **#31 M2 CQAS v1 demotion + hold-background config provenance
  (2026-08-17):** benchmark output now records the failed aggregate as
  `cqas_v1_legacy`, an explicitly diagnostic-only field excluded from
  automated verdicts and radar ranking; individual component diagnostics
  remain visible. `ASP_HOLD_BG_SUB` is now typed and persisted in
  `_CONFIG_SCHEMA` as Advanced-only, experimental and default-off, with the
  current unaligned-median limitation documented. M4 retains the algorithmic
  keep/delete decision.

- **#31 M2 GhostGate telemetry-only candidate (2026-08-17):** Chat's
  design. `SafeAspPolicy.ghost_telemetry_only` / `ASP_GHOST_TELEMETRY_ONLY=1`
  keeps `ghosting_score_v2` as `telemetry_only_inverse_validated` and never
  rejects. Default remains the current reject path — not flipped. Offline
  promotion-ladder replay on `anime_stitch_20260807_045552.json`: five-case
  (04/08/27/38/96), structural red set, and all 97 show **0 Safe ASP
  identity changes** and **0 historic GhostGate-only fallbacks** (the gate
  never fired; max asp/limit = 0.70). Recipe:
  `just bench::asp-ghost-telemetry-screen`. Report:
  `.agent/reports/grok/m2_ghostgate_telemetry_screen_20260817.md`.
- **Range-run JSON merge (2026-08-17):** `merge_run_json.py` unions
  disjoint `anime_stitch_*.json` files by dataset name (later file wins)
  into `anime_stitch_latest_consolidated.json`. `generate_json_results`
  writes that sidecar whenever two or more sibling runs exist. Recipe:
  `just bench::asp-benchmark-merge`.
- **#31 M2 `strip_banding_score` instrumentation (2026-08-17):**
  `_compute_all_metrics` persists CompositeGate's previously unaudited
  input again (0.0 without affines, same as the `scans_sb = 0.0` quirk).
  `audit_gate_correlation.py` can `--recompute-missing` values from
  panoramas + `alignment.affines` for historical JSONs. Live rho against
  the 2026-08-10 labels is **-0.417** (n=97 mixed-vintage dump; true
  composites only -0.365, n=43). CompositeGate now has no audited-correct
  input. No gate default changed. Report:
  `.agent/reports/grok/m2_strip_banding_audit_20260817.md`.
- **#31 M2 gate-signal correlation audit (2026-08-17):** new
  `backend/benchmark/audit_gate_correlation.py` computes Spearman correlation
  between each no-reference benchmark metric and human ASP-vs-SCANS score
  delta across the 97 reviewed cases. Reproduces the previously-unlinked bus
  claim (sharpness -0.47, edge_energy -0.53, ghosting -0.60) exactly, and maps
  it onto the live gates: `GhostGate`'s only signal (`ghosting_score_v2`) is
  the worst-scoring inverse metric found; `SeamVisGate`'s
  `seam_visibility_score` is confirmed correct; `CompositeGate`'s
  `strip_banding_score` component is unaudited (imported, never computed in
  the benchmark); the `cqas` aggregate also fails the audit. No gate default
  changed — see `.agent/reports/claude/m2_gate_signal_correlation_audit_20260817.md`.
- **#30 ungated provenance + deterministic gates (2026-08-15):**
  per-case `safe_asp_counterfactual` (`would_select` / `gate` / `reason`
  / policy snapshot / per-gate decisions) is written into the run JSON
  and the session. Ungated run-internal knobs
  (`ASP_ALIGN_GATE_DX=9999`, `ASP_COV_MIN_MULTI_PCT=0`) are **forced**,
  not `setdefault`. Counterfactual uses frozen product Safe ASP defaults
  so inherited `ASP_GATE_*` cannot change the baseline. Official
  `just bench::asp-benchmark-ungated` recipes export the same knobs.
- **#30 ungated Raw ASP harness (2026-08-15):** `ASP_BENCH_UNGATED=1`
  publishes Raw ASP as the baseline (policy still evaluated for
  telemetry: `policy_would_reject`). Internal no-edge / affine-invalid
  fallbacks still have no raw composite. Recipes:
  `just bench::asp-benchmark-ungated` (97-case) and
  `just bench::asp-benchmark-ungated-verify` (5-test smoke).
- **M1c GUI adapter (2026-08-15, #29):** `_ProgressPipeline.run()`
  defaults to canonical `AnimeStitchPipeline.run()`, forwarding
  `exclusion_masks`, `motion_model`, and HITL `pause_hook`. The three
  legacy-fork `_composite_foreground` calls now pass `exclusion_masks`.
  `ASP_GUI_LEGACY=1` keeps the HITL override fork. Headless parity:
  `backend/test/core/test_entry_parity.py`.
- **M1c HITL default clarified (2026-08-15, #29 Chat/Claude review):**
  a real `pause_cb` (the stitch worker) keeps the 9-checkpoint fork
  until M6. Headless / no-op pause uses canonical `run()` so CLI/bench/GUI
  share bytes. `ASP_GUI_CANONICAL=1` / `ASP_GUI_LEGACY=1` force a side.
  Canonical path still only applies the `masks` pause; #29 is **not**
  "all HITL on canonical."
- **M1b canonical bench adapter (2026-08-15, #28):** default
  `process_dataset` path is `run_canonical_asp()` — product
  `AnimeStitchPipeline.run()` then `SafeAspPolicy`. A `raw_asp`
  filename is written only for a true Raw ASP composite; internal
  SCANS/PANORAMA fallbacks record `raw_asp_available=false` and never
  occupy that path. Published `panorama.png` is Safe ASP.
  `ASP_BENCH_LEGACY=1` restores the pre-adapter orchestrator.
  Tests: `backend/test/core/test_bench_adapter.py`.
- **M1b Safe ASP policy extract (2026-08-15, #28 first slice):**
  CompositeGate / GhostGate / SeamVisGate moved to
  `asp_backend.core.pipeline.safety_policy` with the same env knobs and
  reason strings. Score functions live in `safety_metrics.py`; the
  benchmark re-exports the old names. Canonical `run()` still does not
  call the policy (M2). The 4k-line bench orchestrator is not rewritten
  in this slice. Tests: `backend/test/core/test_safety_policy.py`.
- **M0 case-provenance schema (Claude, 2026-08-15, #24/#41):** new
  `backend/benchmark/evaluation/other/provenance.py` — `CaseProvenance`
  dataclass for case-level fields (`corpus_id`, source URL/board, licence,
  `web_redistribution_ok`, `source_work_nsfw` as a real three-state
  bool/None, `content_tags`, `safety_tier`, `gt_known_defects`), plus C0.5's
  append-only `SafetyObservation`/`SafetyAdjudication` records implementing
  the actual dual-veto logic (`minor_presenting_high_risk`/
  `minor_presenting_includable` — OR to exclude, AND with a human `clear`
  specifically to include). New `RESULT_*`/`CONTENT_TAGS`/`SAFETY_TIERS`/
  `MINOR_RISK_VERDICTS` constants. Found (pre-existing, not caused by this
  change) a `conftest.py` package-aliasing issue blocking direct pytest
  invocation in this environment; verified correctness via a standalone
  harness instead.
- **M0 raw_asp/safe_asp/scans relabeling, closing #24 (Claude, 2026-08-15,
  `64d8829`):** `relabel_corpus()` in
  `backend/benchmark/evaluation/other/relabel.py` cross-references the
  saved 2026-08-07 benchmark run's `render_gate_fallback` codes against
  the human evaluations to make explicit, per case, which identity
  (`raw_asp` vs. a `scans` substitution) the human's "asp" score actually
  rated. Deliberately kept as a separate module joined by `case_id` rather
  than extending `RatingEntry` directly — `RatingEntry` stays the
  bench-facing comparison schema, `RelabeledCase` and `CaseProvenance`
  both reference cases by the same `case_id`, satisfying #24's exit
  criterion ("reconstruct exactly which artifact was rated") without a
  second evaluation schema. Verified against the real 97-case data: 43
  true raw_asp composites (mean 1.326) / 54 safety fallbacks (mean
  2.556), matching the numbers already cited in the roadmap §3. 47 tests
  green across `test_eval_schema.py`, `test_eval_provenance.py`,
  `test_eval_relabel.py` (2026-08-20 verification pass).

- **Phase 3 telemetry lock (Harbinger 2026-08-15):** A+B —
  `TelemetrySink` + opt-in Rerun `.rrd` sidecar (`desktop_quality`) +
  OTel spans/metrics (local OTLP/stdout first). Rerun WASM in
  `docs/website` rejected. Native JSON/NPZ inspector (option C) is a
  fully optional, unscheduled extra. Recorded in
  `analytics_and_interpretability.md` §3, change-roadmap §5/§19.3, and
  `.agent/reports/grok/phase3_rerun_tradeoffs_20260815.md`. No
  implementation in this pass.
- **SFW corpus C0/C0.5 rewrite + outreach roadmap kickoff (2026-08-15):**
  direct Claude+Harbinger design session. `asp_sfw_corpus_roadmap_2026q3.md`
  C0 revised: automation/human split locked (automate bulk filtering/
  dedup/clustering, never the quality judgment itself), GT strategy changed
  to stratified-coverage-over-count with a `gt_known_defects` field
  (more imperfect GT preferred over fewer "perfect" ones). New §C0.5:
  `content_tags` + named `safety_tier` enum (`tier_g`/`tier_pg13`/
  `tier_mature_sfw`/`tier_nsfw`, not a numeric score) replacing the `sfw`
  boolean, with per-context policy kept as separate editable config, not
  baked into the corpus. Non-negotiable minor-presenting hard floor:
  appearance-based (not claimed in-universe age), dual-veto gate (either
  human or automated flag excludes a case entirely, both must clear for
  inclusion), periodic re-audit as the corpus grows. New
  `asp_outreach_roadmap_2026q3.md`: Overmix-blog-style results/reasoning
  outreach, goal and rationale only, design deliberately left to a
  Gemini/Grok/Chat-Codex brainstorm round rather than pre-decided.

- **C0.5 dual-veto scoped (2026-08-15):** automated half is an ensemble of
  weak votes (board tags, official source rating, optional WD14, optional
  later commercial API) plus a required periodic adversarial audit — not a
  single age classifier, which this repo does not have. C0.5 applies to
  SFW intake only, after a Harbinger `nsfw_97` provenance review
  (`source_work_nsfw` is series-taint, not a tier). Outreach O1 waits for
  a preregistered complementary ASP/SCANS split; Lab Notes only until then.
- **M1 video smart-select (#27, 2026-08-15):** `VideoIngestionStream` now
  writes proxy frames to temp paths and calls `smart_select_frames(paths)`.
  `TypeError` / empty selection hard-fail; they no longer fall back to
  uniform. Tests in `test_video_ingestion.py`.
- **M1a shared stage protocol / `PipelineSession` (2026-08-15):** extracted
  `backend/src/core/pipeline/session.py` — inputs, frozen config snapshot,
  ordered stage trace, JSON-safe artifacts, fallback/identity labels, and
  HITL pause-hook storage. Canonical `AnimeStitchPipeline.run()` creates a
  session and records stages/fallbacks next to existing log/return sites;
  image operations are unchanged. `_ProgressPipeline.run()` is not rewritten
  (M1c). No new HITL checkpoints were inserted into the canonical runner.
  Tests: `backend/test/core/test_pipeline_session.py`.
  PANORAMA is recorded as a `safe_asp` fallback with `algorithm=panorama`
  metadata, rather than as a fourth externally visible result identity.
- **ASP 2026 Q3 change roadmap (2026-08-15):** added
  `asp_change_roadmap_2026q3.md`, a concise issue-ready plan that tracks Raw ASP,
  Safe ASP, and SCANS separately; prioritizes one canonical benchmark/backend/GUI
  pipeline; sequences BiRefNet single-pose compositing before heavier temporal
  segmentation; defines staged human-quality promotion gates; targets a 12 GB
  VRAM / 32 GB RAM laptop profile; and records the optional artist review screen
  plus visual redesign as later work. The historical `ROADMAP.md` now links to
  this active change plan.
- **Grok SFW/M2.5 feasibility (2026-08-15):** reviewed
  `asp_sfw_corpus_roadmap_2026q3.md`, parent §4.18, and M2.5. Harbinger
  locked a separate `dump_sfw/` root with local `asp_testNN` names;
  C0 now / C1 harvest now, register on M0 / C2 after M1 only (not M2.5);
  Safebooru waits on a native C++ name after `base` rebuilds; this §4.18
  pass is Rating-on-Danbooru/Gelbooru; M2.5 splits into analytics vs a
  post-M1 proxy spike.

- **Grok feasibility pass (2026-08-15):** reviewed the Q3 roadmap against
  current source and the completed 97-case file. Locked incremental M1
  (protocol → bench adapter → GUI hooks), a discriminating Safe ASP exit,
  default-off §9.2 compositor, schema-first dual inspector, and
  relabel-don't-rerun for M0. Recorded that preferences are now complete
  (14 / 29 / 54), `ASP_HOLD_BG_SUB` is a hidden flag, and the GUI composite
  path still drops `exclusion_masks`. Harbinger answered §16: post-M1
  ungated 97-run; PNG+JSON then later PSD; default ≤20 flags with an
  Advanced configuration control; C++ stays blocked until `base` rebuilds.

- **M2.5 quality-metrics/benchmark-analytics milestone (2026-08-15):** added
  to `asp_change_roadmap_2026q3.md` §5/§9/§18 — per-defect-category
  correlation/impact analysis, anime-adapted CV metrics, a non-gating learned
  human-judgment proxy (revalidated as the corpus grows), and similarity-based
  benchmark subset selection. Depends on M0's per-output schema, runs parallel
  to M1–M4, changes no algorithm default. Closes the review round that
  produced §10 (Claude), §12 (Gemini), and §15 (Grok).

- **ASP SFW benchmark corpus roadmap (2026-08-15):** added
  `asp_sfw_corpus_roadmap_2026q3.md`, a companion, non-blocking track for
  building a ~20–30 case SFW benchmark corpus as a generalization check
  alongside the existing 97-case NSFW corpus. Confirmed `docs/website` and
  `docs/tutorials` contain no NSFW material. Depends on the parent
  Image-Toolkit repo's `new_features.md` §4.18 (crawler rating filter +
  Safebooru board). Frame-sequence auto-detection is explicitly deferred to
  this roadmap's M2.5, not duplicated as a one-off tool. C2 now includes a
  blinded, representative 8–10-case human coherence screen; it remains
  informational and does not gate M0–M6 or default promotion.

- **C/D investigation baseline (2026-08-11):** ran the combined existing
  pose-window, phase-composite, and joint-gain candidates on the five-test
  verification set. The run produced 1 ASP ground-truth win, 2 SCANS wins,
  and 2 comparable results; ASP sharpness and measured ghosting improved, but
  GT-SSIM remained slightly behind SCANS. The candidates remain experimental
  and default-off while candidate frame selection and photometric policy work
  continues.

- Added a **Load Evaluation…** control to the benchmark inspector. It opens
  an existing `asp_evaluations_*.json`, switches the active output to that
  file, and resumes at its first unrated dataset. The existing `--out PATH`
  CLI option remains available for scripted launches.

### Changed
- Fixed the Image-Toolkit ASP benchmark wrapper to use the corpus at
  `submodules/ASP/dump` for all ASP benchmark, resume, range, and cleanup
  recipes. The previous parent-level `dump` path caused verification to report
  zero matched datasets without running the benchmark.

- Stopped tracking `docs/website/tsconfig.tsbuildinfo` (a stray TypeScript
  incremental-build cache left over from before the Vue 3 site removal --
  no `.gitignore` rule covered it). Added `docs/website/.gitignore`
  (`node_modules/`, `dist/`, `*.local`, `*.tsbuildinfo`), matching CSG/CRE.
  `docs/mkdocs.yml`'s `site_dir` now points at `build/gen/site` (was
  `../site`, unreferenced by any `.gitignore` rule); CI workflows
  (`.github`/`.forgejo`/`.gitea`/`.gitlab`) updated to match. GitLab's job
  still ends with `mv build/gen/site public`, since GitLab Pages requires
  that literal directory name.

### Added
- Surfaced seam diagnostic recommendations in-context within the `SeamDiagnosticDialog` based on seam analysis (Phase 6.4: Assisted-use suggestions), replacing the need for a separate tutorial mode.


### Added
- **Phase 0.3**: Added `backend/benchmark/merge_overmix.py` to merge Overmix's per-test artifacts (stitch images, variant logs) into a consolidated benchmark JSON and Markdown report without requiring a full pipeline re-run (Issue #352).
- **Phase 0.5**: Ran full 97-corpus benchmark for Hugin using run_hugin.py. Fixed missing constants via PYTHONPATH and parallelized the script for faster processing (Issue #18).
- **Phase 6.3**: Bundled synthetic sample projects (`test_scroll_gradient`, `test_scroll_pattern`) for the HybridStitch onboarding wizard. Provides non-explicit, simple test sequences to learn the tool (Issue #17).
- **Phase 4**: Added quantitative (non-visual) seam-photometric diagnosis tooling via `backend/benchmark/diagnose_seams.py` to analyze borderline seam_vis_gate/composite_gate_sb fallbacks without rendering adult content (Issue #16).
- **Phase 0.4(d)**: Implemented SI-FID as a reference-free metric in `backend/benchmark/evaluation/si_fid.py` and integrated it into `bench_anime_stitch.py` for non-GT test evaluation.
- **Bug Fix**: Fixed `_refine_masks_with_clicks` dimension mismatch in `backend/src/ingestion/masking.py` by threading per-frame shapes through the live SAM-2 predictor state across the HITL dialog boundary (Issue #15).
- **Bug Fix**: Decoupled `gui/` tests from Image-Toolkit by making `AppSettings` import optional in `_thumbnail_file_picker.py` and removing `continue-on-error` from `lint-test-gui` (Issue #3).
- **Toolchain Cleanup**: Consolidated documentation toolchains to MkDocs and Sphinx. Removed Vue 3 site, Structurizr, and TypeDoc, along with their CI build steps (Issue #4).

- **Bundled sample projects for HybridStitch, closing roadmap Phase 6.3 (issue #17).** `gui/scripts/generate_sample_sequences.py` procedurally draws 3 tall synthetic "scroll page" images with PIL (flat rounded panels, line-art silhouette + speech bubbles, dot-grid/text-label page — no real source art) and slices each into 6 overlapping 480×320 frames, committed under `gui/resources/samples/<name>/frame_NN.png` (~60 KiB total). `gui/src/tabs/stencil/sample_sequences.py::list_sample_sequences` discovers them at runtime; `RealHybridStitchPanel` gets a new non-modal "Try a Sample" toolbar button (an instant-popup menu) that loads a chosen sequence into the sidebar via the existing `load_paths()`, so a new user can explore every tool tab on real (synthetic) data immediately. 12 new tests (`gui/test/tabs/test_sample_sequences.py`) cover generator output validity (frame count, real pixel overlap between consecutive frames, bundle size), sample discovery, and the panel's load-a-sample action. Full gui suite 62/62 (50 pre-existing + 12 new).
- Ran a full-session `/code-review high 0f3196a~1..HEAD` (278 files) as a final Ground Rule check on this session's volume of change. 10 findings surfaced: 7 fixed (the `_compute_fg_masks_sam2`/`_compute_fg_masks_grounded_sam2` frame-0-only resize bug in `backend/src/ingestion/masking.py` — same root cause already fixed for issue #11 in the sibling stateful function; `ASP_USE_SAM2` env-var parsing now treats `"false"/"no"/"off"/""` as falsy; CI's `lint-test-backend` pytest step now has the same `continue-on-error: true` + explanation as `lint-test-gui`'s, for the identical issue #3 coupling; a dropped assertion in `test_frame_selection.py` restored; two onboarding-wizard bugs in `gui/src/tabs/stencil/hybrid_stitch_panel.py` fixed — `showEvent()` now gates first-run tour instead of a bare `singleShot(0, ...)`, and re-opening the wizard no longer false-marks the tour "seen" via the superseded instance's `finished` signal); plus a real test-hermeticity bug found in passing (`test_eval_metrics_view.py`'s 6 `load_test_assets()` calls were missing `results_path`, letting real benchmark JSON leak into fixture-only tests). 2 findings were review-methodology false positives (bare `python3 script.py --help` invocation, not the documented `PYTHONPATH=...` form). 1 finding (`_refine_masks_with_clicks`'s HITL click-refinement has the same dimension issue but needs a real architectural fix, not a bounded one) deferred to issue #15. Full backend suite 911/1(pre-existing)/3 and full gui suite 50/50 unchanged; both packages remain mypy-clean.
- Ran the full 97-test corpus benchmark for the first time since 2026-07-28 (issue #13, RTX 3090 Ti, ~2h40m) — the actual Ground Rule #1 verification of this entire session's infra/bug-fix work, not an assumption. Result: statistically indistinguishable from the pre-session baseline (43 true composites, 54 fallbacks, GT-SSIM 0.6659 vs 0.6931 — both exact/near-exact matches to 2026-07-28's 43/54/0.6656/0.693). Direct corpus-scale confirmation that none of this session's work — packaging fixes, the evaluation-dir move, two real pipeline bug fixes, dead-code removal, or the complete mypy/ruff cleanup across `backend/` and `gui/` — changed default pipeline behavior.

- Fixed `gui/test/conftest.py`'s missing `q_app` fixture (separate from issue #3's cross-repo problem): 33 of 50 tests failed at setup with "fixture 'q_app' not found", not real failures. All 50 pass now via Image-Toolkit's shared interpreter. Does not resolve issue #3 itself — CI's standalone environment still hits the deeper `backend.src.constants` coupling.
- **`gui/`'s mypy is now fully clean too (215 → 0 errors), closing issue #14.** Same playbook as `backend/`'s issue #7: a `TYPE_CHECKING`-only Protocol (`_stitch_tab_protocol.py::_StitchTabHost`) for `StitchTab`'s 11-mixin composition (159 of 215 errors), a root-caused `has-type` fix in `_progress_pipeline.py`, narrow stub-gap `# type: ignore`s, and mechanical local-bind fixes. Test suite (50/50) unchanged. `gui/` and `backend/` are now both fully clean on mypy.
- `gui/` ruff cleanup: 447 → 5 errors (issue #14). Auto-fixed 442 mechanically, manually wrapped 14 non-auto-fixable long lines, left 5 string-literal tooltips/docstrings as-is. 51 files touched, test suite (50/50) unchanged.
- Found `gui/` has the same never-cleaned-up lint/type debt `backend/` had (447 ruff + 215 mypy errors) — `uv sync` never had the CUDA blocker that hid this in `backend/`, so it should have been failing CI this whole time. Filed as issue #14, cleanup in progress.
- **`backend/`'s mypy is now fully clean (137 → 0 errors), and ruff is effectively complete (2264 → 249, all remaining being a deliberate line-length judgment call, not unfinished work).** Root-caused the final 17 `has-type` errors properly (reproduced with the mixin Protocol removed to rule it out as the cause) rather than guessing — mixin methods reassigning attributes to differing types across branches without a local declaration. Full test suite identical throughout (911 passed, 1 pre-existing unrelated failure, 3 skipped). Closes the mypy portion of issue #7.
- Resolved ruff's F401/E741/E402 (280 → 249 errors, only line-length remains): 3 genuinely-used re-exports moved into `__all__` (were flagged only because accessed via module-attribute in tests), 3 redundant re-imports removed, 1 ambiguous variable renamed, 13 intentional `sys.path`-bootstrap imports marked `# noqa: E402`. Assessed but mostly left the 260 line-length violations (wrapped 11 clearly-safe function signatures) — `line-length = 100` is a deliberate, consistent repo convention and most violations are string-literal content that shouldn't wrap. Part of issue #7.
- Fixed the mechanical "mypy can't narrow through a repeated subscript" pattern across 21 files plus 5 cv2-stub gaps (narrowly-scoped `# type: ignore`): mypy 88 → 24. Full test suite unchanged. Part of issue #7; remaining 24 errors are a different, not-yet-triaged symptom (`has-type` on lazy-loaded matcher attributes at `run_stage.py:257`).
- Closed issue #12: `_pass2_pose_refine`'s broken DINOv2 recomputation turned out to feed a helper (`_pose_dist`) that's never called anywhere — dead code, zero behavioral impact. Removed it; added the first unit test coverage `_pass2_pose_refine` has ever had (3 tests) and verified via a standalone script that the live DINOv2 scoring path genuinely works.
- Fixed 28 mixin-composition mypy errors in `core/pipeline/` with a new `TYPE_CHECKING`-only `Protocol` (`_pipeline_protocol.py::_PipelineHost`) describing the composed `AnimeStitchPipeline`'s attributes — zero runtime behavior change, mypy 116 → 88, test suite unchanged. Part of issue #7.
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
- Added `docs/website/` — a Vue 3 + Vite documentation site (same design as Image-Toolkit's own) rendering every `docs/**/*.md` directly, nav/search generated from this repo's own `docs/mkdocs.yml`. Includes a "Related Projects" sidebar section embedding Image-Toolkit's and CSG's own docs sites via iframe. Deployed alongside the MkDocs portal in CI (`.github/workflows/docs.yml` + Forgejo/Gitea/GitLab mirrors) at `/app/`.
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
