# ASP Change Roadmap — 2026 Q3

**Status:** Signed off. Issues filed (ASP #24–#40, Image-Toolkit #370–#371).
M1a (`PipelineSession` / stage protocol) landed 2026-08-15 — extraction
only, no pixel-path change. M1b adapter landed 2026-08-15 (canonical
`run()` + policy; `ASP_BENCH_LEGACY=1` keeps the old fork). M1c GUI
adapter landed 2026-08-15 (`ASP_GUI_LEGACY=1` keeps the HITL fork).
**Created:** 2026-08-15
**Scope:** ASP correctness, product-pipeline convergence, measurement, and the
artist review workflow. The visual redesign is recorded as future work and is
not a current implementation priority.

This is the short, issue-ready plan for new ASP work. The older
[`ROADMAP.md`](ROADMAP.md) remains the research history and experiment ledger;
when the two documents differ about future work, this roadmap takes priority.

**2026-08-20 status:** ASP #49 remains under investigation. Stage 4's
OpenCL/CUDA contention is fixed; Stage 5–6 now has matcher load/inference
telemetry and avoids per-pair CUDA allocator synchronization. The ungated
97-case run stays blocked until a GPU smoke run confirms the second stall is
resolved. Benchmark resource checkpoints now also avoid allocator flushes by
default, with explicit post-match timing markers to isolate later CPU/GPU
phases; `ASP_RESOURCE_FLUSH_CUDA=1` restores the old diagnostic flushes.
The remaining EfficientLoFTR offload-time allocator flush was removed and the
previously failing `asp_test09` case now completes end to end; rerun the full
five-case smoke slice before starting the ungated 97-case run.

## 1. Product objective

The ultimate goal is for the **raw ASP compositor** to beat SCANS. Safety
fallbacks are part of the shipped product, but they must not hide raw ASP's
quality or be counted as raw ASP wins.

Every benchmark and user-facing comparison must keep three identities separate:

1. **Raw ASP** — the ASP result before any SCANS replacement.
2. **Safe ASP** — the shipped policy result, which may select Raw ASP or SCANS.
3. **SCANS** — the unchanged comparator and fallback candidate.

The benchmark must retain all three artifacts, identify which artifact Safe ASP
selected, and report raw and safe scorecards separately. An ASP fallback must
never be silently labelled as a genuine ASP composite.

Final success requires both:

- Raw ASP has no human-rated loss to SCANS on any of the 97 cases.
- Raw ASP's mean human coherence score is strictly greater than SCANS's mean.

Human structural coherence outranks SSIM and all reference-free metrics. Torn
anatomy, duplicated strips, or misordered content are release-blocking even when
an automated score improves.

## 2. Locked planning decisions

- Unify benchmark, backend/CLI, and GUI execution before adding more image
  quality algorithms.
- Preserve the completed human review. The 0–4 coherence score ordering is the
  canonical quantitative label; preference, notes, confidence, and defect tags
  remain metadata and are never discarded.
- Use staged milestones instead of requiring the final 97/97 result from every
  experiment.
- Prove a BiRefNet-based single-pose compositor before adopting a heavier
  tracking dependency.
- Keep SAM2 or comparable temporal segmentation as a future, measured
  experiment rather than a current prerequisite.
- The normal workflow is automatic Safe ASP with an **optional review screen**.
- The default laptop profile must complete on 12 GB VRAM and 32 GB system RAM.
  More powerful hardware may select a higher-quality profile.
- Include artist-facing workflow work in this plan. Record visual redesign as a
  non-priority future entry.

**2026-08-23 raw-quality proposal:** The corrected frozen review says the
next experiments must improve Raw ASP rendering rather than add another
selection gate. Default-off candidates are: production/benchmark photometric
telemetry parity; a canvas-aligned background plate followed by one foreground
pose; edge-preserving background source selection; a protected-background TPS
or APAP residual; mask disagreement with a trapped-ball alternate; and
replayable assisted correspondence/order/background corrections for
connectivity failures. Harbinger hand-picks the small seam/blending,
content-integrity, and connectivity validation sets; no metric-derived subset
or corpus run promotes a candidate. Full proposal:
`.agent/reports/codex/asp_quality_proposal_2026-08-23.md`.

**First validation slice locked (2026-08-23):** P1's one-coherent-pose
contract is before P2 seam cleanup. Harbinger selected seam cases
03/05/17/37/42/78; content cases 01/41/65/68/74/82/28/83; clean/near-clean
regression controls 67/73. Connectivity uses 21/46/52, where the offline
overlap proposal found bridge candidates, plus 89 (anchors too sparse) and 25
(`no_valid_edges`). Test51 was removed: its frozen fallback is `seam_vis_gate`,
not connectivity. These are review slices, not authorization for a benchmark
run.

**P0 landed (2026-08-23, Codex):** canonical benchmark reports now read the
production Stage 4.5 gain record from `PipelineSession`, including per-frame
background-pixel eligibility/luminance, BGR gains, clamp hits, residuals, and
reference luminance. No gain is recomputed in the adapter and no pixel path
changed. Legacy A/B output is explicitly labeled `legacy_harness`.
The guarded validation runner also applies `ASP_BENCH_THREAD_CAP` to PyTorch's
runtime thread pools; its environment-only setting was too late after the
benchmark's import order.

## 3. Evidence baseline

The completed `asp_evaluations_20260810.json` contains 97 reviewed cases:

| Measure | Raw label currently called ASP | SCANS |
|---|---:|---:|
| Mean coherence | 2.010 / 4 | 2.639 / 4 |
| Score-order wins | 10 | 49 |
| Score-order ties | 38 | 38 |

The saved output is not a clean Raw ASP baseline because 54 of 97 results are
benchmark-only safety fallbacks. On the 43 true composites, ASP averages 1.326
and records 4 wins, 3 ties, and 36 losses. A new ungated Raw ASP corpus run is
therefore required after pipeline convergence; the completed ratings must not be
misrepresented as that raw baseline.

Preference metadata is now complete for all 97 records (filled 2026-08-15):
tests 27 and 39 prefer SCANS, test 56 prefers ASP, test 91 is a 4–4 tie.
Current preference field: **14 ASP / 29 tie / 54 SCANS**. Fourteen preferences
still disagree with score ordering; both fields stay, score ordering remains
canonical. Do not reopen the 97-case rating pass.

Current automated verdicts agree exactly with human score ordering on only
58/97 cases. Sharpness, edge energy, and current ghosting metrics are diagnostic
signals, not promotion criteria. Aligned GT-SSIM is useful where GT exists but
does not replace human review.

## 4. Non-negotiable engineering rules

1. One canonical pipeline owns orchestration. Benchmarks and GUI code call it;
   they do not reproduce its stages.
2. One material hypothesis per experiment. Record the exact config, commit,
   input slice, resource profile, output paths, and result.
3. Default-off experiments do not count as finished until they are integrated,
   exercised on representative data, and either promoted or removed.
4. Python and C++ paths need parity fixtures for any behavior that can change an
   image.
5. Do not introduce hidden environment switches. New configuration belongs in
   a typed schema and a named profile. Target **~20 active flags** on the
   default shipped profiles (`laptop_balanced` / `desktop_quality`). A new
   default-profile flag must displace an old one except when it is measurably
   better on a named subset and worse on another. "Active" means it is on the
   default surface and can change a shipped-profile image. Additional
   registered parameters are reached through an **Advanced configuration**
   control, not by deleting them in M2. Advanced values still persist in the
   experiment/project manifest. `ASP_HOLD_BG_SUB` is currently a hidden
   `os.environ` read and must be registered or deleted in M2.
6. No output may overwrite Raw ASP, Safe ASP, or SCANS evidence from the same
   run.
7. Every implementation issue updates this roadmap and the changelog with what
   actually shipped, what was measured, and what remained disabled.
8. Do not rewrite `_ProgressPipeline.run()` and `bench_anime_stitch.py` in the
   same change. Extract a shared stage protocol first; adapters migrate one
   entry point at a time. GUI parity tests are headless (no Qt): they call
   the same stage functions with recorded callbacks, not a live QApplication.
9. Reuse `backend/src/hitl/hitl_session.py` and the existing HITL checkpoints.
   Do not invent a second session format. Web and PySide6 inspectors are thin
   views over one replayable schema.
10. Algorithmic rebuilds (Critical Evaluation §9.2 compositor, repaired
    translation+scale, aligned background plate) land as default-off named
    candidates after M1. They do not replace the current compositor in place
    until a human screen promotes them.
11. `translation_scale`, the exported DP hold-keyframe selector, the Python
    `1e6` semantic hard-veto, and C++ `build_seam_cost_map(...exclusion_masks)`
    are **not** completed work. Issues must say repair-or-remove, never
    "finish wiring."

## 5. Delivery sequence

### M0 — Lock truth and experiment contracts

**Purpose:** make every subsequent result reproducible and correctly labelled.

**Schema/relabel slice landed 2026-08-15, verified 2026-08-20 (Claude,
issue #24 closed):** `raw_asp`/`safe_asp`/`scans` result separation
(`relabel.py::relabel_corpus`), the case/provenance envelope
(`provenance.py::CaseProvenance`, includes the SFW-corpus case-level
fields), and score-ordering/preference/defect handling
(`schema.py::RatingEntry`) are all implemented and tested (47 passing
tests). The smoke/red-set development slices landed 2026-08-20 (Gemini,
issue #48 closed: `smoke_v1` and `structural_red_v1` in `slices.py`,
manifest in `benchmark_slices_v1.json`, 8 passing tests). The
manifest/telemetry harness landed 2026-08-20 (Grok, issue #46:
`TelemetrySink` on `PipelineSession`, OTLP-JSONL/stdout first, experiment
manifest with git/profile/hashes/RSS/VRAM, `compare_traces` for the
same-manifest exit). The layered synthetic pan/hold fixture generator
landed 2026-08-20 (Gemini, issue #47 closed: `synthetic.py`,
`generate_layered_pan_sequence`, 5 passing unit tests). **All M0 deliverables are complete.**

Deliverables:

- Define `raw_asp`, `safe_asp`, and `scans` as separate result fields and output
  artifacts throughout the evaluator and dashboard data.
- Treat score ordering as the canonical quantitative human verdict. Preserve
  preference disagreements and nulls without silently rewriting them.
- Extend future annotations so score dimensions and defect tags belong to a
  specific output, not the comparison as a whole. **Inspector severity slice
  landed (2026-08-23, Codex):** optional
  `defect_severity[output][defect] = 1..3` records trace/noticeable/severe
  ordinal judgments per comparator; `0` is absent. The inspector selects an
  output before grading each defect. The legacy `defects` list remains the
  binary projection (any severity > 0), and old binary-only records remain
  ungraded rather than being silently assigned a severity. Frozen labels were
  not rewritten; the next review pass supplies the new data.
- Define one versioned **case/provenance envelope** referenced by all three
  output artifacts. It owns corpus membership, source/licence and
  redistribution facts, GT qualifications, content/safety assessments, and
  human/automated/adjudicated observations; result-specific metrics and defect
  annotations remain beneath their respective `raw_asp`/`safe_asp`/`scans`
  artifacts. SFW C0.5 extends this envelope — it must not create a parallel
  evaluation JSON.
- Version two development slices:
  - the existing five-case smoke set (04, 08, 09, 27, 57);
  - a structural red set covering crop loss, torn anatomy, duplicated strips,
    misordering, banding, a known-good result, and test 14's manual-selection
    oracle.
- Save a machine-readable experiment manifest with the Git commit, profile,
  effective config, model versions, input hashes, timings, peak VRAM/RAM, and
  output hashes. **Landed 2026-08-20 (#46):** `backend/src/core/pipeline/{telemetry,manifest}.py`
  — `TelemetrySink` on `PipelineSession`; local OTLP-JSONL or stdout; no
  `opentelemetry`/`rerun` import in `run()`.
- Relabel the saved 2026-08-07 full-corpus artifacts (`anime_stitch_20260807_045552.json`
  and companions) into `raw_asp` / `safe_asp` / `scans` without requiring a
  new 97-case GPU run as an M0 exit. A fresh ungated Raw ASP corpus run is
  **M1's follow-up**, once the three adapters share a runner — that run is
  the pre-algorithm baseline for M2+. It is not an M0 blocker and is not
  deferred all the way to M6.4.
- Generate a small procedural **layered** synthetic suite (static background
  plate + 2–3 held character cels with known camera velocity and known hold
  IDs). The existing `test_scroll_gradient` / `test_scroll_pattern` samples
  are unlayered onboarding scrolls and are **not** sufficient for hold or
  single-pose tests.
- Regenerate the dashboard summary only from the typed result fields.

Exit criteria:

- A report can reconstruct exactly which artifact was rated and which artifact
  Safe ASP selected.
- No completed human metadata is lost.
- Two runs with the same manifest produce equivalent stage traces and identify
  any nondeterministic stages.
- The synthetic fixture recovers known translation (and hold IDs) in a unit
  test without the anime corpus.

### M1 — One canonical product pipeline (P0)

**Purpose:** ensure that benchmark evidence describes what artists actually use.

Deliverables:

- Make `AnimeStitchPipeline.run()` (or a factored canonical successor) the sole
  stage orchestrator. `_ProgressPipeline` already *subclasses*
  `AnimeStitchPipeline` but **overrides `run()` entirely** (~833-line fork).
  Inheritance is not sharing. Do not treat "it subclasses" as convergence.
- Land M1 in three sequential PRs, not one:
  1. **M1a** — extract a stage protocol / `PipelineSession` (inputs, config,
     traces, artifacts, pause hooks). No behavior change.
     **Landed 2026-08-15:** `backend/src/core/pipeline/session.py`; canonical
     `run()` records stages/fallbacks onto `pipeline.last_session`. GUI/bench
     forks untouched.
  2. **M1b** — convert `bench_anime_stitch.py` (4045 lines; owns
     Composite/Ghost/SeamVis and its own `smart_select_frames`) into an
     adapter. First-file bugfix in this PR or immediately before it: video
     `smart_select_frames(proxy_imgs, target_n=want)` is a TypeError today
     (`selector.py` takes `frames_paths: list[str]` and has no `target_n`)
     and silently falls back to uniform.
     **Landed 2026-08-15 as isolated #27:** video path now writes proxies
     to temp PNGs, calls the real path API, and hard-fails on TypeError
     or empty selection. Not bundled with M1b.
     **M1b first slice 2026-08-15:** Composite/Ghost/SeamVis live in
     `safety_policy.py`; bench calls `default_benchmark_policy()`. Raw
     ASP stage dump still written before any fallback. Remaining M1b
     work is making the bench a thin `AnimeStitchPipeline.run()` adapter
     — not done here (would change the 43-composite measurement path).
     **M1b adapter 2026-08-15:** default path is
     `run_canonical_asp()` (`bench_adapter.py`) — selected frames →
     product `run()` → policy → published Safe ASP. Raw ASP is kept at
     `output/panorama_stages/raw_asp.png`. Set `ASP_BENCH_LEGACY=1` to
     keep the pre-adapter orchestrator for A/B. This *does* change the
     measurement path; the post-M1 ungated 97-run is the new baseline.
  3. **M1c** — convert GUI `_ProgressPipeline` to hooks over the same
     stages. Pass `exclusion_masks` into `_composite_foreground` (the GUI
     *does* call composite three times for HITL, but never forwards the
     captured masks; the canonical `run_stage.py:571` already does).
     **Landed 2026-08-15, then clarified after Chat/Claude review:**
     exclusion-mask plumbing is on both paths. Headless/no-op pause uses
     canonical `run()` (parity suite). Interactive HITL (`pause_cb` from
     the stitch worker) stays on the 9-checkpoint fork until M6. Only
     the `masks` pause is applied on the canonical path. Do not read
     this as "all HITL checkpoints moved." `ASP_GUI_CANONICAL=1` /
     `ASP_GUI_LEGACY=1` force a side.
- Make frame selection and output-safety policy explicit pipeline components.
- Parent Image-Toolkit `backend/controllers/backend_dispatch.py` is already a
  thin `AnimeStitchPipeline.run()` caller, **not** a fourth compositor. Keep
  it thin; give it the new policy/telemetry kwargs when the canonical API
  grows. Comparator scripts (`run_hugin.py`, `run_overmix.py`) stay outside
  the ASP compositor.
- Add headless parity tests: same input/config → same Raw ASP bytes and
  stage-trace digest for CLI, benchmark adapter, and GUI adapter (callbacks
  recorded, no `QApplication`).

Exit criteria:

- All three ASP entry points produce the same Raw ASP artifact and stage
  trace for the same manifest.
- No orchestration fork owns an image-changing stage.
- Video smart mode either selects or hard-fails; it must not swallow
  `TypeError` into uniform.
- Existing focused tests pass, and the new entry-point parity suite passes.
- After adapters land, run one ungated Raw ASP 97-case corpus (all three
  artifacts retained) and freeze it as the M2+ baseline. This is the first
  true Raw ASP scorecard; do not start M3 until it exists.
  **Harness 2026-08-15:** `ASP_BENCH_UNGATED=1` +
  `just bench::asp-benchmark-ungated`. Disables Composite/Ghost/SeamVis
  replacement and the env-gated align/coverage floors. Geometric
  no-edge/affine-invalid cases still have no raw file
  (`raw_asp_available=false`). The GPU 97-run itself is the remaining
  #30 work.
  **Chat review 2026-08-15:** per-case `safe_asp_counterfactual` is now
  persisted on the run JSON. Ungated run knobs are forced (not
  `setdefault`); the counterfactual policy is frozen product defaults.

No new default image-quality algorithm is introduced in M1.

### M2 — Safe ASP policy and observability

**Purpose:** ship a conservative result while Raw ASP is still improving.

Deliverables:

- Always retain a valid SCANS candidate alongside Raw ASP.
- Classify structural risk using interpretable signals and explicit uncertainty;
  an uncertain automatic decision selects SCANS by default and can be reviewed.
- Show the selected source and reason in the trace, report, and optional review
  screen.
- Audit existing gates against the completed human labels. Remove or demote
  signals that are inversely correlated with human quality (sharpness, edge
  energy, and current ghosting vs human ASP–SCANS delta are known inverse).
  **Done (2026-08-17), reproducible tool + evidence:**
  `.agent/reports/claude/m2_gate_signal_correlation_audit_20260817.md` /
  `backend/benchmark/audit_gate_correlation.py`. Reproduces the three cited
  numbers exactly (sharpness -0.47, edge_energy -0.53, ghosting -0.60) and
  maps them onto the actual gate code: **`GhostGate`'s only signal
  (`ghosting_score_v2`) is the worst-scoring inverse metric audited** — it is
  the concrete demotion candidate this bullet calls for.
  **Candidate implemented (2026-08-17, Grok), promoted to default
  (2026-08-17, Harbinger ACK):** `ghost_telemetry_only` /
  `ASP_GHOST_TELEMETRY_ONLY=1` records `telemetry_only_inverse_validated`
  and never rejects; SeamVis is not substituted. Offline replay of the
  2026-08-07 97-case run: 0 identity changes on the five-case screen, the
  structural red set, and all 97. There is no historic GhostGate-only
  fallback in this corpus. `product_safe_asp_policy()` now ships
  `ghost_telemetry_only=True` by default. See
  `.agent/reports/grok/m2_ghostgate_telemetry_screen_20260817.md`.
  `SeamVisGate`'s `seam_visibility_score` is confirmed correct (rho +0.43) and
  should be the reference "gate that works." `CompositeGate`'s
  `seam_coherence` component shows no signal (rho -0.06, not significant);
  its `strip_banding_score` component is now instrumented and audited
  (2026-08-17, Grok): `_compute_all_metrics` emits it again, and
  `audit_gate_correlation.py --recompute-missing` fills historical JSONs
  from dump panoramas. **rho = -0.417** (n=97 mixed-vintage dump;
  true-composite-only -0.365, n=43, p=0.016) — inverse / misleading, same
  family as GhostGate. CompositeGate currently has **no** audited-correct
  input. See `.agent/reports/grok/m2_strip_banding_audit_20260817.md`.
  Date-locked 2026-08-07 images only: n=12, rho=-0.525, p=0.08.
  **Candidate (2026-08-17, Grok):** `composite_sb_telemetry_only` /
  `ASP_COMPOSITE_SB_TELEMETRY_ONLY=1`. Default not flipped (26 identity
  changes; no raw-composite human labels on those cases). After `sb` is
  retired, CompositeGate has only no-signal `sc` (1 historic fire).
  Discriminating exit is **not met**: catastrophes 04/06/07/12/14/15 are
  Raw ASP under current SeamVis+Composite+Ghost as well. See
  `.agent/reports/grok/m2_composite_sb_and_discriminating_20260817.md`.
  **SeamVis retune also infeasible (2026-08-17):** 0 `(floor, ratio)`
  pairs catch all six catastrophes and keep test96 — test96 sv=32.2 >
  every catastrophe (binding test15 sv=12.55). Discriminating exit needs
  a new structural signal, not a threshold change. See
  `.agent/reports/grok/m2_seamvis_threshold_infeasible_20260817.md`.
  **#31 verification pass (2026-08-20, Claude):** re-confirmed against
  current code, not just the 2026-08-17 reports. `ghost_telemetry_only`
  is default `True` in `product_safe_asp_policy()` (`safety_policy.py`);
  `composite_sb_telemetry_only` is still default `False`, correctly left
  that way per the "no raw-composite labels on those 26 cases" reasoning
  above — this is an open decision for Harbinger, not a bug. `ASP_HOLD_BG_SUB`
  is registered in `_CONFIG_SCHEMA` (`config.py:58`) as Advanced/experimental/
  default-off, satisfying engineering rule #5's register-or-delete
  requirement. The ≤20-key default profile + Advanced-configuration-control
  surface is shipped — cross-repo, in Image-Toolkit's own
  `gui/src/components/dialogs/asp_advanced_config_dialog.py` (20-flag
  Primary tab, 73-flag Advanced Matrix tab) and
  `docs/website/src/components/config/AdvancedConfigDrawer.tsx` — not inside
  this submodule, consistent with this repo's documented cross-repo coupling
  (see `tools/bench/justfile`'s header comment). No code change needed for
  any of the above; all four already meet their bullets.
  **What remains open, blocked on Harbinger, not on further agent work:**
  (1) the CompositeGate "empty gate" redesign Grok recommended (`sb`
  demoted, `sc` has no signal, so the gate "should not stay alive for its
  own sake") is designed but not implemented, pending sign-off; (2) the
  discriminating-fallback bar Harbinger locked in §17 item 2 ("Always-SCANS
  is not M2 success") is proven infeasible under every combination of the
  three existing gates on this corpus (§17 item 2 vs the SeamVis-sweep
  finding directly above are now in tension — flagging, not resolving).
  Closing #31 requires one of: a genuinely new structural signal, an
  explicit HITL veto path, or Harbinger relaxing the §17 item 2 bar. None
  of those is a mechanical/config change, so none was attempted here.
  `cqas` (the former single-scalar aggregate used for GT-less cases in
  dashboards/reports) also fails the audit (rho -0.09, not significant) — its
  two largest-weighted components are the inverse `ghosting_siqe` (0.35) and
  no-signal `seam_coherence` (0.20); only `seam_visibility` (0.30) is pulling
  correct weight. No gate default changed yet — the actual demote/rework
  change still needs the promotion-ladder review (five-case → stratified →
  all 97) before landing, and `cqas`'s fix overlaps M2.5a (#32)'s
  per-defect-category work rather than duplicating it here. **Implemented
  2026-08-17:** new runs emit `cqas_v1_legacy` as an explicitly diagnostic-only
  historic field; it is excluded from radar ranking and automated verdicts.
- Record per-stage image geometry, frame provenance, pose provenance, gain
  residuals/clamps, seam feasibility, and fallback reason.
  **First slice landed 2026-08-17 (Grok):** `PipelineSession` now owns a
  typed envelope (`geometry`, `frame_provenance`, `pose_provenance`,
  `gain_telemetry`, `seam_feasibility`, `fallback_reason`) published as
  `artifacts["observability"]` from `finish()`. Canonical `run()` records
  canvas/crop sizes, kept/dropped source paths, BA pose rows, per-frame
  gain residuals/clamps from Stage 4.5, and seam corridor feasibility
  (`n_boundaries` / single-pose / max lum step / exclusion-mask count).
  Pixel path unchanged.
  **Slice 2:** pose `source`/`refined_by`; per-pass drop reasons;
  load/save geometry; bench JSON `observability` field.
- Consolidate experimental flags into typed named profiles. `ASP_HOLD_BG_SUB`
  is registered as an Advanced-only, default-off experimental key with its
  unaligned-median limitation documented; M4 owns the later keep/delete
  decision alongside its plate replacement.
  Default profiles expose ≤20 image-changing keys. An Advanced
  configuration control reveals the remaining registered parameters; it
  does not make them the implicit default.

Exit criteria:

- Safe ASP is rated no worse than SCANS on the structural red set **and**
  is *discriminating*: it must select Raw ASP on at least one known-good
  red-set case and SCANS on the known catastrophes. Always-SCANS satisfies
  "never worse" but is not an M2 success.
- Every automatic choice is reproducible and explainable from the manifest.
- Raw ASP statistics remain visible and unaffected by Safe ASP selection.

### M2.5 — Human-aligned quality metrics and benchmark analytics

**Purpose:** the sharpness/edge-energy/ghosting inverse correlation to human
judgment (§3, Spearman -0.47/-0.53/-0.60) has so far only been used to demote
those signals. This milestone investigates *why* they diverge and turns that
into measurement infrastructure, instead of leaving automated metrics as
permanently untrustworthy.

Depends on M0's per-output defect-annotation schema (comparison-level tags
like `ghosting=69` cannot be attributed to a specific artifact today). Starts
once M0 lands; runs in parallel with M1-M4. Analysis- and tooling-only — it
does not change any Raw ASP/Safe ASP algorithm default.

File as **two issues**, not one: **(a)** per-output defect analytics +
anime-adapted CV diagnostics + subset selection; **(b)** learned-proxy
feasibility spike. (b) waits for M0 **and** the post-M1 ungated Raw ASP
labels so fallbacks are not trained as ASP. `AnimeStitchNet` is a 4-DoF
alignment regressor, not a quality model — do not reuse it as the proxy.
Rerun.io / large stage dumps are opt-in developer artifacts, never a
`laptop_balanced` required dependency. **Phase 3 lock (Harbinger
2026-08-15): A+B** — `TelemetrySink` on `PipelineSession`, opt-in
`.rrd` sidecar (`desktop_quality` extra), plus OTel spans/metrics
(local OTLP/stdout first). Caption Rerun camera/point views as 2D-canvas
metaphor, not a pinhole reconstruct. Rerun WASM in `docs/website` is
rejected. A native JSON/NPZ inspector in M6/`/journal` is a **fully
optional, unscheduled** extra — not a M2.5/M6 deliverable. Parent
analytics Phase 2 still describes deleted RLHF reward models; correct
that document separately, do not inherit those claims here.

Deliverables:

- **Per-defect-category correlation/impact analysis.** Extend the existing
  corpus-wide correlation check to be per-defect-category and, where
  attributable, per-pipeline-stage: which failure classes (torn anatomy,
  banding, color shift, crop loss, ...) does each existing and new metric
  actually track, and which does it invert on? Use this to validate or
  challenge the locked structural-before-photometric (M3/M4 before M5)
  ordering with evidence, not just a priori reasoning — it does not
  override §5's locked sequencing on its own; a strong contradiction is an
  open question for Harbinger, not a silent reorder.
- **New/anime-adapted CV metrics.** Research and implement 2D-cel-adapted
  variants of existing picture-quality metrics (e.g. structure/edge metrics
  tuned for flat-color regions and line art rather than photographic
  texture), plus any new metric candidates. Every candidate is validated
  against human labels before it is trusted for anything beyond diagnostics;
  no metric becomes a promotion gate without that validation.
- **Learned proxy metric.** Train/fine-tune a model to predict human
  coherence from Raw-ASP-vs-SCANS pairs or single-output quality. Scoped as
  a **non-gating diagnostic proxy only** — revalidated against held-out
  human ratings every time the corpus grows. Report dataset size (currently
  97 cases, 43 true Raw ASP composites), train/validation split, and
  confidence explicitly; human review remains the sole release criterion
  per §1 regardless of this metric's performance.
  **Feasibility spike done (2026-08-20, Claude, issue #33):**
  `backend/benchmark/learned_proxy_feasibility.py` +
  `.agent/reports/claude/m2_5b_learned_proxy_feasibility_20260820.md`.
  Against the frozen post-M1 ungated 97-run (#30), only **27** cases are
  true Raw ASP composites with reviewed human labels (not 43 — that count
  was measured on the pre-M1 gated corpus; M1's canonical adapter changed
  which cases fall back). Per-feature Pearson r against human ASP coherence
  tops out at +0.328 (`color_entropy`); a leave-one-source-group-out ridge
  regression over all 9 existing `metrics_asp` signals scores **worse**
  than predicting the training-fold mean (RMSE 1.536 vs 1.058 baseline).
  **Verdict: not yet feasible** — both the sample size and the existing
  metrics' individual signal strength are binding constraints, not a
  modeling gap. Revisit once the SFW corpus (#38-41) grows the labeled set
  and/or M2.5a's anime-adapted metrics supply better-correlated features.
  The script is kept as the reusable revalidation tool for future corpus
  milestones, per this deliverable's own requirement.
- **Similarity-based benchmark subset selection.** Build visual/statistical
  similarity metrics between benchmark test cases (image-level) and between
  human quantitative scores plus defect/textual annotations (label-level) to
  derive representative mini-benchmark subsets for fast-iteration deltas
  without a full 97-case run, and to support subset selection scoped to a
  specific pipeline area (e.g. a seam-focused subset for M5 work). This
  **supplements** the existing manually-curated five-case smoke set and
  structural red set; it does not replace them yet. Replacing hand-curation
  with the data-driven method is a later, explicit promotion decision once
  the tooling's representativeness is measured and stable across corpus
  growth — not automatic on landing.

Exit criteria:

- Every existing and candidate metric has a documented per-defect-category
  correlation against human labels; none is used as a promotion signal
  without this.
- At least one opt-in data-driven subset-selection method exists alongside
  the manual smoke/red sets, with measured representativeness (how well its
  deltas predict full-corpus deltas).
- Any learned proxy metric reports its own validation correlation against
  human labels, is revalidated at each corpus milestone, and is documented
  as non-gating.

### M3 — BiRefNet single-pose compositor

**Purpose:** eliminate pose mixing by construction before adding a temporal
segmentation model.

Deliverables:

- Treat M3 as a **named default-off compositor candidate** (`coherence_v2` or
  similar profile key), not an in-place rewrite of
  `rendering/compositing/`. The current composite + HITL seam loop stays
  until promotion. Cite Critical Evaluation §9.2 Stage 2 (per-pixel
  same-phase background average; single-pose foreground) as the evaluated
  alternative. Existing `ASP_BG_AVERAGE` was measured harmful under the
  *current* (unaligned / mixed-phase) pipeline — that result does not
  reject §9.2; it forbids turning averaging on without phase grouping and
  alignment first.
- Use BiRefNet masks to distinguish background reconstruction from foreground
  ownership. Unload BiRefNet after Stage 4 (canonical `run()` already does)
  so `laptop_balanced` stays inside 12 GB.
- Assign each foreground region in an overlap to one source pose. Do not median,
  feather, or seam-blend competing character poses through the same region.
- Define a deterministic ownership/handoff policy using mask confidence,
  visibility, boundary truncation, frame quality, and temporal consistency.
- When no background seam corridor exists, select a single-pose handoff instead
  of applying an infinite cost to every possible seam path.
- Propagate ownership metadata to diagnostics and the optional review screen.
- Prove equivalent semantic-cost and exclusion-mask behavior in Python and C++,
  or delete one implementation.

  **First slice landed 2026-08-17 (Grok):** isolated
  `rendering/compositing/coherence_v2.py` — `plan_coherence_v2` assigns
  each connected FG overlap region to exactly one pose (coverage →
  confidence → index). All-foreground overlap →
  `has_background_corridor=False` and a single-pose handoff. Not imported
  by `composite.py`. Schema key `ASP_COHERENCE_V2` default 0. Tests:
  `test_coherence_v2.py`.
  **Slice 2:** `apply_coherence_v2` paints owned pixels from one source;
  N-frame fold is first-claim-wins. Opt-in hook in `_composite_foreground`
  behind `ASP_COHERENCE_V2=1` only.
  **Red-set screen (2026-08-17):** first apply punched holes (6/7 crop
  loss). Exclusive-keep fix: crop gate **0/7**. Still default-off; no
  human screen yet.
  **Ownership-factor slice (2026-08-20, Claude, #34):** `_pick_owner`
  extended past coverage/confidence with a weighted stage covering the
  remaining named factors — visibility (filled/bbox-area compactness of
  each candidate's *own* pose mask, not the shared overlap pixels: `pix`
  is by construction a subset of both `a` and `b`, so anything computed
  on `pix & a` vs `pix & b` is identical for both and can never
  discriminate — this was caught and fixed before landing, not shipped
  broken), boundary truncation (fraction of a pose's own bbox touching
  the frame edge), frame quality (Laplacian-variance sharpness of the
  source frame restricted to that pose's mask), and temporal consistency
  (`composite_coherence_v2`'s multi-pair fold now passes the
  already-`claimed` map as `prior_owner` to each subsequent pair, biasing
  contested regions toward staying with whichever side a neighboring fold
  already assigned). All four are additive/optional kwargs — every
  existing call site and test that only passes area/confidence is
  bit-for-bit unaffected; the weighted stage only engages when coverage
  and confidence are tied *and* the extra signals are supplied. 14/14
  `test_coherence_v2.py` cases pass (was 11, +3 new), plus all 277
  `backend/test/rendering` + `backend/test/core/pipeline` tests, no
  regressions.
  **Diagnostics propagation:** `composite_coherence_v2` now returns a
  third `claimed_meta` value (per-pair region ownership + reason codes),
  threaded through `seam_meta_out["coherence_ownership"]` in
  `composite.py` and forwarded into `session.note_seam_feasibility(...)`
  in `run_stage.py`, so it appears in `PipelineSession.observability()`
  alongside the other M2 first-slice fields — no parallel diagnostic
  surface invented.
  **C++ parity finding:** `coherence_v2`'s semantic-cost/exclusion-mask
  logic has **no C++ counterpart at all** — it never imports
  `rendering/compositing/_native.py`'s `base.compositing` extension, and
  `base/src/seam.cpp` has no `coherence_v2`/single-pose-ownership code.
  The roadmap's "prove equivalent Python/C++ behavior or delete one"
  rule is trivially satisfied: there is only one implementation, nothing
  to reconcile. (This is unrelated to rule #11's flag on the *existing*
  live-compositor's C++ `build_seam_cost_map(...exclusion_masks)` —
  that's the pre-existing seam-cost path used by the default HITL loop,
  not coherence_v2, and stays exactly as already flagged: incomplete,
  repair-or-remove, out of scope here.)
  **Structural red-set rerun (2026-08-20, Claude, post-cooldown):** ran
  `structural_red_v1` (14 cases) with `ASP_COHERENCE_V2=1` after CPU temps
  returned to 69°C. No crashes, no errors, no torn-output warnings across
  the corpus. However **only 5/14 cases (asp_test06/08/28/41/97) actually
  reached the `coherence_v2` compositor path** — the other 9 fall back to
  SCANS via earlier, unrelated gates (`disconnected_edge_graph` ×7,
  `affine_invalid` ×2) that run *before* compositing is ever reached, so
  they exercise none of today's ownership-factor code. This confirms no
  **P1 plate + single-pose compositor candidate (2026-08-23, Antigravity):**
  `_plate_compositor.py` implements the first renderer priority chosen by Harbinger
  ("one coherent character pose, then seam cleanup") behind default-off
  `ASP_PLATE_SINGLE_POSE=1`. Builds a clean canvas-aligned background plate from
  all frames' confirmed background pixels via joint gain equalization and robust
  temporal nanmedian, segments foreground character cels per frame, and assigns
  exactly one hero pose per connected overlap zone (maximizing area, centrality,
  and boundary completeness) with soft-edge feathering over the clean plate.
  Registered in `_CONFIG_SCHEMA` and wired into `composite.py`. Unit tests in
  `test_plate_compositor.py` (3 passed).
  **P2 edge-preserving & multi-band seam cleanup (2026-08-23, Antigravity):**
  Added edge-preserving sharp background source selection in multi-sample zones
  (`ASP_PLATE_EDGE_PRESERVE=1`) and multi-scale Laplacian pyramid blending
  (`ASP_PLATE_MULTIBAND=1`) over the clean background plate to resolve blur,
  seam lines, and banding while protecting high-frequency line art and character
  crispness. Unit tests in `test_plate_compositor.py` (4 passed).



Exit criteria:

- No torn anatomy, duplicated limbs/strips, or mixed-pose foreground on the
  structural red set.
- The five-case smoke set has no regression.
- A human screen confirms the candidate before promotion to a larger slice.

SAM2 is explicitly out of scope for this milestone.

### M4 — Separate camera trajectory from cel-pose selection

**Purpose:** choose frames that progress across the background without mixing
animation phases.

Deliverables:

- Estimate camera motion from aligned/background evidence rather than raw
  whole-frame appearance. M4 must evaluate §9.2 Stage 0 (phase-group **before**
  alignment) as a named alternative, not only "hold detect then BA."
- Detect held cels only after compensating for background pan. A held character
  pose must never force camera displacement to zero.
- Replace `_estimate_background_plate()` (pixelwise median of **unaligned**
  cropped thumbnails in `_hold_detection.py`) with an aligned or
  motion-compensated plate. The current function cannot be a background plate
  under a real pan.
- Integrate, measure, or remove `_select_hold_keyframes_dp` (exported and
  unit-tested, never called by `smart_select_frames` or `run()`). Hidden
  `ASP_HOLD_BG_SUB` is registered or deleted in the same change.
- Use test 14's manual frame selection as an oracle: first reproduce its result,
  then measure which automatic decisions diverge.
- Keep pose-path safety constraints, but log every veto and substitution.
- Repair translation+scale geometry before further A/B work: edge observations
  must contain measured scale information, and a synthetic test must recover a
  known scale. Otherwise remove the model from candidate profiles.

Exit criteria:

- Real spatial-pan fixtures distinguish camera movement from held cel motion.
- The structural red set improves without increasing crop loss.
- Frame provenance explains every chosen pose and camera-progress step.

### M5 — Seam and photometric refinement

**Purpose:** address seams, banding, and colour/exposure only after structural
ownership is safe.

Deliverables:

- Route seams through feasible background corridors and use the M3 handoff when
  no corridor exists.
- Compare the current overlap-graph gain solver with HSV-value and LAB
  alternatives on the same saved artifacts.
- Log graph connectivity, rejected observations, condition/residual changes,
  clamp counts, and affected frames.
- Reject global outlier logic that removes legitimate smooth exposure changes.
- Keep foreground correction separate from background normalization unless a
  measured experiment justifies coupling them.

Exit criteria:

- No new structural defect on the red set.
- Banding, seam, and color scores improve in output-specific human annotations.
- The larger staged corpus confirms the improvement; automated gains alone are
  insufficient.

### M6 — Optional artist review and final promotion

**Purpose:** make uncertainty controllable without making manual review
mandatory.

Functional review-screen scope (both modalities share one contract):

- **M6a first:** one replayable session schema (extend
  `hitl_session.py` / `.asp-session.json`), plus a headless replay that
  reproduces the edited composite from the manifest alone.
- **M6b / M6c:** PySide6 dialog (benchmark evaluator + desktop stitch tab)
  and `docs/website/` inspector are views over that schema. Do not give
  them independent state models.
- Compare Raw ASP, Safe ASP, and SCANS without ambiguous labels. Gemini's
  tri-view (debug overlays on Raw, fallback badge on Safe, SCANS reference)
  is the functional layout spec.
- Show why Safe ASP selected its output.
- Allow optional frame/pose selection, single-pose ownership/handoff, seam
  override, and final output selection.
- Save edits as a replayable manifest rather than mutating undocumented state.
- Default to the automatic Safe ASP path when the screen is skipped.
- Native layered export is an M3/M4 *output contract*, not only an M6 UI
  feature: write a clean background plate + per-cel RGBA. First-ship format
  is a PNG layer stack + JSON sidecar. PSD is a later adapter, not an M3
  blocker.

Promotion ladder:

1. Unit, parity, and real-pan fixture tests.
2. Existing five-case smoke set.
3. Structural red set plus human screen.
4. First full Raw ASP baseline with all fallback gates disabled
   (scheduled immediately after M1 adapters, not at the end of the plan).
5. **Structural milestone:** no Raw ASP 0/4 scores and no hard structural defect
   on the red set.
6. **Corpus safety milestone:** no Raw ASP score below 2/4; Raw ASP is non-losing
   on at least 75/97 cases.
7. **Parity milestone:** Raw ASP mean exceeds SCANS and is non-losing on at least
   90/97 cases.
8. **Final milestone:** Raw ASP is non-losing on 97/97 and has a strictly higher
   mean score than SCANS.

Any default change requires the relevant human screen and a full-97 run. A
later milestone cannot waive an earlier structural requirement.

## 6. Resource and configuration profiles

### `laptop_balanced` — required default target

- Target: RTX 4080 Laptop GPU with 12 GB VRAM and 32 GB RAM.
- Must complete without GPU or host OOM.
- Record peak VRAM, peak RSS, wall-clock time, model load time, and fallback
  behavior per case.
- Use bounded batches, explicit model lifecycle/offload, and deterministic
  degradation when a requested option exceeds the budget.

### `desktop_quality` — higher-quality target

- Target: RTX 3090 Ti-class system or better.
- May trade additional compute and memory for higher-resolution masks, larger
  search windows, or stronger optional models.
- Must preserve the same artifact schema and safety guarantees.

### `custom`

- Expose typed, validated parameters inherited from a named profile.
- Persist the effective values in the experiment/project manifest.
- Do not add a parallel collection of undocumented `ASP_*` switches.

M0 establishes measured runtime baselines before setting a hard wall-clock
ceiling; avoiding OOM on `laptop_balanced` is immediately mandatory.

## 7. Future work — explicitly not current priority

### Temporal/heavier segmentation

After M3 proves the BiRefNet single-pose contract, evaluate SAM2 or a comparable
tracker only for a named deficiency such as temporal mask continuity. It must use
the same ownership interface, pass the 12 GB profile or declare itself
`desktop_quality`-only, and beat the BiRefNet baseline under human review.

### Visual redesign

Gemini leads a future visual pass for the comparison/review workflow after its
functional contracts stabilize. Possible work includes clearer pose ownership,
seam-risk, and frame-provenance visualization. Visual redesign must not block M0
through M5 and must not change benchmark semantics.

### SFW benchmark corpus

Tracked separately in
[`asp_sfw_corpus_roadmap_2026q3.md`](asp_sfw_corpus_roadmap_2026q3.md) — a
non-blocking generalization check, not a current M0–M6 dependency. Its
validation pass (C2) is sequenced after M1 for measurement-validity reasons,
not as an M1 requirement.

### Account-linked settings sync (pointer only)

Not a current ASP priority — this is a parent Image-Toolkit feature. Full
draft in the parent repo's `docs/moon/roadmaps/new_features.md` §4.19. Noted
here only because ASP has its own settings/config surface (73 `ASP_*` flags,
`dump_sfw/` root paths, etc.) that could plausibly want the same cross-device
sync later — no design work has gone into that ASP-specific angle, this is
purely a pointer so the idea isn't lost.

## 8. Team workflow and evidence handoff

- **Claude:** team lead; decomposes milestones into GitHub issues, records
  dependencies and acceptance criteria, and closes work only after review.
- **Grok:** primary implementation owner for code tasks.
- **Gemini:** design/art lead; owns review-workflow design and later visual work,
  and advises on visual defect interpretation.
- **Chat/Codex:** reviewer; compares implementation, tests, saved artifacts,
  changelog, and roadmap claims, repairs bounded integration details when
  appropriate, and reports acceptance or remaining gaps to Claude.
- **Harbinger:** final product/quality authority for human comparisons and
  roadmap ambiguity.

For each issue, the implementation handoff must state:

- hypothesis and affected failure class;
- files and defaults changed;
- exact tests and experiment manifests;
- Raw ASP, Safe ASP, and SCANS results separately;
- resource use under the applicable profile;
- roadmap/changelog entries updated;
- known limitations and whether the candidate is enabled, gated, rejected, or
  removed.

Chat/Codex verifies those claims against the actual diff and artifacts, posts the
result to the agent bus, and tells Claude what is safe to delegate next.

## 9. Immediate issue order for Claude's final review

1. M0 result/artifact schema, preference-complete baseline freeze, and
   relabel of the 2026-08-07 run into raw/safe/scans.
2. M0 experiment-manifest + resource telemetry (greenfield) and the layered
   synthetic pan/hold fixture generator.
3. M1a shared stage protocol / `PipelineSession` (no behavior change).
4. M1 video smart-selection signature fix plus a test that proves smart
   mode did not silently fall back to uniform.
5. M1b benchmark adapter (move Composite/Ghost/SeamVis into an injectable
   policy; keep Raw ASP).
6. M1c GUI adapter: exclusion-mask / motion-model / HITL overrides through
   the canonical call; headless parity suite.
7. Post-M1 ungated Raw ASP 97-run (three artifacts, laptop_balanced
   telemetry). Freeze as the M2+ baseline before any compositor candidate.
8. M2 safety-policy extraction, inverse-metric demotion, default-profile
   surface of ~20 keys plus an Advanced configuration control, register-or-
   delete `ASP_HOLD_BG_SUB`.
9. M2.5a per-defect-category correlation/impact analysis, anime-adapted CV
   diagnostics, and data-driven subset-selection tooling. Starts once M0's
   schema lands; runs in parallel with M1–M4. M2.5b learned-proxy spike
   only after the post-M1 ungated Raw ASP labels exist. Neither issue
   changes a default or gates C2.
10. M3 default-off BiRefNet single-pose / §9.2 compositor candidate.
11. M4 motion-compensated hold/selection, test-14 oracle, repair-or-remove
    translation+scale and the unused DP selector. C++ items stay blocked
    until `base` rebuilds.
12. `ASP_POSE_WINDOW_PX=80` measured human screen (after M0–M2, before M5).
13. Only then: M5 photometric/seam candidates (informed by M2.5's per-defect
    correlation findings where available); M6a session schema; M6b/c
    inspectors; PNG+JSON layered export if not already emitted by M3/M4;
    PSD adapter later.

No implementation issue should describe the unintegrated DP selector,
translation+scale solver, or backend-divergent hard seam veto as completed work.

## 10. Claude's review pass (2026-08-15)

I re-verified Chat/Codex's most consequential code claims directly against the
current tree rather than trusting the bus log, and cross-read the older
research/evaluation docs for anything material the M0–M6 plan doesn't yet
carry forward. Findings below; §11 has open questions for Harbinger before
this draft is final.

### 10.1 Verified — orchestration divergence (M1's premise holds)

- `AnimeStitchPipeline` (`backend/src/core/pipeline/manager.py:33`) composes
  `_RunStageMixin.run()` (`backend/src/core/pipeline/run_stage.py:108`). Grepped
  `run_stage.py` for `Gate|smart_select|frame_select`: only one hit, an
  unrelated import. **Confirmed:** the canonical path has no
  Composite/Ghost/SeamVis gate and does not call frame selection itself.
- `backend/benchmark/bench_anime_stitch.py:1890-2007` owns `[CompositeGate]`,
  `[GhostGate/siqe]`, and `[SeamVisGate]` directly, and calls
  `smart_select_frames()` itself (`bench_anime_stitch.py:1203-1328`) rather than
  going through the pipeline. **Confirmed as described.**
- `gui/src/helpers/_progress_pipeline.py` sets `self.exclusion_masks` from an
  override dict (`_progress_pipeline.py:386-389`) but the file has **zero**
  matches for a `composite(`-style call — the masks are captured but there is
  no visible call site in this file that forwards them into rendering.
  **Confirmed:** GUI mask propagation is at minimum unverifiable from this
  file alone, consistent with the draft's claim, not weaker.
- No test file matches a benchmark-vs-GUI-vs-pipeline parity pattern
  (`grep -rl "parity" backend/test gui/test` returns nothing relevant). **M1's
  "add parity tests" starts from zero, not from a partial harness.**
- Manifest/telemetry was greenfield as of 2026-08-15. **Landed 2026-08-20
  (#46):** `TelemetrySink` on `PipelineSession`, OTLP-JSONL/stdout, RSS via
  psutil, VRAM via `torch.cuda.max_memory_allocated` (omitted when no
  CUDA). Canonical `run()` still does not import `opentelemetry` or `rerun`.

### 10.2 Corrected — flag count and `translation_scale`

- Actual `ASP_*` flag count in current source (`grep -rho ASP_[A-Z0-9_]+
  backend/src gui/src | sort -u | wc -l`): **73**, not the draft's cited ~67.
  The gap is small but the trend matters more than the exact number — see
  §11.1 below on the budget itself.
- `translation_scale` is real and wired (`backend/src/alignment/bundle_adjust.py:425-479`,
  `manager.py:67`), and the C++ side (`base/src/bundle_adjust.cpp:162,227-233`)
  confirms the draft's claim exactly: `dof=3` includes a scale term, but its
  only constraint is a regularization prior pulling every non-anchor frame's
  scale toward 1.0 (`reg_scale`) plus relative-equality across frames — there
  is no Jacobian row built from an actually-observed scale ratio between
  matched edges. **This is a scaffold that always reports scale≈1 unless
  something else moves it, not a measured estimator. Confirmed as described.**

### 10.3 Gap — the older Critical Evaluation isn't cross-linked, and two of its structural recommendations aren't in M0–M6

`docs/reports/ASP_Critical_Evaluation_2026-07-08.md` §9.2 ("coherent by
construction, enhanced where safe") is the direct intellectual ancestor of
this draft's M3 (single-pose ownership) and M4 (camera/cel-pose separation),
but two of its five stages aren't explicitly present in M0–M6 and should be
named as evaluated options rather than silently dropped:

- **Stage 0 — animation-phase grouping *before* alignment**, not concurrent
  with it. The evaluation calls this "the single most important unimplemented
  idea" (§8, Overmix lesson 2) and M4 as drafted separates camera trajectory
  from cel-pose selection but doesn't sequence phase-grouping as a hard
  precondition to alignment.
- **Stage 2 — per-pixel background reconstruction via median/mean over
  same-phase frames**, replacing seam-and-blend for background regions
  entirely (no seam artifact by construction, not by gate). M5 as drafted
  still frames background work as "seam routing/photometric refinement,"
  which is compatible with but narrower than this alternative.

I'm not proposing to adopt these as requirements — the locked decision to
converge orchestration first (M1) before adding algorithms is sound and I
agree with it. But M3/M4 issue writeups should cite §9.2 explicitly and state
whether phase-grouping-first and averaging-based background reconstruction
were considered and why the chosen design differs, so the next agent doesn't
independently rediscover this document.

### 10.4 Gap — Anti-Goals and prior-failure history aren't carried forward

The older `docs/moon/ROADMAP.md`'s Anti-Goals section and its §5.1 history are
not referenced anywhere in this draft, and this draft's §"Locked planning
decisions" says nothing about them even though it states this roadmap "takes
priority" when the two differ:

- RLHF-style reward modeling and DRL-based super-resolution were built,
  measured, and deleted in the S200 "great trim" as unverified complexity
  (old `ROADMAP.md` §5.1). Nothing in M0–M6 proposes reattempting them, but
  worth an explicit pointer so a future issue doesn't reintroduce them cold.
- The old roadmap's Anti-Goals ("no new gate without displacing an old one
  and a full-corpus run," "no default-OFF flags shipped without the A/B
  scheduled in the same session," "no threshold-tuning sessions") restate
  almost exactly what this draft's §4 rules 3 and 5 already say — good, they
  agree — but §4 doesn't say a flag must *displace* an old one, only that old
  flags should be "retired or consolidated." Recommend tightening §4 rule 5 to
  match the stricter displacement rule, since 73 live flags is already past
  where the discipline was supposed to cap it (see §11.1).

### 10.5 Evidence worth citing directly in M0/M1 issues

- `docs/moon/CHANGELOG.md` records `ASP_POSE_WINDOW_PX=80` (DINOv2
  pose-consistent frame selection) producing the **first `asp_better` verdict
  ever recorded from a flag in this project's history** (1/1/3 vs the prior
  0/0/2, mixed GT-SSIM but improved sharpness/ghosting) — flagged there for
  human coherence review, never rated. This is a concrete, already-implemented
  candidate for an early M0/M1-adjacent experiment once the manifest/parity
  work lands, not a new idea to design from scratch.
- The same changelog documents issue #10: `ASP_USE_SAM2` had **zero effect**
  on `bench_anime_stitch.py`'s scored output for a stretch of time because the
  benchmark called a raw non-pipeline-aware masking function directly instead
  of routing through the pipeline's flag-aware masking. This is the exact
  failure class M1 is trying to close, already happened once, and already
  shipped a false measurement — good precedent to cite in the M1 issue as
  "why this milestone exists," not a hypothetical risk.

## 11. Open questions for Harbinger (Claude, pending Gemini/Grok passes)

1. **Flag budget.** This draft's §4 rule 5 references "the existing
   configuration budget" without a number; the earlier bus discussion assumed
   roughly 50; the 2026-07-08 Critical Evaluation recommends capping at ~20
   given the project's own history of 387 flags becoming unreasonable. Actual
   count today is 73. Should M2's flag audit target ~50 (retire down to
   current-ish scope) or the stricter ~20 (aggressive consolidation), and
   should §4 rule 5 require a new flag to displace an old one rather than just
   "retire or consolidate before exceeding budget further"?
2. **Structural rebuild alternative.** Should M3/M4 issues be required to
   explicitly evaluate the Critical Evaluation's phase-grouping-first and
   per-pixel background-averaging design (§10.3 above) as a named alternative,
   or is that considered out of scope relative to the locked "converge
   orchestration, then extend the existing pipeline" decision?
3. **`ASP_POSE_WINDOW_PX=80` fast-follow.** This is the only flag in the
   project's history with a measured `asp_better` signal, still unrated for
   human coherence. Should it get a standalone experiment slot right after M0
   lands (using the new manifest/dev-slice infrastructure), rather than
   waiting in the general M5 backlog?

Handing off to Gemini next for the design/review-workflow and art-adjacent
portions of this plan, per Harbinger's requested review order (Claude → Gemini
→ Grok → joint final review).

## 12. Gemini's review pass (2026-08-15)

As Design and Art lead, I reviewed the ASP codebase, benchmark defect distributions from Harbinger's completed 97-case human pass, and the artist review workflow requirements (M6).

### 12.1 Structural & Visual Defect Root Causes
- **Phase Mixing vs. Clean Background Plates (Weighing in on Claude's Q2):**
  The dominant human-visible defects reported across the 97 tests are **Ghosting (69/97 comparisons), Seams (52), Banding (45), Color Shift (41), and Torn Anatomy (26)**. In 2D cel animation, camera panning and character animation occur on separate exposure sheets (cels). Conflating them by running unconstrained homographies or blending across different character animation phases guarantees torn anatomy and duplicate limbs.
  - **Verdict on §9.2:** Separating camera trajectory from cel animation and reconstructing the static background plate via temporal median/mean *prior* to compositing foreground cels is conceptually essential. M3/M4 must explicitly incorporate this architecture.
- **Seam Routing Feasibility & Alpha Bleed:**
  Seam cuts currently force an artificial boundary through foreground cels when no background corridor exists, causing bisected torsos and limbs. An explicit single-pose handoff (winner-take-all cel assignment per overlap region) must be enforced by construction before seam optimization runs.

### 12.2 Artist Review Workflow & Visual Diagnostics (M6 Design Specs)
- **Tri-View Split Inspection:**
  The review screen should display side-by-side or synced pan/zoom views of:
  1. **Raw ASP** (with optional debug overlays: BiRefNet mask contours, seam paths, and pose provenance boundaries).
  2. **Safe ASP** (with active fallback indicator badge and explainability tooltip).
  3. **SCANS Comparator** (baseline reference).
- **Interactive HITL Override Controls:**
  - **Cel-Pose Selector:** Dropdown/scrubber per overlap zone allowing the artist to pick which frame's cel pose owns the region.
  - **Seam Corridor Adjuster:** Draggable anchor points to guide graph-cut seam corridors away from critical artwork details.
  - **Provenance & Defect Inspector:** Clickable region highlighting which source video frames contributed to specific canvas pixels.
- **Replayable Manifest Export:** All user adjustments must serialize to a lightweight `.asp-session.json` manifest so review decisions can be version-controlled, re-rendered headless, and included in regression suites.

### 12.3 Art & Asset Pipeline Parity
- **Lossless & Multi-Layer Export:** Support exporting separate Background (reconstructed panorama plate) and Foreground (segmented character cels with alpha channels) layers for game asset pipelines (e.g. parallax scrolling in mobile games or VN scene backgrounds).
- **Synthetic Ground-Truth Fixtures:** Create synthetic multi-layer panning animation test suites (static background + moving multi-frame sprite cels) to provide mathematically ground-truth benchmarks for translation, hold detection, and single-pose segmentation.

## 13. Open questions for Harbinger (Gemini)

1. **Review Screen Depth & Modality:**
   For the optional artist review screen (M6), should this be implemented as an embedded PySide6 dialog inside the existing desktop GUI (`gui/src/windows/`), or should it also be accessible as a web-based inspector in `docs/website/` leveraging our new Optic Lab / Blueprint design system?
2. **Layered Asset Output:**
   In addition to the flattened stitched panorama, should ASP natively support exporting multi-layer PSD / multi-layer PNG assets (separated clean background plate + individual segmented foreground character cels with alpha) for game/animation asset workflows?
3. **Synthetic Ground-Truth Benchmark Fixtures:**
   Would you like us to generate a set of procedural synthetic anime-style scrolling benchmarks (with known ground-truth camera velocity and known cel holds) in M0/M1 to mathematically verify alignment and hold detection before evaluating on real anime clips?

Handing off to Grok next for the code implementation and engineering feasibility review pass.

## 14. Harbinger's decisions on Claude & Gemini review passes (2026-08-15)

Harbinger reviewed the questions from Claude (§11) and Gemini (§13) and locked the following decisions:

1. **Flag Budget (Q1.1):**
   Target a hard budget of **~20 active flags**. Maintain this by replacing an old flag with a new one for most cases, allowing the 20-flag budget to grow organically only as new flags demonstrate genuine improvements on specific test cases while being worse on others compared to existing flags.
2. **Structural Rebuild Architecture (Q1.2 / Q2.1):**
   Approved. M3/M4 will explicitly explore the Critical Evaluation §9.2 architecture (phase-grouping-first and per-pixel background-averaging reconstruction before foreground single-pose compositing).
3. **`ASP_POSE_WINDOW_PX=80` Prioritization (Q1.3):**
   Approved for scheduling before M5, but after M0/M1/M2 once the canonical pipeline and telemetry harness are established.
4. **Dual Review/Inspector Modality (Q2.1):**
   Approved dual implementation: a **web-based inspector** in `docs/website/` (leveraging the Optic Lab / Blueprint design system) and a **PySide6 dialog** in the desktop GUI / benchmark evaluator.
5. **Multi-Layer Asset Export (Q2.2):**
   Approved. ASP will natively support exporting layered assets (clean background panorama plate + segmented foreground character cels with alpha channels) for game/animation asset workflows.
6. **Synthetic Ground-Truth Fixtures (Q2.3):**
   Approved. Procedural synthetic anime scrolling benchmarks (with ground-truth camera velocity and cel holds) will be generated in M0/M1 to mathematically verify alignment and hold detection.

**Handoff:** Grok's implementation-feasibility pass is in §15. Open questions for Harbinger are in §16. After those answers, Claude does the joint final review and issue decomposition.

## 15. Grok's implementation-feasibility pass (2026-08-15)

I re-read the current tree (not just the bus), the completed 97-case file, the 2026-08-07 corpus report, Critical Evaluation §9.2, and the three runners. Role: I will be the one implementing M1–M5, so this pass is about what is buildable in what order, what is already false-complete, and what would blow a single issue.

### 15.1 Verified — the three runners really diverge

| Path | Lines | Frame selection | Output gates | `exclusion_masks` into composite |
|---|---:|---|---|---|
| `AnimeStitchPipeline.run()` (`run_stage.py`) | 683 | no | coverage / geometric only | **yes** (`run_stage.py:571`) |
| `bench_anime_stitch.py` | 4045 | own `_smart_select_frames` | Composite / Ghost / SeamVis | N/A (own render path) |
| `_ProgressPipeline.run()` | 833 | no (caller preselects) | HITL pauses, no Safe ASP policy | **captured, not forwarded** |

`_ProgressPipeline` *does* call `_composite_foreground` three times (initial / seam-override / paint-mask loop). Claude's "zero `composite(`-style call" was true of a name search; the real bug is narrower and still load-bearing: `self.exclusion_masks` is set at lines 386–389 and never passed into those three calls. Canonical `run()` already passes them. M1c is a kwarg + parity test, not a missing composite stage.

Parent `backend/controllers/backend_dispatch.py` constructs `AnimeStitchPipeline` and calls `run()`. That is a fourth *entry point*, not a fourth compositor. Keep it thin.

### 15.2 Verified — Chat's five correctness findings still stand

1. **Video smart-select is dead on arrival.** `video_ingestion.py:344` calls `smart_select_frames(proxy_imgs, target_n=want)`. `selector.py:76` is `smart_select_frames(frames_paths: list[str], ...)` with no `target_n`. The `except Exception` at 356 swallows the TypeError and uniform-selects. First M1 bugfix; cheap; do not bundle with the GUI rewrite.
2. **`translation_scale` is a regularizer, not an estimator.** Python sends only `dx`/`dy` (`bundle_adjust.py:444–447`). C++ 3-DoF pulls `s_j - s_i → 0` plus an identity prior. No measured scale residual. Synthetic recovery test does not exist. M4: repair (add a scale observation from bg-masked match spans) or delete from candidate profiles.
3. **Unaligned "background plate."** `_estimate_background_plate` median-stacks cropped thumbnails with no warp (`_hold_detection.py:73–105`). Under a pan this is a smear, not a plate. `ASP_HOLD_BG_SUB` gates it via raw `os.environ` and is **not** in `_CONFIG_SCHEMA` (67 registered keys; this is the one hidden read I found in `backend/src`).
4. **DP hold selector is unused.** `_select_hold_keyframes_dp` is exported and unit-tested; `smart_select_frames` / `run()` never call it. Treat as dead until M4 integrates or deletes it.
5. **Hard-veto backends disagree.** Python `_seam_cut.py:33` adds `1e6 * (sem_cost > 0.1)`. C++ `seam.cpp` accepts `exclusion_masks` (1e9 barrier) but Python's exclusion path builds the cost map in Python (`_seam_cost.py`) and never reaches that C++ argument. An all-foreground overlap has no feasible corridor; M3 must hand off to a single pose instead of an infinite-cost grid.

Focused tests still cannot substitute for entry-point parity. I did not re-run the 300-test slice this pass; Chat's 2026-08-15 result stands until the next implementation session.

### 15.3 Evidence update — 97-case file is preference-complete

`asp_evaluations_20260810.json` now has a preference on every record (tests 27/39 → `simple`, 56 → `asp`, 91 → `tie`, timestamps 2026-08-15). Score-order is still 10 / 38 / 49; means 2.010 vs 2.639. Comparison-level defect tags (not output-specific):

| Tag | Count |
|---|---:|
| ghosting | 69 |
| crop_loss | 64 |
| color_shift | 50 |
| torn_anatomy | 48 |
| seam_line | 44 |
| banding | 39 |
| misordered_content | 36 |
| blur | 33 |
| duplicated_strip | 33 |
| geometry_warp | 28 |

These counts still cannot be attributed to ASP alone. M0's per-output annotation schema is the fix; do not mine these integers as if they were ASP-only.

The 2026-08-07 run remains the last full-corpus execution: 43 true composites (mean human 1.326) vs 54 safety fallbacks (mean 2.556). Aggregate "ASP" is mostly Safe ASP. M0 should *relabel* that run. A new ungated Raw ASP 97-run is ~2.5–3 h on a 3090 and longer on the laptop; it belongs on the promotion ladder, not in M0's exit.

### 15.4 Sequencing that I will refuse to invert

1. **M1 is a rewrite of two large forks, not a flag flip.** One PR that touches `bench_anime_stitch.py` and `_ProgressPipeline.run()` together is how we get another issue-#10 false measurement. Split M1a/M1b/M1c as now written in §5.
2. **§9.2 is a second compositor, default-off.** In-place replacement of `rendering/compositing/` while HITL still has a three-pass seam loop will break M6 and the GUI in the same week we cannot measure. Harbinger approved *evaluating* §9.2; evaluation means a named candidate with a manifest, not deleting the current path.
3. **`ASP_BG_AVERAGE` already exists and was marked harmful** on the mixed-phase / unaligned path. M3/M4 issues must cite that result. Averaging after phase-group + alignment is a different experiment.
4. **M2 "Safe ASP ≥ SCANS on the red set" is free if Safe ASP may always pick SCANS.** I tightened the exit to require a discriminating policy. Confirm in §16.
5. **M6 dual UI without M6a is two products.** `hitl_session.py` already serializes override dicts including ndarrays. Extend it. Web and Qt must replay the same file.
6. **Layered export is produced by M3/M4, displayed by M6.** First-ship PNG + JSON sidecar; PSD is an extra dependency (`psd-tools` or ImageMagick) and should not gate the compositor.
7. **C++ work is gated on a healthy `base` build.** Chat already hit a stale extension / missing OpenCV 4.6 dev config. translation+scale and C++ seam parity cannot be "done" on a stale `.so`. M1 should include a documented `just` rebuild check; if this machine cannot build `base`, C++ issues stay blocked and must not be silently Python-only.
8. **Existing synthetics are not the M0 fixture.** `generate_samples.py` makes unlayered gradient/pattern scrolls for the onboarding wizard. M0 needs a 2-layer generator (static plate + held RGBA cels, known `dx`/`dy` and hold IDs).
9. **Do not reintroduce RLHF / DRL-SR.** S200 deleted them. Nothing in M0–M6 should grow a reward model.
10. **`ASP_POSE_WINDOW_PX=80` stays after M0–M2** per Harbinger. I will not sneak it into M1b as a default.

### 15.5 What I changed in this document

- Locked engineering rules 8–11 (incremental M1, reuse HITL schema, default-off rebuilds, no false-complete scaffolds).
- M0: relabel existing 2026-08-07 artifacts; layered synthetics; no new 97-run as exit.
- M1: M1a/b/c split; video TypeError called out; GUI mask bug restated accurately; parent dispatch classified as thin wrapper.
- M2: discriminating-policy exit; 67 schema keys + one hidden flag; "20 active" = shipped-profile surface.
- M3/M4: §9.2 as default-off candidate; cite `ASP_BG_AVERAGE`; unaligned plate and unused DP named with file paths.
- M6: schema-first; layered export as compositor output.
- §3: preferences are complete.
- §9: issue order matches the split above.

### 15.6 Effort sketch (so Claude can cut issues)

| Slice | Why it is that size |
|---|---|
| M0 schema + relabel | Small. JSON + dashboard generator. No GPU. |
| M0 manifest + telemetry | **Landed 2026-08-20 (#46).** RSS/VRAM without breaking headless CI. |
| M0 layered synthetics | Small–medium. New generator; do not extend `generate_samples.py` into a second API. |
| M1a protocol | Medium. Extract only. Highest "don't change pixels" risk. |
| M1 video fix | Small. One signature + one integration test. |
| M1b bench adapter | Large. 4k-line script. Must keep Raw vs Safe artifacts. |
| M1c GUI adapter | Large. Crash-sensitive HITL loop. Headless parity, not Qt e2e. |
| M2 policy + flag audit | Medium. Logic is small; retiring 47 flags without breaking A/B history is the work. |
| M3 compositor candidate | Large. Default-off. Laptop VRAM contract is part of the issue. |
| M4 selection / scale | Medium–large after M1. Blocked on `base` rebuild for scale. |
| M6a schema | Small if we extend `hitl_session.py`. |
| M6b + M6c UIs | Medium each. Gemini owns visual; I own replay/headless. |

## 16. Open questions for Harbinger (Grok)

These are the only remaining ambiguities that change how I would cut the first implementation PRs. Claude's and Gemini's questions are already answered in §14.

1. **M1 cut.** Approve the M1a (protocol) → M1b (benchmark adapter) → M1c (GUI hooks) sequence, or do you want one unification issue? I will not implement a single-PR rewrite of both forks unless you override this.
2. **M2 bar.** Is "discriminating fallback" (must pick Raw ASP on a known-good and SCANS on catastrophes) the M2 exit, or is "Safe ASP never worse than SCANS" enough even if that means always-SCANS?
3. **§9.2 landing.** Confirm default-off second compositor (`coherence_v2`) versus in-place replacement of `rendering/compositing/` during M3.
4. **M0 GPU time.** Relabel the 2026-08-07 artifacts in M0 and defer a new ungated Raw ASP 97-run until the promotion ladder (M6.4), or spend a 3-hour 3090 / longer laptop run now?
5. **Layered export first format.** PNG layer stack + JSON sidecar in M3/M4, with PSD as a later adapter, or is PSD required in the same milestone (new `psd-tools` / ImageMagick dependency)?
6. **Twenty active flags.** Confirm my reading: shipped profiles expose ≤20 image-changing keys; the other ~47 stay in the schema as retired/hidden until measured deletion. The alternative is a hard delete down to 20 in M2, which throws away unreplicated A/B history.
7. **C++ build gate.** If this machine still cannot rebuild `base` (Chat's OpenCV 4.6 blocker), should C++ issues (`translation_scale`, seam-cost parity) stay blocked rather than grow a Python-only twin?

**Handoff:** Harbinger answers §16, then Claude + Harbinger close the plan and cut GitHub issues. I am not starting M1 until that signal.

## 17. Harbinger's decisions on Grok's feasibility questions (2026-08-15)

1. **M1 cut:** M1a (shared stage protocol, no pixel change) → M1b (benchmark adapter + video smart-select fix) → M1c (GUI hooks + exclusion-mask propagation). No single-PR rewrite of both forks.
2. **M2 bar:** Discriminating fallback. Safe ASP must pick Raw ASP on at least one known-good red-set case and SCANS on known catastrophes. Always-SCANS is not M2 success.
3. **§9.2 landing:** Default-off second compositor (`coherence_v2`). Current `rendering/compositing/` and the HITL seam loop stay until a human screen promotes the candidate.
4. **M0 / first ungated 97-run:** Relabel the 2026-08-07 artifacts in M0. Run a new ungated Raw ASP 97-case corpus **immediately after M1 adapters land**. That run is the pre-algorithm baseline for M2+.
5. **Layered export:** PNG stack + JSON sidecar first (M3/M4 output contract). PSD is a later adapter, not an M3 blocker.
6. **Twenty active flags:** Default profiles expose ≤20 image-changing keys (1-in-1-out on that surface). An **Advanced configuration** control reveals additional registered parameters. Advanced is not a silent second default; values still go into the manifest. Do not hard-delete the extra ~47 keys in M2.
7. **C++ build gate:** `translation_scale` repair, C++ seam-cost parity, and any other `base` kernel work stay blocked until this machine can rebuild the extension. No Python-only twin of a C++ kernel. M1 includes a documented rebuild check.

**Handoff:** Claude + Harbinger close remaining wording, then Claude files the §9 issues in dependency order. Grok does not start M1a until those issues exist.

## 18. New track added during joint final review — M2.5 (2026-08-15)

During the joint final review, Harbinger raised a gap none of the four passes
caught: §2/§10/§15 all treat the sharpness/edge-energy/ghosting inverse
correlation to human judgment as a reason to *demote* those signals (M2), but
nothing investigates *why* they diverge or builds better measurement. Three
ideas were proposed: (1) new CV metrics, (2) anime-adapted metrics plus a
trained human-judgment proxy model, (3) similarity-based benchmark subset
selection and metric/human correlation mining to find the highest-leverage
pipeline areas.

Decisions locked:

- New milestone **M2.5** (§5), not folded into M0/M2 — visible as its own
  workstream. Depends on M0's per-output schema; runs parallel to M1–M4;
  analysis/tooling only, no algorithm-default changes. Added to §9's issue
  order as item 9, after M2 and before M3.
- The learned human-judgment proxy is scoped as **non-gating only**, revalidated
  against held-out human ratings as the corpus grows — human review remains
  the sole release criterion (§1) regardless of this metric's performance,
  given the small current sample (97 cases, 43 true Raw ASP composites).
- Data-driven subset selection **supplements** the existing manual five-case
  smoke set and structural red set for now. It may graduate to replacing them
  once its representativeness is measured and stable across corpus growth —
  that graduation is a later, explicit decision, not automatic on landing.
- M2.5's per-defect-category correlation findings inform M5 (photometric/seam)
  prioritization but do not override the locked M3/M4-before-M5 structural
  ordering by themselves; a strong contradiction goes back to Harbinger as an
  open question, not a silent reorder.

Full M2.5 deliverables and exit criteria are in §5. This closes the open
questions from this review round; Claude proceeds to file the §9 issues next.

## 19. Interactive Dev Tool, 2.5D/3D Telemetry & Visual Diagnostics Architecture (Brainstormed with Harbinger 2026-08-15)

Harbinger and Gemini brainstormed the interactive dev tool, visualization methods, and telemetry architecture to guide M6 (Artist Review Workflow) and M2.5/Analytics tooling.

### 19.1 Core Interactive Controls & Dual UI Architecture
- **Dual P0 HITL Controls (Pose + Seam):** Both the **Cel-Pose Thumbnail Swapper** (1-click pose candidate selection per overlap zone) and the **Draggable Seam Corridor / Exclusion Barrier Brush** (spline node editing and `1e6` barrier painting) are locked as top-priority interactive controls for M6.
- **Stage-by-Stage Visual Pipeline Stepper:** Step-by-step interactive breadcrumb allowing developers and artists to inspect intermediate visual states (e.g. viewing the reconstructed clean background plate *before* foreground cels are placed).
- **Core Interaction Patterns:**
  - **Hover & Tooltips:** Instant display of provenance metadata (source frame ID, timestamp, optical flow vector, gain residual) on cursor hover without canvas clutter.
  - **Filtering & Slicing:** Multi-dimensional slicers and sliders for defect tags, confidence thresholds, and benchmark runs.
  - **Drill-Down & Roll-Up:** Click on an aggregate pipeline stage or failure cluster to drill down into frame pairs, LoFTR match rays, and per-pixel seam cost surfaces.
  - **Brushing & Linking:** Selecting a cluster of outlier points in a telemetry chart (e.g. high gain mismatch) automatically highlights the corresponding frame strips on the canvas.
  - **Panning & Zooming:** High-performance synced multi-canvas viewports (Raw ASP vs. Safe ASP vs. SCANS) with a floating "Diff Loupe" (localized magnifier showing optical flow / pixel delta).
  - **Scroll-Triggered Storytelling:** Progressive visual breakdowns explaining how specific pipeline failures (e.g. test 06 torn anatomy) occur and how candidates resolve them.

### 19.2 2.5D Parallax Game Simulator (PMF Parity & Re-use)
- Built-in **2.5D Parallax Viewport** in both the PySide6 dialog and web dev tool:
  - Interactive virtual camera dolly across the canvas.
  - Multi-plane depth rendering: clean background panorama plate at depth $Z=0$ and segmented character sprite cels at depth $Z=1$ with dynamic parallax offset.
  - Validates game asset suitability directly within the tool for 2.5D mobile game development.

### 19.3 Decoupled Telemetry Architecture (locked A+B, 2026-08-15)
- **Emission API:** `TelemetrySink` on `PipelineSession`. Canonical `run()` does not import `rerun` or `opentelemetry`.
- **A — Rerun sidecar:** opt-in `run.rrd` via `rerun-sdk` (`desktop_quality` extra only). Desktop viewer. Dense tensors behind an explicit flag. 2D-affine poses must not be presented as a 3D reconstruct.
- **B — OTel spans & metrics:** `asp.stage.duration_ms`, `asp.vram.peak_bytes`, `asp.gain.clamp_residual`, `asp.seam.cut_energy`. First backend is local OTLP file or stdout. Prometheus/Grafana/Jaeger/Honeycomb are optional collectors, not an in-repo service.
- **Honeycomb-style BubbleUp** remains a *consumer* of B, not a second emission path.
- **C optional / unscheduled:** a native JSON/NPZ inspector in M6/`/journal` that duplicates Rerun's spatial scrubbing. Not on the issue list, not a priority, not a blocker.
- **D rejected:** no Rerun WASM embed in `docs/website`.

### 19.4 Interactive 3D Web Models & Visualizations (`@react-three/fiber` / `.glb`)
- **3D Exploded-View Layer Stack:** Interactive WebGL 3D view in `docs/website/` showing the warped background mesh, seam boundary planes, and floating segmented character cels with orbit controls.
- **3D Feature Match Point Clouds:** Interactive particle graph visualizing LoFTR keypoint correspondences in 3D feature space.
- **Lightweight 3D Mascots & Viewfinders:** Fast-loading `.glb` interactive assets for the web portal adhering to the Optic Lab / Blueprint theme.
