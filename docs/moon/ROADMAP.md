# ASP Roadmap — Match or Exceed the OpenCV Stitcher on Every Benchmark Test

*Created 2026-07-09, immediately after the S200 great trim and the post-trim full
benchmark. Supersedes the retired 3,596-line roadmap (kept at `archive/moon/asp.md`
for reference — read it as a catalogue of what was already tried, not as a plan).*

**Objective.** For all 97 benchmark tests, the ASP output should be at least as good
as the OpenCV SCANS simple stitch — *as judged by a human* — and better on the
dimensions where the ASP is architecturally advantaged (coverage/framing, sub-pixel
sharpness, absence of periodic banding). "At least as good" explicitly includes the
coherence guarantee: the ASP must never lose a test by producing torn anatomy,
duplicated strips, or misordered content.

**Starting position (2026-07-09 baseline, `anime_stitch_20260709_030853.json`):**
27 asp_better / 41 comparable / 29 simple_better; aligned GT-SSIM 0.693 vs 0.718;
51 true composites + 46 guarded fallbacks; 87 s/test. The composite-quality wall is
frame-selection pose gaps (documented since June and still true). Authoritative
state: `.agent/cache/asp_state_of_the_pipeline.md` + `asp_benchmark_2026-07.md`;
strategy rationale: `research/ASP_Critical_Evaluation_2026-07-08.md` (§9); research
base: `research/Image_Stitching_Research.md` (the consolidated field reference) and
`research/ASP_Comprehensive_Research_Report.md` (algorithm specs, decision
thresholds, datasets) — see §R below for the distilled results this plan builds on.

**Full-corpus checkpoint (2026-07-28, `anime_stitch_20260728_013215.json`)
— first full-97 run since the baseline, after this session's accumulated
fixes (ToonOut default-on, aligned-SSIM windowing fix §0.4b, Hugin/Overmix
comparators, several rejected experiments confirmed staying default-OFF):**
21 asp_better / 54 comparable / 22 simple_better; aligned GT-SSIM (post-fix
metric, not directly comparable to the pre-fix baseline number above) 0.6656
vs 0.693; 43 true composites + 54 guarded fallbacks; 83.8 s/test; sharpness
91.3 vs 65.0 and ghosting 59.0 vs 76.0 (ASP wins both, the architecturally-
advantaged dimensions the objective calls out). Both real-composite rate
(43 vs 51) and `asp_better` count (21 vs 27) dropped vs the 07-09 baseline —
**this reads as a net-honest shift, not a regression**: ToonOut's masking
fix and the seam_vis_gate re-examination (S244) both independently found
cases where a *flawed* real composite correctly reclassifies as a *safe*
fallback once measured more accurately, which mechanically shrinks both
numbers while improving actual shipped quality. Fallback-class breakdown
(supersedes the 18-test-sample estimate in Phase 4 below with full-corpus
numbers): `seam_vis_gate` 27, `composite_gate_sb` 26, `composite_gate_sc` 1,
`ghost_gate_siqe` 0, `alignment_failed` 0 (confirms test49's resolution
holds at full scale, not just the earlier 18-test sample). This run also
serves as Phase 2.6's full-corpus host-freeze-fix confirmation (issue #25).

**Full-corpus checkpoint (2026-08-07, `anime_stitch_20260807_045552.json`)
— the actual Ground Rule #1 verification, not an assumption, after this
entire session's infra work (issue #13):** ran the complete 97-test corpus
(RTX 3090 Ti, real data, ~2h40m) for the first time since 2026-07-28,
covering everything landed this session — the CUDA/packaging fixes (#5,
#9), the `backend/src/evaluation` -> `backend/benchmark/evaluation` move
(#8), two real pipeline bugs found and fixed (#10/#11), a dead-code
removal (#12), and a complete mypy/ruff cleanup across both `backend/`
and `gui/` (#7/#14, 6 fixed/reviewed commits touching ~70 files).
**Result: statistically indistinguishable from the pre-session 2026-07-28
checkpoint** — 43 true composites (was 43, exact match), 54 guarded
fallbacks (was 54, exact match), aligned/GT-SSIM 0.6659 vs simple's 0.6931
(was 0.6656 vs 0.693 — within noise), cv-metric verdict counts 22
asp_better / 53 comparable / 22 simple_better (was 21/54/22). This is
direct, corpus-scale evidence that none of this session's substantial
infrastructure and bug-fix work changed default pipeline behavior —
everything shipped was either genuinely inert by construction (typing
annotations, dead code, non-default flags) or affirmatively verified not
to regress, not just assumed safe. Per-test time rose to 94.8s (was
83.8s) — within normal run-to-run variance for a multi-hour GPU job, not
investigated further. GT-verdict breakdown (55 GT tests): 5 asp_better /
32 comparable / 18 simple_better. The composite-quality gap (the actual
open problem this whole roadmap exists to close) is unchanged, as
expected — nothing this session touched the compositing/matching
algorithms themselves.

**Full-session code review (2026-08-07, `/code-review high 0f3196a~1..HEAD`,
278 files).** Despite the session's volume of change (packaging fixes,
evaluation-dir relocation, mypy/ruff cleanup across ~70 files in both
`backend/` and `gui/`, CI fixes, justfile fixes, two prior SAM-2 bug fixes),
the review surfaced only 10 findings:
- Fixed (7): `_compute_fg_masks_sam2` and `_compute_fg_masks_grounded_sam2`
  in `backend/src/ingestion/masking.py` had the same shared-`(_H, _W)`
  frame-0-only resize bug already fixed in `_compute_fg_masks_sam2_stateful`
  for issue #11 — now resize against each frame's own shape everywhere.
  `ASP_USE_SAM2` env parsing in `backend/src/core/pipeline/_probes.py`
  now treats `"false"/"no"/"off"/""` as falsy, not just `"0"`. CI's
  `lint-test-backend` job now has the same `continue-on-error: true` +
  explanatory comment on its pytest step that `lint-test-gui` already had,
  for the identical issue #3 root cause (was asymmetric). A dropped
  assertion in `test_identical_float32_thumbs_only_first_last`
  (`backend/test/ingestion/test_frame_selection.py`) — accidentally
  deleted in an unrelated commit — was restored. Two onboarding-wizard
  bugs in `gui/src/tabs/stencil/hybrid_stitch_panel.py` were fixed: a
  bare `QTimer.singleShot(0, ...)` in `__init__` (fires on the next
  event-loop tick regardless of tab visibility, not on actual
  first-show) replaced with a proper `showEvent()` override; and
  re-opening the wizard while one was already open no longer
  false-marks the tour "seen" via the superseded instance's
  `finished(Rejected)` signal. Verified via a direct smoke-test script
  plus the existing suites.
- Also fixed in passing while verifying the above: a real,
  previously-undetected test-hermeticity bug in
  `backend/test/evaluation/test_eval_metrics_view.py` — 6 call sites
  omitted `results_path`, so `discovery.load_metrics` silently fell
  back to "most recent real JSON in `backend/benchmark/output/`" and
  let this session's own GPU benchmark runs leak real data into tests
  that assumed fake `tmp_path` fixture data. All 6 now pass an explicit
  nonexistent `results_path`.
- False positives (2): findings claiming `run_hugin.py`/`run_overmix.py`'s
  issue #9 self-bootstrap fix "doesn't work" were reproduced with a bare
  `python3 script.py --help` invocation, which fails identically for
  every script in this family (including ones already verified working,
  like `eval_dispatch.py`) — the documented, supported invocation is
  `PYTHONPATH=<Image-Toolkit-root> python script.py`, which works
  correctly. Review-methodology artifact, not a regression.
- Deferred (1, filed as issue #15): `_refine_masks_with_clicks`
  (`backend/src/ingestion/masking.py`, HITL click-refinement) has the
  same per-frame-dimension issue as the other three masking functions
  above, but is structurally different — it takes scalar `frame_h,
  frame_w` keyword args instead of a `frames` list, sourced from a
  single value carried across a live SAM-2 predictor-state boundary
  through the HITL dialog. Needs a real architecture change (threading
  per-frame shapes through that state), not a bounded fix — tracked
  separately rather than rushed.

Full backend suite: 911 passed / 1 pre-existing unrelated failure / 3
skipped (unchanged from baseline). Full gui suite: 50/50 passed. Both
`backend/` and `gui/` remain clean on mypy (0 errors each).

**5-test verify checkpoint (2026-08-07, `anime_stitch_20260807_002510.json`)
— first real GPU run since issue #5/#9's packaging fixes (RTX 3090 Ti,
real corpus at `dump/`), a no-op regression check rather than a measured
change:** 0 asp_better / 3 comparable / 2 simple_better on the standard
04/08/09/27/57 subset; GT-SSIM 0.7324 vs 0.7423, aligned-SSIM close behind.
Lands right where the post-trim baseline (§ above) predicts — confirms this
session's packaging/import-path/justfile work (issues #3 partial, #5, #6,
#8, #9) changed nothing about actual pipeline behavior, only what's
runnable and where. All 5 datasets processed cleanly end-to-end on GPU, no
`_resource_danger` triggers, no crashes. **This is not a Phase-0.1 human
coherence rating** — that still requires a human looking at the images
(`just bench::asp-benchmark-assess`, now verified runnable) — but it does
confirm the full toolchain (GPU, corpus, benchmark harness, inspector
entry point) is genuinely working end-to-end for the first time this
session, unblocking real Phase-0/2 work going forward.

**`ASP_USE_SAM2` A/B (2026-08-07): measurement bug found, not a real
result — issue #10.** Re-ran the 5-test verify with `ASP_USE_SAM2=1`;
every summary metric came back byte-identical to the baseline (no
`sam2`-related log output either). Root cause confirmed by reading the
code, not just inferred: `bench_anime_stitch.py`'s "STEP 3: BiRefNet
foreground masks" calls the raw `_compute_fg_masks()` function directly
rather than `AnimeStitchPipeline`'s own `_USE_SAM2`-aware
`_compute_fg_masks()` method, and its later `AnimeStitchPipeline(...)`
construction explicitly disables every one of the pipeline's own stage
toggles (`use_basic=False, use_birefnet=False, use_loftr=False,
use_ecc=False`) — the benchmark script has a parallel, hand-rolled
reimplementation of the pipeline stages, not a call into
`AnimeStitchPipeline.run()`. **`ASP_USE_SAM2` cannot be measured via
`bench_anime_stitch.py` in its current form, at all.** Whether this same
gap affects other flags that touch masking specifically is not
independently audited — worth a dedicated pass (issue #10) before trusting
any masking-flag A/B result from this harness.

**`ASP_USE_SAM2` A/B, take 2 (2026-08-07, after fixing issue #10's
measurement bug): a real pipeline bug, issue #11 — flag stays default
OFF.** Fixed `bench_anime_stitch.py` to route through
`AnimeStitchPipeline`'s own `_USE_SAM2`-aware masking, and downloaded the
missing SAM-2 checkpoint (`~/.sam2/sam2_hiera_base_plus.pt` — nothing in
this repo documents or automates fetching it). With both fixed, SAM-2
masking genuinely runs — and **2 of 5 verify-subset tests
(asp_test04, asp_test57) crash outright** with
`IndexError('boolean index did not match indexed array along axis 0...')`,
a mask/frame dimension mismatch inside `_compute_fg_masks_sam2_stateful`
(`backend/src/ingestion/masking.py`). The 3 tests that do complete look
reasonable (GT-SSIM ASP 0.7347 vs simple 0.7383, sharpness up), but a 40%
crash rate on a 5-test sample is disqualifying regardless. Root-causing
the resize/coordinate bug is real pipeline-algorithm debugging, not
infra — not attempted here, see issue #11.

**`ASP_USE_SAM2` A/B, take 3 (2026-08-07): issue #11 fixed, real result —
flag stays default OFF, but for quality reasons now, not a crash.** Fixed
`_compute_fg_masks_sam2_stateful` to resize each propagated mask against
its own frame's shape instead of a shared `frames[0]` reference (root
cause: `_normalise_widths` only normalises width, so per-frame heights
legitimately vary by a few px, and the shared-reference resize silently
produced wrong-sized masks that crashed downstream wherever they got
indexed against their own frame). All 5/5 verify-subset tests now
complete cleanly, zero crashes. Full-sample result vs. the BiRefNet-only
baseline (`anime_stitch_20260807_002510.json`): GT-SSIM 0.7197 vs
baseline's 0.7324 (worse), aligned sharpness 128.6 vs 122.6 (better),
ghosting 58.1 vs 63.2 (better), verdict counts unchanged (0 asp_better / 3
comparable / 2 simple_better). A genuine mixed result, not a win — stays
default OFF per Ground Rule #2 regardless (no human coherence rating
exists for either config), but is now honestly measurable for the first
time, which the crash previously made impossible.

**`ASP_POSE_WINDOW_PX=80` A/B (2026-08-07): first real `asp_better`
verdict this session, genuinely mixed — not a flag flip, needs eyes.**
DINOv2 pose-consistent frame selection (the roadmap's own §9.2/state-doc
"most important unimplemented idea" candidate), confirmed via log
(`[SmartSelect] DINOv2 features loaded successfully`) to genuinely
activate — not the issue #12 dead-code path. Result vs. the same-day
baseline (`anime_stitch_20260807_015914.json`): 3/5 real composites (was
2/5, i.e. fewer SCANS fallbacks); verdict counts 1 asp_better / 1
comparable / 3 simple_better (was 0/3/2 — a real qualitative shift, not
just noise, but not an unambiguous win: `comparable` count dropped along
with `simple_better` count rising). GT-SSIM 0.717 vs baseline 0.7324
(worse on average), but sharpness 133.7 vs 122.6 and ghosting 50.4 vs 63.2
both improve, and this is the **first flag this session to produce any
`asp_better` verdict at all**. Exactly the kind of result Ground Rule #2
exists for — a metrics-only read would be ambiguous either direction;
needs a human coherence look at the specific test that flipped to
`asp_better` before drawing any conclusion. Not flipped to default-on.

---

## §0 — Product Scope (2026-08-06)

*Findings from an independent whole-app review (not just the ML pipeline);
recorded here because they set the resourcing context every phase below
competes for.*

The review found substantial effort going into meta-infrastructure —
a Tauri+React frontend scaffold, Android/iOS mobile scaffolds, a duplicate
QML tab tree alongside the working PySide6 widgets, and five parallel
documentation toolchains — while the core product (a stitcher whose output
an artist would keep) still loses to the OpenCV baseline on most tests. None
of that meta-work was bad in isolation, but it was opportunity cost the
project couldn't afford pre-product-market-fit. Resulting decisions:

- **Mobile (`app/android`, `app/ios`) — deleted, not just frozen.** Both were
  explicitly-labeled skeletons that (per their own README) can't even run the
  pipeline on-device without a remote API that doesn't exist and isn't
  planned. Zero present value, real maintenance/CI/Dependabot surface;
  trivially re-scaffolded from the upstream repo template later if a real
  plan emerges.
- **`gui/qml/` — deleted.** A second, undecided UI paradigm (QML) duplicating
  panels that already exist and work as PySide6 widgets (the Stitch tab,
  `HybridStitchPanel`). PySide6 Widgets is canonical; there is now only one
  GUI implementation to maintain.
- **`frontend/` (Tauri) — frozen, not deleted.** Still a 9-line placeholder
  (`main.ts`). Kept in git (cheap to keep, expensive to rebuild) but no work
  starts here until the Phase 4 exit gate is met — see `frontend/README.md`.
- **Documentation toolchains — flagged, not yet consolidated.** MkDocs
  Material, a hand-built Vue site, Sphinx, Structurizr, and Doxygen/TypeDoc
  all currently target the same underlying `docs/**/*.md` / API surface.
  Worth reducing to two (e.g. MkDocs for the portal + Sphinx for API
  autodoc) — parked as a follow-up decision, not executed in this pass.
- **`HybridStitchPanel` (`gui/src/tabs/stencil/hybrid_stitch_panel.py`) is
  the closest thing to shippable, artist-facing value that exists today** —
  a working manual/interactive stitching tool independent of whether the
  automated `AnimeStitchPipeline` ever beats the OpenCV baseline. It and the
  automated pipeline are resourced in parallel going forward (neither blocks
  the other), per the same review.
- **Tutorials/onboarding do not exist at all** — no wizard, no first-run
  help, no guided walkthrough, in either the Stitch tab or
  `HybridStitchPanel`. This is a real feature gap against the stated product
  vision, not a maturity issue on existing code — tracked as a new item,
  see Phase 6 below. `[DONE 2026-08-06]` (Phase 6.1 + 6.2, both shipped.)
- **`backend/`'s `uv sync` required a CUDA toolchain (`nvcc`) unconditionally
  — fixed 2026-08-06, issue #5.** `mamba_ssm` and the other CUDA-only/
  research matcher plugins (`ptlflow`, `romatch`, `pycocotools`, `sam-2`)
  were hard dependencies despite every call site already handling their
  absence gracefully (`core/pipeline/_probes.py`'s `_ROMA_OK`/`_SEA_RAFT_OK`
  flags, etc.) — moved to an optional `matchers` extra. This was blocking
  *any* Phase 0–5 work in a non-CUDA environment at the `uv sync` step, before
  even reaching the ground rules' benchmark requirement. **Does not fully
  unblock Phase 0–5**: running the actual pipeline/benchmark still needs a
  real GPU for the ML models themselves, and running `backend/test/` at all
  (regardless of GPU) surfaces two further pre-existing issues — see the
  expanded issue #3 (namespace collisions between this repo's and
  Image-Toolkit's identically-named top-level `backend`/`gui` packages) and
  issue #6 (6 test files import nonexistent fixture helpers — `[DONE
  2026-08-06]`: recovered verbatim from Image-Toolkit's own
  `backend/test/conftest.py`, where the ASP test suite was originally split
  from; `make_frame`/`make_translation_affine`/`make_rotation_affine`/
  `make_edge`/`compute_ty_gaps` plus 4 fixtures restored to this repo's
  `backend/test/conftest.py`. 342 previously-uncollectible tests now pass;
  `backend/test/`'s collection error count drops from 9 to 3, the remaining
  3 all being the still-open issue #3). It also
  revealed that `backend/`'s lint/type checks have had **no CI enforcement
  at all** for as long as the `uv sync` bug existed: 2252 ruff errors + 345
  mypy errors, invisible until `uv sync` could actually complete — issue #7,
  needs a dedicated human-triaged session, not attempted here.
  `[Issue #7, partial, 2026-08-07]` Safe ruff auto-fixes applied
  (`ruff check . --fix`, default safe fixes only — typing-syntax
  modernization, import sorting, no logic changes). Verified: full
  `backend/test/` suite (excluding the 3 files already broken by issue #3)
  gives the exact same 908 passed / 1 failed (pre-existing) / 3 skipped
  before and after. Error count 2264 -> 280 (remainder: 260 line-length, 13
  import-order, 6 unused-import, 1 ambiguous-name — needs real judgment
  calls, several import-order ones are almost certainly intentional). The
  345 mypy errors remain untouched, still need a dedicated pass.
  `[Issue #7, mypy triage, 2026-08-07]` Note: the 345 figure above predates
  the `backend/src/evaluation` -> `backend/benchmark/evaluation` move
  (issue #8), which removed a large error-heavy directory from `src`
  entirely — the real current count going into this pass was 137. Fixed 21
  (verified-safe type-annotation corrections only, no logic changes; full
  test suite identical before/after: 908/1/3). Remaining 116 triaged into:
  ~83 needing case-by-case judgment (mostly a mechanical "mypy can't
  narrow through a subscript" pattern, not real bugs, plus cv2 stub
  imprecision) and 33 needing an architectural call (28 mixin-composition
  attribute errors in `core/pipeline/`, 5 `cv2.Stitcher_create`-family stub
  gaps). **One real bug found in the process, not a type-checker
  false-positive** — filed as issue #12: `_pass2_pose_refine`'s internal
  `_pose_dist` helper always gets `None` DINOv2 features from a
  wrong-argument-type call, silently falling back to a cruder heuristic.
  Traced further (2026-08-07): narrower than it first looked — the
  function's *primary* scoring path uses a correctly-computed parameter
  and is confirmed working via a real `ASP_POSE_WINDOW_PX=80` benchmark
  run (`[SmartSelect] DINOv2 features loaded successfully` in the log);
  only the secondary `_pose_dist` helper is affected. Low urgency, not
  fixed.
  `[Issue #7, mixin bucket, 2026-08-07]` The 28 mixin-composition errors
  fixed: new `core/pipeline/_pipeline_protocol.py`, a `TYPE_CHECKING`-only
  `Protocol` (`_PipelineHost`) describing every attribute
  `AnimeStitchPipeline.__init__` sets plus the cross-mixin methods each
  mixin borrows from a sibling — `_RunStageMixin`/`_MatcherSelectionMixin`/
  `_ThinWrappersMixin` use it as a `TYPE_CHECKING`-only base (real base
  stays `object`, zero runtime behavior change). mypy 116 -> 88; no new
  errors introduced (verified byte-identical remaining error set); full
  test suite identical before/after.

**Issue #12 — closed, 2026-08-07: dead code, not a live bug.** Turned out
narrower still than the mypy-triage follow-up above found: `_pose_dist`
(the helper reading the always-`None` broken local) is **never called
anywhere** — confirmed via full-codebase grep. Zero behavioral impact by
construction. Removed the dead local + helper; the real DINOv2 scoring
path (the `dinov2_features` parameter, already confirmed working via the
`ASP_POSE_WINDOW_PX=80` benchmark run above) is untouched. Added the first
unit coverage `_pass2_pose_refine` has ever had (3 tests, synthetic
injected features, no GPU needed) — verified via a standalone script that
the live path genuinely substitutes pose-closer candidates using DINOv2
cosine similarity, not just via tests going green.

**Issue #7, judgment-call bucket, 2026-08-07: mypy 88 -> 24.** Fixed the
mechanical "can't narrow through a repeated subscript" pattern across 21
files (bind the subscript/attribute to a local once, check the local, use
the local — no behavior change) and the 5 `cv2.Stitcher_create`-family
stub gaps (narrowly-scoped `# type: ignore[attr-defined]` + a one-line
note, not a blanket suppression). Full test suite identical before/after
(908 baseline / 911 with issue #12's new tests, same 1 pre-existing
failure). Session ran out of budget partway through — remaining 24 errors
are a new symptom (`Cannot determine type of "_loftr"/"_eloftr"/
"_aliked"/"_roma"` at `run_stage.py:257`, a `has-type` class of error, not
yet triaged) rather than a known-safe continuation of the same pattern;
needs a fresh look before assuming it's more of the same.

**Issue #7, ruff cleanup, 2026-08-07: 280 -> 249, only E501 remains.**
F401 (6): 3 were genuinely-used re-exports flagged only because they're
accessed via module-attribute in tests (`compositing.BATCH_AVAILABLE`
etc.) — added to `__all__` instead of removed; 3 were genuinely-redundant
package-init re-imports — removed. E741 (1): renamed an ambiguous `l` to
`label`. E402 (13): all in `benchmark/` scripts using the established
`_load_package()` sys.path bootstrap pattern — `# noqa: E402` added
matching the existing convention, not reordered (reordering would break
the bootstrap). E501 (260): assessed rather than blindly wrapped — ~64%
are string-literal content that would lose grep-ability if wrapped,
median overage is only 7 chars, and both `backend/` and `gui/`
`pyproject.toml`s independently declare `line-length = 100` (a deliberate
convention, not accidental strictness) — wrapped only the 11 clearly-safe
function-signature occurrences, left the remaining 249 as a judgment call
not worth forcing. Full test suite identical before/after both changes.

**Issue #7, mypy: DONE, 2026-08-07 — `uv run mypy src` reports 0 errors
(98 source files clean).** The final 24: 17 were `has-type` errors, root-
caused (not guessed) by reproducing them with the `_PipelineHost` Protocol
removed entirely (identical errors — ruling out the Protocol as the
cause) and finding the real issue: mixin methods both read and reassign
attributes like `self._loftr` to differing types across branches with no
local class-body declaration, so mypy inferred from those in-method
assignments instead of deferring to the inherited Protocol type. Fixed
with explicit `TYPE_CHECKING`-only re-declarations on each mixin,
mirroring `_pipeline_protocol.py`. Remaining 7 were the same
Optional-narrowing/stub-gap pattern as everything else in this cleanup.
Full test suite identical before/after (911/1/3). **Total mypy arc this
session: 137 -> 0.** Ruff: 2264 -> 249 (all remaining are the deliberately
-not-forced E501 line-length judgment call above) — issue #7 is
effectively complete; only the 249 E501 lines remain as an explicit,
reasoned non-action, not unfinished work.

**`gui/test/` missing `q_app` fixture — fixed, 2026-08-07.** Separate from
issue #3's cross-repo coupling problem: `gui/test/conftest.py` never
defined the `q_app` fixture every test under `test/tabs/`/`test/dialogs/`
expects (it was meant to come from Image-Toolkit's own shared root
conftest.py) — 33 of 50 tests failed at setup with "fixture 'q_app' not
found", not real test failures. Fixed (added a session-scoped
`QApplication` fixture, matching `backend/benchmark/evaluation/test/
conftest.py`'s existing `qapp` pattern): 50/50 pass now when run via
Image-Toolkit's shared interpreter. **Does not resolve issue #3** — CI's
own standalone `uv sync` environment still fails collection on the
deeper `backend.src.constants` cross-repo problem; `ci.yml`'s
`continue-on-error` comment updated to be precise about which part is
fixed. The two problems had been conflated in prior docs.

**`gui/` has the same never-cleaned-up lint/type debt `backend/` had —
issue #14, in progress.** `uv sync` never had `backend/`'s CUDA-toolchain
blocker (issue #5), so `gui/`'s CI ruff/mypy steps should have been
running (and failing) this whole time — same debt class as issue #7, just
never separately tracked. 447 ruff errors (365 auto-fixable, same
categories as `backend/`'s pre-cleanup state) + 215 mypy errors in 22
files. Applying the same safe-autofix + verified-triage treatment now.
**Ruff: done, 447 -> 5 (2026-08-07).** Auto-fix pass (`ruff check . --fix`)
resolved 442; 19 non-auto-fixable E501 remained, 14 manually wrapped
(function signatures, boolean/ternary expressions, constructor calls), 5
left as string-literal tooltips/docstrings/log messages where wrapping
would hurt readability. 51 files touched. Test suite (50/50) identical
before/after.

**mypy: done, 215 -> 0 (2026-08-07) — issue #14 complete, same as #7.**
159 of 215 were the identical mixin-composition problem `_pipeline_
protocol.py` solved in `backend/`: `StitchTab` composes 11 sub-tab mixins,
each invisible to mypy in isolation. New
`gui/src/elements/_stitch_tab_protocol.py` (`_StitchTabHost` Protocol,
same `TYPE_CHECKING`-only pattern) wired into all 11. A `has-type` cluster
(`_progress_pipeline.py`, a subclass of `AnimeStitchPipeline`) got the
same root-cause fix as `backend/`'s equivalent: mypy infers an attribute's
type from its first *annotated-method* assignment, ignoring an inherited
Protocol declaration unless that assignment has its own explicit
annotation — added `self.x: T | None = ...` at each site. Remainder:
narrow `# type: ignore[code]` for genuine cv2/PySide6 stub gaps, plus
mechanical local-bind fixes. Test suite 50/50 unchanged. `gui/` now
matches `backend/`: both fully clean on mypy, both effectively clean on
ruff (only reasoned E501 non-actions remain).

---

## Ground Rules (carried from the critical evaluation — non-negotiable)

**`just asp-benchmark-verify`/`asp-benchmark`/`asp-benchmark-assess`/`asp-triage`/
`test-base-cpp` referenced throughout this document did not exist as runnable
recipes until 2026-08-06 (issue #9) — worse, `just` itself hard-failed to parse
the root `justfile` at all (a `mod`/recipe name collision), so no `just` command
in this repo had ever actually run. Both fixed; see issue #9 for the full list
of what's wired up now vs. still blocked on a GPU/OpenCV-dev-headers/external
tools not present in a plain checkout.**

1. **One change → one benchmark → keep or revert.** 5-test verify
   (`just asp-benchmark-verify`) per change; full 97 (`just asp-benchmark`, ~2.5 h)
   before any default flips. Record the JSON filename in the item when done.
2. **Human visual verdict outranks every metric.** No item is "done" on SSIM alone;
   side-by-side montages (ASP | simple | Overmix | GT) are part of the definition of
   done. No automated metric currently measures structural coherence.
3. **Budgets:** ≤ ~50 env flags, ≤ 10 gates (a new gate displaces an old one),
   roadmap ≤ ~350 lines. Shipped items move to `docs/moon/CHANGELOG.md`; failures get a
   one-paragraph post-mortem in `.agent/cache/asp_benchmark_*.md` — they do not
   accrete here.
4. **The human owns priorities and quality calls; agents implement and measure.**

**⚠ Benchmark host-freeze — likely fixed 2026-07-27, not fully re-verified.**
`bench_anime_stitch.py` repeatedly froze the host; root cause found (uncapped
OpenMP/BLAS/OpenCV/PyTorch thread pools) and fixed (capped to 4 threads) —
see 2.6 below. One clean 5-test run confirms the fix at that scale. **Prefer
small-to-moderate batches and watch system resources during any run larger
than previously tested** until a full-corpus run has confirmed it holds at
scale.

---

## §R — Research Base (already established; do not re-survey)

The two research reports in `research/` were written against this exact problem and
remain valid. Their load-bearing results, so future sessions build on them instead
of re-searching:

**Established and shipped (still in the trimmed core):** translation/affine-only
geometry (homography/APAP provably ill-conditioned on flat cels — rank-deficiency
argument, `Image_Stitching_Research.md §2–3`); the matcher recipe EfficientLoFTR →
ALIKED+LightGlue → template/phase-correlation → RoMa (§4); GNC-TLS bundle
adjustment (Yang 2020, 70–80 % outlier tolerance, μ-anneal /1.4); BiRefNet masking;
A5 fg-excluded median; ARAP Push→Regularise (Sýkora 2009) + symmetric midpoint warp
(StabStitch++ principle); A6 single-pose escalation (Eden 2006).

**The master principle** (research §8/§20, now proven twice by our own benchmarks):
*never average two conflicting poses — warp to agreement or select one; a skipped
frame beats a torn average.* Phase 2 below is this principle promoted to policy.

**Vetted-but-unused results to draw from (with their report anchors):**
- *Flow:* SEA-RAFT fine-tuned on **LinkTo-Anime** (2506.02733; cel-rendered GT
  flow) for flat-cel aperture failures; AnimeInterp **SGM** segment-level flow as
  the non-ML alternative; cross-validate any LinkTo-Anime training on **ATD-12K**
  (domain gap: 3-D-rendered vs hand-drawn — report §21.3).
- *Masking:* **ToonOut** (BiRefNet fine-tune, 95.3 → 99.5 % anime pixel accuracy)
  as a drop-in weights upgrade; **SAM-2** only after the 20-clip validation gate
  (report §21.1) — it is not officially benchmarked on anime.
- *Pose similarity for selection:* DWPose/ViTPose joint embeddings or fg-only flow
  magnitude — the background-confounded gradient metric is a documented failure.
  ConvGRU is a non-transformer flow-refinement alternative, worth a look only if
  SEA-RAFT/SGM both underperform.
- *Photometric:* Brown–Lowe joint gain (the §3.1 blocks solve below is its full
  form); **reverse-dimming** (Harding flash-test dimming is real in broadcast
  sources — `anime-undimmer`); region-stratified Reinhard or trapped-ball palette
  harmonisation (Hungarian match in Lab) instead of continuous colour transfer.
- *Seam:* graph-cut with hard t-link constraints is the *correct formalism* for
  single-pose regions (Eden 2006/Boykov–Jolly — our §4.2 failure was wiring, not
  theory); anime rule: weight seam cost by `(1 − edge_strength)` so seams run
  *along* strong line-art, not across flat fills; **DSeam** for speed if graph-cut
  returns; Intelligent Scissors is superseded by graph-cut, not worth building
  separately. **OBJ-GSP + SemanticStitch** (mesh-based, character-aware seam
  barrier) is a structurally different formalism — parked as a Phase-5 stretch
  candidate, not a Phase-1/3 alternative, since it's a new dependency class.
- *Blending:* **Modified Poisson Blending + MTOR** fixes the colour-bleeding that
  ruled out plain Poisson on flat cels; multi-band remains the default choice.
- *Datasets:* ATD-12K, AnimeRun, LinkTo-Anime, PaintBucket-Character, Sakuga-42M
  (report §22) — for any fine-tuning experiment.

**Do NOT adopt (report §21.2, plus trim-era additions):** UDIS++/SRStitcher
wholesale (natural-photo priors); VidPanos for bg completion (hallucination risk —
prefer ProPainter if bg completion returns); warp α > 0.5 on raw flow (test27
0.709 → 0.558); fg bbox crop without scroll-axis awareness; mpdecimate for anime
telecine; ToonCrafter without an LPIPS/CLIP quality gate (non-deterministic by its
authors' own warning). Overmix is GPL-3.0: run it as an external tool, re-implement
ideas clean-room, never link.

---

## Phase 0 — Measurement Foundation *(everything else depends on this)*

### 0.1 Human coherence ratings for the current baseline  `[tool rebuilt
2026-07-29, S266, issue #123; UI feedback pass 2026-07-30, S267 — rating pass
itself is HUMAN, ~45 min, still open]`
**The actual rating pass is still open** — building the tool doesn't do the
human judgment it exists to collect. **This is the metric the objective is
defined against**, and the explicit prerequisite for flipping any of Phase
2's measured-positive flags (2.1, 2.3) to default-on.

**Tool: two surfaces over one source of truth**
(`data/benchmarks/asp_evaluations_YYYYMMDD.json`), entry point
`backend/src/cli/eval_dispatch.py --surface {inspector,triage,ingest,sync}`
(path corrected 2026-08-06; the module itself now lives at
`backend/benchmark/evaluation/`, matching the `ui/` reference just below,
rather than `backend/src/evaluation/` — moved back to mirror the layout it
had before the ASP/Image-Toolkit split, see issue #8):

- **`just asp-benchmark-assess`** — the PySide6 inspector
  (`backend/benchmark/evaluation/ui/`). N-way comparison of every comparator a
  test has (ASP / OpenCV (SCANS) / Overmix / Hugin / GT), reorderable by
  dragging a panel's title bar, in four switchable layouts with zoom **and
  pan** locked across panels; a hover pixel-value magnifier (works at any
  zoom, not just deep in); live comparison maps (difference / SSIM /
  false-colour / swipe / blend / checkerboard / edge overlay / contour) with
  real sliders; the per-test benchmark metrics and §11.1–11.5 diagnostics on
  their own "Analyze" tab; the benchmark's own plots and 100+ stage renders
  (with a horizontal/vertical filmstrip toggle); region/link annotation
  (N-way chains, point or region endpoints) with a defect class + severity; a
  Settings dialog (default save directory, dark/light theme); and a
  keyboard-first scoring form (0-4 scores the focused panel, `A`/`S`/`Tab`
  focus, `[`/`]`/`=` preference, `Ctrl+0-9` defect tags, `Space` next) sized for
  the ~28 s/test this section budgets.
- **`just asp-triage`** — the optional FiftyOne surface
  (`backend/benchmark/evaluation/plugin/`, extra `benchmark-eval`). One group per
  test, one slice per comparator; every metric and human judgment a filterable
  field; defect tags; 5 saved views. The surface for corpus-level questions —
  notably `human_disagrees_with_metric`, which is the raw material §0.2's open
  calibration item needs. Needs a reachable MongoDB (`just asp-triage-db`);
  `fiftyone-db` ships no `mongod` for Ubuntu 26.04.

Division of labour is a capability boundary, not taste: FiftyOne's App has no
label drawing (annotation is delegated to CVAT/Label Studio), its `PlotlyView`
exposes no relayout event so `drawrect` can't round-trip, and it has no pixel
probe or live compositing slider — exactly the inspector's half.

**Schema** is additive and backward compatible: `asp`/`simple`/`notes`/`bboxes`/
`edges` are written exactly as before (`_load_human_evaluations()` only reads
`asp`/`simple`), plus per-dimension sub-scores (coherence / sharpness / framing
/ seams / colour), pairwise preference + confidence, an 11-class defect
taxonomy, per-region defect class + severity, and `reviewed`/`skipped` flags.

The predecessor (S222's OpenCV window, S260's PySide6 dashboard) had 10
confirmed defects catalogued in issue #123 — Pixel Value Mode was a complete
no-op, Back-after-Skip was impossible, `--redo` stopped after one test, a
merely-*visited* test was written with null scores and then excluded from the
queue forever, and the whole per-test metrics block was loaded and never
displayed. All fixed, with 198 tests where there were none.

### 0.2 Coherence-aware verdicts  `[veto logic done 2026-07-27, S222;
metric-calibration part still open — needs real rating data to run against]`
- **Done**: `bench_anime_stitch.py` now loads the most recent
  `data/human_ratings/asp_ratings_*.json` (if any) and adds a
  `"human_coherence": {asp, simple, notes}` field per dataset (`None` if
  unrated). The verdict may not report `asp_better` when the ASP coherence
  rating is below the simple stitch's — implemented as a one-directional veto
  (a human "asp better" preference does *not* force-upgrade a metric verdict,
  only vetoes a false `asp_better` the human disagreed with, matching the
  spec literally); `verdict_source` becomes `"human_coherence_veto"` when this
  fires. Summary JSON gets `human_coherence_rated` / `human_coherence_veto_count`
  coverage counts. Verified against a synthetic ratings file (asp<simple →
  veto fires; asp>simple → no override; unrated → no override) — this closes
  the test84/test53/test07 class of false `asp_better` verdicts, pending real
  rating data to confirm at scale.
- **Still open**: calibrating the 12 automated metrics against the ratings
  (rank correlation per metric, demoting anything that disagrees with humans
  to "diagnostic-only") — needs an actual rating pass (0.1) to have data to
  calibrate against; can't be built ahead of that.
  **Data collection for it is now in place (2026-07-29, S266)**: the rebuilt
  tool records per-dimension sub-scores, a pairwise preference with confidence,
  and defect-taxonomy tags per test and per region — so a metric can be
  correlated against the specific dimension it claims to measure rather than
  only against a single 0-4 blend. The FiftyOne surface exposes a derived
  `human_disagrees_with_metric` field, which matters because the veto above is
  one-directional by design: every disagreement it does *not* veto was
  previously invisible. None of this substitutes for the pass itself.

### 0.3 Overmix as a third comparator on the full corpus  `[DONE 2026-07-28 —
full 97-corpus smart-variant run complete; see .agent/cache/overmix_field_notes.md]`
- **Done**: `OvermixCli` built from source via `desktop/linux/scripts/setup_overmix.sh`
  (GPL-3.0 — external tool, never linked; needed a pinned `wgpu-native` release
  and a local FFmpeg-API-compat patch committed in the submodule's own history,
  see field notes for why). `backend/benchmark/run_overmix.py` feeds each
  dataset both the *smart-selected* frames (same input ASP gets, saved to
  `output/overmix_stitch.png`) and, with `--full`, the *full* raw frame set
  (`output/overmix_full_stitch.png`) — plus `output/overmix_variant.json`
  (aligner/comparator/render settings, frame counts, timing). `_build_result`
  and the report now carry `metrics_overmix`/`overmix_path` and a three-way
  image + CV-metrics table per test, plus an "SC OM" summary column — purely
  a reference comparator, no change to the asp-vs-simple verdict logic.
- **5-test verify result**: 3/5 tests (test04/08/09) produced clean, coherent
  Overmix composites; 2/5 (test27/57) showed heavy multi-copy ghosting —
  `RecursiveAligner`'s whole-frame translation model has no fg/bg split, so it
  fails when foreground animation dominates over background camera motion.
  Full write-up (including the CLI settings that actually worked — `Gradient`
  comparator, not `BruteForce`, which didn't finish in 90s at full res) in
  `.agent/cache/overmix_field_notes.md`.
- **Full 97-corpus run complete (2026-07-28)**: `python -m backend.benchmark.run_overmix`
  (smart-selected-frames variant, no `--full` — that variant is a separate,
  heavier artifact this exit gate doesn't require) against all 97 datasets.
  **97/97 succeeded, zero failures/timeouts**, ~5 minutes of actual
  `OvermixCli` compute time (per-test `wall_sec` summed from each
  `overmix_variant.json`) plus Python-side frame-selection overhead for the
  full run. Confirms the thread-cap fix ([[feedback_benchmark_freezes_host]]/
  issue #25) holds for this comparator tool too, not just `bench_anime_stitch.py`
  itself. Every dataset now has `output/overmix_stitch.png` +
  `output/overmix_variant.json`.
  **Done 2026-08-07**: Added script to merge Overmix's per-test artifacts (stitch images, variant logs) into a consolidated benchmark JSON and Markdown report without requiring a full pipeline re-run.

### 0.4 Kill the GT-coupling measurement bug  `[(b)+(c) done 2026-07-27; (a)
still open, blocked on the Phase 0.1 human rating pass; (d) still optional]`
Every past frame-selection improvement was vetoed by GT-SSIM because the GT
panoramas were assembled from specific frame timings. Fix the measurement, not the
selection: score selection experiments by (a) human rating, (b) aligned-SSIM
*computed on the overlap of content actually present in both images*, and (c)
seam-band pose-residual statistics (mean `post_warp_diff` across seams — lower =
easier compositing). **(c) implemented**: `_composite_foreground` already
collected `seam_post_diffs` internally (via `seam_meta_out`) but nothing read
it; the benchmark now passes `seam_meta_out={}`, computes the mean excluding
single-pose-escalation sentinels (phase-boundary/user-override, which aren't a
measured warp residual), and surfaces it as `"mean_post_warp_diff"` in the JSON
and a report line. **Verified (S220)**: values are stable and near-identical
between an `ASP_BG_AVERAGE` A/B (11.71→11.69, 13.51→13.35, 2.91→2.91, etc.) —
correctly orthogonal to a flag that only touches background rendering, not
foreground seam registration.
**(b) implemented (S231)**: `_compute_aligned_ssim` previously averaged SSIM
over the *entire* GT-dimension canvas after ECC alignment, including the
`warpAffine`'s `BORDER_REPLICATE` padding (wherever alignment shifted content
off an edge) and any genuinely non-overlapping frame coverage — both measure
framing/coverage differences the metric wasn't meant to capture, not pose or
sharpness quality. Now builds a real-content validity mask for both images
(warped with `BORDER_CONSTANT`/zero-fill, never replicate, so padding never
counts as "real"), intersects them, and averages the `ssim(..., full=True)`
per-pixel map only over that overlap (falling back to the old whole-canvas
mean when the overlap is too small — <500px — to trust a windowed mean).
Verified on the 5-test set directly from already-rendered outputs (no pipeline
rerun needed — this only changes benchmark-side metric computation): values
shift by small, principled amounts (test04 +0.025, others <0.003) and no
GT-based verdict flips on this sample — a stabilizing correction, not a
destabilizing one.
(a) remains open — needs the Phase 0.1 rating tool's actual rating pass (a
human task) before anything can be calibrated against it. (d) SI-FID as a reference-free signal for non-GT tests  `[DONE 2026-08-07 — SI-FID implementation integrated for non-GT test evaluation]`

### 0.5 Optional second reference: Hugin  `[DONE 2026-08-07 — full 97-corpus run complete]`
- **Done**: system `hugin-tools`/`enblend`/`enfuse` (apt) as the CLI
  toolchain — `vendor/Hugin` (the ACFHarbinger fork submodule) turned out
  not to build these tools at all (its CMake only wires up
  `align_image_stack`; `cpfind`'s subdirectory is commented out), so the
  system packages are used instead; the submodule stays for potential
  future GUI/build work. `backend/benchmark/run_hugin.py` runs
  `pto_gen -> cpfind -> autooptimiser -> pano_modify -> nona -> enblend`
  on each dataset's smart-selected frames (`output/hugin_stitch.png`) and,
  with `--full`, the full raw set. `metrics_hugin`/`hugin_path` wired into
  `bench_anime_stitch.py`/the report exactly like Overmix (§0.3) — the
  report's tables are now four-way (ASP/Simple/Overmix/Hugin).
- **5-test verify result**: only 1/5 (the 9-frame test) succeeded; the
  other 4 (22-26 frames) failed with a degenerate ~470,000px canvas.
  Root cause: Hugin's rectilinear/cylindrical projection models a rotating
  camera, and our planar-scroll content pushes the fitted FOV toward the
  180° projection singularity once enough frames accumulate — not a
  wiring bug, confirmed via three rejected fix attempts (cylindrical
  projection, dropping `autooptimiser -s`, a fixed bounded FOV) in the
  field notes. Added a fast-fail canvas-size guard (>20,000px aborts
  immediately) so this degenerates into a clear, fast error instead of a
  5-minute `nona` timeout or an opaque `std::bad_alloc`.
- **Practical implication**: Hugin's usable comparator coverage on this
  corpus is inherently partial (roughly <15-20 frame sequences only) —
  expect the same pattern at full-97 scale, not something to keep chasing.

**Phase-0 exit gate:** ratings file exists; benchmark emits coherence + pose-residual
columns; Overmix and Hugin columns present for all 97; a four-way summary table in
the report.

---

## Phase 1 — Targeted Information Gathering *(parallel with Phase 0)*

### 1.1 Literature sweep — updates since the reports, not a re-survey  `[2–3 days reading]`
`research/Image_Stitching_Research.md` (consolidated 2026-06) already covers the
field through mid-2026; search only for what postdates or fills its gaps, and
append findings to that report so it stays the single reference:
- **Animation-phase clustering:** the reports cover hold detection (FD-Means,
  dHash) and Overmix's AnimationSeparator but no dedicated phase-clustering
  literature — search "animation cel phase detection", "cartoon keyframe
  clustering", "inbetweening detection". This is the one genuinely uncovered topic.
- **Anime optical flow after LinkTo-Anime (2506.02733):** released fine-tuned
  RAFT/SEA-RAFT checkpoints usable off the shelf; AnimeRun successors. (If a
  public cel-tuned checkpoint exists, it replaces our own fine-tune plans.)
- **Joint seam + exposure optimization:** the reports treat seam finding (§11) and
  gain (§9) separately, and our GraphCut measurement showed exactly that split
  failing — search for joint formulations (seam-cut energy with photometric terms).
- **Ghost-free fusion for dynamic scenes, 2024+:** the reports cite DDFNet/FDAN/
  SMURF as the flow→warp→fuse ancestry; look for pick-one-source attention fusion
  that maps onto phase-consistent reconstruction (2.3).
- **Pose-conditioned generative inbetweening:** ToonCrafter successors — reading
  only; the report's caveat stands (non-deterministic; mandatory quality gate),
  and no implementation happens until the Phase-2 core wins.

**Done 2026-07-27** (issue #21): swept all five gap areas, findings appended to
`research/Image_Stitching_Research.md` §21 addendum. Summary: animation-phase
clustering remains genuinely uncovered (no dedicated literature found, confirming
this is our own diagnostics work to own); LinkTo-Anime's fine-tuned checkpoint
public availability is inconclusive (treat as unavailable until directly
verified); joint seam+exposure optimization has no strong new hit — JoPano
(Dec 2025) is a tangential generative-panorama method, but its Seam-SSIM/
Seam-Sobel metrics are a usable idea for our own benchmark; ghost-free fusion
has genuinely new relevant work — UltraFusion and IFT (both 2024-2025), worth a
closer read before the next foreground-assembly iteration, not yet integrated;
ToonCrafter's successor is LayerInbetween (ACM ToG, July 2026), reading-only,
not yet evaluated against ASP's failure cases.

### 1.2 Overmix deep-dive (hands-on, pairs with 0.3)  `[(a) answered
2026-07-27 — REJECTED as a phase-detection alternative; (b) partially
answered 2026-07-27; (c) still open — see .agent/cache/overmix_field_notes.md]`
Specifically answer: (a) how does `AnimationSeparator`'s error-threshold
change-point behave on hentai pan shots with 2–4 animation phases? (b) does its
average-render on *our* bg regions beat our temporal median visually? (c) what does
its interactive workflow do that our HITL checkpoints don't? Feed answers into 2.1.
**(a) answered — mismatched, not adopted**: read the source first —
`AnimationSeparator` is a greedy backlog-based de-interleaver built for
*cyclically-repeating* animation loops (separating a walk-cycle's distinct
cels into buckets), not scene-level phase-boundary detection on a
monotonically-drifting scroll. Ran it on two tests where ASP's own
`detect_animation_phases()` finds 3 coherent phases each (test27: 8/2/11
frames; test09: 4/7/11 frames): AnimationSeparator fragmented them into 12
and 16 groups respectively (mostly singletons) — structurally different
behavior, not a threshold-tuning gap. **Not adopted** as a phase-detection
alternative or cross-check for §2.2; ASP's own dHash+robust-MAD detector is
the right tool for this input structure.
**(b) partial**: on the 5-test verify's 3 clean composites, average-render
backgrounds looked sharp and coherent, no clear win or loss vs ASP's median at
a whole-canvas glance — a rigorous background-only crop comparison wasn't done.
(c) not yet explored (needs the GUI, out of CLI-automation scope).

### 1.3 GraphCut post-mortem experiment  `[REJECTED again, 2026-07-27 — see
.agent/cache/asp_graphcut_postmortem_2026-07-27.md]`
Fixed the two identified wiring gaps (distance-transform feathering + local
per-boundary gain correction; widened `ASP_GC_FEATHER_PX` 8→96px to match the
DP path's scale). 5-test verify: GraphCut's real seam quality got *worse* on
4/5 tests despite the fix, and the one test that passed the gate produced a
visibly corrupted image (dense scan-line artifacts a naive sharpness metric
scored as "great"). Root cause is architectural, not wiring: the low-res
seam-estimation proxy likely fragments into thin alternating ownership bands
on flat anime cel content, which feathering can't repair. `ASP_GRAPHCUT_SEAM`
stays default OFF; do not revisit without addressing the fragmentation
hypothesis first (see the post-mortem for the two candidate fixes, neither
attempted). The anime edge-cost rule was also not attempted — OpenCV's
`GraphCutSeamFinder` exposes no hook for a custom cost function short of
reimplementing the min-cut algorithm.

---

## Phase 2 — Coherence-First Core (the actual quality plan)

*Rationale: the simple stitch wins because adjacent frames ⇒ `A_animation ≈ 0`.
Give the ASP the same property via animation-phase awareness, instead of trying to
warp incompatible poses together. Evaluation §9.2 has the full sketch.*

### 2.1 `ASP_HOLD_AVERAGE=1` A/B  `[done — measured at full-corpus scale, S214/S217]`
Overmix-style ECC sub-pixel averaging within hold blocks. Needed real
engineering before it was even measurable — the benchmark had its own
disconnected frame-selection reimplementation; the S214 session that
consolidated it + the bugs found/fixed (both pre-dated this work and already
affected the GUI path) is documented in the pre-import project history, not
in this repo's `docs/moon/CHANGELOG.md` — see the note at the top of that
file. **Full-corpus (S217, n=96/97, combined with
2.3's flag — not isolated) vs the 2026-07-09 baseline**: CV verdict 31/36/27/2
(was 27/41/29/0); aligned GT-SSIM 0.694 vs 0.719 (was 0.693 vs 0.718 — flat,
expected: preprocessing win, not the pose-gap fix 2.4 targets); sharpness
+56%, ghosting −26%; 44/96 guarded fallbacks (was 46/97). Real,
non-regressive, not a breakthrough. **Stays default OFF** pending the human
visual pass (0.1) the ground rules require before any flag flips.

### 2.2 Animation-phase grouping at ingestion  `[done — S215/S217]`
`detect_animation_phases()` + `phase_spans()` in `ingestion/frame_selection.py`:
dHash change-point detection over the *selected* frame sequence, same
primitive as hold detection one level up. Measurement-only (JSON diagnostics
+ phase-strip PNG); zero behavior change confirmed. Full-corpus phase census
(n=96): 1–6 phases/test, mean 2.18, 60/96 tests multi-phase — the structure
2.3 targets is the common case, not an edge case.

### 2.3 Phase-consistent compositing  `[done — S216/S217; human ratings
(the actual success criterion) still open]`
`ASP_PHASE_COMPOSITE=1` (default OFF): seams whose two frames belong to
different phases skip midpoint-warp entirely and escalate to single-pose
from the dominant phase, via `_dominant_frame_in_band` in `compositing.py`.
`phase_ids` computed once in `AnimeStitchPipeline.run()`, shared by GUI and
benchmark (S216 caught an index-alignment bug here before shipping — again,
pre-import history, not in this repo's `CHANGELOG.md`; see its top-of-file
note). 5-test verify: neutral-to-slightly-better, 8 seams
correctly escalated, spot-checked visually coherent. Full-corpus: see 2.1's
numbers (run combined). Human ratings — the roadmap's actual Phase-2.3
success criterion ("zero coherence-class losses among true
composites" needs eyes on images, not SSIM) — have not run; that's the
next step before either flag can flip to default-on.

### 2.4 Phase-aware frame selection  `[REJECTED, 2026-07-27 — see
.agent/cache/asp_phase_aware_select_postmortem_2026-07-27.md]`
Implemented and 5-test verified: `ASP_PHASE_AWARE_SELECT` (default OFF) adds
a Pass-2 tie-break penalty against camera-step candidates that would cross
into a different candidate-level animation phase than the previous anchor.
Mean seam `post_warp_diff` did not drop (8.90→8.99 across the 5 tests, flat
to slightly worse) and one test (`asp_test57`) flipped from a safe SCANS
fallback to a real ASP attempt with visible seam/registration corruption —
a real regression, not a metric artifact. Flag stays OFF; do not re-enable
without a different mechanism (see post-mortem).

### 2.5 Background quality: Overmix-style averaging  `[REJECTED again,
2026-07-27 — see .agent/cache/asp_bg_average_postmortem_2026-07-27.md]`
`ASP_BG_AVERAGE=1` (default OFF) in `rendering.py` blends the per-pixel
temporal median toward the mean as confirmed-background sample count grows.
First measurement (S220) attributed the visible strip-banding to an abrupt
mean/median switch at the count boundary; this session fixed that (smooth
blend weight, `ASP_BG_AVERAGE_FULL_AT`) and re-measured. The banding
persists — root cause is unrelated to the switch's abruptness: some source
frames carry fine block-structured overlay content whose block boundaries
don't survive warp-then-average across misaligned frames, regardless of
blend weight. Flag stays OFF; do not re-attempt without addressing frame
content alignment for such overlays specifically, not the blend curve.

**Phase-2 exit gate:** on the 55-GT subset, ASP human coherence ≥ simple on every
test; aligned-SSIM gap ≤ 0. (Coverage wins like test96 should start flipping
`comparable` → `asp_better` once coherence losses stop cancelling them.)

---

## Phase 2.6 — Benchmark-Harness Host Freeze *(FIXED, confirmed at full-corpus
scale 2026-07-28 — issue #25 closeable)*

`bench_anime_stitch.py` repeatedly froze the host badly enough to force a
hard restart, on a 128GB RAM / 24GB VRAM machine — a real bug, not
underpowered hardware. **Root cause**: nothing in the codebase capped
OpenMP/BLAS/OpenCV/PyTorch thread pools, so each defaulted to one thread per
CPU core independently, stacking uncoordinated (numpy BLAS, OpenCV
`parallel_for_`, PyTorch intraop, the C++ `base` extension's own OpenMP in
`canvas.cpp`) — confirmed by the user's own `htop` observation. **Fix**:
`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS`
capped to 4 (`ASP_BENCH_THREAD_CAP`) before numpy/cv2/torch import, plus
`cv2.setNumThreads`/`torch.set_num_threads`. **Confirmed**: the exact 5-test
combination that previously froze the host completed cleanly with the fix —
RSS/RAM/VRAM flat across all 5 datasets, thread count settled at 22–34 (not
growing). Permanent diagnostics also added: `_resource_snapshot()`/
`_resource_danger()` abort guardrail (RAM≥80%/VRAM≥85%) and `_log_resource(tag)`
per-stage checkpoints in `process_dataset` — note these *failed* to prevent
one freeze before the thread-cap landed, so they're a backstop, not the fix.

**Confirmed at full-corpus scale, 2026-07-28**: user-authorized full 97-test
run (`anime_stitch_20260728_013215.json`) completed cleanly end-to-end —
8540.6s total (~2h22m, 83.8s/test avg), zero `_resource_danger()` triggers
across the entire run, no host freeze. This supersedes the earlier 5-test
and 18-test partial confirmations from this session and closes the "not
fully trusted at scale" caveat this phase carried since S218-S220.
Escalation path if a freeze ever recurs (lower `ASP_BENCH_THREAD_CAP`
further; audit `frame_selection.cpp`/`fg_register.cpp`/`compositing.cpp`/
`seam.cpp`/`exposure.cpp`; check model `.offload()` VRAM release) remains
documented below as a reference, not because it's still believed needed.

## Phase 3 — Photometric & Seam Parity with OpenCV

*(Only after Phase 2 — these polish composites that must first be coherent.)*

- **3.1 Joint canvas-space blocks-gain solve** `[MIXED, 2026-07-27 — see
  .agent/cache/asp_joint_gain_solve_postmortem_2026-07-27.md; stays default
  OFF]` — implemented the full Brown–Lowe 2007 formulation as specified
  (`ASP_JOINT_GAIN_SOLVE`, one least-squares system over all overlapping
  frame pairs with a bg-only, luminance-scalar, clamped gain-prior term,
  replacing §4.10's sequential chain). 5-test verify: genuine wins on 2/5
  (test08/09 — reduced visible banding, one gained a fallback→real
  composite with more coverage), roughly neutral on 2/5, and one new
  regression (test04: nudged the aggregate composite-gate score just past
  its threshold, producing a real composite with a local banding defect the
  finer-grained seam_visibility metric catches but the gate's aggregate
  statistic didn't). The approach works; the gate needs a matching
  finer-grained local check before this can default-on — not attempted here
  to keep this one measurement to one change.
  **Follow-up REJECTED, 2026-07-27** (see
  `.agent/cache/asp_dense_band_scan_gate_2026-07-27.md`): built the
  recommended finer-grained local check (`_dense_band_scan_score`, a full
  sliding-window scan vs `_strip_banding_score`'s frame-boundary-anchored
  sampling), wired into the composite gate. 5-test verify across both
  conditions (default OFF, `ASP_JOINT_GAIN_SOLVE=1`): the new metric never
  crossed its calibrated floor on any test — zero measured effect on any
  verdict. Bigger finding: **test04's original regression no longer
  reproduces** against the current codebase — the ToonOut masking fix
  (S230, landed after this postmortem was written) already changed test04's
  pixel composition enough that the *pre-existing* `sb` check alone now
  correctly gates it (`sb=40.4>35`) without any new metric. Reverted per
  the anti-goal against unproven new gates; the architectural sampling gap
  is real but currently unexercised by any known failing case, so it's
  parked. `ASP_JOINT_GAIN_SOLVE`'s own disposition should be re-measured
  fresh in a future session against the current ToonOut-inclusive baseline
  rather than inherited from this now-partially-stale postmortem.
  **Re-verified same day**: 5-test re-run against the current baseline is
  cleaner than the original — test04 now safely reverts to SCANS instead of
  shipping the defect, test08/57 unaffected (already SCANS via ToonOut
  either way), test09 still flips SCANS→real (coverage win), test27 still
  real→real (modest win). Zero regressions on this sample. **But** a direct
  look at test09's output still shows visible banding despite passing every
  gate — the same "gates pass, defect remains" pattern, just below current
  thresholds. **Stays default OFF**: ground rule #1 requires a full-97 run
  before any default flip regardless, and ground rule #2's actual gating
  requirement (Phase 0.1's human coherence rating pass) is still open — this
  5-test re-verify is encouraging, not a substitute for either.
- **3.2 GraphCut revisit** — moot; 1.3 did not survive its second measurement
  (2026-07-27).
- **3.3 Multi-band blend on final boundaries — ALREADY DONE, stale roadmap
  text corrected 2026-07-27.** This bullet's premise ("reintroduce the
  deleted C++ `multiband_blend`") is factually wrong: `base::laplacian_blend`
  (`base/src/animation/compositing.cpp`) is a genuine 5-level Laplacian
  pyramid multi-band blend — not deleted, not a stub, and already the live
  default (`LAPLACIAN_BANDS=5`, `backend/src/constants/animation.py`),
  called from `_laplacian_blend` at Stage 11's per-seam blend
  (`compositing.py:2295`). It was never removed; this bullet appears to
  predate that check, or confused it with a different retired component.
  **This reframes Finding-3's visible test09 banding** (Phase 4 analysis,
  `.agent/cache/asp_phase4_fallback_class_analysis_2026-07-27.md`): since a
  5-band blend is already active and the banding still shows through, the
  defect is **not a blending-method gap** — multi-band blending is already
  the technique in place. It points back toward an upstream photometric
  (gain/exposure) mismatch as the real remaining lever, consistent with why
  §3.1's joint gain solve (not a blending change) is what actually moved
  test09 to a real composite. No code change from this correction — closing
  this item as already satisfied rather than re-implementing something that
  exists. (The "MPB + MTOR for colour bleeding" caveat remains relevant
  research-base context if colour-bleeding specifically — as opposed to
  luminance banding — is ever observed, but is not itself evidence of a gap
  to fill.)
- **3.4 Cheap photometric candidates from the research base** `[done
  2026-07-27 — see .agent/cache/asp_toonout_fix_2026-07-27.md]`
  — **ToonOut weights**: the "mirror is gone" blocker was itself stale — a
  live mirror exists (`joelseytre/toonout` on HF). Locating it surfaced a
  real, previously-unknown bug: `TOONOUT_MODEL`/`BIREFNET_MODEL` were
  swapped, the intended fallback repo ID was never a valid HF model repo,
  and even fixed, the wrapper's weight-loading didn't handle this repo's
  filename or its `module._orig_mod.`-prefixed checkpoint — ToonOut had
  **never** actually been loaded, ever; every run silently used plain
  generic BiRefNet. Fixed all three layers and verified a real mask-level
  difference (not just "loads without erroring"). 5-test verify: 3 tests
  improve or trade a flawed real composite for a clean fallback, 1
  unaffected, 1 (test04) exposes the same gate-threshold fragility the
  §3.1 postmortem already documented (any quality improvement nudges it
  just past the gate without fixing the underlying local defect — a
  gate-design gap, not specific to this fix). Kept as the new default (bug
  fix restoring intended behavior, not a speculative feature — no flag).
  — **Reverse dimming**: checked as instructed before building. Sampled 3
  tests' gain plots for Harding's signature (a sudden luminance *drop* on
  risky frames) — found only smooth monotonic drift or isolated
  *brightening* spikes (opposite direction), already handled by the
  existing coherence gate. This R18/OVA corpus was never likely subject to
  Japanese broadcast-safety dimming in the first place. Not built.

---

## Phase 4 — Convert the Fallback Classes

The 54 guarded fallbacks (full-corpus count as of the 2026-07-28 checkpoint,
`anime_stitch_20260728_013215.json` — supersedes the 46-count pre-trim
census) are wins-by-safety, not wins. Full-corpus class breakdown:
`seam_vis_gate` 27, `composite_gate_sb` 26, `composite_gate_sc` 1,
`ghost_gate_siqe` 0, `alignment_failed` 0 — confirms the 18-test sample's
proportions held at scale (seam_vis_gate and composite_gate_sb dominant,
the other three classes negligible-to-zero). Reclassify each:
- **seam_vis_gate class (27 at full-corpus scale, 2026-07-28; re-examined
  2026-07-27 on an 18-test targeted sample — see
  `.agent/cache/asp_phase4_fallback_class_analysis_2026-07-27.md`):** **not
  uniformly "mostly pose-blend artifacts" as originally assumed.**
  Cross-referencing `seam_visibility` against the existing
  `mean_post_warp_diff` metric splits this class into two different root
  causes: some are genuinely pose-blend-driven (high on both metrics,
  e.g. test41: sv=63.8, post_warp_diff=44.7), but several of the *worst*
  seam_visibility scores in this sample occur on tests with *low*
  post-warp-diff (test08: sv=143.3 but post_warp_diff=15.3; test37: sv=65.6
  but post_warp_diff=6.7) — meaning registration is fine and the defect is
  photometric (banding/exposure), not torn or misaligned content. Tested
  whether `ASP_JOINT_GAIN_SOLVE` (§3.1, still default OFF) rescues the
  photometric subset: 2/7 flip to real composites (test09, test32), 3/7
  improve without crossing the gate (test37/08/57), but 2/7 actually
  *regress* under gain solve (test41, and notably **test51**, whose
  post-warp-diff is comparably low to test32's yet responds in the
  opposite direction) — `mean_post_warp_diff` correlates with outcome but
  isn't clean enough on its own to build an automatic per-test dispatch
  rule from. **Follow-up checked same day**: tried frame/pair count as a
  second candidate discriminator (`_joint_gain_solve` builds its
  least-squares system over all sufficiently-overlapping pairs, not just
  adjacent frames, so fewer selected frames means fewer constraining
  pairs) — test51/test41 (the two regressions) do have the fewest pairs
  (28, 45), but **test08 breaks the pattern** (36 pairs, second-fewest, yet
  *improved*). Two independent cheap heuristics now each checked and each
  falls short on a real counter-example — not a matter of finding the
  right one-line rule; closing this specific follow-up, a full per-test
  measurement pass is what's actually needed. No code shipped from this
  round; still needs full-97 data and a real per-test triage pass before
  this class can be meaningfully "converted" rather than just
  better-understood.
  — **Full-97 triage pass built and run, 2026-07-28** (`backend/benchmark/triage_fallback_classes.py`,
  new tool — parses every fallback's `fallback_reason` gate values,
  cross-references `mean_post_warp_diff`/pair count, sorts by
  margin-over-limit; full report:
  `.agent/cache/asp_phase4_fallback_triage_full97_2026-07-28.md`).
  Deliberately a triage table for human review, not a new automatic
  dispatch rule — both candidate one-line rules above already failed on
  real counter-examples, and the anti-goals rule out shipping a third
  guess as pipeline behavior. Full-corpus qualitative split (by the same
  `mean_post_warp_diff` bucketing as the 18-test sample, threshold <10 vs
  ≥30): only **1/27** seam_vis_gate tests is clearly pose-blend-leaning
  at full scale (test41) — the "mostly pose-blend artifacts" framing this
  whole section opened with is even more wrong at 97-test scale than the
  18-test sample suggested. 10/27 are clearly photometric-leaning, 15/27
  fall in the ambiguous middle this bucketing can't resolve (which is the
  same "no clean rule" finding, not a new gap). **14/27 (52%) are
  borderline** — within 10 points of the 35.0 limit — meaning a modest,
  targeted photometric-quality improvement (not a new gate, an actual fix
  to whatever's causing the banding/exposure defect in the photometric
  subset) could plausibly flip a meaningful fraction of this class without
  needing to solve the "mixed/moderate" cases at all. `composite_gate_sb`
  (26 tests) is structurally similar: median margin only 9.6 over its 35.0
  limit, 8/26 within 5 points. **This is now a "what's the actual
  photometric defect" investigation, not a "which heuristic dispatches
  correctly" one** — the next real step is picking 2-3 of the most
  borderline photometric-leaning tests (e.g. test87 margin=0.4, test10
  margin=1.3, test71/test69 margin~3.5) and visually diagnosing what
  specifically is wrong with their seam photometrics, not further
  cross-metric correlation attempts.
- **render-gate class (21) — clarified 2026-07-27:** this is the benchmark's
  own umbrella term (`timings["render_gate_fallback"]`, set to 1 or 2
  whenever *any* post-render quality check fires) covering
  `composite_gate_sc`/`composite_gate_sb`, `ghost_gate_siqe`, and
  `seam_vis_gate` combined — not a separate, unexamined class. The
  seam_vis_gate re-examination above (Findings 1-5,
  `.agent/cache/asp_phase4_fallback_class_analysis_2026-07-27.md`) already
  covers a meaningful chunk of this umbrella at 18-test scale, including
  4 `composite_gate_sb` and 1 `composite_gate_sc` case from this batch.
  The original fg-dominant high-animation-scene concern is still open and
  still needs the deliberate policy decision below — best single-phase
  reconstruction (2.3 degenerate case) vs SCANS, picked per-test by
  measurement. Accepting SCANS permanently for some tests is a valid
  outcome **if a human confirms SCANS is the best achievable** — the
  objective is "never worse than the OpenCV stitcher", which a coherent
  fallback satisfies; but each such test must beat the *raw* cv2 stitcher
  (our SCANS-on-preprocessed-frames already tends to, via selection +
  photometric prep + no crop failures).
- **alignment_failed (test49): SUPERSEDED, 2026-07-27** — test49 no longer
  fails at alignment. It now reaches the composite stage and fails at
  `composite_gate_sb` instead (`sc=39.3/limit 53.2, sb=47.1/limit 35.0`,
  `mean_post_warp_diff=20.3`), most likely as a side effect of the ToonOut
  masking fix (S230) changing the BiRefNet output that feeds bundle
  adjustment and the alignment health check. The corpus's one dedicated
  `alignment_failed` case is resolved; test49 now belongs to the
  `composite_gate_sb` class discussed above like any other test in it — no
  further individual diagnosis needed under this heading.

**Phase-4 exit gate = the objective:** for every one of the 97 tests, ASP output
human-rated ≥ the simple stitch, with `asp_better` on the coverage/sharpness
dimensions wherever a true composite ships.

---

## Phase 5 — Exceed (stretch, unscheduled)

Only once Phases 0–4 hold: per-phase super-resolution output (Overmix's actual
specialty — √N sub-pixel averaging), optional Real-ESRGAN anime_6B or APISR finish
(the two report-vetted anime SR models, §15), GC/multi-band refinements, OBJ-GSP +
SemanticStitch mesh-based seam barrier as a third seam candidate (§R), and
revisiting generative seam synthesis with whatever 1.1 found — each as a measured
A/B, and any generative step behind the report-mandated LPIPS/CLIP quality gate.

### 5.1 Bounded RL / mathematical-optimization reintroduction

**History, so this isn't attempted blind:** PSO and DRL (for multi-frame
super-resolution) and a full RLHF reward-model/trainer/feedback-tab stack
(for compositing quality) were all built during the pre-trim era and deleted
in the S200 "great trim" as unverified complexity — see
`research/ASP_Critical_Evaluation_2026-07-08.md` and
`.agent/cache/asp_state_of_the_pipeline.md` §1. They failed under exactly the
conditions that made everything else in that era fail: no coherence metric
to optimize against, no A/B discipline, default-shipped without measurement.
That is settled negative evidence about *how* these methods were used, not
about whether RL/mathematical-optimization has a place here — the two are
different questions, and this section is deliberately scoped to keep them
distinct.

Reintroduction is allowed only narrow-and-bounded, under the same Ground
Rules as everything else (one change → one benchmark → keep or revert; no
default-ON without a full-97 run; a human coherence rating, once Phase 0.1
lands, as the success criterion — never a proxy metric alone). Candidates,
in priority order:

1. **RL for pose-consistent frame selection.** The evaluation's own §9.2 and
   this roadmap's §R call frame/phase-consistent selection "the single most
   important unimplemented idea" — closer to the actual bottleneck (pose
   gaps at frame selection, not compositing) than RLHF-for-compositing ever
   was. A policy trained against Phase 0.1's human coherence ratings
   (unavailable until that rating pass runs) to pick frame groups by
   animation phase, evaluated the same way 2.4's phase-aware selection
   attempt was (and was rejected) — this must beat that attempt's numbers,
   not just exist.
2. **PSO/genetic search over the remaining gate thresholds and warp
   parameters** (the ~8 gates in `.agent/cache/asp_state_of_the_pipeline.md`
   §3, plus ARAP regularization weights) — a much smaller, more legible
   search space than "learn compositing end-to-end," with a clear objective
   (the human coherence rating) once it exists.
3. **Evolutionary search over blend/photometric parameters** (gain-solve
   priors, feather curves) — lowest priority; §3.1's joint gain solve
   postmortems suggest the current gap there is architectural (local vs.
   global fixes), not a tuning problem search would help with.

Do not revisit RLHF-style reward modeling or DRL-based super-resolution in
their prior forms without first reading why they were removed — if
reattempted, they must be re-justified against the same discipline as (1)–(3)
above, not shipped as a wholesale subsystem.

**Gate:** none of this starts before the Phase 4 exit gate, per the
Anti-Goals below — this section documents the plan, it does not schedule it.

---

## Phase 6 — Tutorials & Onboarding *(unscheduled; parallel-eligible with
Phase 2+, does not block or get blocked by pipeline-coherence work)*

Grepped `gui/` and `frontend/` for tutorial/onboarding/wizard/guide
(2026-08-06 whole-app review): zero hits. This is a genuine gap against the
stated product vision ("tutorials to help beginner artists"), not a
maturity issue on existing code — nothing exists yet, not even a stub.
Candidate approaches (not mutually exclusive, roughly cheapest-first):

1. **In-app guided walkthrough** — a first-run `QWizard` overlay in the
   PySide6 GUI walking a new user through `HybridStitchPanel`'s tool tabs
   once, dismissible, re-invocable from a Help menu. Lowest effort, most
   direct. `[DONE 2026-08-06, issue #1]`
   `gui/src/tabs/stencil/onboarding_wizard.py::HybridStitchOnboardingWizard`
   — welcome page + one page per tool tab (sequence sidebar, Control
   Points, Color Correct, Seam Painter, Mesh Warp, Render); paging through
   switches the panel's live `QTabWidget` to match, so the tour points at
   real UI. First-run only via `AppSettings` (same convention
   `_thumbnail_file_picker.py` already uses), re-invocable from a new "?"
   toolbar button in the panel (no menu bar exists in `HybridStitchPanel`
   to hang a Help menu off of). Non-modal — never blocks the panel
   underneath. 10 new tests (`gui/test/tabs/test_onboarding_wizard.py`).
2. **Content-first tutorials** on the docs site (`docs/website/` — already
   functional) — written/video walkthroughs, cheaper to iterate than in-app
   UI but doesn't meet users inside the tool. `[DONE 2026-08-06, issue #2]`
   `docs/tutorials/getting-started-hybridstitch.md` (a tab-by-tab walkthrough
   of `HybridStitchPanel`'s actual UI — sequence loading, control points,
   color correct, seam painter, mesh warp, render, and handing frames to the
   automated Stitch tab) and `docs/tutorials/pipeline-overview.md` (a
   beginner distillation of the 13-stage `AnimeStitchPipeline`, when to use
   it vs. HybridStitch), wired into `docs/mkdocs.yml`'s nav and linked from
   `docs/index.md`.
3. **Bundled sample projects** — 2–3 example frame sequences shipped with a [DONE 2026-08-07, issue #17]
   guided "try HybridStitch on this" mode; teaches by doing.
4. **Assisted-use suggestions** — surface `gui/src/elements/_stats_recommendations.py`'s
   existing diagnostics as explained, in-context suggestions ("why this seam
   adjustment") rather than a separate tutorial mode. The most
   vision-aligned option (ML-assisted teaching) but should wait until the
   automated pipeline's suggestions are trustworthy enough to be worth
   following — i.e. not before the Phase 4 exit gate.

No A/B-benchmark discipline applies here (this isn't pipeline-quality work);
normal UI review and user testing is the right bar.

---

## Parked (real gaps, explicitly deferred until the Phase-4 exit gate)

*Carried forward from the archived roadmap — these are engineering/research items
with unimplemented value that were never benchmarked or rejected, unlike the §5.x
gate factory. They are out of scope until every one of the 97 tests clears Phase 4,
but should stay visible rather than silently vanish.*

- **Full 2D canvas geometry for horizontal/diagonal scroll** (Category F/H in the
  archived taxonomy). Non-vertical scroll only gates to SCANS today; touches
  canvas/warp/seam code that assumes 1D vertical layout — high blast radius,
  multi-week, do not start before Phase 4.
- **StabStitch++ multi-axis trajectory smoothing** — wave correction is linear
  today (`np.polyfit` deg=1, §4.3); pairs naturally with the 2D-canvas item above,
  probably not worth doing alone.
- **HITL manual-correction primitives** (Fourier-Mellin manual align, arrow-based
  flow override). The evaluation credits Overmix's interactive workflow as good
  product design (§8); revisit only if Phase 2's automated compositing leaves a
  test class no automated policy can resolve.
- **ASP → HybridStitch handoff** — integration into the main app's general stitch
  tab, once quality no longer needs the benchmark harness to validate every change.

---

## Anti-Goals (do not do these; the archive documents why)

- No new quality gates or per-strip/per-seam statistics without a displaced gate
  and a full-corpus run (§5.x factory, 104 items, zero measured value).
- No threshold-tuning sessions (±0.002 SSIM outcomes, weeks lost).
- No new default-OFF flags "for later" — implement behind a flag only with the
  A/B run scheduled in the same session.
- No Phase-2-era ambitions (video ingestion, dataset harvesting, RLHF, 4K hybrid)
  until the Phase-4 exit gate is met.
- No trusting `asp_better` without looking at the image.
