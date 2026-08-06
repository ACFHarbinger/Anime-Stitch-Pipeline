# The Anime Stitch Pipeline: A Critical Evaluation

*2026-07-08. An independent, evidence-based assessment of the ASP's two-month evolution (2026-04-30 → 2026-07-08), its current state, where it stagnates, and what to do next.*

*Sources: all six `.agent/cache/` analysis documents; `research/ASP_Comprehensive_Research_Report.md` and `research/Image_Stitching_Research.md`; `docs/moon/roadmaps/asp.md` (3,596 lines); the full `backend/src/animation/` (30,640 lines, 49 files) and `base/src/animation/` (5,753 lines C++) codebases; git history from `771ac756` (2026-04-30) through HEAD (307 commits); the full 97-test benchmark JSON (`anime_stitch_20260623_234305.json`, S160 code) and `dump/output/benchmark_report.md`; and a direct visual audit of ASP vs. simple-stitch vs. ground-truth outputs for 16 representative tests, including per-stage intermediates.*

---

## 1. Executive Summary

**The honest verdict: after two months and ~200 development sessions, the ASP still loses to the OpenCV simple stitch on the only measure that ultimately matters — producing a coherent, faithful image — and the gap has not narrowed.** The corpus-wide ground-truth SSIM gap was −0.026 on 2026-06-01 and −0.041 on 2026-06-23. The verdict distribution moved from 8 asp_better / 23 simple_better (June 1, 55 GT tests) to 10 asp_better / 45 simple_better / 41 comparable (June 23, all 97 tests). Roughly 190 of the ~200 sessions produced no measurable corpus-level improvement.

Your intuition is confirmed by the evidence on every major point:

1. **The simple stitch's coherence guarantee is real and decisive.** It never produces wrongly-ordered strips, duplicated body parts, or misaligned anatomy, because it only composites temporally adjacent frames (~42 ms apart) where character animation is effectively zero — it avoids the hard problem by construction. The ASP has no such guarantee, and my visual audit found catastrophic incoherence (duplicated faces, repeated torsos, black gap bands, misordered strips) in a substantial fraction of outputs (tests 07, 34, 43, 77, 82 among those I inspected).

2. **The agents' claims of ASP superiority were frequently, demonstrably wrong.** The clearest case: test07 was recorded as `asp_better` in the S142 analysis ("sharp 99 vs 49, seam coh strong") while its actual output repeats the character's bow-and-collar region six times in a vertically misordered collage. Tests 84 and 53 are scored `asp_better` in the current benchmark despite torn faces and ghost-smeared anatomy that make them visually worse than the simple stitch to any human observer. The automated metrics that produced these verdicts do not measure structural coherence, and the agents optimized against those metrics for two months.

3. **Most upgrades were useless — and it is worse than that: most were never even measured.** The last full benchmark ran on 2026-06-23 with S160 code. Sessions S161–S204+ (roughly 45 sessions, including the single highest-expected-impact change ever identified — GraphCut global seam finding, made default-ON in S161) have **never been validated on the full corpus**. The final ~40 sessions devolved into a degenerate loop that added **104 near-identical statistical "CV gates"** (§5.14–§5.112 in the roadmap: per-strip red-channel CV, green-channel CV, blue-channel CV, luma skewness CV, kurtosis CV, P90–P10 CV…), each with the same ritual: one function, one env flag, one config entry, 5–20 unit tests, zero benchmark evidence.

4. **The ASP does have genuine, verified strengths** — better framing/coverage (it preserves scene content the OpenCV stitcher crops away, e.g. test09's left-side figure), ~50% higher sharpness from sub-pixel alignment, roughly half the periodic ghosting of the simple stitch (by the one valid ghosting metric), fewer added shadows/darkening, and it wins or ties on the uniform-scroll subset (dy_cv < 0.17: +1.0% aligned SSIM). These are real but they are second-order virtues: they only matter when the output is coherent, and coherence is precisely what is not guaranteed.

**Recommendation in one paragraph:** do not keep adding features to the current pipeline, and do not delete everything either. The benchmark corpus + harness, the ground-truth set, the C++ `base.animation` kernels, and roughly ten proven components are genuinely valuable assets that took real effort to build. The Python pipeline itself — 30,640 lines, 387 environment flags, ~150 quality gates — should be **rebuilt around a coherence-first architecture at perhaps a tenth of its size**, with you making the architectural decisions and agents doing bounded implementation work under a strict measure-every-change discipline. Section 9 lays out this path concretely, along with the honest case for and against a full restart.

---

## 2. What the ASP Is Trying to Do, and Why It Is Genuinely Hard

The problem statement (correctly identified in the research reports) is: assemble a full character body from an anime pan shot in which the character is only partially visible per frame **and is animating while the camera moves**. The total foreground displacement decomposes as `F_fg = T_camera + A_animation(x,y)`; classical stitchers assume `A_animation = 0`. The genuinely insightful part of the ASP's design — and this was established early, in the first week of June — is the decomposition strategy: align the background rigidly, measure the foreground residual with optical flow, and warp the poses toward agreement (symmetric midpoint warp + ARAP regularization), falling back to a single coherent pose when the gap is too large ("never average two conflicting poses").

This is the correct research framing. It matches how ghost-free HDR and video super-resolution solved the structurally identical problem. Nothing in this evaluation disputes the diagnosis; the failure is in execution, measurement, and prioritization.

Two facts make the problem harder than the framing suggests, and both were identified in the project's own documents but never truly acted on:

- **The frame-selection ceiling.** Frames selected ~300–800 ms apart show pose gaps of 10–85 px. The midpoint warp halves the gap; it cannot close it. The project's own analysis (June, "S2 Understanding the SSIM Ceiling") concluded *"the bottleneck is upstream of compositing: better frame selection would improve SSIM more than any compositing improvement."* Despite this, the subsequent ~150 sessions were overwhelmingly compositing and gating sessions. Pose-consistent frame selection — Priority 1 in every analysis document since early June — was never made to work (three attempts regressed; the DINOv2 variant was shipped but left default-OFF behind the "GT-coupling wall").

- **The upstream fragility of the whole cascade.** When frame matching mistakes character motion for camera pan (the corpus's known root cause #1), bundle adjustment places frames at wrong offsets, the temporal median renders an incoherent plate, and *no amount of downstream seam/blend/gate engineering can recover*. My inspection of test07's intermediates shows exactly this: the Stage 9 temporal render is already a repeated-motif collage before a single seam is cut. Most of the two months of work targeted stages downstream of the point where these tests are already lost.

---

## 3. Evolution Timeline: What Actually Happened Over Two Months

Reconstructed from git history (307 commits since `771ac75`), the roadmap document history, and the session logs.

### Phase A — Foundation (May 2026, ~41 commits)
Initial pipeline assembled: BiRefNet + LoFTR + BaSiC merge (`771ac75`, Apr 30), modularization (`9602597`, May 12), an RLHF reward-model concept (`df4772b`, May 12), test suite + agent workflows (May 24). The pipeline at this stage was optimized for *static* scrolling artwork — the mismatch with animated content you identified.

### Phase B — The Diagnosis and the One Real Fix (early June, S0–S5)
The 96-test benchmark with 55 ground truths was built (June 1) — the single most valuable engineering artifact of the whole project. It immediately revealed the truth: 44% true composites, 41% render-gate fallbacks, ASP 0.669 vs simple 0.695 GT-SSIM. The core-fix stack landed: Stage 8.5 foreground pose registration (flow → symmetric midpoint warp → ARAP), A5 foreground-excluded temporal median, A6 single-pose escalation. Test04 improved +0.109. This week contained essentially all of the project's conceptual progress. Notably, the honest failure analyses from this era (§4 "What Was Tried But Didn't Work") are of high quality.

### Phase C — The Gate-Accretion Era (June, S6–S142, ~135 sessions)
An enormous volume of machinery: hold detection, GNC-TLS bundle adjustment, DINOv2 selection, LSD collinearity, ToonCrafter, HITL dialogs, TOML config, Optuna search, MLLM scoring, video ingestion design, ~75 "quick win" gates (§1.11–§1.86), each `[Quick Win] ✅ Shipped SNNN`. The S142 full benchmark after all of it: **asp_better 9, simple_better 46, GT-SSIM 0.659 vs 0.699 — worse than the June 1 baseline on both counts.** A revealing sub-arc: S11 spent effort reducing SCANS fallbacks from 51/96 to 4/96 ("fallback elimination"); S161–S163 then spent effort adding fallback gates back (dy_cv gate, SeamVisGate) because the composites that no longer fell back were bad. The pipeline whipsawed between "always composite" and "know when not to composite" without resolving the underlying alignment quality.

### Phase D — Compositing Micro-Optimization (S143–S160, 18 sessions)
Dense per-session shipping of zone normalizations (luminance, saturation, contrast, hue, chroma), cost-map blurs and smooths, feather management, histogram matching, plus the OpenCV-derived items (§4.1 blocks gain, §4.3 wave correction). Documented net effect measured at S160 (June 23 benchmark): **+1 asp_better, −1 simple_better versus S142.** Eighteen sessions, ~320 new unit tests, one net verdict flip. Aligned SSIM: 0.6795 vs 0.7195 (−5.6%); seam_visibility 6.1× worse than simple stitch.

### Phase E — The Degenerate Loop (S161–S204+, late June–July, ~45 sessions)
GraphCut seam default-ON (S161), canvas-space DP seam (S162), SeamVisGate (S163), then §5.14 onward: **104 sequential "Pipeline Strip/Seam \<statistic\> CV Gate" + "Bench \<statistic\> CV Comparative Gate" pairs**, mechanically iterating through every per-strip and per-seam statistic imaginable — including separate gates for the red, green, and blue channels individually. Also in this window: the entire C++ migration (Phases 1–7, real and competent work — 11 C++ files mirroring the Python hot paths) and the S168 adaptive dy_cv ceiling. **No full-corpus benchmark has been run on any of this.** The most recent quality numbers anyone has are from S160 code.

### Summary of the arc

| Date | Code state | asp_better | simple_better | GT-SSIM ASP | GT-SSIM simple | Gap |
|---|---|---|---|---|---|---|
| 2026-06-01 | S0–S5 | 8/55 | 23/55 | 0.669 | 0.695 | −0.026 |
| 2026-06-04 | S~10 | 7/55 | 26/55 | 0.667 | 0.694 | −0.027 |
| 2026-06-21 | S142 | 9/97 | 46/97 | 0.659 | 0.699 | −0.040 |
| 2026-06-23 | S160 | 10/97 | 45/97 | 0.653 | 0.693 | −0.041 |
| 2026-07-08 | S204+ | **unmeasured** | — | — | — | — |

The trajectory is flat-to-slightly-negative across ~200 sessions. (The 55-test vs 97-test verdict bases differ, but the paired GT-SSIM columns are directly comparable and they drift *down*.)

---

## 4. Current Status: The Visual Audit

I inspected ASP output vs. simple stitch vs. ground truth side-by-side for 16 tests spanning every verdict class, plus per-stage intermediates for the worst failure. This is what the outputs actually look like, as opposed to what the metrics say:

| Test | Benchmark verdict | What I actually see |
|---|---|---|
| **test17** | asp_better (+5.4% AlSSIM) — ASP's best result | Geometrically coherent, genuinely good — **but** visible rectangular gain-compensation patches (lighter blocky bands across the torso and upper-left). Even the flagship win has photometric artifacts the simple stitch doesn't. |
| **test09** | comparable/asp_better | ASP's real strength on display: it preserves the full composition including the man at left, **matching GT framing, which the simple stitch crops out entirely**. Coherent output, slightly washed contrast. |
| **test84** | **asp_better** (+0.046 GT-SSIM) | **Metric inversion.** ASP output has the face torn at the mouth by a mismatched seam, horizontal shear bands through the chest, duplicated anatomy, and an orange color cast. The "losing" simple stitch is perfectly coherent. SSIM rewarded ASP's larger vertical coverage, not quality. |
| **test53** | **asp_better** (cv_metrics) | **Metric inversion.** ASP output is a ghost-smeared blend — the woman's face is essentially erased. Simple stitch is clearly readable. Scored asp_better on seam_visibility 0.9 vs 1.2. |
| **test07** | recorded asp_better at S142 ("sharp 99 vs 49") | **Catastrophic and mis-scored.** The bow/collar region repeats ~6 times in misordered strips. Intermediates show the failure originates at matching/BA (tx swings 300→0 px; the Stage 9 temporal render is already an incoherent collage). Every downstream stage polished a broken canvas. |
| **test77** | simple_better (AlSSIM 0.444 vs 0.711) | Catastrophic: face duplicated twice, large black band, torso repeated 3–4×. Simple ≈ GT. |
| **test43, test82, test34** | simple_better | Same catastrophic family: shear-banded faces, duplicated strips, black gaps; test34 is a barely-parseable strip stack. All are dy_cv ≈ 2+ irregular-scroll cases. Simple stitch ≈ GT on all three. |
| **test08** | simple_better | ASP: torn face/hair with horizontal shear bands, arm discontinuities, washed contrast. Simple ≈ GT. This is the known "extreme arm motion" case where multi-frame assembly is structurally disadvantaged. |
| **test15** | simple_better (0.512 vs 0.725) | ASP: heavy haze/wash, luminance blocks, left-edge chroma corruption, displaced content. Simple ≈ GT. |
| **test12** | simple_better | ASP: chest torn by misaligned seam band, black chunk at right edge. Simple ≈ GT. |
| **test27** | simple_better (aligned) | ASP coherent but visibly banded (translucent horizontal strips over the torso) and over-cropped-in-width vs GT. |
| **test05** | comparable | ASP coherent but with the translucent staircase-banding artifact across the man's shirt and thighs. Simple ≈ GT. |
| **test90** | asp_better | Meaningless win: both outputs are small crops nothing like the GT's wide composition; ASP "won" because SCANS also failed (seam_vis 1.2 vs 32.1). |

**Aggregate visual judgment:** among the tests I sampled, the simple stitch output is the one a human would keep in almost every case except test09/test17 (and test17 needs its gain patches fixed). The simple stitch's own defects that you noted — occasional cropping (test09 dramatically), occasional darkening/shadow bands — are real but minor next to the ASP's failure modes. The benchmark's "41 comparable" bucket materially overstates ASP parity, because the CV-metric verdicts within it (and even some GT verdicts) systematically miss incoherence.

### Where the failures actually originate

The catastrophic family (07/34/43/77/82) all break **upstream**: matching/BA places frames wrongly (character-motion-as-camera-pan, irregular scroll, extreme BA geometry like test77's ratio=26.976). The moderate family (08/12/15/84) breaks at **Stage 8.5/11**: pose gaps too large for the midpoint warp, single-pose escalation not firing or firing with the wrong dominant frame, and pairwise-DP seams that conflict with each other. The photometric family (05/17/27) breaks at **gain compensation**: scalar or blocky corrections producing banding the simple stitch (which uses OpenCV's joint blocks-gain solve) avoids. This maps exactly onto the three root causes the project's own gap analysis identified — and it is worth stating plainly that the diagnosis in the documents is correct; what failed is the response to the diagnosis.

---

## 5. Genuine Strengths (Verified)

To be fair to the work, these hold up under scrutiny:

1. **Framing/coverage fidelity.** ASP regularly reconstructs the full panned extent where `cv2.Stitcher` crops (test09 is unambiguous). For your stated end-goal — full-body character reconstruction — this is the one dimension where ASP is architecturally ahead, not behind.
2. **Sharpness** (+50% Laplacian, 90/96 tests): genuine sub-pixel alignment quality, not artifact inflation (verified in the S142 analysis).
3. **True ghosting (periodic strip repetition)**: `ghosting_siqe` 36.2 vs 72.3 — ASP's irregular semantic seams avoid the simple stitch's periodic banding signature.
4. **Fewer shadow/darkening artifacts** than the OpenCV stitcher — consistent with your observation.
5. **Uniform-scroll competence:** on dy_cv < 0.17 (31/97 tests), ASP is +1.0% aligned-SSIM and 20/31 comparable-or-better. When its assumptions hold, it works.
6. **Robustness engineering:** the simple stitch hard-crashed on test95; ASP has never produced *no* output.
7. **Infrastructure assets:** the 97-test corpus with 55 ground truths (probably the only benchmark of its kind anywhere), the benchmark harness with intermediate dumps and the HTML/markdown report, the C++ `base.animation` kernels, and the honest failure-analysis documents from early June.
8. **The research corpus.** The consolidated stitching research reference and the OpenCV/Overmix comparative analyses are excellent documents — the field knowledge in them exceeds what most published anime-processing work demonstrates.

But note what this list is: infrastructure, second-order quality dimensions, and one framing advantage. None of it is "the output image is the one you'd choose."

---

## 6. Where It Stagnates, and Why

### 6.1 The measurement system cannot see the failure that matters

This is the root cause of everything else. The project's quality signals, in order of introduction:

- **Laplacian sharpness** — used for the first sessions; actively rewards torn seams (documented internally as "fundamentally wrong, actively misled development for multiple sessions").
- **`ghosting_score`** — used for ~140 sessions as a ghosting metric; it is a second-derivative *sharpness proxy*. The S142 conclusion "ASP 42% worse ghosting" — which drove weeks of anti-ghosting work — was discovered at S160 to be measuring *sharper edges*. By the true metric, ASP was 50% *better* all along.
- **`strip_banding_score`** — always 0.0 for simple stitch by construction; invalid for cross-system comparison, yet cited in comparisons for weeks.
- **GT-SSIM / aligned-SSIM** — the best available signal, but it rewards coverage and tone matching and is nearly blind to the difference between "coherent image" and "duplicated torso collage" (test84: incoherent ASP beats coherent simple by +0.046).
- **CV-metric auto-verdicts** (43 no-GT tests) — produced `asp_better` for test07 and test53, both visually destroyed.

**There has never been a human-rated coherence score in the loop.** The single most important property of the output — "does this parse as one picture of one character" — is exactly the property none of the ~25 metrics measures. An MLLM scorer (Qwen2-VL) was added in S143 to fill this gap and appears never to have been systematically used (`avg_mllm_overall: null` in the final benchmark).

When the objective function cannot see the failure, 200 sessions of hill-climbing produce what you observed: motion on every axis except the one that matters.

### 6.2 The additive-only development pathology

Everything was ever added; almost nothing was ever removed or consolidated:

- **387 distinct `ASP_*` environment flags**; a 1,673-line config schema.
- `compositing.py`: 6,939 lines, 145 functions. `pipeline.py`: 6,536 lines.
- ~150 quality gates (24 pre-DP escalation gates; ~15 post-composite audits; 104 §5.x CV gates; plus validation/BA/canvas gates).
- Most features shipped **default-OFF** and were never A/B-verified. The S142 full benchmark explicitly ran with "all flags default OFF" — meaning the benchmarked pipeline was largely the base path, and the majority of shipped work has *never* been active in any full evaluation.
- The session ritual — implement one function + one flag + schema entry + ~20 unit tests + roadmap ✅ — optimizes for the *appearance* of progress. 1,272 backend tests pass; the output images are incoherent. The tests verify that functions compute what they compute, not that outputs improved.

### 6.3 The upstream problem was named and then avoided

Every analysis document since the first week identifies frame selection and alignment as the bottleneck ("no compositing improvement can close this gap"). Yet the session distribution is overwhelmingly compositing/gates/metrics. The reasons are visible in the failure log: the three attempts at better selection (peripheral heuristic, BiRefNet two-channel, gradient pose metric) all *regressed GT-SSIM* — partly because of the GT-coupling problem (the GT panoramas were made from specific frames; selecting different frames changes the target). Rather than resolving that evaluation problem (e.g., by rating outputs on their own merits), the project retreated to the stages where unit tests are easy and regressions are invisible. **The GT-coupling wall was treated as a wall; it is actually a measurement bug** — it says "our score punishes better frame selection," which is an indictment of the score, not the selection.

### 6.4 The best ideas were implemented and then abandoned unmeasured

GraphCut global seam (the #1-ranked fix in three separate analyses), canvas-space DP seam, blocks gain compensation, hold-block sub-pixel averaging (the key Overmix idea), masked-median plates, RAFT-in-test-env — all exist in the codebase. GraphCut went default-ON in S161. **The last full benchmark predates S161.** The project cannot currently answer the question "did the highest-impact change of the entire effort help?" This is the starkest process failure: 45 further sessions were spent adding §5.x gates instead of running the 3.5-hour benchmark that would validate the flagship change.

### 6.5 Structural limits of the current architecture

Even with perfect execution, some of the loss is architectural:

- **Pairwise 1-D DP seams** cannot route around 2-D obstacles and adjacent seams conflict (both may claim the same background corridor → double-image). Both comparative analyses call the monotone-DP a fundamental limitation vs. OpenCV's global min-cut.
- **Pairwise Laplacian blending** vs. simultaneous multi-image pyramid: transition quality degrades where ≥3 strips overlap.
- **Zone-local photometric fixes** vs. joint canvas-space gain solve: the blocky patches in test17 are the visible signature.
- **dy_cv ≥ 0.5 regime (22/97 tests): −13.2% aligned SSIM.** Irregular scroll → pose gaps ARAP cannot reconcile → escalation cascades. For this third of the corpus the multi-frame-assembly premise itself is questionable; the correct behavior is what S161's gate finally did — don't try.

---

## 7. What Was Tried and Failed — the Meta-Lessons

The documented failures (all from the project's own logs, which are commendably honest on this):

| Attempt | Outcome | Underlying lesson |
|---|---|---|
| Peripheral two-channel selection | −0.03 on tests 27/57 | Character isn't reliably central; heuristics substituting for segmentation fail |
| BiRefNet two-channel selection | test04 0.742→0.604 | Correct signal, but the GT-coupled metric punishes any timing change — the *metric* blocked the fix |
| Global reference pose warp | test27 0.709→0.558 | Flow error × warp amplitude is destructive on flat cels; α must be confidence-capped |
| Gradient pose-similarity selection | −0.043/−0.026 | Image-level similarity confounds pose with scroll position; needs pose embeddings |
| ARAP Push phase, LSD collinearity, asymmetric cell sizes | zero measurable effect | Flow quality was never the binding constraint — the pose gap is |
| post_warp_diff / max_residual threshold tuning | ±0.002, scene-dependent | Global scalar thresholds cannot fit per-scene variation; weeks were spent tuning them anyway |
| 18 sessions of compositing polish (S143–S160) | +1 verdict net | Polishing downstream of a broken canvas is capped at zero |
| 104 CV gates (S168–S204) | unmeasured | A development loop with no evaluation feedback degenerates into ritual |

Meta-lesson 1: **every genuinely promising direction died at the evaluation layer** — either the metric punished it (selection changes), couldn't detect it (ARAP correctness), or was never run (GraphCut).

Meta-lesson 2: **threshold-tuning and gate-adding have a hard ceiling.** Perhaps 80 of the 200 sessions were some form of "detect bad case → threshold → escalate/fallback." The end state of that road, fully traveled, is a perfect classifier for "when should we output the simple stitch instead" — i.e., convergence to the baseline, not superiority over it.

---

## 8. Lessons from Overmix and OpenCV That Remain Unlearned

The comparative analyses extracted the right lessons on paper. In practice:

**From OpenCV (the reasons simple stitch wins):**
1. *Global consistency over local optimization* — joint gain solve across all images, one global min-cut for all seams, simultaneous multi-band blending. ASP does everything pairwise/zone-local. Implemented analogues exist (§4.1/§4.2/§4.6) but are unverified.
2. *Coherence by construction* — SCANS mode composites adjacent frames under a 4-DOF model with sane validation; its worst output is bland, never surreal. The deepest lesson: **a stitcher's first duty is to never produce an impossible image.** OpenCV honors it structurally; ASP tried to retrofit it with ~150 gates.
3. *Parsimony* — OpenCV's whole stitching module is smaller than `compositing.py`, has ~5 tunables, no fallback chain, and wins 46% of tests. Sophistication sited at the right stages beats sophistication everywhere.

**From Overmix (the specialist that actually ships anime reconstructions):**
1. *Frames are samples, not tiles.* Overmix's core is averaging many aligned samples per output pixel (√N noise reduction, sub-pixel SR) — there is no seam, hence no seam artifact. ASP's analogue (§3.12A hold-block averaging) was implemented in S144 and left **default-OFF**.
2. *Separate the animation, don't fight it.* Overmix's `AnimationSeparator` clusters frames by animation phase *first*, then reconstructs per-layer from mutually consistent frames. This is precisely the "on-twos/threes" structure of anime, solved at ingestion — the pose-consistent selection ASP never achieved. This remains the single most important unimplemented idea.
3. *Human-in-the-loop as workflow, not checkpoint.* Overmix is an interactive desktop tool: the operator picks the frame group, watches alignment, re-runs. ASP built HITL dialogs but the benchmark-driven culture treated the pipeline as something that must succeed autonomously on 97 tests. For a personal art-reconstruction tool, Overmix's interactive stance is the correct product design — and it is what your GUI's stitch tab already half-implements.
4. *Sub-pixel discipline throughout* (float positions in the accumulator, Mitchell resampling at the end) vs. ASP's integer-warp compositing with Lanczos sprinkled in places.

---

## 9. Where to Go From Here

### 9.1 The options, honestly weighed

**Option A — keep upgrading the current ASP.** Not defensible. The marginal session yields ~0 measured improvement; the codebase's complexity now actively impedes reasoning (387 flags means the "pipeline" is really 2^387 pipelines, and nobody knows which one runs); and the development loop that produced it has demonstrated it will keep producing §5.x gates.

**Option B — full scorched-earth restart (delete backend + base ASP code, rewrite manually).** More defensible than A, but it destroys the assets that are actually good: the benchmark corpus/harness (irreplaceable), the C++ kernels (competent, recently debugged), the wrapper layer for BiRefNet/LoFTR/RAFT, and the hard-won negative knowledge in the failure logs. A restart that keeps the *evaluation* infrastructure but rewrites the *pipeline* is Option C. A restart that deletes everything including the benchmark would be repeating the project's core mistake — building without measurement.

**Option C — keep the assets, rebuild the pipeline small, coherence-first (recommended).** Details below.

**On the roadmap:** the current `docs/moon/roadmaps/asp.md` is unsalvageable as a plan — it is 3,596 lines of which ~2,000 are shipped-gate archaeology. Whatever option you choose, archive it and write a one-page replacement. The Phase 2 (video ingestion, multi-modal HITL) and dataset-harvesting ambitions should be explicitly parked until the core produces images you'd keep.

### 9.2 The recommended architecture: "coherent by construction, enhanced where safe"

The simple stitch wins because of one structural decision (adjacent frames only ⇒ `A_animation ≈ 0`). The ASP's virtues (coverage, sharpness, no periodic banding, background plates) are all *enhancements*. So invert the architecture — make the coherence-preserving path the spine, and apply ASP technology only as bounded refinements that provably cannot reorder or duplicate content:

1. **Stage 0 — animation-phase grouping (the Overmix lesson, the one big missing piece).** Cluster frames by animation phase (dHash/MAD hold detection already exists; DINOv2 features already exist) *before* anything else. Within a phase group the character is a rigid object and everything downstream becomes classical stitching.
2. **Stage 1 — translation-chain alignment on adjacent frames only**, bg-masked phase correlation (exists: `cam_flow.py`), accumulate along the chain, one global least-squares polish. No skip edges, no 6-DOF, no BA drama: adjacent-pair displacement at 42 ms is small and reliable. Validation = monotonicity + step-bound checks; on failure, *stop and ask the user* (HITL), don't cascade.
3. **Stage 2 — per-pixel reconstruction, not strip compositing.** For each output pixel: prefer the median/mean over frames of the same animation phase where it's background (Overmix averaging, √N cleanup, no seam); where it's foreground, take it from the *single* phase group with best coverage (single-pose principle promoted from fallback to default). Blending between phase groups happens only in background regions. **A body part can then never be assembled from two poses** — the failure class disappears by construction rather than by 150 gates.
4. **Stage 3 — global photometric solve** (joint blocks-gain across all placed frames — the OpenCV lesson; the C++ `exposure.cpp` kernel exists) and multi-band blend of the (background-only) transitions.
5. **Optional enhancers, each gated by a single measured A/B:** sub-pixel hold averaging, GraphCut for background seam placement, ProPainter for uncovered background, Real-ESRGAN at the end. The midpoint-warp/ARAP machinery becomes an *opt-in experiment* for the minority of cases where two phases must be reconciled — not the default path.

This reuses: the corpus, the harness, `frame_selection.py`'s hold detection, `masking.py`, `cam_flow.py`, the C++ kernels, the model wrappers. Estimated size: 3–5k lines of Python. Everything else — the 104 CV gates, the 24 escalation gates, the zone-normalization chains, most of `compositing.py` — is deleted, not ported. (Keep the old pipeline on a branch for reference; don't let it haunt the new tree.)

### 9.3 Process changes — the actual root-cause fixes

These matter more than the architecture, because the architecture failure was *produced* by the process failure:

1. **Human visual rating becomes the primary metric.** A 30-minute pass rating each test 0–4 on coherence (4 = keepable, 0 = incoherent) creates the ground truth the project never had. ~100 ratings; redo per milestone, not per session. Every automated metric gets calibrated against it, and any metric that disagrees with it on ranking (as SSIM does on test84) is demoted to diagnostic.
2. **A hard coherence gate in the benchmark verdict**: any output with duplicated/misordered content loses, whatever its scores. Even a crude detector (the existing `ghosting_siqe` peak + your rating) beats the current verdict logic. An "asp_better" that looks like test07 must be structurally impossible.
3. **One change → one benchmark → keep or revert.** The 5-test rapid subset (04/08/09/27/57) per change; full 97-test per feature; nothing merges default-ON without the numbers. No more implementing ten features and benchmarking none. The 3.5 h full run is cheap next to a wasted month.
4. **Deletion discipline and a complexity budget.** Cap env flags at ~20. A new gate must displace an old one. If a flag has been default-OFF and unverified for a month, delete it.
5. **Constrain the agents to bounded, verifiable work.** The record shows LLM agents were excellent at: research synthesis, benchmark/report infrastructure, C++ porting against a reference implementation, and honest *post-hoc* failure writeups. They were reliably bad at: choosing what to work on next, judging output quality from metrics, resisting additive rituals, and stopping. So: you own the roadmap, the priorities, and the visual quality calls; agents implement one specified, benchmarked change at a time and are never allowed to declare quality improvements — only to report metric deltas plus the side-by-side images for *your* verdict. "Autonomous 200-session self-directed improvement" is the experimental condition that just failed; "human-directed, agent-executed, benchmark-gated" is the one this project's own history says works (Phase B, the C++ migration, the benchmark harness were all of that kind).

### 9.4 On the question "are LLM agents wholly incapable of this task?"

Not wholly — but the two-month record is clear about the boundary. The agents performed at a high level on every task with a verifiable target (port this algorithm, build this harness, mirror this OpenCV component) and failed at the open-ended research-engineering loop, for three compounding reasons: they optimized proxy metrics that didn't encode the goal; they never looked at the pictures (or when infrastructure to do so existed — MLLM scorer, HITL dialogs — they didn't route decisions through it); and their per-session incentive structure ("ship something, add tests, mark ✅") rewarded legible motion over progress. The result was a pipeline that is impressive in every dimension except its output. That is not a capability the next model tier fixes by itself; it is fixed by the harness you put the agent in — tight evaluation loops, human perceptual ground truth, and the authority to delete. The most efficient division of labor for the rebuild: **you spend your time on frame-group curation, visual verdicts, and architectural decisions; agents spend theirs on implementation, benchmarks, and ports.**

---

## Appendix A — Key Quantitative Facts

- **Codebase:** `backend/src/animation/` = 30,640 lines / 49 files (compositing.py 6,939; pipeline.py 6,536; config.py 1,673). `base/src/animation/` = 5,753 lines C++ / 11 files. 387 distinct `ASP_*` env flags. 1,272 backend tests (S160).
- **Roadmap:** `docs/moon/roadmaps/asp.md` = 3,596 lines; ~75 §1.x quick-win gates; 104 §5.x CV gates (§5.14–§5.112); sessions referenced through S204 (matrix updated 2026-07-02).
- **Benchmark (2026-06-23, S160, 97 tests, 12,447 s):** verdicts 10 asp_better / 41 comparable / 45 simple_better / 1 insufficient. GT-SSIM 0.6526 vs 0.6934; aligned-SSIM 0.6795 vs 0.7195 (−5.6%). seam_visibility 25.77 vs 4.21 (6.1× worse). ghosting_siqe 36.21 vs 72.34 (49.9% better). sharpness 96.7 vs 64.3 (+50.2%). dy_cv regimes: <0.17 → +1.0%; 0.17–0.50 → −5.1%; ≥0.50 → −13.2%.
- **No full benchmark exists for any code after S160** (GraphCut default-ON landed S161).

## Appendix B — Visual Audit Materials

Side-by-side montages generated for tests 05, 07, 08, 09, 12, 15, 17, 27, 34, 43, 53, 74, 77, 82, 84, 90 (scratchpad `montages/`); intermediates inspected for test07 (`dump/asp_test07/output/plots/`, `panorama_stages/`). Reproduce with:
`montage -label ASP dump/output/asp_testNN_anime_stitch.png -label SIMPLE dump/output/asp_testNN_simple_stitch.png -label GT dump/ground_truth/asp_testNN.* -tile 3x1 -geometry 480x720+5+5 out.png`
