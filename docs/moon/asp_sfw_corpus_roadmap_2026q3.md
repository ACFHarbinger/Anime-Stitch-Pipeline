# ASP SFW Benchmark Corpus Roadmap — 2026 Q3

**Status:** Draft + review round 1 + C0/C0.5 rewrite, then **Grok review
round 2 (2026-08-15)**: automated dual-veto scoped as an ensemble of weak
votes plus a periodic adversarial audit; C0.5 applies to SFW intake only
after Harbinger's `nsfw_97` provenance sanity check (series-taint vs
case-level content is an open policy, not a classifier output). Final
review + issue filing still waits on Claude/Harbinger close.
**Scope:** building a second, SFW-only benchmark corpus for ASP, as a
generalization check alongside the existing 97-case corpus. Companion to
[`asp_change_roadmap_2026q3.md`](asp_change_roadmap_2026q3.md) and to
`new_features.md` §4.18 (Image Board Crawler — Rating Filter & SFW Board
Support) in the parent Image-Toolkit repo, which this roadmap depends on.

## 1. Why this exists

The existing 97-case ASP benchmark corpus is sourced entirely from sexually
explicit content. This was a deliberate, already-scoped choice — Harbinger
confirmed 2026-08-15 that ground-truth-quality stitchable sequences are far
easier to source for NSFW anime (booru tag search, generous APIs, rich
metadata) than SFW: a prior attempt at a SFW corpus needed ~300 candidates
screened to find 10 usable ground-truth-quality stitches, versus ~10 found in
a 15-minute pass through 3 booru pages for NSFW content. The corpus stays
NSFW for benchmarking and research purposes; that decision is not reopened
here.

What this roadmap addresses is a real gap that decision leaves: every quality
gate defined in `asp_change_roadmap_2026q3.md`'s M0–M6 (structural red set,
five-case smoke set, the eventual 97/97 parity milestone) is tuned and
validated against one content distribution. A pipeline that looks good there
could be silently overfit to that domain's specific visual characteristics
(skin-tone palettes, particle/sparkle overlay effects, specific line-art
conventions) with nothing to catch it. A second, independently-sourced SFW
corpus is a generalization check the current promotion ladder doesn't have.

Separately, and non-blocking to the above: an eventual public-facing set of
real examples (not just abstract tutorial markdown) requires SFW content —
`docs/website` and `docs/tutorials` were confirmed clean of NSFW material on
2026-08-15 and should stay that way.

## 2. Locked decisions (Harbinger, 2026-08-15)

- **Initial corpus target: ~20–30 cases**, not corpus parity with the
  existing 97. Prove the pipeline generalizes to SFW content at all before
  investing in matching scale. Cheap to validate, cheap to abandon if the
  domain gap turns out larger than expected.
- **Frame-sequence auto-detection (similarity clustering/dedup) is deferred
  to `asp_change_roadmap_2026q3.md`'s M2.5.** Curate manually for now, using
  the same booru-tag-browsing workflow already used for the NSFW corpus.
  M2.5's Rust `graph`/`dim_reduce`/`distance` primitives are the natural fit
  for this once they exist; don't build a duplicate one-off tool first.
- **Crawler-engine work lands in the parent Image-Toolkit repo**
  (`new_features.md` §4.18): rating-filter retrofit for the existing
  Danbooru/Gelbooru crawlers, plus a new Safebooru board. Zerochan is an
  unscoped follow-up pending investigation (not an API-compatible booru
  clone).
- **Issue tracker split**: crawler-engine issues file in Image-Toolkit;
  corpus-curation issues file in ASP, matching where the code/work actually
  lives.

## 3. This roadmap does not gate M0–M6 (but C1/C2 have inbound deps)

The SFW corpus is diagnostic, not a release gate. **No M0–M6 default,
promotion step, or 97-case target waits on this corpus.** The reverse is
not a free-for-all:

| Slice | May start now? | Hard inbound dependency |
|---|---|---|
| **C0** rubric + first 3–5 candidates | Yes | None. Hand-type `rating:safe` today. |
| **C1** harvest / folder drop | Yes | **Registration** uses M0's case/provenance schema when it lands. Do not invent a parallel JSON. |
| **C2** pipeline run + 8–10 human screen | No | **After M1 adapters** (same runner as the post-M1 ungated 97-run). Pre-M1 `bench_anime_stitch.py` is the issue-#10 measurement fork. Does **not** wait on M2.5 new metrics. |

SFW/NSFW, source, licence, and web-redistribution permission are
**case-level** fields, referenced by the three output artifacts. They are
not per-output and not a second evaluation schema.

**Storage (Harbinger, 2026-08-15):** a **separate data root**
(`dump_sfw/` or equivalent) with the same local names (`asp_testNN`).
Do **not** continue the NSFW series as `asp_test98+` and do **not** mix
the trees under one default `dump/`. The bench glob is `asp_test*`
(`bench_anime_stitch.py:3861`); isolation is the data-dir, not the
prefix. Every 97-case recipe, dashboard generator, and C2 command must
take an explicit `--data-dir` / corpus root and must never default to a
merged tree. `corpus_id` in the M0 schema (`nsfw_97` vs `sfw_q3`) is the
record-level distinguisher when two `asp_test04` names exist.

## 4. Delivery sequence

### C0 — Curation workflow and rubric

**Purpose:** define what counts as a usable SFW test case before spending
curation time, since the NSFW corpus's original curation criteria were never
written down (confirmed absent from `ROADMAP.md`/`CHANGELOG.md`).

**Automation/human split (locked 2026-08-15):** do not try to automate the
quality judgment itself ("is this GT-quality") — that call is cheap and
reliable for Harbinger and expensive/unreliable to automate well. Automate
the bulk elimination *before* human review instead: tag-based pre-filtering,
near-duplicate detection, and sequence clustering (reuses M2.5's
similarity-clustering machinery once it exists) to hand Harbinger
pre-organized candidate groups, not raw search results to assemble by hand.

Deliverables:

- Write down the actual selection rubric: what makes a candidate booru
  post/sequence "ground-truth-quality" for stitching purposes (a real
  panorama/scan image as an optional GT target, source frames identifiable
  or derivable, sufficient overlap/pan structure).
- Sample Gemini's visual-style strata so C2's 8–10 human screen can be
  balanced: (1) cel-animation screencaps, (2) webtoon / monochrome manga
  vertical scrolls, (3) parallax / game-background pans. Record the
  stratum per case.
- Identify productive tag search patterns for pannable SFW sequences
  (background pan panels, scan/multi-page doujin-adjacent structure minus
  the NSFW content, official PV/BD bonus footage as a non-booru source
  worth evaluating separately).
- Track **case-level** fields on the M0 schema: `corpus_id` (`nsfw_97` /
  `sfw_q3`), source URL/board, licence, `web_redistribution_ok`, and
  `source_work_nsfw` (series-taint, distinct from `safety_tier`). **`sfw`
  (boolean) is superseded by the content-tag/safety-tier system in
  [C0.5](#c05--content-tags-safety-tiers-and-the-minor-presenting-hard-floor-2026-08-15)
  below** — do not add new `sfw: bool` writes going forward, the tier system
  is the actual schema now.
- Require a second human SFW/content pass; do not trust the board tag alone
  — this is now formalized as one half of C0.5's dual-veto gate, not a
  separate informal step.
- **GT strategy**: prioritize *stratified* GT coverage (some GT in each of
  the three visual-style strata) over raw GT count or ratio — GT is a
  diagnostic/fast-iteration aid, not the release gate (human coherence
  rating is), so a lower GT ratio than the NSFW corpus's ~57% is acceptable
  as long as every stratum has some. Prefer **more GT cases with tagged
  known defects** over fewer "perfect" ones — GT's job is validating
  *structural* correctness (alignment, no torn anatomy/duplication), which
  a cosmetically-flawed reference can still do. Add `gt_known_defects` per
  case so a flawed GT is never mistaken for ground truth on a dimension it
  can't actually validate (e.g. don't score color-shift against a GT that
  itself has a color cast).

Exit criteria: a written rubric exists; the first 3–5 curated candidates
were selected against it, not ad hoc.

### C0.5 — Content tags, safety tiers, and the minor-presenting hard floor (2026-08-15)

**Why this exists:** a binary SFW/NSFW split can't answer "is this
appropriate for a GitHub README vs. a private Discord showcase" — those are
different questions with different acceptable content, and forcing one
global calibration (strict all-ages vs. more permissive) for the whole
corpus was a false choice. A binary split is a degenerate one-tag case of
the system below, not something this competes with — building the richer
system doesn't cost meaningfully more than the simple one, and the case-level
schema groundwork (C0) already exists to extend.

**Schema:**

- `content_tags` (multi-valued enum per case): `violence`, `gore`,
  `nudity_implicit`, `nudity_explicit`, `fanservice`, `dark_themes` (mature/
  gray-morality content), and others added as needed — objective content
  description, not a policy judgment.
- `safety_tier` (named, not numeric — a raw score invites false precision
  the same way an unexplained numeric site-score would, see
  `asp_change_roadmap_2026q3.md`'s reasoning against unlabelled composite
  scores): `tier_g` (no tags beyond fully benign) / `tier_pg13` (mild
  fanservice/violence, no explicit content) / `tier_mature_sfw` (suggestive,
  dark themes, not explicit) / `tier_nsfw` (explicit).
- **Per-context policy is separate, editable config, not baked into the
  corpus.** E.g. `docs/website` public examples = `tier_g`/`tier_pg13` only,
  further excluding `violence`/`gore` even within `tier_pg13`; a private
  showcase context may allow up to `tier_mature_sfw`. Tag content once,
  objectively; change what's shown where without re-tagging anything.
- `safety_assessment` preserves both the intake decision and the policy it was
  evaluated under: `content_tags`, `safety_tier`, `policy_version`, and
  independent human/automated observations. Any later adjudication is a
  separate record, not a rewrite of either observation. This extends M0's
  case/provenance envelope; it is not an output-specific metric.

**The minor-presenting hard floor — non-negotiable, not tier-calibrated:**

- Defined by **apparent appearance, not claimed in-universe age** — an
  "actually thousands of years old" character that presents as a minor is
  still minor-presenting. No exception, no per-case override.
- **Evidence-based dual-veto gate**: human and automated assessments are stored
  independently with `high_risk` / `clear` / `uncertain`, evidence, and
  provenance. A **high-likelihood** `high_risk` finding from either assessor
  permanently excludes the case — not a low tier, a hard drop. Uncertainty is
  not itself a veto. Inclusion requires either both assessments to clear, or
  the less-uncertain assessor to select a controlled one-sided acceptance
  justification (for example, a PEGI-3 source rating) with adequate supporting
  provenance. Remaining cases are sent to a manual-review queue or rejected by
  an explicitly versioned strictness policy; that policy may use quality
  thresholds such as SSIM/PSNR, but must never disguise uncertainty as a
  confirmed risk.
- **Periodic independent re-audit as the corpus grows**, not just a
  one-time gate at intake — labeling-pipeline failure at scale (trusted
  upstream human+automated curation missing real problems as volume grows)
  is a documented, real failure mode in this exact problem space, not a
  hypothetical. Re-audit cadence TBD by whoever owns C0.5's implementation.
- This floor applies regardless of `safety_tier` or which context a case is
  used in. A case cleared for `tier_mature_sfw` still must independently
  clear the minor-presenting gate — the two systems are orthogonal, not
  nested.

#### C0.5 implementation scope (Grok, 2026-08-15 — Harbinger-locked)

There is **no apparent-age classifier in this repo**. DINOv2 (pose
embeddings), BiRefNet (fg masks), LoFTR (matches), and `AnimeStitchNet`
(4-DoF alignment) cannot be renamed into one. Photo-trained age APIs
systematically fail on stylized anime faces; treating a numeric "age 16.2"
as evidence would be false precision of the same kind this roadmap already
rejects for site scores.

**Locked strategy: ensemble of weak votes + periodic adversarial audit.**
Do not ship a single named "the automated half" that hard-drops from one
unvalidated model.

| Vote | What it actually is | What it is not |
|---|---|---|
| Upstream board tags / rating | Cheap prior (`rating:safe` plus a denylist of child-coded tags). Already available. | Not a safety guarantee. Taggers lie; style tags collide. |
| Official source rating | PEGI / CERO / TV-Y7 / BD extras rating, stored as provenance. | Only exists for licensed stills/PV, not most booru posts. |
| Local WD14-style tagger (SmilingWolf or successor) | Optional `desktop_quality` extra. Emits tags we already know how to denylist. | Not an age model. High-confidence child-coded tags are a *vote*, not a diagnosis. |
| CLIP / DINOv2 nearest-neighbour prompts | Last-resort local signal. Default output is `uncertain`. | Must never emit a numeric age. |
| Commercial anime-capable endpoint | Named later, only after a written eval on a planted holdout. | Not assumed to exist. Hive/Sightengine/Rekognition are photo-first until proven. |

**Decision rule (inclusion)** — compatible with the dual-veto paragraph
above, not a second policy:

- A **high-likelihood** `high_risk` from the human assessor is a permanent
  hard drop. No override.
- Cheap ensemble members (board tags, WD14, CLIP) **cannot emit
  high-likelihood `high_risk`** until that source passes the planted
  holdout eval below. Until then they emit `uncertain` or a queue-only
  flag. This stops a tag collision from laundering itself into a permanent
  exclude, and it keeps "either assessor may hard-drop" from applying to
  sensors we already know are noisy.
- A later-validated named source (commercial endpoint after holdout, or
  WD14 after a measured miss-rate) *may* emit high-likelihood `high_risk`
  and then the existing dual-veto paragraph applies.
- `uncertain` is never a veto and never stored as `high_risk`.
- Inclusion in `dump_sfw/` requires a human `clear` and no unresolved
  high-likelihood `high_risk`.
- Store every vote independently under `safety_assessment.observations[]`
  with `source`, `verdict`, `evidence`, `model_or_rule_id`,
  `policy_version`. Later adjudication appends; it does not rewrite.

**Periodic adversarial audit (required, not optional):**

- Before calling C0.5 "implemented", plant a small red-team set: obvious
  excludes, obvious adult-cast clears, and near-miss stylized cases.
- Re-run the ensemble every N newly registered cases *and* on a calendar
  cadence (start: every 25 cases or 90 days, whichever first).
- Publish miss/false-drop counts into the experiment manifest. If the
  planted-exclude miss rate is non-zero, freeze new registrations until
  the human re-reviews the last intake window.
- This is the actual automated safety mechanism. The models are only
  sensors feeding it.

**Where C0.5 applies (Harbinger, 2026-08-15):**

- **Intake gate for the SFW corpus only** (`dump_sfw/` / `corpus_id=sfw_q3`).
- **Not** a silent re-rate of `nsfw_97` and **not** a website/journal
  filter by itself. Public surfaces already require
  `web_redistribution_ok` plus a per-context policy.
- **Blocked on a Harbinger provenance review of `nsfw_97`.** Several
  cases may be case-level SFW (or borderline) while still coming from a
  work that is NSFW as a series. `asp_test97`'s evaluation record has
  quality notes only — no content-type field — so "is this truly NSFW?"
  cannot be answered from `asp_evaluations_20260810.json`. That review
  is human, not a Grok classifier pass.
- Add a case-level field `source_work_nsfw` (bool | unknown), distinct
  from `safety_tier`. Series-taint is a policy input ("enough to keep
  this out of `dump_sfw/` / public journal") and must not be collapsed
  into `tier_nsfw` or a dual-veto `high_risk`.

### C1 — First curated pass (~20–30 cases)

**Purpose:** the actual corpus-building work.

Deliverables:

- Manually curate ~20–30 SFW test cases via existing Danbooru/Gelbooru
  boards with `rating:safe` (hand-typed or the §4.18 Rating control). Store
  under the SFW data root as `asp_testNN`. The Safebooru *preset* waits on
  a native C++ crawler name after `base` rebuilds; it is not a C1 blocker.
- Record ground-truth availability per case honestly — most NSFW-corpus
  cases lack true GT frames and rely on human coherence judgment alone (only
  55/97 have GT, per `asp_change_roadmap_2026q3.md` §3); expect a similar or
  worse ratio here and don't overstate GT coverage.
- Store under the separate `dump_sfw/` root with C0.5 `safety_tier` /
  `content_tags` / `source_work_nsfw` filled in. Do not write a new
  `sfw: bool`, and do not drop files into the NSFW `dump/` tree.

Exit criteria: ~20–30 cases curated and stored with correct corpus-
composition metadata; C0's rubric applied consistently, documented
deviations noted per case.

### C2 — First validation pass (informational only)

**Purpose:** answer the actual question this roadmap exists to answer — does
ASP's current behavior generalize, or is it overfit to the NSFW corpus's
visual profile?

Deliverables:

- Run the corpus through the **post-M1 canonical adapter** (same runner and
  three-artifact contract as the post-M1 ungated 97-run). Do not run C2
  against pre-M1 `bench_anime_stitch.py`. Do not wait for M2.5's new CV
  metrics or the learned proxy — those may be attached later as optional
  columns.
- Compare score distributions and defect-tag frequency between the NSFW and
  SFW corpora. A large divergence is a finding to report to Harbinger, not
  something to silently tune away.
- Run a blinded human coherence screen for a predeclared representative subset
  of **8–10 SFW cases**, balanced across C0's visual-style strata. This is an
  informational generalization check, not a promotion gate and not a request
  to repeat the completed 97-case review. Automated distributions may describe
  domain shift; only this human subset supports a quality-generalization claim.
- No promotion-ladder or default-change decision is gated on this result by
  itself — it's a generalization signal, reported alongside the primary
  97-case corpus, not a replacement for it.

Exit criteria: comparative report delivered with the blinded subset's human
results; explicitly not a pass/fail gate.

## 5. Team workflow

- **Harbinger:** curates candidates (has the domain fluency this requires)
  and makes final calls on rubric ambiguity.
- **Grok:** crawler-engine implementation (§4.18, parent repo), C0.5
  ensemble/audit spec (above), and pipeline runs for C2 once M1 lands.
- **Claude:** schema alignment with `asp_change_roadmap_2026q3.md`'s M0, C2
  comparative reporting.
- **Chat/Codex:** verifies curation rubric was actually followed and that
  corpus-composition metadata is correctly tracked before C1 is called done.
- **Gemini:** not currently claimed on this roadmap; may have a role if/when
  public-facing SFW examples are surfaced on the website, out of scope for
  C0–C2.

## 6. Open questions

None currently blocking issue *drafting*. Revisit after C1's first pass if
the curation ratio turns out far worse than the ~300:10 baseline Harbinger
described. Sequencing/ID/crawler questions for Harbinger are in §8.

## 7. Grok's implementation-feasibility pass (2026-08-15)

Verified against current parent crawler + ASP bench glob, not the bus.

### 7.1 §4.18 is a convenience layer — and Safebooru is Gelbooru, not Danbooru

- `DanbooruCrawler` / `GelbooruCrawler` / `SankakuCrawler` are thin wrappers.
  `ImageBoardCrawler.run()` sends `self.__class__.__name__.replace("Crawler","").lower()`
  into C++ `base.run_board_crawler`, which only accepts
  `"danbooru" | "gelbooru" | "sankaku"` (`board_crawler.cpp:170–183`).
- Tags are a free-text field (`_config.py` `board_tags` → `config["tags"]`).
  Hand-typing `rating:safe` already works. C0 does not wait on §4.18.
- **Safebooru speaks Gelbooru dapi** (`/index.php?page=dapi&s=post&q=index`),
  not Danbooru `/posts.json`. A `SafebooruCrawler` that only sets a Danbooru
  URL will 404. A new class named `SafebooruCrawler` would dispatch as
  `"safebooru"` and **throw** in C++.
- C++ kernel work is blocked until `base` rebuilds (change-roadmap §17.7).
  Therefore §4.18 must be **Python/GUI only**: a Safebooru preset that
  reuses the **Gelbooru** engine name plus `url=https://safebooru.org`
  (and `resource=post`). Do not add a C++ crawler name in this issue.
- Rating control needs a **per-board tag map**, not one literal appended
  to every board. Danbooru uses `rating:g` / `general` / `sensitive` /
  `questionable` / `explicit`. Gelbooru's current enum is the one place
  §4.18 already flagged. Safebooru is SFW-by-site; the control is
  redundant there but must not emit a Danbooru-only token.

Effort stays Low if we follow that path. Zerochan stays unscoped.

### 7.2 C1/C2 sequencing holds — with the inbound deps in §3

From the implementer seat:

- **C0 now** is correct.
- **C1 harvest now, register on M0** is correct. I will not write a
  second evaluations JSON.
- **C2 after M1** is mandatory. The post-M1 ungated 97-run is the first
  trustworthy runner; C2 should be the same binary/adapter, different
  `--data-dir` / name prefix.
- **C2 must not wait for M2.5.** New metrics are optional columns on a
  report, not a C2 exit. Coupling them would park the generalization
  check behind a research spike.
- **Hidden coupling Chat called:** if SFW cases share the default
  `dump/` + `asp_test*` glob, every full/range bench absorbs them.
  Isolation is a **separate data root** (`dump_sfw/`), locked in §9.
- Showcase-on-website is Gemini's later surface and needs
  `web_redistribution_ok`. I will not copy third-party frames into
  `docs/website` because they are SFW.

### 7.3 M2.5 (cross-read, for Claude's issue split)

Agree with Chat: file **(a)** per-output defect analytics + interpretable
CV / subset selection separately from **(b)** a learned-proxy feasibility
spike. (b) waits for M0 schema **and** the post-M1 ungated Raw ASP
labels (otherwise we train on fallbacks labelled as ASP). 43 true
composites is a spike, not a product model.

`AnimeStitchNet` is a Siamese **4-DoF alignment regressor**
(`stitch_net.py`). It is not a human-coherence predictor. Do not rename
analytics Phase 2's RLHF item onto it. Phase 2 in
`docs/moon/roadmaps/analytics_and_interpretability.md` still says
"Reward Models in RLHF" (§2.0 and §9 TLA+). That is a parent-docs issue,
not an M2.5 algorithm issue.

Rerun.io is opt-in developer telemetry (`desktop_quality` / extra), never
a `laptop_balanced` runtime or required package. Parent analytics Phase 3
is locked A+B (Rerun sidecar + OTel); a native inspector is unscheduled.

## 8. Open questions for Harbinger (Grok)

Answered in §9.

## 9. Harbinger's decisions on Grok's SFW/§4.18 questions (2026-08-15)

1. **Storage:** separate data root (`dump_sfw/`) + local `asp_testNN`
   names. Isolation is `--data-dir`, not a name prefix. Never merge with
   the NSFW `dump/` default.
2. **Safebooru:** wait for a **native C++** `safebooru` crawler name after
   `base` rebuilds. This §4.18 pass is the Rating control on existing
   Danbooru/Gelbooru only. C0/C1 harvest via those boards + `rating:safe`.
3. **C2 vs M2.5:** C2 runs after M1 only (canonical adapter + 8–10 human
   screen). It does not wait for M2.5a metrics or the M2.5b proxy.

**Handoff:** Claude may file the SFW + §4.18 issues with this wording.
Grok implements §4.18 Rating (Danbooru/Gelbooru) when that parent issue
exists; Safebooru C++ stays blocked with the other `base` work.
