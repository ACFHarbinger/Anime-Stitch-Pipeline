# ASP SFW Benchmark Corpus Roadmap — 2026 Q3

**Status:** Draft, brainstormed 2026-08-15 (Claude + Harbinger)
**Scope:** building a second, SFW-only benchmark corpus for ASP, as a
generalization check alongside the existing 97-case corpus. Companion to
[`ASP_CHANGE_ROADMAP_2026Q3.md`](ASP_CHANGE_ROADMAP_2026Q3.md) and to
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
gate defined in `ASP_CHANGE_ROADMAP_2026Q3.md`'s M0–M6 (structural red set,
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
  to `ASP_CHANGE_ROADMAP_2026Q3.md`'s M2.5.** Curate manually for now, using
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

## 3. This roadmap does not block M0–M6

The SFW corpus is diagnostic, not a release gate. It does not block, and is
not blocked by, `ASP_CHANGE_ROADMAP_2026Q3.md`'s M0–M6 sequence. Curation can
start immediately — `rating:safe` can already be typed by hand into the
existing crawler's tags field today, working ahead of §4.18's convenience
layer — but no M-milestone in the ASP change roadmap depends on this corpus
existing, and this roadmap does not gate any default change there.

## 4. Delivery sequence

### C0 — Curation workflow and rubric

**Purpose:** define what counts as a usable SFW test case before spending
curation time, since the NSFW corpus's original curation criteria were never
written down (confirmed absent from `ROADMAP.md`/`CHANGELOG.md`).

Deliverables:

- Write down the actual selection rubric: what makes a candidate booru
  post/sequence "ground-truth-quality" for stitching purposes (a real
  panorama/scan image as an optional GT target, source frames identifiable
  or derivable, sufficient overlap/pan structure).
- Identify productive tag search patterns for pannable SFW sequences
  (background pan panels, scan/multi-page doujin-adjacent structure minus
  the NSFW content, official PV/BD bonus footage as a non-booru source
  worth evaluating separately).
- Track **SFW/NSFW as an explicit corpus-composition field** in whatever
  schema `ASP_CHANGE_ROADMAP_2026Q3.md`'s M0 defines for per-output
  metadata — this corpus should extend that schema, not invent a parallel
  one.

Exit criteria: a written rubric exists; the first 3–5 curated candidates
were selected against it, not ad hoc.

### C1 — First curated pass (~20–30 cases)

**Purpose:** the actual corpus-building work.

Deliverables:

- Manually curate ~20–30 SFW test cases via Safebooru/existing boards with
  `rating:safe` filtering (§4.18 crawler work, or manual tag entry if that
  lands first).
- Record ground-truth availability per case honestly — most NSFW-corpus
  cases lack true GT frames and rely on human coherence judgment alone (only
  55/97 have GT, per `ASP_CHANGE_ROADMAP_2026Q3.md` §3); expect a similar or
  worse ratio here and don't overstate GT coverage.
- Store alongside the existing corpus with the SFW/NSFW field set, not in a
  separate untracked location.

Exit criteria: ~20–30 cases curated and stored with correct corpus-
composition metadata; C0's rubric applied consistently, documented
deviations noted per case.

### C2 — First validation pass (informational only)

**Purpose:** answer the actual question this roadmap exists to answer — does
ASP's current behavior generalize, or is it overfit to the NSFW corpus's
visual profile?

Deliverables:

- Run the corpus through whichever pipeline is canonical at the time (ideally
  post-M1, once benchmark/backend/GUI orchestration has converged — running
  it against the pre-M1 forked benchmark script would repeat the same
  measurement-validity problem `ASP_CHANGE_ROADMAP_2026Q3.md` §10.1 already
  found).
- Compare score distributions and defect-tag frequency between the NSFW and
  SFW corpora. A large divergence is a finding to report to Harbinger, not
  something to silently tune away.
- No promotion-ladder or default-change decision is gated on this result by
  itself — it's a generalization signal, reported alongside the primary
  97-case corpus, not a replacement for it.

Exit criteria: comparative report delivered; explicitly not a pass/fail gate.

## 5. Team workflow

- **Harbinger:** curates candidates (has the domain fluency this requires)
  and makes final calls on rubric ambiguity.
- **Grok:** crawler-engine implementation (§4.18, parent repo) and pipeline
  runs for C2 once M1 lands.
- **Claude:** schema alignment with `ASP_CHANGE_ROADMAP_2026Q3.md`'s M0, C2
  comparative reporting.
- **Chat/Codex:** verifies curation rubric was actually followed and that
  corpus-composition metadata is correctly tracked before C1 is called done.
- **Gemini:** not currently claimed on this roadmap; may have a role if/when
  public-facing SFW examples are surfaced on the website, out of scope for
  C0–C2.

## 6. Open questions

None currently blocking — this roadmap can proceed to issue filing. Revisit
after C1's first pass if the curation ratio turns out far worse than the
~300:10 baseline Harbinger described, since that would change whether C2's
~20-30 case target is realistic on the original timeline.
