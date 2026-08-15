# ASP Outreach & Results Blog — 2026 Q3

**Status:** Gemini architecture + Harbinger §6 locks (2026-08-15), then
**Grok review round 2:** O1 / performance-claim articles wait for a
preregistered complementary split. Until that split exists, only Lab Notes
(instrumentation, failures, protocol). Not a public-quality blog yet.

## Goal

An Overmix-blog-style outreach presence: write up actual results, the
reasoning behind real decisions (the kind of thing already happening in
`asp_change_roadmap_2026q3.md` and this repo's `docs/reports/`), not
polished marketing copy. Spillerrec's Overmix blog is the explicit
reference point — rigorous, unusually formal methodology for this exact
problem space (anime/game frame stitching), genuinely useful to the kind of
person who'd want to read it, not written to persuade.

## Why (Harbinger's stated rationale, 2026-08-15)

Two purposes, not one:

1. **Collaboration/contribution magnet.** ASP is GitHub-only, not
   commercially available — the audience this could attract is people who'd
   read real technical reasoning and want to contribute (PRs, ideas), not
   passive consumers. The tag/safety-tier system from
   `asp_sfw_corpus_roadmap_2026q3.md` §C0.5 is directly relevant here: what
   gets shown publicly is tier-gated (`tier_g`/`tier_pg13` by default),
   separate from the fuller internal corpus.
2. **A testbed for outreach approach, feeding a different, later project.**
   Explicitly stated as a secondary goal: learning what kind of outreach/
   content approach actually works here is meant to inform marketing
   strategy for the separate PMF mobile game project. Whatever gets tried
   here should be measured/observable enough to actually learn something
   from — not just "post things and hope," if that's achievable without
   over-engineering a small blog.

## 4. Proposed Architecture & Design (Gemini Brainstorm, 2026-08-15)

### 4.1 Platform & Visual Integration (The "Optic Lab Journal")
- **Integrated Route in `docs/website/`:** Host the journal under `/journal` (or `/lab-notes`) within the existing React 19 + Vite website.
- **Visual Identity:** Reuses the established **Optic Lab (Obsidian/Silver/Cyan-Magenta)** and **Blueprint Lab** aesthetic. Articles read like high-end research lab notebooks (similar to *Distill.pub* or *Overmix*), pairing rigorous technical prose with deep slate typography.
- **Zero Additional Hosting Overhead:** Bundled with the main documentation site build; version-controlled in the repository alongside source code.

### 4.2 Distill-Style "Explorable Explanations"
- Rather than static blog posts with flat PNGs, articles feature **live interactive widgets**:
  - Embedded **Synced Diff Loupe**: Readers can hover over composite seams to inspect pixel deltas and optical flow vectors directly in the browser.
  - Interactive **Cel-Pose Timeline Slider**: Demonstrates how changing keyframe hold selection alters foreground cel composition in real time.
  - Embedded **2D Stitching Viewfinder / 3D Exploded-View**: Readers can interactively orbit the 3D layer stack to understand how background plates and foreground cels separate.

### 4.3 Direct Pipeline Bridge from M6 Review Screen
- **1-Click Case Study Exporter:** The M6 Review Screen (PySide6 / Web Inspector) will support an "Export Case Study Bundle" action.
- Serializes only publication-approved derived assets, anonymized telemetry,
  comparative crops, and a redacted `.asp-session.json` manifest into a
  self-contained JSON/MDX bundle. It must not export raw corpus frames, source
  URLs, reviewer data, or personal browsing data by default.

### 4.4 Content Safety Tiering & Showcase Policy
- **Default Public Baseline:** Only cases tagged `tier_g` or `tier_pg13` with `web_redistribution_ok = True` are displayed publicly by default.
- **Opt-in Technical Edge Cases:** For complex technical stitching challenges involving `tier_mature_sfw` content (e.g. dynamic combat sequences or stylized dark fantasy plates), the journal provides an opt-in warning banner before revealing the interactive comparison.
- **Minor-Presenting Hard Veto:** Zero tolerance; cases with apparent minors are excluded unconditionally from all public journal entries.

### 4.5 PMF Outreach Testbed & Lightweight Analytics
- **Goal:** Learn what technical storytelling formats resonate with developer and game creator communities to build an audience playbook for the PMF mobile game.
- **Measurement Strategy (Privacy-Preserving & Lightweight):**
  - Track GitHub traffic referrals and star velocity correlated with journal publication dates.
  - Track reader engagement on interactive widgets (anonymous local interaction event counts, e.g. % of readers who interact with the seam slider).
- Measure qualitative inbound contribution (issues, discussions, PRs citing specific article methodology).
- Do not identify or profile individual readers. Use aggregate, privacy-
  preserving measures only; this work does not imply a public corpus release
  or create an obligation to operate one.

---

## 5. Delivery Sequence

### O0 — Journal Framework & Layout (`docs/website/`)
- Implement `/journal` index and article view inside `docs/website/`.
- Build the markdown/MDX parser and embeddable React interactive widget harness (`DiffLoupe`, `TimelineScrubber`, `LayerStack3D`).

### O1 — Foundational Article #1: The Geometry of Cel Animation
- **Topic:** *"Why Classical Homography Fails on 2D Cel Animation: Deconstructing Pose-Mixing, Seam Tearing, and the Clean Background Plate."*
- Uses SFW showcase test cases (e.g. Test 14 manual hold selection oracle) to clearly articulate the problem ASP solves.
- **Evidence gate (Harbinger, 2026-08-15 — tightened):** do **not** publish
  O1 or any other performance-adjacent article until a **preregistered**
  subset shows a complementary split — ASP and SCANS each winning a real
  slice, not ASP merely being less-bad on already-easy SCANS cases. The
  current 97-case score-order (10 / 38 / 49) and the 43 true-composite
  mean of 1.326 are the opposite of that bar.
- **Until the split exists, public writing is Lab Notes only:** M1
  session protocol, metric inversion (sharpness/ghosting anti-correlated
  with humans), fallback-carries-the-mean, C0.5 dual-veto design. Lab
  Notes may use synthetic layered fixtures and diagrams. They must not
  embed third-party frames unless `web_redistribution_ok` and C0.5-clear.
- A complementary split is a valuable result even when ASP does not win
  every case. Improvements on already near-perfect SCANS cases do not
  count as that split.

### O2 — M6 Case Study Exporter Integration
- Wire the 1-click case study export in the M6 review screen to output journal-ready interactive bundles.

### O3 — Staged Benchmark & Experiment Writeups
- Post-milestone deep-dives (e.g. M3 BiRefNet single-pose results, M4 motion-compensated camera trajectory findings) presenting empirical before/after evidence.
- Keep experiments milestone-driven and non-obligatory: no public corpus
  release, regular marketing cadence, or outreach expansion is in scope unless
  future circumstances create sustained niche demand and the core research work
  remains healthy.

---

## 6. Harbinger's Locked Decisions (2026-08-15)

1. **Interactive Focus:**
   - **Full Explorable Explanations:** Lean heavily into Distill.pub-style interactive widgets (interactive seam scrubbers, diff loupes, optical flow vectors, 3D layer stacks) rather than passive flat images.
2. **Publication Cadence:**
   - **Milestone-Driven Core + Selective Lab Notes:** Primary cadence aligns with major completed milestones (M1, M3, M4, M6). Shorter "Lab Notes" are reserved for genuinely high-value intermediate breakthroughs or failure analyses that must never queue up or block core development work.
3. **PMF Game Synergies:**
   - **Careful Exploration:** Explore 2.5D asset pipeline and parallax background applications where technically natural, while keeping the core technical rigor focused on the animation stitching problem.
4. **M6 Case Study Exporter Formats:**
   - **Dual Output:**
     1. **JSON / MDX Bundle:** Drop-in component and telemetry payload for direct rendering in the `/journal` web application.
     2. **Multi-Layer PNG + Metadata Archive:** Clean static background plate + alpha-masked character cels with layer bounding boxes for external validation and 2.5D game engine importing.

---

## 7. Team

- **Gemini:** Lead for visual presentation, layout, interactive widgets, and journal design.
- **Grok:** Lead for technical result verification, algorithm telemetry, and M6 exporter integration.
- **Chat/Codex:** Lead for factual accuracy review, tone consistency, and schema verification.
- **Harbinger:** Final editorial sign-off on case study publications.
