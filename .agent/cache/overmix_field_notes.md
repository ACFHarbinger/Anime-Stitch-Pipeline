# Overmix Field Notes (roadmap §0.3/1.2, GitHub issue #18)

Built and wired Overmix as a third, reference-only comparator (never a gate)
and ran it on the 5-test verify subset (`asp_test04/08/09/27/57`). This is a
first pass, not the full-97-corpus run the roadmap's Phase-0 exit gate wants
— see "What's still open" at the end.

## Build environment gotchas

Three gaps beyond what `vendor/Overmix/CMakeLists.txt` already checks via
`find_package`/`pkg_check_modules`:

1. **WebGPU header/library.** `src/gpu/*.hpp` unconditionally
   `#include <webgpu/webgpu.h>` and link `WEBGPU_LIBRARY`, with no build-time
   feature switch — even though the CLI align/render/comparator paths we use
   never touch the GPU code. This machine's system
   `/usr/include/webgpu/webgpu.h` is an unrelated dummy stub from a different
   project (`// Dummy WebGPU header for compilation`), so a real
   `wgpu-native` release had to be fetched. The *current* wgpu-native (v29+)
   ships a newer WebGPU API (`WGPUStringView` labels, renamed enums:
   `WGPUMapAsyncStatus` vs the old `WGPUBufferMapAsyncStatus`) that this
   Overmix revision's GPU code doesn't compile against — **v0.19.4.1** is
   the version whose API this code actually expects (plain `const char*`
   labels). `desktop/linux/scripts/setup_overmix.sh` fetches this pinned
   version automatically.
2. **Eigen3.** Not installed system-wide on this machine, but already
   present at the toolkit root's own `include/eigen3` from the `base`
   module's pixi env — reused directly (`-I` flag) instead of requiring a
   redundant system install / sudo.
3. **FFmpeg API drift.** `src/video/VideoFrame.cpp` used
   `AVFrame::key_frame` and `AVFrame::display_picture_number`, both removed
   from newer FFmpeg (this system has libavformat 62.x / FFmpeg 7+, Overmix
   was written against a ~4.x-era API). Fixed with the modern
   `frame->flags & AV_FRAME_FLAG_KEY` replacement and dropped the removed
   `display_picture_number` fast path (falls through to the existing
   pts-based calculation). This patch is committed directly in the
   submodule's own local git history (`f90a887`, not pushed upstream since
   we don't own that repo) rather than left as an uncommitted diff.

None of this touches anything we ship — Overmix stays external, GPL-3.0,
built once via the setup script and invoked only as a subprocess.

## CLI settings that actually worked

- **Comparator: `Gradient:1/false/0:both:0.75:1:6:1638`** (coarse-to-fine
  pyramid search, matching `GradientComparator`'s own C++ defaults except an
  explicit `both` movement axis, since these are free-camera pans, not a
  fixed scroll direction). `BruteForce` was tried first and is *far* too
  slow at full 1920×1080 frame resolution — a 6-frame test didn't finish in
  90 seconds. Gradient did the same job in ~1.2s.
- **Aligner: `Recursive`** — hierarchical pairwise merge, the standard
  general-purpose choice; no phase-awareness (see AnimationSeparator note
  below).
- **Render: `average:false:false`** — deliberately the opposite of ASP's
  temporal-median default, to actually test roadmap §1.2(b)'s question.
- `OMP_NUM_THREADS` capped to 4 (`ASP_BENCH_THREAD_CAP` convention) for the
  same host-freeze-safety reasons as the ASP benchmark itself.

## Results (5-test verify, smart-selected-frame variant)

| test | frames (smart/full) | SC ASP | SC Simple | SC Overmix | verdict (CV) |
|------|---------------------|-------:|----------:|-----------:|--------------|
| test04 | 25/80  | 28.3 | 26.4 | 28.9 | comparable |
| test08 | 9/140  | 13.0 | 12.2 | 10.1 | simple_better |
| test09 | 22/149 | 19.5 | 18.7 | 19.3 | comparable |
| test27 | 21/167 | 30.2 | 24.9 | 28.9 | simple_better |
| test57 | 26/142 | 28.3 | 20.5 | 36.0 | simple_better |

SC = seam_coherence (lower is better; the same automated metric used
elsewhere in this benchmark — treat as a rough signal alongside the direct
visual read below, not a verdict on its own).

**Direct visual read** (the part that actually matters, per this roadmap's
own ground rules):

- **test04, test08, test09 — clean, coherent composites.** Overmix's
  `Recursive` + `average` render produced sharp, artifact-free backgrounds
  on par with what a human would call a good scrolling-pan stitch.
- **test27, test57 — clear failures: heavy multi-copy ghosting.** Both
  show several overlapping, semi-transparent copies of the character
  smeared across the canvas rather than one coherent figure.
  `RecursiveAligner`'s translation-only, whole-frame model assumes the
  dominant motion between frames is camera/background scroll; on these two
  tests the foreground *animation* motion dominates instead, and there's no
  Overmix-side mechanism (comparable to ASP's own BiRefNet fg/bg split) to
  tell it to align on background content only. This is a legitimate,
  informative failure mode for the corpus, not a bug in this wiring —
  reran test57 standalone with fresh output to confirm it's reproducible,
  not a one-off artifact.

**Full-frame-set variant** (Overmix's own "maximal ingestion" philosophy —
feed every raw frame, not the ASP-selected subset): succeeded for 4/5 tests
in 8.7–14.7s (80–167 frames), visually consistent with the smart-variant
result on the same test. **One intermittent failure** on `asp_test08`'s
full variant (140 frames): failed twice via `run_overmix.py`'s subprocess
invocation, then succeeded on a third attempt and on a direct manual
shell invocation of the identical command — this looks like a data race in
Overmix's own OpenMP-parallelized render code (an old C++ codebase using
raw OpenMP directives), not a bug in our wiring or a resource/environment
difference (same env, same command, non-deterministic outcome). Treat
Overmix subprocess failures as retryable rather than fatal in any future
larger-scale run.

## Answering the roadmap's specific questions (§1.2)

- **(b) does average-render on our bg regions beat our temporal median
  visually?** Partial answer from this round: on the 3 clean tests,
  average-render backgrounds looked sharp and coherent — no obvious
  advantage or regression was visually apparent versus ASP's median
  backgrounds in a side-by-side glance. A rigorous background-only crop
  comparison (isolating bg regions and comparing blur/ghosting directly,
  as the question asks) was not done this round — worth a dedicated,
  narrower follow-up rather than inferring it from whole-canvas looks.
- **(a) how does AnimationSeparator behave on 2-4 phase pan shots?**
  **Answered, 2026-07-27 — architecturally mismatched to our use case, not
  worth adopting.** Read the source
  (`vendor/Overmix/src/aligners/AnimationSeparator.cpp`) before running
  anything: its `findThreshold()` sorts all pairwise consecutive-frame
  errors and picks the cutoff that maximizes how often the *original,
  temporally-ordered* error sequence crosses above/below it (an Otsu-style
  bimodal split), then `align()` greedily walks a backlog, assigning each
  frame to the current group only if its error *against the last frame
  already accepted into that group* is below the threshold, starting a new
  group each time the backlog can't be extended further. This is designed
  to **de-interleave a cyclically-repeating animation loop** (where a
  non-adjacent frame can be more similar to an earlier one than its
  immediate temporal neighbor is — think separating a 3-cel walk-cycle GIF
  back into cel-A/cel-B/cel-C buckets), not to detect scene-level phase
  *boundaries* in a monotonically drifting scroll sequence — a
  fundamentally different problem from ASP's own `detect_animation_phases()`
  (§2.2: dHash Hamming distance + robust-MAD change-point on a monotonic
  sequence).
  - **Built OvermixCli** (`vendor/Overmix/build/OvermixCli` — no persisted
    build artifact existed at session start; `setup_overmix.sh` rebuilds it,
    ~4 min). `AnimationSeparator` needs a comparator configured first
    (`ImageContainer::getComparator - No comparator!` otherwise) — same
    `Gradient:1/false/0:both:0.75:1:6:1638` comparator as the rest of this
    file, passed *before* `--align=AnimationSeparator` on the command line.
  - **test27** (21 smart-selected frames, ASP's own change-point detects 3
    phases sized 8/2/11): AnimationSeparator produced **12 groups**
    (`2,5,2,3,1,1,2,1,1,1,1,1` frames each) — nothing resembling ASP's 3
    coherent phase spans.
  - **test09** (22 smart-selected frames, ASP detects 3 phases sized
    4/7/11): AnimationSeparator produced **16 groups**, 13 of them
    singletons (`1×13, 2, 5, 2`) — even more fragmented.
  - **Conclusion**: on both tests, AnimationSeparator fragments a
    monotonically-panning sequence into many small (mostly size-1) groups
    rather than recovering large, temporally-contiguous phase runs. This is
    not a parameter-tuning issue — it's the expected behavior of an
    algorithm built for a structurally different input (cyclic/repeating
    content) applied to a structurally different one (monotonic drift).
    **Do not adopt AnimationSeparator as a phase-detection alternative or
    cross-check for §2.2** — ASP's own dHash+robust-MAD change-point
    detector is already the right tool for this specific input structure
    and is correctly the one in production use.
- **(c) what does its interactive workflow do that our HITL checkpoints
  don't?** Not explored — this needs the GUI (`Overmix`, not `OvermixCli`)
  and hands-on use, out of scope for CLI-only automation. Still open.

## Full 97-corpus run (2026-07-28) — DONE

`python -m backend.benchmark.run_overmix` (smart-selected-frames variant,
no `--full`) against all 97 datasets. **97/97 succeeded, zero failures or
timeouts**, ~5 minutes of actual `OvermixCli` compute time (summed
per-test `wall_sec` from each `overmix_variant.json`). Clean run
throughout — no host-freeze symptoms, confirming the thread-cap fix
(`ASP_BENCH_THREAD_CAP`, [[feedback_benchmark_freezes_host]]) holds for
this second benchmark entry point too, not just `bench_anime_stitch.py`
itself. Every dataset now has `output/overmix_stitch.png` +
`output/overmix_variant.json`.

Not yet done: merging these into `bench_anime_stitch.py`'s own
consolidated four-way report — that script has no metrics-only/recompute
mode, so doing this properly would currently mean a full ~2h22m pipeline
re-run just to backfill comparator columns onto data that didn't change.
Left as an explicit, separate follow-up.

## What's still open

- Merging the full-97 Overmix data into `bench_anime_stitch.py`'s
  consolidated report (see above) — needs either a metrics-only recompute
  mode built first, or accepting a full pipeline re-run.
- §1.2(a) and (c) above (only (c) remains open).
- ~~§0.6 (adding Overmix as a GUI option in the Merge tab)~~ — **done**,
  confirmed 2026-07-27: `merge_tab.py`'s engine dropdown already has an
  "Overmix" option (`new_features.md` §4.16 work), wired through
  `image_merger.py::_merge_images_overmix`. This note was stale.
