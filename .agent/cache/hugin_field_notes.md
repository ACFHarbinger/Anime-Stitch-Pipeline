# Hugin Field Notes (roadmap §0.5, GitHub issue #20)

Wired Hugin as a fourth, reference-only comparator (never a gate) and ran it
on the 5-test verify subset (`asp_test04/08/09/27/57`). Unlike Overmix
(§0.3), this surfaced a fundamental architectural mismatch, not just a
wiring gap — see "The core finding" below.

## Why system packages, not the vendored submodule

The user added `vendor/Hugin` (an ACFHarbinger fork) as a submodule expecting
a from-source build mirroring Overmix's. Investigating its CMake showed this
isn't practical for that fork as committed:

- `src/CMakeLists.txt` only `add_subdirectory`s `foreign`, `hugin_base`, and
  `tools` — `hugin_cpfind` (the control-point finder), `celeste`,
  `deghosting`, and `translations` are all commented out.
- `src/tools/CMakeLists.txt` — the directory containing `pto_gen.cpp`,
  `autooptimiser.cpp`, `nona.cpp` — only has one `add_executable`:
  `align_image_stack`. The other tools' sources exist on disk but aren't
  wired into any CMake target at all.
- Building even what *is* wired needs VIGRA≥1.9, EXIV2, PANO13≥2.9.19,
  SQLite3, and (optionally) FLANN/FFTW — four of which aren't installed on
  this machine and needed a sudo install this session couldn't run
  non-interactively.

Given the user's own confirmed preference (this session, when asked): use
the system `hugin-tools` + `enblend`/`enfuse` apt packages instead, which
ship a complete, working `pto_gen`/`cpfind`/`autooptimiser`/`pano_modify`/
`nona`/`enblend` toolchain — no submodule build needed. The submodule stays
as-is for potential future GUI/build work.

## Wiring: the CLI chain and its two bugs

`pto_gen -> cpfind --linearmatch -> autooptimiser -a -l -s -> pano_modify
--fov=AUTO --canvas=AUTO --crop=AUTOOUTSIDE --output-cropped-tiff -> nona
-m TIFF_m -> enblend`. Two real bugs surfaced building this, both fixed:

1. **enblend's overlap-check safety guard always trips.** `enblend`
   refuses to blend with "excessive image overlap detected; too high risk
   of defective seam line" on every test, even pairwise. Traced to
   `src/nearest.h` (via GitHub code search on the enblend-enfuse mirror):
   `overlap_threshold = parameter::as_unsigned("overlap-check-threshold",
   2U) * 2 * (h+w)` — a heuristic tuned for photography with *partial*
   overlap between frames. Anime pan frames overlap almost entirely by
   design (small per-frame camera motion relative to frame size), so the
   check is guaranteed to trip regardless of content quality.
   `--parameter=overlap-check-threshold=0` disables just this check; the
   blend itself is unaffected once it runs. (`--no-optimize`, the
   web-documented workaround for a *similar*-sounding message, did **not**
   work here — this is a distinct check from the one that flag controls.)
2. **`pano_modify --crop=AUTO` throws away most of the pan.** The first
   attempt used `--crop=AUTO` (autocrop to the region *every* frame
   covers), which is correct for photography stitching but wrong for a
   scan/pan sequence — it discarded the leading/trailing frames' unique
   content, shrinking a 6-frame test's canvas from what Overmix/ASP
   render at roughly the same extent down to a much smaller crop.
   `--crop=AUTOOUTSIDE` (crop only the fully-transparent outside border)
   fixed this — full pan extent retained, trapezoidal/keystone shape from
   the rectilinear projection visible as expected.

## The core finding: Hugin's projection model doesn't fit long scroll sequences

**5-test verify result: 1/5 (test08, 9 frames) succeeded cleanly; 4/5
(test04 25 frames, test09 22 frames, test27 26 frames, test57 26 frames)
failed with a canvas size of 450,000–480,000px in both dimensions** —
several hundred thousand pixels, not a wiring bug, a genuinely degenerate
optimizer result. `nona` either timed out (300s) or crashed with
`std::bad_alloc` trying to render layers at that size before this was
diagnosed.

Root cause: Hugin models every input as a ray from a fixed viewpoint at some
field-of-view, and fits everything into a rectilinear (or cylindrical)
projection — a model built for a camera *rotating* on a tripod. Our content
is a *translating* 2D scroll (a scrolling video capture), which Hugin's
optimizer tries to explain as an enormous implied rotation once enough
frames accumulate enough parallax-free translation — and rectilinear
projection has a hard mathematical singularity at 180° field of view. Long
sequences push the fitted FOV toward that singularity, and the canvas size
blows up as a direct consequence (confirmed via the `.pto` project file's
`p` line: `w503005 h503005` for a 22-frame test, vs. a normal few-thousand
for the 9-frame one). Tried and rejected as fixes:

- **Cylindrical projection instead of rectilinear**: still degenerated
  (`w13824 h504226`) — cylindrical has its own limits and this sequence's
  implied vertical extent exceeded them too.
- **Dropping `autooptimiser -s`** (its own "auto-select projection/size"
  flag, suspected of being the culprit): no change — the degeneracy comes
  from `pano_modify --fov=AUTO` fitting the optimized camera positions, not
  from `-s`.
- **A fixed, bounded `--fov=50x140`** (instead of `AUTO`): avoided the
  blow-up but caused `pano_modify` to mark most images "inactive" (outside
  the bounded FOV window) — `nona` then had nothing to render
  ("Project does not contain active images"). Not a usable middle ground:
  either the FOV is unbounded and the canvas is degenerate, or it's bounded
  and most of the pan gets discarded.

**Disposition**: rather than attempt a deeper fix (would mean reformulating
Hugin's placement model around pure 2D translation instead of rotation — a
structurally different formalism, not a wiring fix, matching this roadmap's
own precedent for parking that class of change), added a fast-fail guard:
after `pano_modify`, parse the `.pto` file's canvas dimensions and abort
immediately with a clear message if either exceeds 20,000px, rather than
letting `nona` hang for 5 minutes or crash with an opaque `std::bad_alloc`.
This turned four confusing multi-minute failures into four immediate,
diagnosable ones. **Practical implication: Hugin is usable as a reference
comparator only for short sequences (this corpus's evidence points to
roughly <15-20 frames)** — most of the 97-test corpus's smart-selected
frame counts exceed that, so Hugin's comparator coverage will be partial by
nature of this projection mismatch, not a bug to chase further.

## Disposition

- Both engine functions (backend `_merge_images_hugin` in
  `image_merger.py`, and the standalone `run_hugin.py` benchmark script)
  carry the same fast-fail guard.
- `metrics_hugin`/`hugin_path` wired into `bench_anime_stitch.py` exactly
  like Overmix — reference column only, never a gate. The report's summary
  table, per-test image table, and per-test CV-metrics table are now
  four-way (ASP/Simple/Overmix/Hugin).
- Full 97-corpus run still open (this round covers the 5-test verify
  subset only, per this project's established safe-scaling discipline) —
  expect a similar partial-coverage pattern at full scale given the finding
  above.
