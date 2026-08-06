# ASP Pipeline Overview

ASP gives you two different ways to turn a folder of scrolling capture
frames into one panorama: the **automated pipeline** (`AnimeStitchPipeline`,
the **Stitch** tab) and **HybridStitch** (`HybridStitchPanel`, see
[Getting Started with HybridStitch](getting-started-hybridstitch.md)). This
page is a beginner-level summary of what the automated pipeline actually
does, so you know what's happening when you click "run" — and when to reach
for HybridStitch instead.

## What it's for

Point it at an ordered sequence of frames from a scrolling anime/manga
capture (a webtoon, a scrolling gallery, a game's scrolling map — anything
where consecutive screenshots overlap along one axis) and it produces a
single stitched image, fully automatically. No control points, no manual
seam painting — you give it frames, it gives you a panorama or tells you it
couldn't build one safely.

## The shape of the pipeline

Under the hood it's a fixed sequence of stages, roughly in three phases:

1. **Get the frames ready.** Pick the sharpest, least-redundant frames from
   your capture (near-duplicate "hold" frames common in on-twos/on-threes
   anime are detected and skipped), normalize their width and exposure, and
   separate foreground characters from background art.
2. **Figure out how they line up.** Match overlapping content between
   neighboring frames using a cascade of matching techniques (falling back
   through several methods if the first ones don't find enough
   correspondences), then jointly solve for every frame's offset at once
   with a robust bundle adjustment that tolerates a few bad matches without
   the whole solve going sideways.
3. **Build the panorama.** Construct the output canvas, render the
   background, then composite foreground characters into it — this is the
   trickiest part, since two overlapping frames of a moving character can't
   just be blended (that produces ghosting or duplicated limbs). ASP looks
   for a single frame's pose to use in ambiguous regions rather than
   averaging two conflicting poses, and finds a seam through the overlap
   that follows the path of least visible difference between frames.

## The safety net: quality gates

At several points the pipeline checks whether what it's about to do is
actually trustworthy — irregular scroll speed, disconnected frames that
don't chain together, sparse coverage, and so on. If a check fails, the
pipeline **doesn't force a bad result**: it falls back to a simpler
`cv2.Stitcher`-based stitch (referred to as **SCANS** in the project) rather
than shipping a broken panorama. Seeing a SCANS-style result instead of the
full pipeline output means one of these safety checks tripped — it's a
deliberate "safe fallback," not a crash.

## When to use which tool

| | Automated pipeline (Stitch tab) | HybridStitch |
|---|---|---|
| Effort | One click, no manual work | You align, paint, and warp by hand |
| Best for | Regular vertical/horizontal scroll sequences | Anything the automated pipeline gets wrong, or when you want precise manual control |
| What happens on a hard case | Falls back to a simpler SCANS stitch automatically | You solve the alignment/seam yourself, however hard it is |
| Speed | Fast, fully automatic | Slower — you're doing the work the pipeline normally does |

A practical workflow: run the automated pipeline first. If the result looks
right, you're done. If it fell back to SCANS, or the composite has visible
seam problems or duplicated anatomy, load the same frames into HybridStitch
(you can hand a HybridStitch sequence over to the Stitch tab too, and vice
versa — see the HybridStitch tutorial's last section) and align/seam/warp
the problem pair by hand instead.

## Further reading

- `.agent/cache/asp_state_of_the_pipeline.md` — the full 13-stage
  breakdown, gate list, and current benchmark numbers, for anyone working on
  the pipeline itself rather than just using it.
- [Glossary](../GLOSSARY.md) — definitions for terms like `dy_cv gate`,
  `GNC-TLS`, `single-pose escalation`, and `SCANS` used above.
- [Architecture](../ARCHITECTURE.md) — how the pipeline's code is organized
  across `base/`, `backend/`, and `gui/`.
