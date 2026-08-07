# Getting Started with HybridStitch

HybridStitch (`HybridStitchPanel`) is ASP's manual/interactive stitching tool.
Where the automated `AnimeStitchPipeline` runs matching, bundle adjustment,
and compositing on its own (see the [ASP Pipeline Overview](pipeline-overview.md)),
HybridStitch puts every one of those decisions in your hands: you place your
own control points, correct color per frame, paint seam constraints, warp a
mesh, and render the composite yourself. Reach for it when the automated
pipeline falls back to SCANS on a sequence, or when you just want full manual
control over a stitch.

This tutorial walks through the panel tab by tab, in the order you'd
normally use them.


## Try the bundled samples

If you don't have your own frame sequence, you can try Hybrid Stitch using one of the bundled synthetic sample sequences:

- **Sample A (`test_scroll_gradient`)**: Located in `data/samples/test_scroll_gradient`. A simple vertical color gradient scrolling scene demonstrating a basic scroll that HybridStitch can stitch.
- **Sample B (`test_scroll_pattern`)**: Located in `data/samples/test_scroll_pattern`. A simple geometric pattern that scrolls, providing a slightly more complex test.

You can load these samples directly through the in-app onboarding wizard or manually add the frames from the `data/samples/` directory.

## Layout

The panel splits into two halves:

- **Left sidebar** — a **Sequence** list of your frames (with thumbnails), a
  **Working Pair** selector, and an **Use as Stitch List** button.
- **Right side** — a tabbed tool area: **Control Points**, **Color Correct**,
  **Seam Painter**, **Mesh Warp**, and **Render**.

## 1. Load your frames

In the **Sequence** group on the left:

- **Add Frames…** opens a file picker (PNG/JPG/JPEG/WebP/BMP/TIFF) — pick all
  the frames of your scrolling capture. They're appended in the order you
  select them.
- Drag rows in the list to reorder frames directly.
- **Move Up ↑** / **Move Down ↓** nudge the currently-selected frame instead,
  if you prefer buttons to dragging.
- **Remove Selected** drops the highlighted frame; **Clear All** empties the
  whole sequence.

Each row shows a thumbnail and the file name; hovering shows the full path.

## 2. Pick a working pair

HybridStitch aligns frames **one adjacent pair at a time**. In the
**Working Pair** group, use the **Frame A** / **Frame B** dropdowns to choose
which two frames you're currently aligning (selecting a row in the sequence
list also updates these automatically to that frame and its next neighbor).
Click **Load Pair →** to send both frames into the Control Points tab.

## 3. Control Points — align the pair

The **Control Points** tab shows Frame A on the left and Frame B on the
right, each in its own pannable/zoomable canvas (mouse wheel zooms).

- The **✎ Add Points** toggle button controls the canvas mode: checked means
  clicking places a new point; unchecked (**✋ Pan/Zoom**) switches to
  click-and-drag panning instead. Existing points can be dragged in either
  mode.
- Click a spot on Frame A to place a numbered point. HybridStitch
  immediately drops a matching placeholder point on Frame B at the same
  pixel coordinates — drag that placeholder onto the true corresponding
  feature in Frame B (rather than clicking a fresh point, since one has
  already been created for that index).
- Right-click near a point on either canvas to remove it — removing a point
  from one side removes its numbered match on the other side too, so the
  point lists stay in sync.
- **⚡ Auto-Detect (ORB)** runs ORB feature matching + RANSAC across the
  pair and populates the canvases with the resulting correspondences as a
  starting point you can then refine by dragging.
- **💡 Suggest Next** proposes the next-best unused ORB match — specifically
  the one farthest from your existing points — as a fast way to add
  well-distributed correspondences without eyeballing the whole image.
- **Clear All** removes every point from both canvases.

You need **at least 4** point pairs before you can solve. Pick a **Solve**
mode from the dropdown:

- **DLT (exact)** — passes the homography through every point exactly; use
  it when you trust every click.
- **DLT + RANSAC** — robust to a few bad clicks; RANSAC discards outliers.
- **Auto + Manual** — re-solves using your manual points plus every
  auto-detected ORB pair from Auto-Detect, not just the ones you kept on
  screen.

Click **🔧 Solve Homography**. The status line reports the pair count,
inlier count, and mean reprojection error in pixels. Once solved, the
overlap between the two frames is automatically warped and pushed into the
Seam Painter tab as a live preview.

## 4. Color Correct — match exposure/tone

The **Color Correct** tab adjusts the currently-selected Frame A with five
sliders, each with its own reset (↺) button:

- **Brightness** (−100…100, default 0)
- **Contrast** (0.1…3.0, default 1.0)
- **Saturation** (0.0…3.0, default 1.0)
- **Gamma** (0.2…5.0, default 1.0)
- **Temperature** (−50…50, default 0 — shifts the blue/red balance)

The preview above the sliders updates live. **Reset All** zeroes every
slider back to its default. **Match Adjacent →** compares this frame's
lightness (Lab L channel) mean and standard deviation against Frame B and
sets Brightness/Contrast to approximate a match automatically — a fast
starting point you can then fine-tune by hand.

Corrections are stored per source-frame path, so they carry forward
automatically into the Seam Painter, Mesh Warp, and Render tabs for that
frame.

## 5. Seam Painter — control the blend boundary

Once a homography is solved, the **Seam Painter** tab shows the two aligned
frames blended 50/50 over their overlap. Use the brush toolbar to guide
where the seam goes:

- **Force A (red)** — paint regions that must come from Frame A.
- **Force B (blue)** — paint regions that must come from Frame B.
- **Erase** — clear paint in a region.
- **Size** — brush radius, 2–80 px.

Click **🔧 Compute Seam** to run a minimum-cost seam search across the
overlap: your red/blue paint acts as hard constraints (zero cost for the
side you forced, near-infinite cost for the side you excluded), and
everywhere else the seam follows the path of least pixel difference between
the two frames. The result replaces the preview so you can see the seam
in context. **Preview Blend** resets the view back to the plain 50/50 blend;
**Clear Paint** wipes your brush strokes without recomputing anything.

## 6. Mesh Warp — local deformation

The **Mesh Warp** tab loads Frame A of the current working pair (with its
color correction applied) as soon as you switch to it, overlaid with a grid
of draggable orange pins.

- **rows** / **cols** spinboxes (2–20 / 2–30, default 5×8) set the grid
  density; **Rebuild Grid** applies a new size.
- Drag any pin to deform the image locally — HybridStitch fits a thin-plate
  spline from your pin displacements and remaps the image through it.
- **Reset Pins** snaps every pin back to its original grid position without
  discarding the loaded image.
- **⚡ Apply Warp** bakes the current pin deformation into the image (the
  warped result becomes the new base image, so further pin drags compose on
  top of it).

Use this for local corrections control points and the seam alone can't fix —
e.g. a body part that's slightly the wrong shape after alignment.

## 7. Accept the pair, repeat, then render

Back in the left sidebar's **Working Pair** group, click **✔ Accept H** to
save the solved homography (and any painted seam mask) for the current
pair. The status dialog reports how many pairs are saved so far. Repeat
steps 2–6 for each adjacent pair in your sequence.

When you're ready, open the **Render** tab. It shows how many frames and
homographies are currently staged. Configure:

- **Blend mode** — `Seam mask` composites using your painted/computed seam
  boundaries for each pair; the other options blend all overlapping frames
  by simple weighted averaging instead.
- **Apply color corrections** (on by default) — bakes in your per-frame
  Color Correct adjustments.
- **Use painted seam masks** (on by default) — required for `Seam mask`
  blending to actually use your seam paint rather than falling back to
  plain averaging.

Click **⚡ Render Panorama**. Frames are composited using the *translation*
component of each pair's homography chained end-to-end (rotation/scale
solved by Control Points isn't carried into the final placement), so the
render works best when your sequence is close to pure vertical/horizontal
scroll — which is the common case for scrolling anime/manga capture.
**Save…** writes the result out as PNG, JPEG, or WebP.

## 8. Handing frames to the automated pipeline instead

The **✔ Use as Stitch List** button at the bottom of the sidebar does
something different from Render: it takes your ordered frame *sequence*
(not your homographies, corrections, or seam masks — those stay local to
HybridStitch) and hands it to the main **Stitch** tab, switching you over to
it with a confirmation dialog. From there you can run the automated
`AnimeStitchPipeline` on the same frames instead of, or before, doing a
manual HybridStitch pass. See the [ASP Pipeline Overview](pipeline-overview.md)
for what that automated path does.
