# ASP Research Web Search — CV Literature Survey

**Intent:** Direct a research agent to perform a comprehensive, targeted web search across computer vision literature, preprint servers (arXiv), and empirical engineering resources to find methods that directly address the specific, quantified limitations of the Anime Stitch Pipeline (ASP). The agent should surface both foundational theory and practical implementation tips that could be adapted to our architecture.

**Last updated: 2026-06-07 (S32 baseline — 262 unit tests, 4/96 genuine SCANS fallbacks)**

---

## Background: What the Pipeline Does and Where It Fails

The ASP is a 13-stage research pipeline that assembles **sequential anime video frames into a vertical panorama** showing a character's full body from a pan shot. The canonical problem: the camera pans while the character simultaneously animates, so any two adjacent selected frames show the same background region but the character in a completely different pose. Naively stacking strips creates "torn/doubled" character edges at every seam.

### Current Architecture Summary (S28 baseline, 2026-06-07)

```
Frames (58–333 source, ~18 selected)
  → Frame selection (phase correlation greedy, hold detection, near-dup luma filter,
      DINOv2 pose selection [optional])
  → BiRefNet fg/bg mask generation
  → EfficientLoFTR / LoFTR / ALIKED+LightGlue pairwise matching + bundle adjustment
      (LM, GNC cauchy loss, 5-retry validation chain)
  → Post-match spatial dedup (§1.9A: scans_frames synced)
  → ECC sub-pixel affine refinement
  → SEA-RAFT/DIS optical flow → ARAP Push+Regularise (LSD collinearity) → midpoint
      warp (Stage 8.5, fg_register.py)
  → Canvas construction (translation-only model) + Stage 10.5 multi-frame coverage
      gate (falls back to SCANS when < 30% rows have ≥2-frame coverage)
  → Temporal median render (background-only pixels)
  → Hard-partition seam-DP composite (Laplacian + DSFN blend, Poisson opt-in)
      + single-pose soft-edge + seam color match + adaptive feather refinement
  → TELEA border fill fallback
  → Crop
  → (Optional) TOML config via asp_config.toml (§1.8A)
```

### Quantified Performance (96-test corpus, 55 with ground-truth SSIM, S11 baseline)

| Metric | Value | Meaning |
|--------|-------|---------|
| True ASP composites | 92/96 (96%) | SCANS fallbacks reduced from 51→4 genuine (S11) |
| Genuine SCANS fallbacks | 4/96 | Tests 54, 59, 73, 89 — structural failures |
| Avg ASP GT-SSIM | ~0.667 | vs simple stitch 0.694 — pipeline still loses on average (S4 last full run) |
| asp_better verdict | 7/55 (13%) | Tests where ASP > simple stitch by GT-SSIM (S4 baseline) |
| simple_better verdict | 26/55 (47%) | Tests where simple stitch wins |
| Best ASP score | test17=0.887 | Near-perfect case |
| test09 ceiling | raw=0.787, aligned=0.832 | 0.045 gap = frame-timing mismatch |
| test27 ceiling | raw=0.709, aligned=0.748 | 0.040 gap = same cause |
| Unit test suite | **262 tests** | `pytest backend/test/animation/ -q`, ~7s, no GPU |

### Key Implemented Features (do NOT re-suggest these)

1. **SEA-RAFT dense optical flow** (ptlflow) — seam-band crops only, 1280px downscale
2. **ARAP Push + Regularise** (Sýkora 2009) — per-cell SAD block matching → rigid mesh interpolation; LSD collinearity constraint on boundary cells
3. **BiRefNet foreground/background segmentation** — runs on all frames before compositing
4. **Symmetric midpoint warp** (StabStitch++ principle) — α=0.5 both directions
5. **post_warp_diff escalation** — single-pose fallback when blend residual > 22 lum units
6. **Background-only temporal median** — prevents fg ghosting on background plate
7. **Laplacian-blend seam-DP** (vectorised via `minimum_filter1d`; parallel pre-compute with ThreadPoolExecutor) — graph-cut seam finding with fg-penalty weight + tiered cost map
8. **Smart frame selection** — phase-correlation greedy, 50px min step; hold detection (`_detect_hold_blocks`); DINOv2 submodular pose selection (optional); `_near_dup_luma_filter` post-filter (default OFF)
9. **Fg pixel L1 pose metric** — per-frame gain-normalised masked pixel L1 (background-invariant)
10. **Alignment stability gate** — 75th-pct |dx| > 50px fires SCANS fallback (2D-motion detection)
11. **Ghosting ratio gate with absolute floor** — ASP ghosting > max(40, 2× simple stitch) triggers fallback
12. **GNC robust loss in bundle adjustment** — `loss='cauchy', f_scale=10.0`
13. **5-retry affine validation chain** — escalating min_step/max_ratio retries before fallback
14. **Multi-frame canvas coverage gate** (Stage 10.5) — falls back to SCANS when < 30% canvas rows have ≥2-frame coverage
15. **Adaptive feather refinement** — post-registration seam_post_diffs widens/narrows feathers; gain-adaptive feather minimum
16. **Single-pose soft-edge + seam color match** — ±6px ramp + per-channel band mean shift
17. **Per-pixel DSFN blend ramp** — bg-mask-aware; fg-vs-fg sim forced to 0 after Gaussian blur
18. **Poisson seam blend** — `cv2.seamlessClone` in ±20px band (optional, `ASP_POISSON_SEAM=1`)
19. **Continuous adaptive gain clamp** — `clamp_width = 0.26 − 0.12 × (ref_lum/255)`; per-pair coherence gate
20. **TELEA border fill** — `cv2.INPAINT_TELEA` fallback for residual black corners
21. **TOML config loader** — `load_asp_config()` via stdlib `tomllib`; env vars always win
22. **scans_frames spatial dedup sync** (§1.9A) — fallback paths always receive the same frame subset as the main pipeline
23. **Two-channel phase correlation** — bg-only phase correlation for camera displacement (built, disabled — regressions)

### Definitively Ruled Out (do NOT re-suggest these)

| Approach | Why it failed |
|----------|--------------|
| RAFT vs DIS comparison | Identical SSIM outcomes — flow quality is not the bottleneck |
| Global reference warp (α→1) | Amplifies RAFT noise for flat cel regions; test27 collapsed -0.151 |
| Gradient-based pose similarity | Background structure (lockers, walls) confounds gradient L1 |
| ARAP cell_size tuning (8/16/32px) | No measurable SSIM change |
| Character bounding-box crop | Removes essential background content on vertical pans |
| Lowering post_warp_diff threshold | Scene-dependent; net negative on 5-test corpus |

---

## The Three Quantified Ceilings You Must Help Break

### Ceiling 1: Animation Timing Mismatch (PRIMARY — affects all tests)

**The core problem in precise terms:** At 24fps video with 50px camera advance per selected frame (~300ms between selections), an animating character moves 10–85px between consecutive selected frames. The midpoint warp reduces this to 5–42px residual at the seam. This residual creates the dominant SSIM penalty.

**Aligned SSIM** (ECC-aligned before comparison): test09 raw=0.787, aligned=0.832 (+0.045). The 0.045 gap is purely from selecting frames at temporal positions that don't match the GT reference. This is irreducible without (a) a better frame selector that picks GT-matching frames, or (b) a better compositing method that eliminates seam artifacts regardless of pose mismatch.

**Specific question:** What is the state of the art in **pose-consistent multi-frame fusion** when:
- The source is animated character video ("on twos/threes" — same cel held 2–3 frames)
- The metric is strict (SSIM vs a specific GT reference)
- No GT temporal metadata is available
- BiRefNet masks for fg/bg separation are available
- RAFT flow is available but insufficient for flat cel regions (aperture problem)

### Ceiling 2: Aperture Problem on Flat Cel-Shaded Regions (SECONDARY)

**The problem:** Anime characters have large flat-color regions (uniform skin, costume). Both RAFT and DIS produce chaotic/zero flow vectors in these regions because there is no intensity gradient to track. The ARAP Push phase (block-matching) helps for textured regions but also fails on flat color. The result: in large flat regions, the optical flow is unreliable, causing the midpoint warp to produce incorrect displacements for exactly the largest visible body parts.

**Specific question:** What methods exist for estimating dense correspondence in **textureless/flat-color regions** in animation or stylized content?

### Ceiling 3: GT Reference Coupling in Frame Selection (STRUCTURAL)

**The problem:** Any change to which frames are selected changes the content of the assembled panorama relative to what the GT reference shows. Even if pose-consistent selection improves visual quality, it diverges from the GT's specific temporal sampling, causing measured SSIM to sometimes decrease. The fg pixel L1 metric (our current best) improved test27 by +0.010 but regressed test04 by -0.024.

**Specific question:** What methods exist for **reference-free quality estimation** of multi-frame composites that don't require a GT comparison? What perceptual quality metrics (IQA) are appropriate for stitching/compositing evaluation?

---

## Search Instructions

You are a computer vision research assistant. Perform a **comprehensive, deep web search** across arXiv, Google Scholar, NeurIPS/CVPR/ICCV/ECCV proceedings, and engineering blogs. For each topic below, find:

1. The most recent (2020–2025) high-quality papers
2. Any empirical tips/tricks from practitioners (blog posts, GitHub issues, implementation notes)
3. Whether the method has open-source code available
4. Estimated implementation complexity (lines of code, dependencies)
5. Whether it has been applied to anime/cartoon/stylized content
6. Any negative results or known failure modes that match our constraints

**Search methodology:**
- Use multiple query variants per topic (synonyms, acronyms, author names, workshop names)
- Check arXiv cs.CV, cs.GR, and eess.IV categories
- Check SIGGRAPH and NPAR proceedings specifically for cartoon/anime methods
- Check recent workshops: AIM, NTIRE, PBVS
- Include 2024–2025 preprints even if not peer-reviewed

---

## Topic A: Pose-Consistent Multi-Frame Fusion for Animation

**Goal:** Select or synthesize frames such that consecutive seam boundaries show the same character pose, reducing animation residuals to near-zero without relying on optical flow.

### A1. "On Twos" Animation Hold Detection

Search for methods that detect **animation holds** — runs of identical or near-identical frames that arise from "on twos/threes" drawing techniques in hand-drawn animation. These holds are natural sampling points where the camera has advanced but the character hasn't changed pose.

Queries to try:
- "animation hold detection temporal consistency"
- "cel animation frame clustering pose identical"
- "anime video temporal segmentation hold frames"
- "2D animation repetition detection inbetween"
- "cartoon frame sampling temporal coherence optical flow"

What to find:
- Methods for detecting frame holds in compressed or raw animation video
- Whether perceptual hashing or feature-distance clustering can identify "same cel" frames
- Literature on "temporal redundancy" in animation encoding (MPEG, H.264 can exploit this)
- Any connection between MPEG P-frame prediction and "hold detection"

### A2. Pose Embedding for Frame Selection

We need a **background-invariant** pose similarity metric to drive frame selection. Gradient L1 and pixel L1 fail because background texture changes as the camera pans.

Queries to try:
- "human pose estimation feature matching anime 2D cartoon"
- "DWPose ViTPose frame selection temporal consistency video"
- "DINO features temporal video frame selection pose"
- "optical flow masked foreground pose similarity selection"
- "whole body pose estimation lightweight inference frame rate"
- "skeleton similarity metric temporal video frame sampling"
- "RAFT foreground-only optical flow masked region flow estimation"

What to find:
- Lightweight pose estimators that can run on 256×256 thumbnail crops (~5ms/frame)
- Whether DINO/DINOv2 features on masked foreground are sufficiently discriminative for "same pose" vs "different pose" at animation-frame granularity (~2–10px body part shift)
- Any papers on using pose embeddings specifically for video key-frame selection
- Papers on "temporal attention" in video transformer models for character re-identification

### A3. Temporal Coherence in Video Compositing

How do video production tools and research methods maintain temporal consistency when compositing animated characters across frames?

Queries to try:
- "video compositing temporal coherence animation frame selection"
- "optical flow guided video stitching temporal alignment 2D"
- "anime character stitching multi-frame temporal consistency"
- "StabStitch++ video stitching temporal stability method"
- "video panorama stitching moving objects temporal alignment"
- "multi-frame image registration animation deformable"

---

## Topic B: Correspondence in Textureless and Flat-Color Regions

This is the **aperture problem on cel-shaded content**. Optical flow methods (RAFT, DIS, PWC-Net) fail on large flat regions because the cost volume is uninformative with no gradient. Block-matching (ARAP Push) also fails when the patch is uniform.

### B1. Segment-Level Correspondences

Instead of per-pixel flow, compute correspondences at the **segment level** — find the corresponding color segment in frame B for each color segment in frame A.

Queries to try:
- "superpixel correspondence flat color region textureless matching"
- "segment-level optical flow cartoon anime animation"
- "AnimeInterp segment guided motion estimation anime"
- "SLIC centroid tracking correspondence textureless region"
- "semantic flow flat region matching keypoint-free"
- "image warping textureless region guided by semantic segmentation"
- "AnimeInterp++ segment-guided motion estimation interpolation"

What to find:
- AnimeInterp (CVPR 2021) and its sequel — these specifically target anime frame interpolation with segment-guided motion. Are the segment correspondence methods transferable to our registration use case?
- SLIC-based centroid tracking for flat regions
- Any color-segment-level matching methods from cartoon colorization or recoloring

### B2. Deep Learning Flow for Stylized Content

Optical flow trained on natural images fails on anime because the domain gap is too large. Are there flow networks trained specifically on animated content?

Queries to try:
- "optical flow anime cartoon stylized content domain adaptation"
- "optical flow synthetic training data anime cartoon domain gap"
- "neural flow estimation flat color textureless deep learning"
- "FlowNet RAFT domain transfer anime animation training"
- "DeepStitch animation frame interpolation temporal"
- "cost volume textureless region flat color optical flow"
- "CRAFT optical flow cartoon animation"

What to find:
- Synthetic training data approaches for animation-specific flow
- Whether the SynthFlow or MPI-Sintel datasets have cartoon-adjacent splits
- Any fine-tuned RAFT variants for anime/cel-shaded content
- Explicit handling of textureless regions in optical flow (e.g., cross-bilateral filtering or confidence-guided regularization)

### B3. Contour/Edge-Guided Deformation

Anime line art is characterized by clear silhouette contours. Can we use contour matching (LSD lines, sketch segments) as a proxy for pose correspondence where pixel-level flow fails?

Queries to try:
- "line segment matching deformation cartoon animation registration"
- "contour correspondence image morphing warping anime"
- "ARAP as-rigid-as-possible cartoon line art collinearity constraint Sykora"
- "LSD line segment detector matching correspondence 2D animation"
- "sketch-based image morphing line-art deformation"
- "structure-aware image warping line art"
- "thin plate spline TPS warp guided line segment anime"

What to find:
- The full Sýkora 2009 NPAR paper with LSD collinearity constraint (we have Push+Regularise but not the LSD term)
- Subsequent work extending Sýkora's method
- Papers using detected line segments as control points for anime warping
- Methods from sketch/line-art morphing that are specifically robust to the aperture problem

---

## Topic C: Reference-Free Quality Metrics for Composites

We need metrics that evaluate composite quality WITHOUT requiring a ground-truth reference, because GT-coupling prevents us from using GT-SSIM as the optimization target.

### C1. No-Reference Image Quality Assessment (NR-IQA)

Queries to try:
- "no-reference image quality assessment double image ghosting artifact"
- "BRISQUE NIQE PIQUE stitching artifact detection"
- "deep learning no-reference quality stitch seam artifact"
- "perceptual quality metric image compositing artifact"
- "IQA metric ghosting blending artifact seam visible"
- "MANIQA TOPIQ CLIPIQA no-reference neural IQA"
- "anime stylized image quality assessment no-reference"

What to find:
- NR-IQA metrics specifically trained to detect ghosting/double-image artifacts
- Whether CLIP-based IQA (e.g., CLIP-IQA, TOPIQ) can detect seam artifacts in anime
- Seam-coherence metrics beyond our current row-mean luminance std (seam_coherence)
- Whether there's a "pose coherence" metric — measuring if the visible body in a panorama is consistent

### C2. Seam Quality Metrics from Video Stitching

Queries to try:
- "video stitching seam quality metric evaluation temporal consistency"
- "panorama stitching seam visibility metric no-reference"
- "ghost-free image stitching evaluation metric"
- "HDR ghost detection metric exposure fusion quality"
- "seam coherence perceptual quality image stitching gradient domain"
- "image stitching quality assessment benchmark evaluation"

What to find:
- Seam detection metrics from the video stitching / HDR exposure fusion literature
- Whether warping error maps (flow residual after compositing) correlate with perceptual quality
- Any benchmark datasets for stitching quality that include animated/stylized content

---

## Topic D: Novel Compositing Methods for Moving Foreground

We currently use: bidirectional midpoint warp + ARAP regularization + single-pose fallback. What are we missing?

### D1. Ghost-Free HDR Fusion (structural analogues)

The multi-exposure HDR problem is structurally identical to our multi-frame fusion: multiple captures of a moving subject from a fixed viewpoint. Ghost removal methods from HDR may transfer directly.

Queries to try:
- "ghost-free HDR exposure fusion moving subject deghosting"
- "DDFNet deep deghosting exposure fusion moving object"
- "multi-exposure image fusion moving foreground HDR deghost"
- "reference-based HDR fusion temporal alignment foreground"
- "patch-based deghosting multi-exposure fusion deep learning"
- "optical flow guided exposure fusion ghost removal"

What to find:
- Methods for handling large-displacement foreground (character moving 10–85px between exposures)
- Whether confidence-weighted blending (high confidence = "use this frame's version") outperforms midpoint warping for large misalignments
- FDAN or similar alignment networks for HDR that handle large motions
- Any "hard decision" methods (select one frame's pixel, no blending) that outperform soft blending for moving foreground

### D2. Video Inpainting / Outpainting for Seam Completion

When pose differences at a seam are too large to warp, could we **synthesize** a transitional pose rather than using one frame's raw content?

Queries to try:
- "ToonCrafter anime frame interpolation inbetweening key frame synthesis"
- "video inpainting seam completion moving foreground synthesis"
- "diffusion model video temporal consistency frame interpolation anime"
- "conditional diffusion inpainting seam artifact removal"
- "anime frame interpolation large motion occlusion synthesis"
- "neural rendering temporal gap fill body pose synthesis"
- "AnimateDiff consistent pose temporal video generation"

What to find:
- ToonCrafter (2024) and successors — cartoon-specific frame interpolation models
- Whether ToonCrafter can synthesize a "transitional pose" that would be placed at the seam zone, replacing the hard-partition composite with a synthesized in-between
- Computational cost: is it feasible at seam-resolution crops (600×400px, ~10ms budget)?
- Any methods that generate temporally consistent character poses for anime specifically

### D3. Scene Decomposition for Layered Compositing

The pipeline composites background and foreground in a two-layer model. Are there better decompositions?

Queries to try:
- "layered video decomposition foreground background character animation"
- "omnimatte video scene decomposition foreground layer"
- "layered neural rendering video separation transparent layers"
- "video matting temporal consistency character background separation"
- "alpha matting video anime character foreground extraction temporal"
- "robust video matting temporal coherence foreground layer"

What to find:
- Omnimatte (CVPR 2021) and similar layered decomposition methods
- Whether per-frame alpha matting (not just binary segmentation) would allow smoother cross-seam blending
- Temporal consistency guarantees in foreground extraction that would make the composite more coherent
- Any method that produces "soft" foreground masks suitable for gradient-domain blending at seam boundaries

---

## Topic E: Improving Frame Selection Beyond Phase Correlation

### E1. Background-Separated Motion Estimation

We want the camera displacement at each frame pair **without the character animation contaminating the estimate**.

Queries to try:
- "background-only motion estimation foreground occluded video"
- "dominant motion estimation masked region optical flow background"
- "video stabilization foreground masked motion estimation"
- "scene flow separation foreground background dynamic video"
- "homography estimation with moving objects robust RANSAC video"
- "camera motion estimation dynamic scene moving objects robust"
- "two-layer motion model foreground background video separation"

What to find:
- Robust homography estimation methods that explicitly handle dynamic foreground (treat fg as outliers)
- Whether RANSAC-on-background-points gives significantly better displacement estimates than whole-frame phase correlation
- The "Farneback + background mask" pipeline — any papers showing this outperforms whole-frame for displacement estimation?
- Whether FlowNet/PWCNet's background-flow extraction can be repurposed for this

### E2. Canonical View Normalization for Pose Comparison

The GT-coupling problem: selecting frame i vs frame i+1 (same pose hold) changes the canvas position, shifting the GT comparison by ~50px. What methods normalize for this shift before pose comparison?

Queries to try:
- "canonical view normalization character pose comparison video"
- "camera-invariant pose representation character recognition video"
- "person re-identification appearance feature background invariant"
- "appearance feature extraction video background subtraction independent"
- "temporal video frame sampling pose consistency quality"
- "exemplar-based video frame selection human pose clustering"

What to find:
- Person re-identification methods that produce camera-invariant appearance features
- Whether these can be applied at the anime/stylized-content domain (significant domain gap)
- "Appearance normalization" methods that subtract background before feature extraction

---

## Topic F: Architectural Improvements

### F1. Homography / Similarity Transform Models

The pipeline uses translation-only canvas placement. Test05 fails due to zoom+pan (scale_dev=0.121). What's the right generalisation?

Queries to try:
- "deep homography estimation real-time stitching panorama"
- "content-aware stitching spatially varying warp perspective"
- "as-projective-as-possible image stitching spatially varying homography APAP"
- "meshflow video stitching spatially varying mesh deformation"
- "affine similarity transform stitching scrolling 2D animation"
- "deep video stitching temporal homography network"

What to find:
- APAP (TPAMI 2014) and its successors — spatially varying warps for stitching
- MeshFlow (ECCV 2016) — smooth mesh warp for video stitching
- Whether any of these have been adapted for 2D animation with moving foreground
- Computationally: APAP adds ~0.5s, is it worth it for the 13/96 affine-failure tests?

### F2. Deep Stitching End-to-End Networks

Are there end-to-end learned stitching networks that could replace or augment stages 5–11?

Queries to try:
- "deep image stitching end-to-end neural network 2024 2025"
- "UDIS UDIS2 deep learning video stitching unsupervised"
- "diffusion model image stitching seam inpainting learned"
- "transformer image stitching attention seam blending"
- "neural image stitching implicit neural representation"
- "learning-based panorama stitching moving foreground"
- "STITCH diffusion model video stitching 2024"

What to find:
- UDIS++ and similar unsupervised deep stitching methods — do they handle moving foreground?
- Whether any deep stitching method was specifically trained on animated/2D content
- Recent diffusion-based approaches that could replace the seam-finding + Laplacian blend stage
- Inference time constraints: our pipeline allows 95s/dataset total; Stage 11 currently takes 24–42s

---

## Topic G: Insights from Adjacent Fields

### G1. Panorama Stitching in the Wild

Production panorama stitchers (Google StreetView, photo sphere) handle moving objects. What's their specific approach?

Queries to try:
- "google street view moving object stitching panorama deghost"
- "moving object panorama stitching graph cut seam selection"
- "optical flow temporal panorama ghost-free deep stitching"
- "photogrammetry point cloud reconstruction moving objects outlier"
- "LivePhotos image stitching temporal consistency Google Apple"

### G2. Video Conferencing Background Replacement Temporal Consistency

Background replacement with temporal consistency is similar: segment fg, composite over different background, maintain seam coherence frame-to-frame.

Queries to try:
- "video background replacement temporal consistency matting smooth"
- "Zoom background replacement temporal flickering artifact suppression"
- "neural video matting temporal stability boundary coherence"

### G3. Comic/Manga Panel Assembly

Comics and manga are the static predecessors of anime. Are there methods specifically for assembling manga panels (which also have character art with flat cel-shaded regions)?

Queries to try:
- "manga panel stitching comic assembly image processing"
- "webtoon infinite canvas stitching vertical scroll assembly"
- "digital comic scan alignment page assembly"
- "manga colorization deformation registration alignment"

---

## Output Format Requirements

For each method found, provide a structured entry:

```
### [Method Name] ([Year]) — [Conference/Journal]

**Authors:** [first author et al.]
**arXiv/URL:** [link if available]
**Code:** [GitHub link or "none found"]
**Relevance:** [1–2 sentences on why this addresses our specific limitation]
**How it applies:** [concrete description of what we'd change in the pipeline — be specific about stages/functions]
**Estimated impact:** [low/medium/high and why — be skeptical, reference our specific failure modes]
**Implementation cost:** [rough LOC estimate, new dependencies]
**Domain applicability:** [does it work on anime/cartoon/stylized content, or natural images only]
**Key limitation for our use case:** [the specific reason it might fail given our constraints]
**Combines with:** [other methods from this list that would synergize]
```

After all individual method entries, provide:

1. **Priority ranking** — top 5 methods to implement first, with brief justification for each
2. **Synergy map** — which methods are complementary vs redundant
3. **Negative findings** — any promising-sounding approaches that have published negative results for our specific constraint (flat-color animation, moving foreground, GT-coupled evaluation)
4. **Open questions** — specific research questions that the literature does NOT yet answer but that would directly unblock our pipeline if answered
5. **Surprising findings** — any non-obvious connections between our problem and a different field that seems worth exploring

---

## Constraints and Priorities for the Search

**Hard constraints (must respect):**
- Inference must run on a single consumer GPU (RTX 3090 Ti, 24GB VRAM)
- Per-dataset budget: 95s total (current), ideally <120s with improvements
- No new training required (use pretrained weights or classical methods only)
- Pipeline is Python 3.11+, PyTorch, OpenCV, scipy
- Must work on anime/stylized content (domain gap is a real concern for methods trained on natural images)

**Priority ordering:**
1. Methods that address the **animation timing / pose consistency** ceiling (most impactful, affects all 55 GT-scored tests)
2. Methods for **flat-color region correspondence** (enables better flow → smaller midpoint-warp residuals)
3. **No-reference quality metrics** (unblocks enabling pose selection without GT-coupling regressions)
4. Methods for **2D motion / diagonal pan detection and handling** (affects the 2D-motion gate failure cases)
5. **Synthesis-based seam fill** (would eliminate single-pose fallback artifacts entirely)

**Special interest:**
- Any method that specifically addresses the **"on twos" anime animation** problem by name
- Methods from the anime/webtoon/comic computer vision literature (SIGGRAPH Asia, NPAR)
- Any **negative results** or reproducibility failures for methods that sound applicable — knowing what definitely doesn't work is as valuable as knowing what does
- Methods that have been validated at the **5–30fps video frame rate** (as opposed to single-image pairs)

**The ideal finding** would be a method that:
1. Can estimate pose similarity between two frames without requiring background-free input
2. Is lightweight enough to run on 30 frame pairs at thumbnail scale (~5ms/pair)
3. Has been tested on animated/cartoon content
4. Produces a distance metric (not just a binary yes/no) for integration into our greedy frame selector
