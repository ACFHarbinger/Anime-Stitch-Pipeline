# Image Stitching — Comprehensive Research Reference

*Consolidated 2026-06-03. This document is the single, complete reference for image-stitching research in Image-Toolkit. It covers the entire field — geometric foundations, feature matching, registration, spatially-varying warps, photometric correction, segmentation, seam-finding, blending, background reconstruction, deep/unified frameworks, super-resolution, video stitching, evaluation — with deep, dedicated sections on the hardest case the project targets: **panoramic and full-character reconstruction from sequential 2D cel-animation (anime) frames.** It replaces the 14 prior stitching reports listed in Appendix C; read this instead of them.*

---

## Table of Contents

1. Problem Definition & Taxonomy
2. Geometric Foundations (transformation models & DoF)
3. The Two Paradigms — Perfect Stitch vs Scan Stitch (with the anime mathematical audit)
4. Feature Matching (classical → learned → detector-free)
5. Registration, Outlier Rejection & Sub-Pixel Refinement
6. Optical Flow for Stitching (incl. flat-shaded domains)
7. Spatially-Varying Warps & Structure Preservation
8. **Foreground Assembly — The Core Anime Problem** (motion decomposition, ARAP, midpoint warp)
9. Photometric Correction (broadcast dimming, flat-fielding, gain, color transfer)
10. Foreground/Background Segmentation
11. Seam Finding (graph-cut, MRF, learned)
12. Blending (multi-band, Poisson/MPB, soft-seam diffusion)
13. Background Reconstruction & Generative Inpainting
14. Deep & Unified Stitching Frameworks
15. Super-Resolution & Restoration
16. Video Stitching & Temporal Stability
17. Shot Boundary Detection (pre-stitch)
18. The Full Pipeline Specification (anime, 13–14 stages)
19. Evaluation & Metrics
20. Failure Modes & Principled Fallbacks
21. Implementation Status in Image-Toolkit (ASP)
22. Appendices — research index, paper list, source reports

---

## 1. Problem Definition & Taxonomy

Image stitching fuses a set of partially-overlapping images into one larger, coherent image. The classical pipeline is **registration → warping → seam-finding → blending → (rectangling)**. Stitching subdivides by:

- **Input geometry:** rotational pano (camera rotates about optical centre), translational scan (camera/scene translates in a plane), general 6-DoF with parallax.
- **Scene dynamics:** static scene vs. **dynamic scene with moving objects** (ghosting risk).
- **Content domain:** natural photo (rich texture, continuous depth, realistic lighting) vs. **2D cel animation** (flat colour, sparse high-frequency line art, multi-plane parallax, broadcast-dimmed exposure, independently animating foreground).

**Anime is the hardest combination:** translational/curvilinear pans over multi-plane backgrounds, textureless interiors, crisp line art that makes any misalignment visible as "edge doubling," exposure that shifts frame-to-frame from broadcast dimming, and a foreground character that **animates while the camera pans**. The three "perfect stitch" conditions for anime are: (1) no stacking/tearing/line-bending from forcing global transforms onto multi-plane scenes; (2) no lighting mismatch / visible seams; (3) no ghosting/double-images from temporally-averaged moving foreground.

The project's objective is a superset of classical anime mosaicking: not only **background reconstruction** (clean panoramic plate, character erased) but **full foreground assembly** (a complete articulated character body pieced from frames where it never fully fits and changes pose). §8 treats this.

---

## 2. Geometric Foundations — Transformation Models & Degrees of Freedom

The geometric relationship between two overlapping frames is a transformation matrix; choosing the right model is the single most consequential decision for anime.

| Model | DoF | Preserves parallel lines | Use case | Vertical-drift risk | Top/bottom crop |
|---|---|---|---|---|---|
| **Homography (projective)** | 8 | No | 3-D natural scenes, rotational pans | High (compounds iteratively) | Severe |
| **Affine** | 6 | Yes | 2-D document scan, anime slides | Low | Minimal |
| **Similarity (SRT)** | 4 | Yes (+ isotropic scale, rotation) | Handheld pan+zoom | Low | Minimal |
| **Pure translation** | 2 | Yes | Perfect 2-D digital pans | None | Zero |

**The anime argument.** A homography's 8 DoF let parallel lines converge. Micro-inconsistencies — MPEG macroblock noise, subtle character motion, dust — make the estimator hallucinate perspective where none exists; over a long sequence these microscopic errors **compound logarithmically**, drifting/skewing the panorama and guaranteeing data loss at the final crop. Downgrading to **affine (6 DoF)** preserves parallel lines; constraining further to **pure translation (2 DoF)** for linear pans preserves full source height and prevents warping entirely. In OpenCV this is the shift from `PANORAMA` to `SCANS` mode (bypasses spherical projection; bundle adjustment minimises translation cost, not ray-reprojection cost). The project's ASP uses a **translation-only (2 DoF) bundle adjustment** for exactly this reason.

**Homogeneous-coordinate DLT.** A homography `H` is solved from ≥4 correspondences via the Direct Linear Transform: stack `A h = 0` where each correspondence contributes rows from the cross-product `x' × Hx = 0`; the solution is the right singular vector of the smallest singular value of `A` (SVD). Rank deficiency of `A` (collinear points) is the failure mode on anime line art (§3.3).

---

## 3. The Two Paradigms — Perfect Stitch vs Scan Stitch

### 3.1 "Perfect Stitch" (homography / local-homography)

Aligns by minimising reprojection error of matched features under a perspective transform; foundational for satellite/medical/consumer panoramas. For local variation/parallax it is extended by **As-Projective-As-Possible (APAP)** warping via **Moving DLT** (§7.1).

### 3.2 "Scan Stitch" (pushbroom / Crossed-Slits geometry)

Abandons projective warping. Models the virtual camera as a **linear pushbroom** or **Crossed-Slits (X-Slits)** sensor: extract a narrow central strip (slit) from each consecutive frame and concatenate. Formally, treat the sequence as a 3-D spatio-temporal volume `V(x, y, t)`; for a horizontal pan at velocity `s`, the panorama is a slice taking the central column of each frame, projecting orthographically along the pan axis. Sub-pixel inter-frame translation is recovered by **Phase Correlation** (§5.4) to 1/50-px precision.

**Why Scan Stitch is mathematically superior for anime:**
1. **Affine-property preservation** — no homography is computed, so lines stay parallel and stroke aspect ratio is exact; spherical bulging is impossible.
2. **Robustness to foreground occlusion** — sampling only a thin central slit minimises the moving character's footprint; over time the slit "looks past" a character moving at a different velocity than the background, sampling the true background as it crosses centre.
3. **Inherent multi-perspective unwrapping** — for curvilinear "banana pans" (artist-drawn shifting vanishing points), a global homography forces an impossible consensus across contradictory vanishing points; Scan Stitch samples the artist's intended local viewpoint at each `t`, unwrapping the multi-perspective drawing onto a Euclidean plane without global geometric consensus.

### 3.3 Why APAP/homography *fails* on anime manifolds (rank-deficiency proof)

Anime backgrounds are vast homogeneous flat-colour regions bordered by sharp strokes. For a pixel `x*` inside a flat region, the distance to the nearest reliable feature is large; APAP's offset-Gaussian weights `w_i(x*) = max(exp(-‖p_i - x*‖²/σ²), γ)` decay to the floor `γ`, so the weighted design matrix becomes **ill-conditioned**. Minor aliasing noise in distant line-art keypoints then induces massive fluctuations in the local homography → wavy distortion, localised tearing, geometric bulging. Providing 8 DoF *per pixel* over-parameterises a medium that is a synthetically-manipulated 2-D plane lacking dense texture — a violation of parsimony. **Conclusion:** translation/affine + Scan Stitch geometry is the correct geometric prior; spatially-varying projective warps are reserved for genuine multi-plane background parallax (§7), never for the foreground.

---

## 4. Feature Matching

The background `T_camera` is recovered from correspondences in **background-only** regions (foreground masked, §10). Matching must survive textureless flat colour and exposure shifts.

### 4.1 Classical detectors/descriptors (and why they struggle)
- **SIFT** — DoG scale-space extrema + 128-D gradient-histogram descriptors. Robust on natural texture; on anime it fires thousands of indistinguishable keypoints along a single ink line and **zero** in flat sky/skin/cloth → biased, near-collinear distribution → ill-conditioned RANSAC → singular/unstable homography → "homography estimation failed."
- **SURF/ORB/BRISK** — faster; ORB/BRISK are intensity-pattern binary descriptors, fragile under exposure change (relevant across broadcast-dimmed/scene-cut frames).
- **AKAZE** — features in a **nonlinear diffusion scale-space** (PDE-based) that respects edges instead of blurring across them; M-LDB binary descriptors. Empirically the **best classical detector for cartoon edges**.
- **MSER** — Maximally Stable Extremal Regions detect flat-colour blobs — exactly anime's dominant feature — complementing line/corner detectors.

### 4.2 Learned sparse matchers
- **SuperPoint** (CVPRW 2018) — self-supervised CNN keypoints; far denser in low-texture regions than SIFT.
- **SuperGlue** (CVPR 2020) — GNN matcher with self/cross-attention + Sinkhorn assignment; joint matching + outlier rejection. Outdoor weights generalise surprisingly well to non-photographic content; fine-tuning on anime pairs (ATD-12K, AnimeRun, PaintBucket-Character) recommended.
- **LightGlue** (ICCV 2023) — adaptive-depth (early-exit) SuperGlue successor, 40–60% cheaper. Native ALIKED/DISK/SIFT backends. *LightGlue + Agglomerative Clustering* empirically gives **+26.2% matching efficiency** on broadcast-artifact anime.
- **ALIKED** (IEEE TIM 2023) — Sparse Deformable Descriptor Head (SDDH) positions sampling adaptively around each keypoint → keypoints on anime line-art edges SIFT misses. Pair with LightGlue.
- **XFeat** (CVPR 2024) — CPU-real-time sparse + semi-dense; emergency GPU-free fallback.

### 4.3 Detector-free (semi-dense / dense) matchers — the anime workhorses
- **LoFTR** (CVPR 2021) — detector-free coarse-to-fine Transformer matching; produces matches in flat regions where detectors fail entirely. *The single most relevant matcher for anime.* Caveat from the project's legacy implementation: a fixed internal resolution (e.g. 480²) imposes ~27% aspect distortion on 16:9 frames and degrades localisation; outdoor (MegaDepth) weights are out-of-distribution for flat art.
- **EfficientLoFTR** (CVPR 2024, 2403.04765) — two-stage adaptive-span coarse attention + fine local refinement; **2.5× faster** at equal/better AUC; drop-in replacement. The project's primary matcher.
- **RoMa v2** (2511.15706) — dense warp on **frozen DINOv2 features** (style-agnostic) + fine ConvNet features, per-pixel reliability. Correspondences on flat-shaded art where keypoint detectors yield nothing. Dense last-resort matcher.
- **JamMa** (CVPR 2025, 2503.03437) — O(N) Mamba joint-scan replaces O(N²) attention; 4K matching without tiling.
- **EDM** (ICCV 2025, 2503.05122) — correlation-injection + axis-based subpixel regression head; strong quality/speed.
- **Region-wise correspondence** (e.g. manga line-art region prediction, 2509.09501) — matches at the **segment** level, the natural unit for anime; pairs with palette/positional rules for unmatched regions.

**Pragmatic recipe:** EfficientLoFTR (dense) → ALIKED+LightGlue (sparse fallback) → template / phase-correlation → RoMa v2 (dense last resort), with background masking throughout and translation-only fitting.

---

## 5. Registration, Outlier Rejection & Sub-Pixel Refinement

### 5.1 Robust model fitting
- **RANSAC / MAGSAC++** — consensus estimation; MAGSAC++ removes the inlier-threshold tuning via marginalisation, more robust at high outlier ratios. Use a translation/affine model for anime (not homography).
- **Rank-deficiency guard** — reject solutions when matched points are near-collinear (anime line-art clustering).

### 5.2 Global bundle adjustment
Translation-only (2 DoF) LM (`scipy.optimize.least_squares`) minimising `Σ_(i,j∈overlaps) Σ_k ‖(x_k^i + t_i) − (x_k^j + t_j)‖²`. **2-pronged outlier rejection:** after the initial solve, prune edges whose per-edge residual exceeds 3× the median, then re-solve; additionally drop near-zero / wrong-direction edges. **Robust loss upgrade:** swap L2 for Cauchy/Geman-McClure (`loss='cauchy'`) — Graduated-Non-Convexity guarantees up to 70–80% outlier tolerance, eliminating single-catastrophic-edge failures. **Trajectory regularisation (StabStitch):** add a second-order temporal-smoothness term `‖t_{i+1} − 2t_i + t_{i-1}‖²` to suppress per-frame warping shake.

### 5.3 ECC sub-pixel refinement
Enhanced Correlation Coefficient maximisation (`cv2.findTransformECC`) on image gradients via ZNCC, over a 4-level pyramid, to ~1/N-px precision. **Limitation:** ECC needs non-zero gradients; anime's flat regions make its Hessian near-singular → divergence (caught by an 80-px drift guard, but ~30% of flat-region pairs gain nothing). Prefer learned flow (§6) for sub-pixel on flat content.

### 5.4 Phase correlation (Fourier shift theorem)
For `f_2(x) = f_1(x − d)`, `F_2 = F_1 e^{-i2π(...)}`. The normalised cross-power spectrum `R = (F_1 F_2*)/|F_1 F_2*|` isolates the phase; its IDFT is a localised pulse whose peak (2-D Gaussian fit) gives sub-pixel `d` to ~1/50 px. Discards amplitude (the flat-colour data that confuses spatial matchers) → ideal translational estimator and Scan-Stitch slit-alignment tool; also the project's tertiary matching fallback.

---

## 6. Optical Flow for Stitching

Dense flow drives both background sub-pixel refinement and **foreground residual extraction** (§8). Two anime-specific obstacles: the **aperture problem** inside flat colour (no interior gradient → arbitrary vectors), and exaggerated, non-linear, large motions that break temporal-smoothness/small-displacement assumptions.

- **RAFT** (ECCV 2020) — per-pixel features + multi-scale 4-D all-pairs correlation volumes; a GRU iteratively refines a high-res flow field. Tracks anime structure via line-art motion even over textureless interiors. Backbone of **ProPainter** inpainting (§13).
- **SEA-RAFT** (ECCV 2024 Oral, 2405.14793) — Mixture-of-Laplace loss + **direct initial-flow regression** (fast convergence) + **rigid-motion pretraining** (confident flow over flat regions; ideal for pure-translation pans). 2.3× faster, 3.69 EPE. The recommended flow backbone; run on overlap-zone crops, take the background-pixel trimmed mean for sub-pixel translation.
- **AnimeInterp** (CVPR 2021, 2104.02495) — domain-specific. **Segment-Guided Matching (SGM):** flow over piece-wise-coherent colour segments (hair/skin/shirt as blocks) bypasses the aperture problem. **Recurrent Flow Refinement (RFR):** resolves large non-linear displacements. The flat-region flow method of choice.
- **GMFlow** (CVPR 2022), **MambaFlow** (2503.07046, O(N) for 4K), **EpicFlow** (sparse-to-dense edge-aware interpolation — accepts user/detector matches as seeds).
- **Datasets:** **AnimeRun**, **LinkTo-Anime** (2506.02733, cel-rendered GT flow from 3-D models) — for fine-tuning SEA-RAFT/EfficientLoFTR to anime; off-the-shelf Sintel/KITTI models degrade on flat cels.

---

## 7. Spatially-Varying Warps & Structure Preservation

For genuine **multi-plane background parallax** (multiplaning: foreground/mid/back layers slide at different velocities), a single transform cannot align all planes. Spatially-varying warps help — but must protect line art.

### 7.1 APAP / Moving DLT (Zaragoza, CVPR 2013)
Per-vertex local homography on a mesh; Moving DLT weights each correspondence by an offset-Gaussian of spatial distance to the cell. Absorbs parallax where texture is soft. **Risk:** straight lines bend in non-overlap regions (each cell warps slightly differently). **AANAP** linearises toward a global similarity in non-overlap regions to curb extrapolation distortion. Reference impl: `EadCat/APAP-Image-Stitching`.

### 7.2 Content-Preserving Warps, TPS, MLS
- **Thin-Plate Splines (TPS):** minimise bending energy subject to interpolation `f(p_i)=q_i`; affine part + radial basis `U(r)=r² log r`; soft (regularised) variant trades exact interpolation for `min Σ‖f(p_i)−q_i‖² + λ J(f)` — ideal for user-pinned warps.
- **Moving Least Squares (MLS)** (Schaefer 2006) — closed-form affine/similarity/rigid deformations from control points; Rust crate `mpizenberg/rust_mls`.
- **Content-Preserving Warps (CPW)** (Liu 2009) — mesh warp regularised by a similarity prior; the base of many video-stabilisation/stitching warps.

### 7.3 Line preservation (the anime essential)
Mesh warping bends straight strokes — instantly visible in line art. Inject a **Line-Segment-Detector (LSD)** collinearity energy `E_line` into the warp objective:
`E = α_d Σ‖T_k V̂ − q_k‖² + α_s E_sim + α_l E_line + α_g E_grid (+ α_h hard pins)`
`E_line` penalises any warp that breaks the collinearity of detected segments; `E_grid` prevents uneven mesh-slope skew. (Line-Point Consistency, CVPR 2021; "Line Meets APAP with Moving DLT.") Solved as a sparse SPD QP (Cholesky). Guarantees straight strokes stay straight while flat regions flex to absorb parallax.

### 7.4 Seam-driven multi-hypothesis (parallax-tolerant)
Generate `K` warp hypotheses (global homography, APAP, dual-homography for distant+ground planes per Gao 2011), seam-cut each (§11), keep the lowest-seam-cost warp. **SEAGULL** (ECCV 2016) iterates between seam estimation and local mesh refinement.

---

## 8. Foreground Assembly — The Core Anime Problem

> This is the project's central, hardest problem and the dominant source of visible artifacts. It warrants the deepest treatment.

### 8.1 The motion-decomposition model
The background moves rigidly with the camera (`T_camera`); the foreground character moves with both the camera **and** its own drawn articulated deformation. The total foreground displacement is
`F_fg(x,y) = T_camera + A_animation(x,y)`.
Classical pipelines assume `A_animation = 0`. When a rigid transform aligns the **backgrounds**, the residual `A_animation` is unresolved; a hard seam through the overlap **bisects two different poses** → limb tearing, severed sleeves, doubled line-art edges. The fix: **measure `F_fg` with dense optical flow, subtract the isolated `T_camera`, and warp out the residual `A_animation`** before seam blending.

### 8.2 Why neither shortcut works
- **Erasing the character (temporal median only)** defeats the goal — we want the body, not a clean plate.
- **Single-frame foreground** fails — no single frame contains the full body (the camera pans across it).
  So the foreground must be **assembled from multiple frames whose poses are first reconciled.**

### 8.3 Why "simple stitch" wins benchmarks but is not the answer
OpenCV SCANS stitches only temporally-adjacent frames (~42 ms apart); over 42 ms `A_animation ≈ 0`, so the foreground looks aligned. But chaining hundreds of micro-stitches to cover a long pan accumulates global drift, exposure banding, and edge distortion; widening the baseline (300–800 ms) to cover distance faster re-exposes `A_animation`. Simple stitch *avoids* the problem; it does not solve it.

### 8.4 Production fact to exploit — animation on "twos"/"threes"
Anime is usually animated on twos/threes (8–12 drawings/s) while the camera pans every frame at 24 fps. Across 2–3 frames the background moves but the **character is frozen**. An optimised selector prefers frames within the same animation phase, minimising the warp burden.

### 8.5 Conceptual ancestor — Unwrap Mosaics (SIGGRAPH 2008)
Rav-Acha et al. reconstruct a continuous 2-D representation of a **deforming object** (not the background), modelling image formation as a 2-D-texture→render transform modulated by an occlusion mask, recovering a flattened "unwrap" of the object. The blueprint: treat the character as an independent deforming entity to track, unwrap to a stable pose, and re-composite.

### 8.6 The flow-guided, ARAP-regularised assembly stage
1. **Mask** the character (BiRefNet/ToonOut/SAM-2) in the overlap zone.
2. **Dense flow** between the two canvas-aligned frames over the masked overlap (SEA-RAFT; AnimeInterp SGM for flat regions). Because the frames are already background-aligned, the residual foreground flow **is** `A_animation` (no explicit subtraction needed).
3. **Symmetric midpoint warp** — warp frame `i`'s foreground by `+½·A` and frame `i+1`'s by `−½·A`, meeting at the interpolated midpoint pose. This **halves** the maximum distortion on either frame (StabStitch++ bidirectional principle) and reduces tearing risk.
4. **ARAP + LSD regularisation** — apply the flow not as raw pixel warp (which tears line art into fluid smears) but as attraction forces in a **Sýkora As-Rigid-As-Possible** lattice (NPAR 2009): embed the image in a coarse square control mesh; oscillate a **Push phase** (decoupled block-matching by SAD — points jump arbitrarily to escape the local minima that paralyse gradient methods) and a **Regularise phase** (per-square optimal rigid transform + centroid update). Local rigidity is preserved while global articulation is allowed — the body bends at joints, not like fluid. Convergence is monitored by control-point displacement (not SAD, constant over non-overlapping background). The LSD `E_line` term holds straight strokes rigid through the interpolation. ARAP's coarse lattice caps sub-pixel precision → finish with a dense flow refiner.
5. **Composite** the re-posed foreground onto a background-only temporal-median plate (§13), with semantic graph-cut seam routing (§11) and soft-seam blending (§12).

### 8.7 Two-channel pose-consistency frame selection
Whole-frame phase correlation conflates camera pan with animation (a "50-px displacement" can be 5-px camera + 45-px limb swing). Use **two channels:** (1) **camera** — background-only displacement; select when `d_camera ≥ min_step` (canvas progress); (2) **animation** — among camera-qualifying candidates pick the one minimising foreground residual `‖A_animation‖` (most pose-consistent, exploiting on-twos/threes). This reduces at selection time the quantity §8.6 corrects at warp time — defence in depth.

**Implementation status (2026-06-03):** A two-pass selector is built in `backend/src/animation/frame_selection.py`. Pass 1 runs the standard greedy first-past-threshold. Pass 2 checks ±2 frames around each v1-selected frame for a better pose-similarity candidate. The pose metric `_fg_center_diff()` currently uses **gradient-magnitude L1 on the central 50% crop**, which proved unreliable: Sobel gradients in the central region include background edges (lockers, walls) that change as the camera pans, confounding pose similarity with scroll-position similarity. Disabled by default (`ASP_POSE_WINDOW_PX=0`). The architecture is ready for a proper pose metric — foreground-only RAFT flow (BiRefNet mask applied before comparison) or DWPose/ViTPose joint coordinates.

### 8.8 Principled fallback — single coherent pose (Eden 2006)
When flow confidence collapses (fast action, motion blur, huge pose gap), **abort the warp.** Per connected foreground component, select one coherent pose (one frame) and route the seam entirely around it through background via graph cut. A skipped animation frame is always better than a torn average. **Never average two conflicting poses.**

### 8.9 The HDR/VSR analogy
Foreground assembly is structurally identical to **ghost-free HDR** (DDFNet: flow-warp moving content to a reference, attention-fuse) and **video super-resolution alignment** (FDAN: flow-warp then deformable-conv refine; SMURF: multi-frame RAFT with full-image warping). The whole field converged on *flow → warp-to-reference → fuse* — the same recipe applied here to the foreground.

---

## 9. Photometric Correction

### 9.1 Broadcast dimming reversal (Harding)
Post-1997 "Pokémon Shock," Japanese broadcasters auto-dim/ghost high-contrast/flashing/fast-pan scenes to pass the **Harding Flash-and-Pattern** test (measures flash luminance, screen area, temporal frequency). Consecutive extracted frames therefore have varying global luminance — a massive exposure gradient at seams. **Reverse-dimming** restores each frame's luminance (per-frame piecewise-linear/scalar multiplication, e.g. `L_true = L_observed · (L_max_pre / L_max_dimmed(t))`) before registration so pixel-wise alignment sees consistent illumination. (Refs: `anime-undimmer`, VapourSynth `pvsfunc`.)

### 9.2 Spatial flat-fielding (BaSiC)
Model `I_i(x) = S(x)·O_i(x) + B_i` (static flat-field `S`, temporal baseline `B_i`). Apply the **spatial flat-field `S` only**, deliberately omitting the per-frame baseline `B_i` to preserve the animator's intended brightness transitions. **Trade-off:** skipping baseline harmonisation leaves frame-to-frame brightness differences that blotch the temporal median into horizontal exposure banding — hence the background-only scalar gain in §9.3 at composite time.

### 9.3 Global gain compensation (Brown–Lowe 2007)
For images `i=1…n`, overlaps `R_ij`, scalar gains `g_i`, minimise
`E(g) = Σ_(i,j) Σ_(p∈R_ij) (g_i I_i(p) − g_j I_j(p))²/(2σ_N²) + Σ_i (1−g_i)²/(2σ_g²)`
(second term is a Gaussian prior toward `g=1`, preventing the trivial all-zero solution). A tiny linear system. Per-channel → colour correction; per-channel-per-block → `BlocksGainCompensator` for spatially-varying drift. **Anime rule (empirical):** apply gain from **background pixels only**, **scalar luminance** (BT.601), **tightly clamped** (±7%; widen to `[0.80, 1.25]` only when `ref_lum < 80`). Per-channel/full-frame gain shifts hue and bands flat surfaces — avoid.

### 9.4 Colour transfer & palette harmonisation
- **Reinhard 2001** (`lαβ` mean/std matching) works for natural images but **not anime** (bimodal histograms, background-dominated stats). Fixes: **region-stratified Reinhard** (k-means clusters in Lab, greedy cluster match, per-pair Reinhard); **CDF histogram matching** over user-selected reference regions.
- **Palette harmonisation (anime-specific):** trapped-ball segmentation → per-region palette mode → bipartite (Hungarian, Lab) palette matching across frames → per-region constant offsets. Far more faithful than continuous colour transfer for cel paint.
- **Smooth lighting field:** model per-image lighting as a low-order polynomial/RBF surface `L_i(x,y)`; jointly solve `{g_i, L_i}` over overlap discrepancy (generalised vignetting/exposure calibration, Goldman–Chen 2005).

---

## 10. Foreground / Background Segmentation

- **BiRefNet** — Dichotomous Image Segmentation; ViT Localisation Module (ASPP global context) + Reconstruction Module (Sobel gradient-prior aligns the mask to 1-px line art). The ASP's masker. Apply ~16-px dilation as a safety buffer for flyaway hair/motion-blur — but over-dilation in low-res/crowded scenes erodes too much background, starving the matcher.
- **ToonOut** — BiRefNet fine-tuned on anime; **95.3% → 99.5%** pixel accuracy. Best clean fg/bg split at hair wisps, transparent FX.
- **SAM-2** — promptable segmentation; full-object contours for semantic seam routing.
- **Trapped-ball segmentation** (Zhang 2009; SIGGRAPH Asia 2024 leak-resistant variant) — clean line-art region IDs for palette work.
- **Anime instance/semantic seg** — SkyTNT ISNet, CartoonSegmentation Mask-R-CNN, AniSeg — per-frame foreground masks for exclusion/median-stacking.

Note: classical background subtraction (GMM, temporal median over raw frames) is invalid for **panning** shots because background pixels are not stationary in the temporal axis — alignment must precede median.

---

## 11. Seam Finding

Optimal seam = a path through the overlap where inter-image pixel differences are minimal, making the cut invisible.

- **MRF / graph-cut (Boykov–Kolmogorov max-flow; α-expansion for >2 sources).** Energy `E(L) = Σ_p C_d(p,L(p)) + Σ_(p,q∈N) C_i(p,q,L(p),L(q))`. Data term `C_d` = source-assignment cost (incl. user "image objectives"); interaction term `C_i` = seam cost, e.g. `‖I_a(p)−I_b(p)‖ + ‖I_a(q)−I_b(q)‖` (color match) or the gradient-augmented Graphcut-Textures form. The seam routes around salient objects through flat regions. **Hard constraints:** "pixel must come from frame k" = `t-link=∞`; "seam must/avoid curve" = `n-link` capacity 0/∞; "exclude this moving character" = paint mask → `t-link=∞` to the other source (Boykov–Jolly 2001; Davis 1998; Eden 2006). This is the **single-pose fallback** mechanism (§8.8).
- **Semantic seam routing** — **SemanticStitch** (Vis. Comp. 2025) and **OBJ-GSP** (AAAI 2025, SAM-based) penalise seams crossing foreground contours; feed BiRefNet/SAM-2 edge-confidence as `sem_cost` so the ASP seam never bisects a character.
- **DSeam (Deep Seam Prediction)** — a network predicts the seam as a mask via a selection-consistency loss; ~15× faster than graph-cut at comparable quality, real-time.
- **Anime line-art rule:** weight `C_i` by `(1 − edge_strength)` so seams run cheaply *along* strong edges (where doubling is invisible) and expensively across flat colour (where any tear shows).

---

## 12. Blending

- **Feathering/alpha** — fast; blurs and ghosts under parallax.
- **Multi-band (Laplacian-pyramid) blending** (Burt–Adelson 1983) — blend low frequencies over wide zones (lighting drift) and high frequencies over narrow zones (sharp line art). OpenCV `MultiBandBlender`. Can halo at anime's high-contrast edges.
- **Poisson / gradient-domain** (Pérez 2003) — solve `Δf = div v` over Ω with Dirichlet boundary `f|∂Ω = f*|∂Ω`; copies the source gradient field so colour shifts smoothly to match. **Color bleeding** on stark flat colours because Dirichlet conditions use only target boundary pixels.
- **Modified Poisson Blending (MPB)** — PDE depends on **both** source and target boundary pixels; Mean-Value-Coordinates / multi-spline solve + alpha-compositing step restricts colour propagation → no bleeding on flat cels. Accelerated by **MTOR** (Modified Two-parameter Over-Relaxation) for fast PDE convergence on large panoramas.
- **Soft-seam diffusion (DSFN, NeurIPS 2025)** — replaces the hard graph-cut partition with a confidence-weighted diffused soft-seam zone, absorbing residual misalignment and eliminating micro-tearing at the boundary.
- **Seamless-Through-Breaking (ACCV 2024)** — when alignment and warp-continuity are mutually exclusive, intentionally tear holes to force perfect foreground alignment, then inpaint the holes (foreground-first compositing).

---

## 13. Background Reconstruction & Generative Inpainting

- **Temporal median, foreground-excluded** — a moving character occupies any pixel <50% of the time; the median over **aligned** frames recovers the static background. **Exclude the foreground mask from the median** or varying poses average into a translucent ghost. Needs high overlap (>95%) for full coverage.
- **ProPainter (RAFT-guided video inpainting)** — propagates background pixels from distant unoccluded frames into masked voids via flow; dual-domain spatiotemporal Transformers synthesise remaining context-aware texture/line art. For narrow occlusion voids.
- **Generative panoramic outpainting (latent diffusion)** — for massive missing regions (4:3→16:9, out-of-frame). Estimate the coarse layout of references on a global canvas; fine-tune a diffusion model to outpaint missing regions **conditioned on positional encodings** of each view's location, iteratively outpainting while staying faithful to the cel-shading palette and perspective. **VidPanos** (SIGGRAPH Asia 2024) conditions video generation on known canvas regions to complete dynamic panoramic video.

---

## 14. Deep & Unified Stitching Frameworks

Staged pipelines suffer **cascading error** (registration error compounds through fusion + rectangling). Unified neural frameworks collapse the stages:
- **UDIS / UDIS++** (ICCV 2023) — unsupervised deep stitching; cascaded warp + multi-scale seam prediction; trained on UDIS-D. Domain shift on anime (trained on natural images) → fine-tune.
- **Implicit Neural Image Stitching (NIS, 2309.01409)** — estimates Fourier coefficients to produce quality-enhancing warps and blends colour/exposure/sub-pixel mismatch **in latent feature space**, decoding back to RGB; preserves high-frequency line art via arbitrary-scale-SR theory.
- **SRStitcher** (NeurIPS 2024) — treats fusion + rectangling as one **diffusion inpainting** pass guided by weighted masks; **zero-shot**, no fine-tuning, robust to registration error. An anime-diffusion backbone (Illustrious/Pony/Anything-v5) yields style-consistent seam/border synthesis.
- **UniStitch / SuperUDIS** — latent-space unsupervised stitching shifting focus to reliable features where matching fails.

---

## 15. Super-Resolution & Restoration

- **Real-ESRGAN anime_6B** — trained on anime degradation (JPEG blocks, cel-gradient loss, line thinning); preserves outlines where photo-SR over-smooths; built-in tile-and-stitch. 2× for sharp inputs, 4× for blurry sources. Shared by the ASP SR stage and the generation pipeline.
- **APISR (Anime Production-inspired SR)** — replicates multi-frame video-compression degradation (JPEG/WebP/AVIF/H.264-H.265 intra-prediction) in a prediction-oriented degradation model; inverts non-linear degradation while preserving 2-D topology (no photoreal hallucination in flat regions).

---

## 16. Video Stitching & Temporal Stability

- **Warping shake (StabStitch, ECCV 2024)** — per-frame independent alignment produces micro-jitter in warped non-overlap regions even on stable input. StabStitch jointly optimises spatial alignment + temporal smoothness via a warp-trajectory model (unsupervised, no poses). Add its second-order trajectory term to the ASP bundle adjustment (§5.2).
- **StabStitch++ (TPAMI 2025)** — **bidirectional midplane projection:** project both frames onto a virtual midplane (average position) instead of warping all into frame 0's frame, distributing distortion symmetrically. The principle behind the §8.6 symmetric midpoint warp; halves max per-frame distortion on long pans.
- **Unwrap Mosaics** (§8.5) — the deforming-object representation.

---

## 17. Shot Boundary Detection (pre-stitch)

A continuous stream must be parsed into coherent shots first. Anime breaks frame-differencing/3-D-CNN cut detectors (variable pacing, "sudden jumps" without formal cuts). **OmniShotCut** treats parsing as structured relational prediction: spatio-temporal Transformer encoder (3-D positional embeddings, ResNet18 backbone) + a decoder with 24 learnable shot-query tokens (cross-attention) jointly estimating shot ranges + continuity relations. Trained on synthetic **OmniShotCutBench** (300k videos, 11.9M parameterised transitions). Range prediction uses a differentiable 1-D **Wasserstein (Earth Mover's) distance** over temporal CDFs (instead of orthogonal-class cross-entropy) so gradients scale with temporal error. Pair with FFmpeg/PySceneDetect extraction.

---

## 18. The Full Anime Pipeline Specification (13–14 stages)

| Stage | Operation | Key detail |
|---|---|---|
| 0 | Shot detection (OmniShotCut) | parse stream → coherent pan shots |
| 1 | Load & dark-border trim | temporal-variance border detection (not intensity threshold); strip logos/subs/timecodes |
| 2 | Width/height normalisation | Lanczos-4 to a shared target (min adjacent height); guards anamorphic ref-frame distortion |
| 3 | BaSiC flat-field + reverse-dimming | spatial flat-field only; undo Harding dimming before registration |
| 4 | BiRefNet/ToonOut/SAM-2 masking | DIS mask + ~16-px safety dilation; fg isolation for exclusion + assembly |
| 4.5 | Background photometric normalisation | bg-only, scalar luminance, clamp ±7% (or `[0.80,1.25]` dark) |
| 5–6 | EfficientLoFTR match → edge filter | translation-only; reject affine; geom-consistency + direction-consensus + min-step + velocity filters; fallbacks: template NCC → phase correlation → RoMa |
| Post-6 | Spatial dedup | drop adjacent frames with displacement <25 px (iterative) |
| 7 | Translation-only bundle adjustment + affine-health gate | LM; 2-pronged outlier rejection (+ Cauchy loss upgrade); ratio/min-gap/rotation/scale validation; 3-tier retry → SCANS fallback |
| 8 | SEA-RAFT sub-pixel refine | overlap-zone crops; bg-pixel trimmed mean (replaces gradient-divergent ECC on flat regions) |
| **8.5** | **Foreground pose registration** | mask → dense flow → residual `A_animation` → **symmetric midpoint warp** → **ARAP+LSD** regularise (§8.6) |
| 9 | Canvas construction + bidirectional midplane | symmetric distortion; global offset |
| 10 | **Foreground-excluded** temporal median | background plate only; or Laplacian/Poisson render |
| 11 | Foreground assembly composite | semantic graph-cut seam (BiRefNet/SAM cost) → DSFN soft-seam → MPB/MTOR blend; bg-only scalar gain |
| 12 | Background inpainting | ProPainter (narrow voids) / latent-diffusion outpaint (large) |
| 13 | Largest-inscribed-rectangle crop / RecDiffusion rectangling | clean aspect; generative corner fill |
| opt | Real-ESRGAN anime_6B / APISR SR; ToonCrafter ghost-fill | final-quality mode |

### 18.1 Dual-channel frame selection
Replaces whole-frame phase correlation with the two-channel (camera vs animation) selector of §8.7.

---

## 19. Evaluation & Metrics

- **Laplacian-variance "sharpness" is actively misleading** — hard seam edges and color-band discontinuities *inflate* it, so a torn stitch can outscore a clean one. Do not rank by sharpness.
- **Ground-truth SSIM / PSNR** against reference panoramas — the valid quality signal when GT exists.
- **Seam coherence** = std of per-row mean luminance; high → horizontal colour banding. A reliable banding detector.
- **Strip-banding score** = max adjacent-strip luminance jump.
- **Coverage** = non-black fraction (crop completeness).
- **Ghosting** = 2nd-order vertical-derivative energy (proxy; partly conflated with content).
- **Render quality gate** — measure seam-coherence/strip-banding on the temporal render *before* compositing; fall back to SCANS if already banded.

---

## 20. Failure Modes & Principled Fallbacks

| Failure | Symptom | Fallback |
|---|---|---|
| Flow fails (fast action/blur/huge pose gap) | mangled warp | Single-pose graph-cut around one frame's character (Eden 2006); skip a frame |
| >40% bad matches / catastrophic BA | ratio/min-gap gate fails | 3-tier retry → SCANS on original (pre-ML) frames |
| Multi-plane background parallax | mid-ground tearing | APAP/Moving-DLT + LSD background warp (not foreground) |
| Flat-region flow chaos | arbitrary interior vectors | AnimeInterp SGM (segment-level flow) |
| Permanent occlusion / diagonal corners | black canvas gaps | ProPainter / latent-diffusion outpaint / VidPanos |
| ECC divergence on flat regions | zero sub-pixel gain | SEA-RAFT flow refine |
| Color bleeding at seam | Poisson interior discoloration | Modified Poisson Blending + MTOR |

**Master principle:** never average two conflicting poses — warp to agreement (ARAP midpoint) or select one (graph-cut). A skipped frame beats a torn average.

---

## 21. Implementation Status in Image-Toolkit (ASP)

- **Pipeline modules:** `backend/src/animation/` — `pipeline.py` (orchestrator), `matching.py`, `bundle_adjust.py`, `validation.py`, `canvas.py`, `ecc.py`, `flow_refine.py` (SEA-RAFT), `rendering.py` (temporal median), `compositing.py` (Stage 11), `fg_register.py` (Stage 8.5 — **shipped prototype:** DIS dense flow → residual → symmetric midpoint warp, integrated into compositing; validated on test09), `masking.py`, `photometric.py`, `mfsr/`, `sr_stitcher.py`, `super_res.py`, `anim_fill.py` (ToonCrafter).
- **Model wrappers:** EfficientLoFTR, LoFTR, ALIKED+LightGlue, RoMa, JamMa, BiRefNet, BaSiC.
- **Benchmark:** `backend/benchmark/bench_anime_stitch.py` — 96-test corpus, 55 with ground truth; GT-SSIM + seam-coherence metrics; render quality gate; selective runner.
- **Current GT baseline:** ASP 0.669 vs simple-stitch 0.695 SSIM-vs-GT — the foreground-assembly track (§8) is the path to surpass it.
- **Roadmap:** `docs/moon/roadmaps/asp.md` Phase 0 (foreground assembly: A1 SEA-RAFT engine, A3 full ARAP+LSD, A5 bg-only median, A6 single-pose fallback, two-channel selection, segment-guided flow).

### Addendum (2026-07-27) — Literature Sweep for Issue #21 Gap Areas

Targeted web search across the four gap areas the issue flagged as needing a fresh look, rather than a general resurvey of the whole field:

- **Animation-phase clustering** (grouping frames by shared background/pose phase before stitching, per `asp.md`'s phase-clustering diagnostics work): still genuinely uncovered in the literature. The only tangential hit is DBSCAN-based trajectory clustering used for a different purpose (motion-path segmentation in live-action video stabilization, not phase-boundary detection in cel animation). No paper addresses the specific problem ASP's own §2.2 diagnostics work had to solve from scratch.
- **LinkTo-Anime's fine-tuned RAFT checkpoint** (2506.02733, already cited in Appendix A for its dataset): public availability of the actual fine-tuned optical-flow weights (as opposed to the dataset) is inconclusive from search — the paper's project page references code/data release but no direct confirmation of a downloadable checkpoint was found. Treat as unavailable until directly verified against the repo, not as a ready-to-use asset.
- **Joint seam+exposure optimization** (a single joint objective over seam placement and photometric gain, vs. ASP's current sequential Brown-Lowe-then-seam approach): no strong new 2024-2025 result beyond what's already in §9/§11. The closest hit, **JoPano** (Dec 2025), is tangential — it's a generative panorama-synthesis method (diffusion-based), not classical multi-image stitching, so its joint objective doesn't transfer directly. Its **Seam-SSIM** and **Seam-Sobel** metrics (measuring seam quality independent of the generation method) are a usable idea for `bench_anime_stitch.py`'s own metric suite even though the generation approach itself isn't applicable.
- **Ghost-free fusion** (deghosting for a moving foreground during background-plate reconstruction, ASP's hardest sub-problem per §8): genuinely new relevant work exists — **UltraFusion** and **IFT** (both 2024-2025) both target exposure-fusion deghosting more directly than the HDR-deghosting literature already cited (DDFNet, Sensors 2022). Worth a closer read before the next foreground-assembly iteration; not yet integrated into ASP.
- **ToonCrafter's current successor**: **LayerInbetween** (ACM Transactions on Graphics, July 2026) supersedes ToonCrafter (already cited in §13 for `anim_fill.py`'s generative inpainting) — layer-aware anime inbetweening, published after `anim_fill.py` was originally built against ToonCrafter. Not yet evaluated against ASP's actual failure cases; flagged as the module to re-benchmark against if/when `anim_fill.py` is revisited.

None of the above required or received a code change — this is a research-currency update only, keeping Appendix A/B's citation set from going stale on the four areas issue #21 specifically asked about.

---

## Appendix A — Research Index (technique → paper)

Geometry: APAP/Moving-DLT (Zaragoza CVPR 2013); AANAP; X-Slits/pushbroom; Unwrap Mosaics (SIGGRAPH 2008). Matching: SIFT, SURF, ORB, AKAZE, MSER; SuperPoint (2018), SuperGlue (2020), LightGlue (ICCV 2023), ALIKED (TIM 2023), XFeat (CVPR 2024); LoFTR (CVPR 2021), EfficientLoFTR (CVPR 2024, 2403.04765), RoMa v2 (2511.15706), JamMa (CVPR 2025, 2503.03437), EDM (ICCV 2025, 2503.05122). Flow: RAFT (ECCV 2020), SEA-RAFT (ECCV 2024, 2405.14793), AnimeInterp (CVPR 2021, 2104.02495), GMFlow (CVPR 2022), MambaFlow (2503.07046), EpicFlow (CVPR 2015); datasets AnimeRun, LinkTo-Anime (2506.02733), ATD-12K, PaintBucket-Character. Warps: TPS, MLS (Schaefer 2006), CPW (Liu 2009), Line-Point Consistency (CVPR 2021), SEAGULL (ECCV 2016). Foreground: Sýkora ARAP (NPAR 2009); DDFNet HDR-deghost (Sensors 2022); FDAN/SMURF VSR (2021). Photometric: Brown–Lowe gain (2007), Reinhard (2001), BaSiC, Harding/anime-undimmer, Goldman–Chen vignetting (2005). Segmentation: BiRefNet, ToonOut, SAM-2, trapped-ball (2009 / SIGGRAPH Asia 2024). Seam: Boykov–Jolly (2001), Agarwala Photomontage (2004), Eden (CVPR 2006), DSeam, SemanticStitch (2025), OBJ-GSP (AAAI 2025), DSFN (NeurIPS 2025), Seamless-Through-Breaking (ACCV 2024). Blend: Burt–Adelson (1983), Pérez Poisson (2003), Modified Poisson + MTOR. Inpaint/complete: ProPainter, VidPanos (SIGGRAPH Asia 2024), generative panoramic latent diffusion, RecDiffusion (CVPR 2024). Unified: UDIS/UDIS++ (ICCV 2023), NIS (2309.01409), SRStitcher (NeurIPS 2024). SR: Real-ESRGAN anime_6B, APISR. Video: StabStitch/++ (ECCV 2024 / TPAMI 2025). Shot detection: OmniShotCut.

## Appendix B — Recommended Reading Order

Brown–Lowe (2007) → Agarwala (2004) → Pérez (2003) → Burt–Adelson (1983) → Boykov–Jolly (2001) → Schaefer MLS (2006) → Zaragoza APAP (2013) → Lin SEAGULL (2016) → Sun LoFTR (2021) → Sarlin SuperGlue (2020) → Sýkora ARAP (2009) → Eden (2006) → SEA-RAFT (2024) → AnimeInterp (2021). The math composes cleanly across these — read before the recent learned work.

## Appendix C — Source Reports Consolidated (now removable)

This document replaces: *Anime Frame Stitching Pipeline Research*; *ASP_CV_Research_and_Improvement_Plan*; *ASP_Foreground_Assembly_Research*; *Advanced Methodologies for Flawless Image Stitching in Digital Animation*; *Anime Image Stitching Pipeline Analysis*; *Anime Image Stitching: Methods and Challenges*; *Anime Screenshot Stitching Improvement Research*; *Anime Stitching Pipeline Optimization Report*; *Image Stitching Methods for Anime Screenshots*; *Image Stitching Pipeline Optimization Research*; *Image Stitching: Methods and Artifacts*; *Semi-Interactive Image Stitching for Anime Frame Imagery*; *AI_OR Research Assistant for Anime Stitching*; *AI_OR Research Assistant Pipeline Development* (the extraction/augmentation/segmentation/stitching audit — its shot-detection, BaSiC, and APISR material is folded into §3, §9, §15, §17).
