# Anime Stitch Pipeline: A Comprehensive Research Report

## From Sequential Frame Extraction to Multimodal Panoramic Stitching with Human-in-the-Loop Supervision

*Consolidated 2026-06-15. Merges: Image_Stitching_Research.md, Multi-modal Anime Panorama Stitching.md, Multimodal_ASP_HITL_Research.md, ASP Consolidated Research Plan.md, Image_Generation_Research.md, and Upgrading Anime Stitch Pipeline.md. Serves as the foundation for both academic research and production implementation of the Anime Stitch Pipeline (ASP).*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Scope and Domain Framing](#2-scope-and-domain-framing)
   - [2.2.1 2D Video Game Screenshots: A Third Domain](#221-2d-video-game-screenshots-a-third-domain)
3. [Stage 0 — Video Ingestion and Frame Extraction](#3-stage-0--video-ingestion-and-frame-extraction)
4. [Geometric Foundations](#4-geometric-foundations)
5. [The 13-Stage ASP Architecture](#5-the-13-stage-asp-architecture)
6. [Failure Mode Taxonomy](#6-failure-mode-taxonomy)
7. [Frame Selection: The Primary Bottleneck](#7-frame-selection-the-primary-bottleneck)
8. [Foreground Segmentation](#8-foreground-segmentation)
9. [Feature Matching and Bundle Adjustment](#9-feature-matching-and-bundle-adjustment)
10. [Sub-Pixel Refinement and the Aperture Problem](#10-sub-pixel-refinement-and-the-aperture-problem)
11. [Canvas Construction, Seam Routing, and Blending](#11-canvas-construction-seam-routing-and-blending)
12. [Background Completion and Generative Compositing](#12-background-completion-and-generative-compositing)
13. [Quality Assessment Metrics](#13-quality-assessment-metrics)
14. [Empirical Lessons from Hobbyists and Practitioners](#14-empirical-lessons-from-hobbyists-and-practitioners)
15. [Multi-Modal Pipeline Architecture](#15-multi-modal-pipeline-architecture)
16. [Human-in-the-Loop Architecture](#16-human-in-the-loop-architecture)
17. [Data Serialization and Dataset Harvesting](#17-data-serialization-and-dataset-harvesting)
18. [Progressive Automation via Fine-Tuning and RLHF](#18-progressive-automation-via-fine-tuning-and-rlhf)
19. [Generative Models for Anime Domain](#19-generative-models-for-anime-domain)
20. [Implementation Roadmap](#20-implementation-roadmap)
21. [Decision Thresholds and Anti-Patterns](#21-decision-thresholds-and-anti-patterns)
22. [Reference Datasets](#22-reference-datasets)

---

## 1. Executive Summary

Anime panoramic stitching is not a stitching problem with motion noise. It is a **frame-selection-conditioned multi-cel reconstruction problem** operating on a domain that violates every assumption baked into natural-scene computer vision.

The Anime Stitch Pipeline (ASP) currently achieves a corpus-wide GT-SSIM of 0.667 against 0.694 for naive simple stitching — a paradox where a sophisticated 13-stage pipeline underperforms a one-shot baseline. This metric inversion is not a tuning artifact; it reflects fundamental architectural limitations: a phase-correlation frame selector that conflates character animation with camera panning, and gradient-based optical flow that hallucinates on flat cel fills (the aperture problem).

**Benchmark (96-test corpus, 55 with ground truth):**

| Outcome | Proportion |
|---|---|
| ASP better than simple stitch | 14.5% |
| Comparable | 43.6% |
| Simple stitch better | 41.8% |
| Fallback to SCANS (render gate fired) | 40.6% (39/96) |
| Validation failure (BA outlier dominance) | 13.5% (13/96) |

**The four highest-leverage upgrades ordered by expected impact:**
1. Pose-consistent frame selection — replaces phase-correlation heuristic with background-subtracted Overmix-style cel clustering + DP optimizer
2. GNC-TLS robust bundle adjustment — replaces static residual pruning with Geman–McClure continuation
3. SAM-2 video segmentation — replaces per-frame BiRefNet with temporally consistent video predictor
4. AnimeInterp SGM + LinkTo-Anime-finetuned SEA-RAFT — solves the aperture problem on flat fills

The long-term architectural trajectory moves toward a **multimodal pipeline** that accepts not just static frames, but video files, 4K keyframe scans, and natural language prompts, while exposing HITL checkpoints at every stage that allow human experts to rectify automated errors. These corrections are simultaneously serialized into COCO JSON annotations that fuel SAM-2 fine-tuning and RLHF reward model training — progressively automating the pipeline toward zero-shot production quality.

---

## 2. Scope and Domain Framing

### 2.1 The Anime Stitching Problem Defined

The Anime Stitch Pipeline assembles complete character bodies and continuous background panoramic plates from sequential pan shots where the character is only partially visible at any given moment and is actively animating while the camera pans. The input is a sequence of broadcast or digital anime frames; the output is a single high-resolution composite showing the character in full, against a continuous panoramic background.

This problem is superficially similar to landscape panorama stitching (used in tools like Hugin, Microsoft ICE, or Google Photos) but is structurally different in every significant dimension.

### 2.2 Photography vs. Anime: A Domain Contrast

The most important theoretical insight in this document is that **every standard assumption of photographic stitching breaks when applied to 2D cel animation**. Understanding each failure mode is prerequisite to understanding why the ASP makes the algorithmic choices it does.

| Property | Natural Photography | 2D Cel Animation |
|---|---|---|
| **Texture density** | Rich, continuous spatial gradients everywhere | Vast flat-color regions; gradients confined to ink outlines |
| **Feature detectability** | SIFT/SURF/SuperPoint find thousands of stable keypoints | Keypoints cluster on collinear ink lines; flat fills return zero features |
| **Geometric model** | 3D scene → perspective projection → homography | 2D planes panning laterally → orthographic translation |
| **Motion type** | Camera rotation + parallax from 3D depth | Multi-plane camera rig; pure translation with negligible parallax |
| **Foreground behavior** | Static objects or rigid bodies | Actively animating characters with non-rigid articulation |
| **Temporal structure** | Continuous, smooth motion | Discrete "holds" on twos/threes (12 or 8 unique drawings/second) |
| **Color representation** | Continuous gradients, HDR | Flat fills, pure black ink lines, high-saturation gamut |
| **Compression artifacts** | JPEG quantization on rich textures | MPEG macroblock edges on flat regions; chroma subsampling creates false gradients |
| **Background composition** | Single connected 3D scene | Multi-plane art: separate hand-painted layers at different depths |
| **Depth of field** | Optically consistent | Artificially shallow; different background elements may be independently animated |

### 2.2.1 2D Video Game Screenshots: A Third Domain

2D video game screenshots share the most prominent surface properties of cel animation — flat-color regions, hard ink-outline analogues (sprite edges), limited palettes — but they introduce a distinct set of failure modes that diverge from both photography and anime.

**Parallax-scrolling layers.** Classic 2D games (side-scrollers, RPG maps) construct scenes from independently scrolling layers that move at different fixed ratios (typically 0.25×, 0.5×, 1.0× the camera speed). The geometric relationship between layers is therefore a piecewise-constant translation field, not a single global translation. Phase correlation on the full frame will return a dominant displacement that is neither the background nor the foreground shift, but a confusion of both. This is structurally similar to multi-plane camera parallax in anime, but with more predictable integer-ratio depth tiers. The correct handling is per-layer segmentation (which is easier in games because layer boundaries are compositionally clean) followed by independent alignment of each layer — the same multi-plane compositor approach advocated for anime, but with simpler segmentation.

**Tile repetition and phase correlation ambiguity.** Background environments are typically constructed from repeated tiles (8×8 to 32×32 pixels). The normalized cross-power spectrum (§6.1) will place a peak at the tile period — which can have a higher correlation magnitude than the true global displacement, especially when the tile grid is perfectly aligned. This is a fundamentally different failure from anime's aperture problem: instead of too few features, there are too many spurious matches at regular intervals. The fix is to restrict phase correlation to a unique landmark ROI (a character, a HUD element masked out, a distinctive background object) rather than performing full-frame correlation.

**Palette quantization and dithering.** 8-bit and 16-bit era games are limited to 16–256 colors with hard palette constraints. Regions that appear smooth are rendered with ordered (Bayer) or error-diffusion dithering, producing high-frequency spatial noise at sub-4-pixel scales. Unlike anime's flat fills (which produce zero gradient) or photographic textures (which produce spatially rich, non-repetitive gradient fields), dithered regions produce a dense, periodic, detector-confusing gradient pattern. SIFT and SURF respond strongly to this periodic noise, generating unstable, scale-ambiguous keypoints. Deep feature matchers (SuperPoint, LoFTR) trained on real-image datasets will generalize poorly here; fine-tuning on game-screenshot corpora is necessary.

**HUD and UI overlays.** Heads-Up Display elements (health bars, score counters, mini-maps, dialogue boxes) occupy fixed screen positions independent of scene content. They must be masked before registration because they are not part of the scene geometry: their pixels move with the viewport, not with the world. Unlike cel animation, where all visible layers have meaningful depth relationships, HUD elements have no depth semantics at all. The implication for the ASP is that a dedicated HUD-detection mask step (simple template matching or a fast semantic classifier) must precede the foreground mask computation that drives SemanticStitch seam routing.

**Sprite articulation vs. cel animation.** Both domains animate characters as discrete drawings, but game sprite cycles are typically shorter (4–8 unique frames per walk cycle) than traditional cel animation (12–24). This produces more pronounced temporal aliasing: optical flow estimators encounter discontinuous jumps between sprite frames more frequently relative to the animation cadence. The MAD-based hold detector in VideoIngestionStream (§3.3) must therefore use a lower threshold for game content to avoid treating every sprite-frame transition as a valid keyframe.

**Practical implication.** The ASP translation-only pushbroom model (§6) applies unmodified to game screenshots *within a single scrolling layer* — the geometry is identical. The primary adaptation needed is: (1) per-layer segmentation before registration, (2) landmark-constrained rather than full-frame phase correlation to avoid tile-period ambiguity, and (3) a HUD mask step in the preprocessing stage. All three are straightforward extensions of existing ASP infrastructure.

### 2.3 Why This Document Exists

This report serves two audiences simultaneously:

**For academic researchers:** It provides a rigorous treatment of the geometric foundations, algorithm specifications with mathematical formulations, and empirical benchmarks that should inform the next generation of animation-aware computer vision papers. The contrast between natural-scene methods and anime-specific adaptations highlights an under-studied problem space with well-defined failure modes, large-scale benchmark potential, and clear opportunities for novel contributions.

**For the ASP implementation team:** It provides a unified specification of every algorithm the pipeline does or should implement, with concrete code architecture guidance, dependency tables, and an ordered implementation roadmap tied to expected GT-SSIM improvements.

---

## <a id="3-stage-0--video-ingestion-and-frame-extraction"></a>3. Stage 0 — Video Ingestion and Frame Extraction

Before any stitching algorithm runs, frames must be extracted from video. The choice of extraction strategy has profound downstream consequences that practitioners frequently underestimate.

### 3.1 FFmpeg: The Standard Baseline

The dominant practitioner workflow begins with FFmpeg, the open-source multimedia framework. At its simplest:

```bash
ffmpeg -i input.mkv -vf fps=24 frames/%04d.png
```

This extracts every frame at 24 fps into sequential PNG files. However, several subtleties complicate this for anime:

**Variable frame rates and telecine pull-down.** Broadcast anime is typically produced at 24 fps (often "on twos" — 12 unique drawings per second) but broadcast at 29.97 fps via 3:2 telecine pull-down. FFmpeg's `mpdecimate` filter attempts to strip duplicates:

```bash
ffmpeg -i input.ts -vf "mpdecimate=hi=64*12:lo=64*5:frac=0.33,setpts=N/FRAME_RATE/TB" frames/%04d.png
```

However, `mpdecimate` applies a fixed cycle assumption. Anime produced on "threes" (8 unique drawings/second) will be mishandled by a decimation filter calibrated for twos. In scenes with both held frames and fluid motion, autonomous decimation fails entirely. The solution is not better FFmpeg flags but a higher-level duplicate detection layer in the pipeline itself (see §3.3).

**I-frame seeking accuracy.** Seeking to a specific frame with `ffmpeg -ss <timestamp>` can return the wrong frame if the requested position is not an I-frame (Intra-coded frame) in the video's GOP (Group of Pictures) structure. The standard workaround is to seek to the preceding keyframe and decode forward, but this introduces variable latency.

**Non-linear frame access.** The ASP's multi-pass frame selector must evaluate hundreds of candidate frames in a non-sequential, stride-based pattern. Launching a new FFmpeg process for each seek is prohibitively slow. FFmpeg's pipe interface (`-f rawvideo`) supports sequential streaming but not random access.

### 3.2 Python Video Reader Benchmarks

The Python ecosystem offers several alternatives to raw FFmpeg, each with distinct trade-offs:

| Library | Random Access | GPU Decode | Memory Safety | VFR Support | Anime Suitability |
|---|---|---|---|---|---|
| `cv2.VideoCapture` | Poor (drift) | No | OK | Poor | Low |
| `Decord` | Good | Yes (CUDA) | Memory leaks on batch | Poor | Medium |
| `PyAV` | Good (seek + forward decode) | No | Good | Good | **High** |
| `torchvision.io` | Adequate | Limited | Good | Adequate | Medium |

`PyAV` (a Pythonic binding to FFmpeg's low-level C libraries) is the recommended choice for the ASP. It exposes granular keyframe seeking and precise PTS-based temporal decoding. The cost of seeking to a non-keyframe requires decoding forward from the nearest I-frame, but this is acceptable given the ASP's I-frame-first proxy pass (§3.3).

### 3.3 Hybrid Memory-Mapped Ingestion Architecture

The optimal video ingestion strategy for the ASP is a two-pass hybrid architecture:

**Pass 1 (proxy pass):** Decode only I-frames at ¼ resolution. This produces a fast, coarse representation of the entire video for frame selection. I-frame-only decoding is at least 10× faster than full decode and requires no seek-forward penalty.

```python
class VideoIngestionStream:
    def get_proxy_frames(self, stride: int = 5) -> List[np.ndarray]:
        """I-frame-only pass at ¼ resolution for fast first-pass frame selection."""
        frames = []
        for packet in self._container.demux(self._stream):
            if packet.is_keyframe:
                frame = packet.decode()[0]
                arr = frame.to_ndarray(format="rgb24")
                frames.append(arr[::4, ::4])  # ¼ resolution
        return frames
```

**Duplicate decimation.** Before frame selection, drop telecine duplicates using Mean Absolute Difference (MAD) on the proxy stream:

```python
def decimate_duplicates(self, mad_threshold: float = 0.01) -> List[int]:
    proxies = self.get_proxy_frames(stride=1)
    unique = [0]
    for i in range(1, len(proxies)):
        mad = np.mean(np.abs(
            proxies[i].astype(float) - proxies[unique[-1]].astype(float)
        ))
        if mad > mad_threshold * 255:
            unique.append(i)
    return unique
```

This adaptive MAD decimation handles both "twos" and "threes" animation without a fixed cycle assumption. Failure to decimate exact duplicate frames introduces zero-displacement vectors that artificially skew the Levenberg-Marquardt global consensus (§9.2).

**Pass 2 (precision decode):** Once the frame selector has identified the optimal temporal coordinates from the proxy pass, trigger full-resolution decoding only for those specific frames. This minimizes memory overhead and maximizes throughput.

### 3.4 Hybrid 4K/1080p Ingestion

The highest-fidelity variant accepts both a compressed video for computation and sparse high-resolution keyframes for final compositing:

- The 1080p video drives all heavy computation: phase correlation, SAM-2 segmentation, LoFTR matching, bundle adjustment, ECC refinement, optical flow.
- Locked affines from bundle adjustment are then mapped onto the 4K keyframes before canvas construction.
- Final Laplacian pyramid blending runs on 4K pixel data, achieving near-4K output quality at 1080p compute cost.

This paradigm is adapted from UAV remote sensing and VR rig workflows where a sparse set of high-resolution stills is augmented by a continuous compressed video for temporal tracking.

---

## 4. Geometric Foundations

### 4.1 Classical Panoramic Stitching: Homographies and the DLT

Standard photographic panorama stitching assumes that consecutive images are related by a 2D projective transformation (homography) **H ∈ ℝ³ˣ³**, operating with 8 degrees of freedom (up to scale). The Direct Linear Transform (DLT) solves for **H** by stacking cross-product constraints derived from matched point pairs:

For a pair of corresponding points **x** ↔ **x'**, the constraint is:

```
x' × Hx = 0
```

Expanding this yields a linear system **Ah = 0** where each point pair contributes 2 independent equations. The solution **h** (the vectorised homography) is the right singular vector corresponding to the smallest singular value of **A**, obtained via SVD:

```
A = UΣVᵀ,  h = last column of V
```

In natural scenes with rich feature distributions, **A** is well-conditioned and the SVD solution is stable. The estimated homography warps one image into the coordinate frame of the other, allowing seamless alignment despite camera rotation and scene parallax.

### 4.2 Why Homographies Fail Catastrophically on Anime

**Feature collinearity.** SIFT, SURF, FAST, and their learned successors detect stable keypoints at intensity gradient extrema. In anime, gradients are densely concentrated along ink outlines — thin, nearly parallel curves — while vast interior regions (sky, skin, fabric, flat-color backgrounds) produce zero detectable features. This extreme spatial clustering causes the design matrix **A** to become severely ill-conditioned. The smallest singular value approaches zero, amplifying any observation noise into catastrophic estimation error.

**Hallucinated perspective from MPEG artifacts.** Even when homography estimation converges, the result is wrong. MPEG macroblocking on flat-color regions creates artificial 8×8 pixel boundary patterns. These spurious gradients generate false keypoint correspondences, causing the DLT to "see" a perspective transformation that does not physically exist. The estimator hallucinates a projective warp (spherical bulging, line convergence) that is mathematically consistent with the erroneous matches.

**Compounding drift.** Even sub-pixel homography errors compound logarithmically over long pan sequences. After 20–30 frames, the panorama drifts and skews vertically, requiring severe destructive cropping that eliminates the very artwork the pipeline is attempting to preserve. APAP (As-Projective-As-Possible) spatial weighting does not fix this; its Gaussian spatial decay degrades to a uniform floor wherever the nearest reliable feature is far away — which, in anime, is everywhere in the interior.

### 4.3 Translation-Only Pushbroom Geometry

Anime multi-plane camera rigs produce **pure lateral translations** in 2D plane space. There is no 3D scene geometry, no perspective projection, no depth parallax. Enforcing a **2-DoF translation-only model** is not a simplification — it is the geometrically correct model for the domain.

The translation model is mathematically equivalent to a **linear pushbroom camera** (also called a Crossed-Slits or X-Slits geometry): the final panorama is an orthographic projection of a thin temporal slit sweeping through the spatio-temporal video volume.

**Pushbroom objective.** The Levenberg-Marquardt optimizer minimizes:

```
min_{t₁,...,tₙ} Σᵢⱼ ρ(‖(xⱼ + tᵢ) − (xⱼ' + tᵢ')‖²)
```

where **tᵢ** is the 2D translation of frame *i*, **xⱼ, xⱼ'** are matched keypoint coordinates, and ρ is a robust loss function (Cauchy or GNC-TLS; see §9.2). This guarantees that stroke aspect ratios remain exact and multi-plane parallax artifacts do not trigger structural bending.

**Why this preserves anime geometry:**
- Parallel ink lines remain parallel (no projective distortion)
- Flat-color proportions are preserved exactly (no anisotropic scaling)
- Top and bottom crop is zero for perfectly vertical pans (no warping-induced clipping)

### 4.4 Phase Correlation and the Fourier Shift Theorem

Sub-pixel translation estimation between two frames exploits the **Fourier Shift Theorem**: a spatial shift **Δ = (Δx, Δy)** corresponds to a linear phase ramp in the frequency domain:

```
F{f(x − Δx, y − Δy)} = F{f(x,y)} · e^{−2πi(Δx·u + Δy·v)}
```

The **normalized cross-power spectrum** isolates the phase difference:

```
C(u,v) = F{A}(u,v) · conj(F{B}(u,v)) / |F{A}(u,v) · conj(F{B}(u,v))|
```

Applying the inverse DFT to **C** produces a Dirac delta impulse at the true shift **Δ**. Fitting a 2D Gaussian over this peak yields sub-pixel accuracy to 1/50th of a pixel.

**Critical advantage for anime:** Phase correlation operates entirely in the frequency domain. It explicitly discards the amplitude spectrum — which carries the flat-color information that confuses spatial feature matchers — and focuses exclusively on phase relationships, which encode pure structural shift. Flat regions contribute no phase noise; they are structurally invisible to this estimator.

**Failure mode.** Phase correlation measures the aggregate displacement of the entire frame. When a large foreground character occupies 60%+ of screen space and is mid-animation, the measured displacement conflates camera motion with character motion. This is the root cause of Category A failures (§6.1) and motivates the background-subtracted phase correlation approach (§7.2).

### 4.5 Transformation Model Decision Table

| Model | DoF | Preserves Parallel Lines | Vertical Drift Risk | Anime Suitability |
|---|---|---|---|---|
| Projective (Homography) | 8 | No | High (compounds iteratively) | Discarded |
| Affine | 6 | Yes | Low | Fallback gate only |
| Similarity (SRT) | 4 | Yes (isotropic) | Low | Limited use |
| **Pure Translation** | **2** | **Yes** | **None** | **Primary** |

---

## 5. The 13-Stage ASP Architecture

The current ASP implements a 13-stage translation-only pushbroom compositor:

| Stage | Module | Algorithmic Function |
|---|---|---|
| 1 | Trim & Detect | Canvas initialization, letterbox detection, dark-border trim |
| 2 | Width Normalization | Lanczos-4 resize to unified coordinate space |
| 3 | Photometric Correction | BaSiC flat-field correction for luminance gradients |
| 4 | Semantic Masking | BiRefNet foreground isolation (pixel-level) |
| 5 | Background Normalization | Scalar photometric adjustment on BG-only pixels |
| 6 | Matching & Filtering | EfficientLoFTR + ALIKED+LightGlue + Phase Correlation + RoMa; spatial dedup gate |
| 7 | Bundle Adjustment | Translation-only Levenberg-Marquardt with GNC Cauchy loss + adaptive f_scale |
| 8 | Validation Gate | Min gap, aspect ratio, off-diagonal rotation checks; retry chain → PANORAMA fallback → SCANS |
| 8.8 | Hires Injection | Optional 4K keyframe substitution before canvas construction |
| 9 | Sub-pixel Refinement | SEA-RAFT + ECC maximization |
| 10 | Foreground Registration | RAFT flow on seam-band crop + ARAP regularization + symmetric midpoint warp |
| 11 | Canvas Construction | Bidirectional midplane projection |
| 12 | Background Assembly | Foreground-excluded temporal median |
| 13 | Compositing & Blending | Laplacian blend + DSFN adaptive ramp + DP seam + Poisson (optional) |

**Frame selection precedes Stage 1.** The smart selector (`_smart_select_frames`) reduces hundreds of raw frames to a manageable subset before the full pipeline runs. It uses phase correlation displacement ≥ 50px, direction consistency, and a hold-detection filter.

---

## 6. Failure Mode Taxonomy

### 6.1 Category A — Render Quality Gate (40.6% of corpus)

**Trigger:** seam coherence > 35 or inter-strip color difference > 25 luminance units.

**Root cause:** Phase correlation measures whole-frame displacement. When a character occupies a large fraction of the frame and is mid-animation, the phase response conflates the camera's panning vector with the character's localized motion vector. The selector algorithm accepts a frame where the background has moved correctly but the character's animation phase has advanced — presenting disparate anatomical poses at the seam cut. ARAP (§10.3) cannot bridge 300–800 ms morphological gaps between poses.

**Scale of impact:** Over 30 tests in the 96-test corpus. This single failure mode drives the 40.6% fallback rate.

**Fix vector:** Pose-consistent frame selection (§7).

### 6.2 Category B — Bundle Adjustment Outlier Dominance (13.5% of corpus)

**Trigger:** affine validation dimension ratio > 2× (e.g., 11.1× in test13, 4.2× in test64).

**Root cause:** A single erroneous LoFTR match lands within the 3.0× median residual threshold, corrupting the median itself and hijacking the LM solver. Static edges (fixed patterns in the image) produce high-confidence but geometrically incorrect feature matches.

**Fix vector:** GNC-TLS bundle adjustment (§9.2).

### 6.3 Category C1 — Seam Cutting Character

**Trigger:** seam gradient > 12 on the seam visibility metric.

**Root cause:** DP seam optimization uses only photometric cost; the foreground BiRefNet penalty is insufficient when the character spans the full strip width, and the DP seam routes directly through anatomy.

**Fix vector:** OBJ-GSP + SemanticStitch hard barrier (§11.2).

### 6.4 Category C2 — SSIM Ceiling (structural)

**Observation:** Aligned SSIM exceeds raw SSIM by 0.04–0.05 (framing gap). The remaining gap of 0.17–0.25 is driven by animation timing: the ASP selects a different animation phase than the studio ground-truth composite.

**Root cause:** This is not a bug but a fundamental constraint of autonomous operation — without knowing the studio's intended pose for the composite, no automated selector can reliably match it.

**Fix vector:** Pose-consistent frame selection (§7) + ToonCrafter generative seam synthesis (§12.2) + HITL pose anchor selection (§16.2).

### 6.5 Category D — Aperture Problem in Foreground Registration

**Root cause:** Gradient-based optical flow (RAFT, DIS, TV-L1) estimates pixel displacement by solving the optical flow constraint:

```
∇I · v + ∂I/∂t = 0
```

where **∇I** is the spatial intensity gradient and **v** is the velocity field. Inside uniform flat-color regions, **∇I ≈ 0** everywhere, making this equation degenerate — the Hessian matrix of the patch becomes singular. The estimator outputs either chaotic noise or exactly zero.

When the ASP attempts to warp the foreground character using these noisy flow fields, it amplifies error proportionally to the asymmetric alpha parameter. Post-warp luminance differences exceeding 22 units trigger the ghost-prevention escalation gate, forcing a fallback to a single-pose hard cut.

**Fix vector:** Segment-guided flow (AnimeInterp SGM, §10.2) + SAM-2-tracked segment centroids (§8.2).

---

## 7. Frame Selection: The Primary Bottleneck

### 7.1 Why Frame Selection Is the Most Important Stage

The ASP's architectural insight is that **anime stitching is a frame-selection-conditioned problem**. Given a perfect set of pose-consistent frames, almost every subsequent stage succeeds: the background aligns geometrically, the seam routes easily through empty space, and the foreground requires minimal warp. Given a frame set with incompatible poses, no amount of optical flow refinement, ARAP regularization, or generative inpainting will produce a high-quality composite.

Re-architecting the frame selector is therefore likely to do more for the 40.6% fallback rate than any single optical-flow or segmentation upgrade.

### 7.2 Overmix: The Practitioner Baseline

Overmix (spillerrec, GPL-3.0, `github.com/spillerrec/Overmix`) is the closest existing analogue to the ASP and the algorithmic baseline for frame selection research. Key findings from its implementation:

**Animated aligner:** Computes pixel-difference between consecutive aligned frames. Static spans → low diff; animation transitions → spikes. Threshold segmentation divides the sequence into clusters (one per drawn cel). This is the "hold detection" approach: identify periods where the pixel delta drops below a threshold to isolate frames belonging to the same animation phase.

**Sub-pixel precision:** Integer upscaling images 4× before alignment, then standard pixel-based SAD search, then divide the result. This cheap technique achieves sub-pixel accuracy without Fourier methods.

**JPEG-aware iterative render:** Starts from the average frame; repeatedly DCTs the estimate within each input's 8×8 grid and replaces quantized coefficients matching observations. Recovers detail that MPEG compression destroys.

**Limitation:** Overmix fails during continuous fluid animation where no holds exist. Users must manually slice and layer the output in image editing software.

**Overmix is GPL-3.0.** Algorithmic ideas may be re-implemented freely; the source code may not be linked into a non-GPL ASP.

### 7.3 Background-Subtracted Phase Correlation

The key improvement over raw phase correlation is computing displacement on **background-only pixels**:

1. Compute SAM-2 foreground mask for frame *i* and frame *i+1*
2. Zero out all foreground pixels in both frames
3. Apply phase correlation on the background-only image pair
4. The resulting displacement vector is clean camera motion, uncontaminated by character animation

This is the "Overmix improvement" that separates camera motion estimation from pose consistency evaluation:
- Background difference → camera motion magnitude (for displacement gate)
- Foreground difference → pose consistency score (for cel cluster assignment)

### 7.4 DINOv2 + SigLIP Submodular Frame Selection

For selecting the optimal subset of *k* frames under a budget constraint, submodular maximization provides a principled solution with theoretical guarantees:

**Query Relevance** (SigLIP): how strongly a frame aligns with a target action or natural language prompt.

**Semantic Representativeness** (DINOv2): ℓ²-normalized patch embeddings define a facility-location coverage objective. Adding frame *f* to selected set *S* is rewarded only if its embedding occupies a distinct latent region not covered by any frame already in *S*.

**Objective:**
```
maximize L(S) = λ₁ · L_SigLIP(S) + λ₂ · L_DINOv2(S)
subject to |S| ≤ k
```

Greedy maximization of this monotone submodular objective provides a **(1 − 1/e) ≈ 0.632** approximation to the optimum. This is the best polynomial-time guarantee achievable for NP-hard submodular maximization.

Already partially implemented in the ASP (Session 8 DINOv2 frame selection, enable with `ASP_POSE_WINDOW_PX=80`). Proven superior to uniform sampling on MLVU and LongVideoBench benchmarks.

### 7.5 Pose-Consistent DP Frame Selector

The full pose-consistent selector combines three signals in a shortest-path optimization:

**Signal 1 — Camera motion magnitude:** Phase correlation on background-masked images (foreground subtracted before cross-correlation). Ensures selected frames represent sufficient spatial displacement.

**Signal 2 — Pose consistency score:** Either ViTPose-Base fine-tuned on AnimePose data, or Sýkora Push residual as a pose-distance proxy. Frames are clustered by animation phase; all selected frames should belong to the same cluster.

**Signal 3 — Foreground-crop SSIM:** Between adjacent candidate frames. High SSIM → same animation state → pose compatible.

**DP formulation:** For each strip column *c*, select the frame from a candidate window that maximizes:

```
score(f, c) = foreground_SSIM(f, neighbours(c)) − λ · background_displacement_penalty(f, c)
```

This is a shortest-path problem on a DAG (frame × column → cost), solvable in O(N·C·k²) by standard DP, where *N* is the candidate frame count, *C* is the strip column count, and *k* is the local candidate window size.

---

## 8. Foreground Segmentation

### 8.1 Current Approach: BiRefNet

BiRefNet (Bilateral Reference Network) provides the current foreground segmentation. It is run per-frame with no temporal consistency, resulting in mask jitter between frames that creates seam-pixel inconsistencies. When the BiRefNet mask bleeds into the background — which happens on thin hair strands, translucent magical effects, and complex multi-character overlaps — downstream stages amplify the error.

### 8.2 SAM-2: Video-Consistent Segmentation

**SAM-2** (Segment Anything Model 2, Meta AI, Ravi et al. arXiv:2408.00714) is the targeted replacement for BiRefNet.

**Architecture:** A streaming-memory Transformer (Hiera-B+ or Hiera-L backbone) that propagates a session memory module across video frames. A single bounding box or point prompt on frame 1 produces a "masklet" that is propagated automatically across the remaining frames via temporal memory attention.

**Performance:** Hiera-B+ at 43.8 FPS on A100; Hiera-L at 30.2 FPS. Both are inference-efficient enough for the ASP's typical 30–150 selected frame sequences.

**Training data:** SA-V (51K diverse videos, 643K spatio-temporal masklets). Not anime-specific — community reports positive results on clean-outline characters, mixed results on translucent effects and extremely thin elements (ahoge hair strands, magical particle systems).

**Integration strategy:**
1. Run BiRefNet on frame 1 → extract bounding box from mask
2. Feed bounding box as SAM-2 prompt on frame 1
3. SAM-2 propagates temporally consistent masklet across all frames
4. For translucent effects: use SAM-2 ∪ BiRefNet union mask as fallback per-frame

**Validation caveat:** SAM-2 on anime is not officially benchmarked. Run 20-clip validation against BiRefNet before full replacement.

### 8.3 Grounded SAM-2: Natural Language Segmentation

For the HITL extension (§16), SAM-2 can be prompted with natural language via GroundingDINO:

1. User types: `"the girl with the blue sailor uniform and red hair"`
2. GroundingDINO (DINO weights ~700MB) detects the character region → bounding box
3. SAM-2 propagates the masklet across all frames using the DINO bbox as initial prompt

**Expected improvement:** Eliminates character misidentification by BiRefNet in multi-character scenes (estimated +0.02–0.05 GT-SSIM).

### 8.4 Click-Based Refinement (FocalClick Paradigm)

After SAM-2 produces an initial mask, the HITL dialog exposes iterative click-based correction:

- **Left-click** on missed foreground → adds positive SAM-2 prompt point → mask expands
- **Right-click** on incorrectly included background → adds negative prompt → mask shrinks
- SAM-2 re-propagates the corrected segment across all frames (~0.5s)

This approach (FocusCut / FocalClick paradigm) provides a self-improving correction loop: each user correction immediately propagates and reduces the next correction's scope.

---

## 9. Feature Matching and Bundle Adjustment

### 9.1 Multi-Matcher Ensemble

The ASP employs an ensemble of matching algorithms to maximize coverage across the highly variable appearance space of anime:

**EfficientLoFTR:** Dense detector-free matching using coarse-to-fine correlation volumes. Produces 400–2000 matches per frame pair. Most important for background registration.

**ALIKED + LightGlue:** Keypoint detector (ALIKED) + graph-neural-network matcher (LightGlue). Faster than LoFTR on sparse scenes. Better recall on ink-line junctions.

**Phase Correlation:** Pure spectral shift estimation. Provides a reliable displacement prior even when feature matchers fail (§4.4).

**RoMa:** Dense matching via foundation model features. Slower but more robust in textureless regions.

**EDM (ICCV 2025 Highlight):** Deeper CNN backbone + Correlation Injection Module + bidirectional axis-based sub-pixel regression head. Drop-in LoFTR replacement with improved translation-specific sub-pixel accuracy. **Recommended first replacement when LoFTR quality is insufficient.**

**JamMa (CVPR 2025):** Joint Mamba state-space model replacing LoFTR's quadratic Transformer; <50% parameters and FLOPs. Consider when inference speed is the constraint.

### 9.2 Bundle Adjustment: GNC and GNC-TLS

**Current state:** The ASP uses GNC (Graduated Non-Convexity) with Cauchy loss and adaptive f_scale. This represents a significant improvement over naive RANSAC but has two remaining failure modes: (1) a single bad match that falls within the static residual threshold corrupts the median; (2) convergence can be trapped in local minima when initialization is poor.

**GNC-TLS upgrade:**

GNC-TLS (Yang et al., IEEE RA-L 2020, arXiv:1909.08605) parameterizes a surrogate loss ρ_μ(r) that starts (μ→∞) as a convex quadratic and is annealed toward Truncated Least Squares (TLS):

```
ρ(r) = min(r², c²)
```

or Geman-McClure:

```
ρ(r) = r² / (r² + c²)
```

**GNC-TLS algorithm for ASP translation-only BA:**

1. Initialize μ₀ such that `2μ₀c² ≥ max(rᵢ²)` — initial problem is effectively convex
2. For each outer iteration:
   - One LM step on weighted problem `Σ wᵢⱼ rᵢⱼ²`
   - Update weights in closed form (Geman-McClure): `wᵢ = (μ·c² / (μ·c² + rᵢ²))²`
3. Divide μ by 1.4 (Yang et al. schedule)
4. Terminate when `‖Δx‖ < tol` or `μ < μ_min`
5. Set `c ≈ 3σ` (Yang et al. recommend `c = 1.0` for normalized pixel coords with `σ ≈ 0.3px`)

**Advantage over current Cauchy loss:** The continuation schedule explicitly avoids local minima from bad initialization. Tolerates 70–80% outliers vs. RANSAC-style 50%.

**Reference implementation:** TEASER++ (`github.com/MIT-SPARK/TEASER-plusplus`) ships a GNC solver in C++/Python.

**Adaptive GNC (AGNC):** Cho et al. 2023 (arXiv:2308.11444) extends GNC by dynamically adjusting the annealing schedule based on Hessian positive-definiteness monitoring. Claims perfect stability at 99% outlier rate. Worth evaluating if standard GNC-TLS still fails on degenerate matches.

### 9.3 Pre-Bundle Static Edge Rejection

Implemented in Session 32: `_reject_static_edges()` identifies fixed-pattern image features (broadcast watermarks, subtitle bars, macroblocked edges) that produce consistently high-confidence but geometrically incorrect matches. These are filtered before the BA runs, preventing them from corrupting the residual distribution.

---

## 10. Sub-Pixel Refinement and the Aperture Problem

### 10.1 ECC Maximization

Enhanced Correlation Coefficient (ECC) maximization (Evangelidis & Psarakis, 2008) refines the affine/translation parameters by maximizing the normalized cross-correlation between a reference patch and a warped template:

```
max_W ECC(T_W, I) = (T_W^T I) / (‖T_W‖ · ‖I‖)
```

where **T_W** is the reference template warped by transformation **W**, and **I** is the target patch. ECC is solved by iteratively linearizing the warp and solving the resulting normal equations. It converges in 5–20 iterations and achieves sub-pixel accuracy by optimizing directly in the gradient domain.

**Anime suitability:** ECC works well on ink-line regions where spatial gradients exist. It fails in the same flat-color interiors where LoFTR struggles.

### 10.2 The Aperture Problem and Segment-Guided Flow (SGM)

**The fundamental limit.** The optical flow constraint equation:

```
Iₓu + I_yv + I_t = 0
```

is underdetermined even in texturized regions (the "aperture problem" for thin edges), and completely degenerate in flat-color regions where Iₓ = I_y = 0. RAFT, DIS, TV-L1, FlowFormer, and all gradient-descent flow estimators fail here for the same mathematical reason.

**AnimeInterp Segment-Guided Matching (SGM)** (Li Siyao et al., CVPR 2021, arXiv:2104.02495) resolves this by abandoning pixel-level gradient tracking in favor of region-level geometric matching:

**SGM Algorithm:**
1. **Segment** each frame via SLIC-style super-pixelation gated by Laplacian-of-Gaussian contour detection → each closed near-uniform-color region becomes one segment
2. **Describe** each segment: color histogram + centroid + area + deep CNN features (VGG-19 at relu1_2, relu2_2, relu3_4, relu4_4) → d-dimensional feature vector per segment
3. **Globally match** segments across frames via Hungarian assignment minimizing L2 cost in feature space; one-to-many fallback for region splits
4. **Propagate** piece-wise flow: every pixel inside a source segment receives the centroid-to-centroid displacement of its matched segment
5. **Matching Degree Matrix:**
   ```
   M = α·Affinity + β·DistancePenalty + γ·SizePenalty
   ```
   Empirical values: β=0.25, γ=0.25

**Why this works on anime:** The flat-color region that kills gradient-based flow is exactly the region that SLIC segments most accurately — uniform regions become perfect clean segments. The centroid displacement of the matched segment provides a noise-free, geometrically consistent flow field for the interior.

**ASP integration:** Run SGM on the SAM-2 foreground crop pair between LoFTR and RAFT. Use SGM output as warm-start initialization for SEA-RAFT (which supports flow initialization natively). For ARAP control points in flat regions, use SGM segment-centroid displacement directly.

**Important caveat:** SGM segmentation is sensitive to anti-aliasing artifacts. Disable chroma upscaling before segmentation or apply LineArtDetector pre-processing.

### 10.3 LinkTo-Anime SEA-RAFT Fine-tuning

**Dataset** (Feng et al., arXiv:2506.02733, 2025 — Macau UST/CUHK): 22+ open-source 3D character models with Mixamo skeletons, rendered with flat toon shading (uniform fills + ink contours). Ground-truth pixel-perfect forward/backward optical flow, occlusion masks, and skeleton annotations. 395 sequences; 24,230 training frames, 720 val, 4,320 test.

**Training recipe:**
1. Start from SEA-RAFT-L checkpoint (Sintel + FlyingThings pre-training)
2. Freeze feature pyramid for first 10k steps (prevent catastrophic forgetting)
3. Fine-tune for ~50k steps on LinkTo-Anime training split
4. Evaluate on LinkTo-Anime test EPE + ATD-12K validation + held-out ASP corpus

**Critical caveat:** LinkTo-Anime is 3D-rendered, not hand-drawn. Cross-validate on ATD-12K (real animation studio frames) to catch the domain gap between rendered and hand-drawn anime. Recommended strategy: pretrain on LinkTo-Anime (largest, geometry-correct), then fine-tune on ATD-12K (real animation distribution).

### 10.4 SAIN — Sketch-Aware Interpolation Network

For uncolored line-art stages (before colorization):
- Multi-stream U-Transformer with integrated self-attention + cross-attention
- Region-level, stroke-level, and pixel-level guidance
- Trained on STD-12K (30 sketch animation series)
- Use when ASP must interpolate raw line-art in pre-colorization workflows

### 10.5 ARAP Foreground Registration

As-Rigid-As-Possible (ARAP) regularization (Sýkora 2009, NPAR '09) provides the current foreground pose reconciliation between adjacent strips.

**Full Sýkora algorithm:**
1. Embed source image in a coarse square lattice (~16px), respecting foreground articulation
2. **Push phase:** for every lattice point, find: `t* = argmin_{t ∈ M} Σ_{p ∈ N} |S(p+t) − T(p)|`, where |N|=16px search area, |M|=48px motion range. Points move independently — no shape constraint.
3. **Regularize phase:** per lattice square, compute optimal 2D rigid transform via Schaefer 2006 closed form. Replace each lattice point with the centroid of its transformed instances across all incident squares. Iterate 5–20 times.
4. **Stopping criterion:** monitor `d_avg = (1/|P|) Σ ‖pᵢ − qᵢ‖`. Stop when d_avg is unchanged over 20 iterations. (Do NOT use SAD — it plateaus spuriously.)

**Speed-up:** Replace brute-force block matching in the Push phase with PatchMatch (Hao et al. 2019, 10–50× speed-up at comparable quality).

**DeepLSD Collinearity Extension (Novel, not in Sýkora 2009):**

Detect line segments via DeepLSD (Pautrat et al., ECCV 2022, MIT-licensed) on source and target frames. Match line segment pairs by descriptor + endpoint geometry. For each matched pair (ℓ_src, ℓ_tgt) with target line normal **n** and offset **d**, add a collinearity penalty to the Regularize step:

```
λ_lsd · Σ_{pᵢ ∈ ℓ_src} (n · (R* pᵢ + t*) − d)²
```

Use `λ_lsd ≈ 0.1 × rigidity weight`. Expected impact: moderate — primarily helps on costume edges and prop outlines with strong ink lines but flat fills.

---

## 11. Canvas Construction, Seam Routing, and Blending

### 11.1 Canvas Construction: StabStitch++ Bidirectional Midplane

**Current limitation:** The ASP warps each strip onto its left neighbor sequentially, accumulating warp error with each step. The asymmetric distortion grows monotonically from left to right.

**StabStitch++** (Nie et al., TPAMI 2025, arXiv:2505.05001) reformulates canvas construction as a bidirectional midplane projection:

1. Compute a virtual midplane **T(t)** — a 2D translation trajectory shared across all strips
2. Project each strip onto the midplane (not onto its neighbor), distributing distortion symmetrically
3. Regularize with warp smoothing:
   ```
   min_T ‖∇²T‖₁
   ```
   (L1-trend-filter on second differences of the trajectory — appropriate for anime pans, which are piecewise-constant velocity)

**Bidirectional midplane** eliminates asymmetric distortion of the current "warp one strip to neighbour" approach.

**Horizontal and diagonal scroll support:**
1. Full 2D phase correlation for (tx, ty) displacement estimation — not projected onto a single axis
2. Generalize BA to estimate (tx, ty) per frame
3. Orient strip seams perpendicular to the estimated camera-motion direction per pair
4. Detect scroll axis: if `ty_range < 0.1 × tx_range`, apply horizontal strip mode or fall back to SCANS
5. For rotation/scale: augment LM state with global rotation per frame, strong prior toward zero

### 11.2 Seam Routing: From DP to Semantic Hard Barriers

**Current DP seam.** The ASP uses Dynamic Programming graph-cut to find the minimum-cost seam path through the overlap region. Cost is based on photometric dissimilarity between the two strips at each pixel position. This produces good seams in background regions but routes through character anatomy when no photometrically quiet background corridor exists.

**SemanticStitch hard barrier** (Jin et al., arXiv:2511.12084):

Set the seam cost of all pixels inside the SAM-2 foreground mask to `10⁶ × photometric_cost`:

```python
def _build_seam_cost_map(fa_zone, fb_zone, fg_mask=None, ...):
    cost = np.abs(fa_zone.astype(float) - fb_zone.astype(float)).sum(axis=-1)
    if fg_mask is not None:
        cost[fg_mask.astype(bool)] = 1e6
    return cost
```

This mathematically guarantees the seam routes around the character. Where the seam must cross the character (vertical pan, character occupies full strip width): don't seam-blend at all — select the single best frame for that strip column (combined with pose-consistent frame selection).

**OBJ-GSP mesh shape preservation** (Cai & Yang, AAAI 2025, arXiv:2402.12677): SAM-2 segments are converted to triangular meshes during warping. Each mesh balances projective + similarity transformations, keeping object shapes intact during the warp. Apply to the character foreground mesh to prevent shape distortion.

**Graph-Cut with Intelligent Scissors fallback (HITL):**
- Graph-Cut on the panoramic canvas as a pixel-node graph weighted against character silhouette ink lines
- Intelligent Scissors: user-provided spatial waypoints force the seam through designated empty background space
- Guarantees seam integrity without requiring perfect automated foreground detection

**Natural Language Seam Routing:**
User types `"route seam around the right arm"`. GroundingDINO detects the named region → returns pixel-space bounding box → injected as `cost[mask] = 1e6` hard barrier into `_build_seam_cost_map()`. This allows non-technical users to provide spatial constraints using natural language.

### 11.3 Blending

**Multi-band Laplacian pyramid blending:**

At each scale *l* of the Laplacian pyramid, blend independently using the mask pyramid:

```
Blend_l(p) = M_l(p) · A_l(p) + (1 − M_l(p)) · B_l(p)
```

where M is the smooth mask, A and B are the two strips. Low frequencies transition gradually; high frequencies transition sharply. This eliminates the haloing artifacts of simple alpha blending.

**Poisson seamless cloning** (optional, `ASP_POISSON_SEAM=1`):

Solves the Poisson equation to find the blended output *f* such that:

```
∇²f = ∇²g,  f|∂Ω = f*|∂Ω
```

where *g* is the source patch, *f** is the target, and *∂Ω* is the seam boundary. This matches gradient fields rather than pixel values, producing smooth transitions through any seam regardless of absolute color difference.

**DSFN adaptive ramp:** Per-pixel gain normalization using a spatially varying feather ramp that widens or narrows based on photometric compatibility between adjacent strips. Prevents over-brightening at seam edges.

---

## 12. Background Completion and Generative Compositing

### 12.1 ProPainter — Deterministic Background Completion

**ProPainter** (Zhou et al., ICCV 2023, arXiv:2309.03897) is a video inpainting model that fills masked regions via dual-domain (image + feature space) flow-guided propagation + mask-guided sparse video Transformer.

**Performance:** ~192 FPS on NVIDIA V100. +1.46 dB PSNR over FuseFormer/FGT baseline.

**Pipeline integration:**
1. SAM-2 foreground mask per frame
2. Align frames with translation-only BA
3. Project to panoramic canvas with foreground masked out
4. ProPainter on masked panoramic video to complete background under character footprint
5. Collapse temporal axis with existing temporal median

**Why ProPainter over generative alternatives:** ProPainter is deterministic and does not hallucinate new content into flat-color backgrounds. Generative models (e.g., VidPanos, Stable Inpainting) are inappropriate for background completion in anime — they introduce photoreal textures into regions that should be flat fills, destroying the artistic style.

### 12.2 ToonCrafter — Generative Midpoint Synthesis

**ToonCrafter** (Xing et al., SIGGRAPH Asia 2024/ACM TOG 43(6), arXiv:2405.17933) is an anime-specific generative inbetweening model.

**Architecture:**
- Adapts live-action image-to-video diffusion priors (DynamiCrafter) to cartoon domain via **toon rectification learning** — fine-tunes image context projector + spatial layers on curated cartoon data while freezing temporal layers, eliminating "content leakage" (realistic textures on flat cels)
- **Dual-reference 3D VAE decoder:** injects uncompressed pixel data from input keyframes as structural anchors during decoding, preserving crisp ink-line detail
- **Sparse sketch guidance:** human-drawn intermediate sketches as ControlNet anchors for non-linear pose control

**ASP use case:** When two adjacent strips have character poses that ARAP cannot bridge without ghosting (post-warp seam difference > 22 luminance units), synthesize an intermediate frame conditioned on the two adjacent strip character crops, then use it as the seam transition source.

**Inference:** ~24s per 320×512 clip at 50 DDIM steps on A100; 10–17 GB VRAM with fp16.

**Critical caveats:**
- Authors' README: "*due to the variety of generative video prior, the success rate is not guaranteed.*"
- **Always gate output** by LPIPS or CLIP similarity to input strips — mandatory quality check
- Wired in ASP as `ASP_TOONCRAFTER_SEAM=1` (default OFF) for worst single-pose seam per test
- Also serves as the ASP ghost-fill / occlusion-completion model (`animation/anim_fill.py`)

### 12.3 OmnimatteZero — Training-Free Layer Decomposition

(arXiv:2503.18033, SIGGRAPH Asia 2025) — 0.04 s/frame on A100. Encodes the original video and a clean background estimate into diffusion latent space; computes latent difference to isolate the foreground; self-attention maps inpaint associated effects (shadows, reflections). Use to strip animated character + shadow from complex pan shots, leaving a pristine background for global motion estimation when BiRefNet/SAM-2 masks are unreliable.

---

## 13. Quality Assessment Metrics

### 13.1 Reference Metrics

**GT-SSIM (Ground Truth SSIM):** Structural Similarity Index Measure against a studio-produced ground truth composite. Computed at 55 of the 96 benchmark tests. Current corpus-wide GT-SSIM: ASP = 0.667 vs. simple stitch = 0.694.

**Aligned SSIM:** SSIM after rigid re-alignment of the ASP output to the GT. Exceeds raw GT-SSIM by 0.04–0.05, indicating that framing differences (not structural quality) account for part of the gap.

**PSNR:** Peak Signal-to-Noise Ratio. Less perceptually meaningful than SSIM for anime composites but useful for regression testing.

### 13.2 No-Reference Metrics

**Seam Coherence (current pipeline gate):** Inter-strip color difference at the seam line. Threshold > 35 triggers the render quality gate. A hard-coded heuristic that will eventually be replaced by learned quality assessment.

**SIQE — Ghosting Detection:**
- Multi-scale steerable pyramids (2 scales, 6 orientations → 12 subbands) decompose the target image
- Ghosting manifests as unnatural high-frequency edges in tight spatial clusters
- Gaussian mixture model models pyramid statistics → quantifies probability of structural inconsistency
- Local optical-flow variance establishes perceptual geometric error gradient
- 94.36% precision vs. mean human opinion in empirical studies

**SI-FID:** Contrastive training with artificial stitching noise injections → Fréchet distance between generated vs. pristine feature distributions. Rank correlation coefficient ≥25% higher than competing NR-IQA indicators.

**DiFPS (DINOv2 Features Perception Similarity):** Uses DINOv2 semantic completeness rather than patch similarity — evaluates geometric coherence of the stitched result.

**DLNR-SIQA — Localized Artifact Detection:** Fine-tuned Mask R-CNN segments and localizes discrete stitching error regions. Provides precise pixel coordinates of ghosting and seam artifacts — enabling targeted re-synthesis by ProPainter or ToonCrafter on the detected error regions.

### 13.3 MLLM-Based Quality Scoring

**SIQS (Single-Image Quality Score):** VLM-based assessment detecting semantic contradictions — severed torsos, four arms, mismatched lighting.

**MICQS (Multi-Image Comparative Quality Score):** Multi-image comparison using large multimodal models (Qwen-VL, GLM). Captures catastrophic logical failures that statistical metrics like SI-FID overlook.

These represent the natural long-term replacement for hard-coded seam coherence gates — a fully learned perceptual quality model that can evaluate anime composites with human-level semantic understanding.

---

## 14. Empirical Lessons from Hobbyists and Practitioners

### 14.1 FFmpeg Workflows and Their Limits

The anime enthusiast community has long engaged in manual and semi-automated screenshot stitching. The prevailing workflow:

```bash
ffmpeg -vf "decimate=cycle=2,setpts=N/FRAME_RATE/TB" -i input.mkv frames/%04d.png
```

This works for content animated consistently on twos. It fails for variable-timing anime (different scenes use different animation intervals), for content with telecine pull-down where the cycle is not uniform, and for high-action scenes with genuine motion on every frame.

**Practitioner wisdom:**
- Extract at full frame rate, then decimate in post using the pipeline's MAD-based approach (§3.3)
- Never trust `mpdecimate` to handle anime telecine reliably — always validate the decimated output manually
- Use 16-bit PNG or lossless formats for intermediate frames; JPEG artifacts interact catastrophically with feature matchers

### 14.2 Hugin and Panorama Stitching Software

**Hugin** (open-source, `hugin.sourceforge.io`) is the primary tool advanced hobbyists use for anime stitching. Its `cpfind` utility generates control points and `nona` warps images into cylindrical/spherical projections.

**Limitation:** Hugin's default projection models introduce severe spherical bulging on flat orthographic anime scenes. Advanced users mitigate this by forcing translation-only mode via `--affine --no-crop`, but Hugin still cannot handle animating foreground characters — it either ghosts them (blurs multiple poses) or truncates limbs at seam boundaries.

**Lesson for ASP:** Translation-only enforcement (which Hugin supports poorly) is a hard requirement, not an option. The ASP's explicit 2-DoF BA with rejection of rotation/scale components is the correct approach.

### 14.3 Microsoft ICE (Image Composite Editor)

ICE's "video panorama" feature autonomously ingests video and tracks camera motion. Highly praised for speed and handling subtle exposure changes.

**Limitation:** No granular control over frame selection or seam routing. When it encounters animating characters, ICE treats changing poses as temporal noise, rendering a blurred, translucent approximation of the character against a sharp background — exactly the ghosting Category D produces.

**Lesson for ASP:** Automated frame selection that doesn't account for pose consistency is fundamentally wrong for animated content. The "temporal average" approach is only valid for completely static subjects.

### 14.4 Overmix: The Domain-Specific Baseline

Overmix (`github.com/spillerrec/Overmix`, GPL-3.0) is the best existing tool for anime screenshot stitching. It uses pixel-difference thresholding to group frames by animation phase ("animated aligner"), then renders the median of aligned frames within each group.

**Strengths over general-purpose stitchers:**
- Domain-specific hold detection
- Sub-pixel alignment without Fourier methods (4× upscaling trick)
- JPEG-aware iterative render recovers compression detail

**Limitations:**
- Falls apart during continuous fluid animation with no holds
- No semantic foreground/background separation
- Manual post-processing required when holds don't exist
- GPL-3.0 license prevents code reuse in non-GPL projects (only algorithmic ideas are reusable)

**Lesson for ASP:** The animated aligner concept (frame clustering by pixel-difference spikes) is correct and should be the foundation of the pose-consistent frame selector (§7.5). The key extension is running it on background-subtracted images to avoid the animation-conflation problem.

### 14.5 Why All General-Purpose Tools Fail on Anime

The common thread across Hugin, ICE, Overmix, and every other general-purpose stitcher:

1. **Projection model mismatch:** Spherical/cylindrical projections for what is a 2D orthographic pan
2. **Feature detector failure:** SIFT/SURF/ORB cannot find stable keypoints in flat-color interiors
3. **Temporal model mismatch:** Tools assume temporally smooth motion; anime has discrete phase jumps
4. **Foreground ignorance:** No tool except Overmix makes any attempt to isolate the animating character
5. **No pose awareness:** Frame selection by geometric criteria only, ignoring animation phase

The ASP addresses all five failure modes. The remaining gap (fallback rate 40.6%, SSIM < naive baseline) is driven by the two that are hardest to solve without human involvement: pose consistency (requires knowing the studio's intended composite) and flat-color aperture problem (requires segment-level tracking).

---

## 15. Multi-Modal Pipeline Architecture

### 15.1 Video Ingestion as Native Input

The current ASP paradigm requires FFmpeg pre-extraction — a destructive step that:
- Fractures temporal continuity
- Bloats storage requirements (100 frames at 1080p ≈ 600 MB PNG vs. 30 MB in source video)
- Discards temporal metadata (PTS, frame type, GOP structure)
- Prevents access to the video's native temporal prior (frame-to-frame compression residuals that encode motion)

Native PyAV ingestion (§3.2) resolves all four problems. The `VideoIngestionStream` class wraps the full ingestion lifecycle, from proxy pass to precision frame decode.

### 15.2 Hybrid 4K/1080p Pipeline (§9C, Session 119)

The `hires_keyframes: Dict[int, str]` parameter to `AnimeStitchPipeline.run()` enables hybrid operation:

1. All heavy computation (phase correlation, SAM-2, LoFTR, BA, ECC, ARAP) runs on 1080p video frames
2. The locked affine grid from Stage 7 (BA output) is preserved
3. Stage 8.8 (new): replaces proxy frames with 4K hires images; scales affines proportionally:
   - Linear 2×2 sub-matrix (rotation/scale) is preserved exactly
   - Translation components (tx, ty) are scaled by the upscale ratio (e.g., 3840/1920 = 2.0)
4. Stage 11 Laplacian/Poisson blend runs at 4K → near-4K output quality at 1080p compute cost

**Affine scaling formula:** For an upscale factor s, given affine matrix A with translation (tx, ty):

```
A_hires = [[A[0,0], A[0,1], tx*s],
           [A[1,0], A[1,1], ty*s],
           [0,      0,      1   ]]
```

The 2×2 linear sub-matrix is invariant to upscaling (it encodes rotation/scale, which do not depend on the image resolution). Only the translation components scale.

### 15.3 The Academic 6-Stage Generative Architecture

The frontier multimodal approach (Karunratanakul et al., SIGGRAPH 2026) extends beyond classical stitching into generative scene re-composition:

| Stage | Function | Key Models |
|---|---|---|
| 1 | Video Restoration | AnimeSR / APISR super-resolution |
| 2 | Instance-Aware Segmentation | RTMDet-Ins + SAM-2 video propagation |
| 3 | 2.5D Layer Decomposition | See-Through framework (23 anatomical layers) |
| 4 | Layout Estimation + Feature Alignment | DINOv2 cross-frame correspondence |
| 5 | Generative Stitching | ControlNet-Inpaint + IP-Adapter + PanoDiffusion |
| 6 | Temporal Motion Synthesis | CoTracker + ToonCrafter + PhysAnimator |

**Stage 3 (See-Through decomposition)** is the most novel component: it decomposes the scene into 23 anatomical character layers (each body part, costume element, and background plane as a separate transparent layer), enabling independent manipulation of each element.

**Stage 5 (PanoDiffusion)** enforces spherical wraparound consistency — the left-right boundary of the panorama is forced to converge, enabling seamless looping panoramas without explicit stitching at the boundary.

**Current availability:** The See-Through framework (`shitagaki-lab/see-through`) is not yet pip-installable. PanoDiffusion weights are available but not anime-fine-tuned. ControlNet-Inpaint + IP-Adapter are available and wired in ComfyUI (`§4.8A in new_features.md`).

### 15.4 Generative View Stitching (GVS)

**GVS** (arXiv:2510.24718, "Omni Guidance") presents a non-autoregressive alternative to sequential video generation. It partitions the target video into non-overlapping chunks and denoises every target chunk jointly with its neighboring chunks, conditioning on both past and future context via a guided score function:

```
score_guided = score_original + λ · score_bidirectional
```

This **Omni Guidance** mechanism maintains frame-to-frame consistency over long camera trajectories without autoregressive drift.

**ASP application:** When the camera pan outpaces the bounds of the original background artwork, exposing empty canvas space, a GVS-adapted model conditioned on the ASP's locked affine grid could outpaint missing background — extending wooden floors, painted skies, or corridor interiors while adhering to spatial continuity.

---

## 16. Human-in-the-Loop Architecture

### 16.1 Philosophical Foundation

HITL is not a fallback for when automation fails — it is the primary architectural decision that determines the ceiling of achievable quality. Pure automation cannot resolve Category C2 failures (SSIM ceiling) because the "correct" answer (which animation phase the studio intended for the composite) is not in the data. It requires human artistic judgment.

The HITL architecture shipped in Session 79 (QWaitCondition/QMutex pause-resume with 4 checkpoints) provides the structural foundation. The multi-modal extension adds foundation model interaction at each checkpoint.

### 16.2 HITL Checkpoints

**Stage 2.1: Multi-Modal Frame Selection**

The selection dialog presents a navigable timeline of the video. Frames where the camera displacement is mathematically optimal (phase correlation gate passed) but DINOv2 cosine distance flags a severe pose mismatch are highlighted. The user can:
- Override frame selection by dragging to an alternative frame
- Draw a bounding box to define the character pose that should serve as the global reference anchor
- Type a natural language description (`"standing with arms at sides"`) to filter candidates

**Stage 4: Interactive Segmentation Refinement**

The HITL dialog shows the BiRefNet/SAM-2 mask overlaid on frame 1 in a `QDialog` with `QGraphicsView`. The user applies corrective clicks (§8.4). SAM-2 re-propagates the corrected masklet across all frames (~0.5s).

**Stage 7: Bundle Adjustment Override**

Presents the matched keypoints overlaid on the background. The user can:
- Delete erroneous matches that are visibly wrong
- Provide manual correspondences on background elements (landmark-based BigWarp correction)
- Accept or reject the proposed affine grid before the pipeline proceeds

**Stage 10: Foreground Registration Review**

Shows the ARAP-warped foreground character overlaid on the background. The user can:
- Accept the current registration
- Apply scribble-based trajectory hints to guide the warp direction
- Force a hard-cut (single pose, no warp) for irreconcilable gaps

**Stage 13: Seam Routing Override**

The EdgeReviewDialog (shipped S79) shows the current DP seam path. The user can:
- Provide spatial waypoints to force the seam into safe background space
- Type `"route seam around the right arm"` for NL seam routing
- Accept the current seam

### 16.3 Natural Language Seam Routing (Phase A3)

```python
def _build_seam_cost_map(fa_zone, fb_zone, exclusion_masks=None, ...):
    cost = np.abs(fa_zone.astype(float) - fb_zone.astype(float)).sum(axis=-1)
    if exclusion_masks:
        for mask in exclusion_masks:
            cost[mask.astype(bool)] = 1e6
    return cost
```

The NL exclusion mask is computed as:
1. User types: `"route seam around the right arm"`
2. GroundingDINO detects the arm region → bounding box
3. SAM-2 segments the arm → binary mask
4. Mask injected as `cost[mask] = 1e6` into `_build_seam_cost_map()`

### 16.4 Benchmark JSON Import (§1.10E, Session 119)

The `bench_import.py` module provides RLHF-oriented result import:

```python
def suggested_rating(metrics: Optional[Dict[str, Any]]) -> float:
    """
    composite = coverage*0.35 + sharpness_norm*0.25 + (1−ghosting)*0.20 + seam_coh*0.20
    Returns float in [0.0, 10.0]
    """
```

This formula reflects the empirically validated weighting from the 55-test GT corpus:
- Coverage (0.35): whether the panorama fills the canvas without missing regions
- Sharpness (0.25): whether the composite maintains ink-line crispness
- Anti-ghosting (0.20): absence of double-image artifacts at seam boundaries
- Seam coherence (0.20): photometric continuity across strips

Pre-computed ratings from benchmark runs are shown in the StitchFeedbackTab as a dataset list with verdict badges, enabling operators to efficiently review large benchmark results and identify cases that merit human rating.

---

## 17. Data Serialization and Dataset Harvesting

### 17.1 Strategic Value of HITL Data

Every human interaction in the HITL pipeline is an exceptionally high-value data point:
- Corrected bounding boxes → SAM-2 fine-tuning data
- Frame selection overrides → pose embedding contrastive pairs
- Segmentation clicks → mask decoder fine-tuning data
- Seam waypoints → seam routing preference data
- Pairwise composite preferences (A vs. B) → RLHF reward model training data

Small labeling errors in training data teach models the wrong pattern. Human expert corrections provide the accurate ground truth required to correct systemic biases — this is the critical advantage of domain-expert HITL over crowd-sourced annotation.

### 17.2 COCO JSON Schema

```json
{
  "info": {"session_id": "...", "operator_id": "...", "timestamp": "..."},
  "images": [
    {"id": 1, "file_name": "frame_0042.png", "temporal_id": 42, "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": {"counts": "RLE...", "size": [1080, 1920]},
      "bbox": [x, y, w, h],
      "attributes": {
        "prompt": "blue sailor uniform, red hair",
        "correction_type": "positive_click",
        "pre_correction_iou": 0.72,
        "post_correction_iou": 0.94
      }
    }
  ],
  "categories": [
    {"id": 1, "name": "character"},
    {"id": 2, "name": "background"},
    {"id": 3, "name": "seam_exclusion"}
  ]
}
```

**Storage:** `~/.image-toolkit/hitl_annotations/session_{timestamp}.json`

### 17.3 Label Studio Multi-Modal Export

Label Studio format adds the model-vs-human delta:
- `predictions` array: SAM-2's pre-correction mask (what the model proposed)
- `annotations` array: human's post-correction accepted mask (what the human chose)

This precisely captures the RLHF preference supervision signal — not just the correct answer, but the model's wrong answer paired with the human's correction. Frameworks like SAMannot demonstrate how to partition large video sequences into memory-conscious blocks for scalable annotation serialization.

---

## 18. Progressive Automation via Fine-Tuning and RLHF

### 18.1 SAM-2 Anime Domain Fine-Tuning (Phase C1)

**Architecture:**
- Frozen: ViT-H image encoder (domain-agnostic; too large for per-domain FT on 24 GB)
- Fine-tuned: Mask decoder + memory module on collected COCO anime masks

**Training objective:**
```
L = BCE(corrected_mask, sam2_prediction)
```

Binary cross-entropy on the human-corrected mask vs. SAM-2's pre-correction prediction. The collected HITL corrections serve as the training pairs: model output is the negative example, human correction is the target.

**Expected outcome:** After 100+ sessions, SAM-2 correctly delineates semi-transparent magical effects, thin ahoge hair strands, and complex multi-character overlaps without human prompting.

**Data gate:** Requires ~100+ COCO-formatted annotation sessions before fine-tuning is meaningful.

### 18.2 Pose Contrastive Fine-Tuning (Phase C2)

Human frame selection overrides become contrastive triplets:
- Rejected frame: anchor embedding
- Accepted frame: positive embedding
- Random frame: negative embedding

Loss: InfoNCE on DWPose/ViTPose embedding space:

```
L_NCE = −log [exp(sim(a, pos)/τ) / Σₖ exp(sim(a, kₖ)/τ)]
```

Effect: `_compute_dinov2_features()` → `_pose_dist()` learns to rank anime poses by visual coherence in the specific production style, reducing frame selection errors without requiring explicit pose labels.

**Data gate:** Requires ~500+ selection override pairs.

### 18.3 RLHF Reward Model (Phase C3)

**Preference data collection:** The ASP generates two stitching hypotheses for the same sequence (e.g., Laplacian blend vs. DP seam with Poisson). The operator selects the superior result. This pairwise preference is serialized to Label Studio format.

**Reward model architecture:** CNN or Transformer classifier that takes a stitched panorama and outputs a scalar quality score. Trained via Bradley-Terry model on collected pairwise preferences:

```
P(A > B) = σ(r(A) − r(B))
```

**PPO integration (Phase C3):** Once the reward model is calibrated, Proximal Policy Optimization controls compositing parameter selection:
- **Action space:** discrete compositing parameters (feather width, seam cost weights, blend method, gain cap)
- **State:** current composite encoded by the reward model's backbone
- **Objective:** maximize `E[r(composite)]` across the 55 GT-test corpus

Expected effect: non-obvious parameter combinations emerge from PPO exploration that outperform manual tuning.

**Data gate:** Requires calibrated reward model (Pearson r ≥ 0.7 with human ratings on held-out composites).

---

## 19. Generative Models for Anime Domain

### 19.1 Diffusion Model Foundations

**ε-prediction (DDPM):** Predicts the noise added at each step. Standard for SD 1.5 / SDXL-eps. Weakness: over-weights high-frequency noise at the expense of low-frequency structure → washed-out flat fills and greyness in pure-color anime backgrounds.

**v-prediction:** Predicts a velocity vector `v = αₜε − σₜx₀` combining noise and data. Objective: `L = E‖v − v_θ(xₜ,t)‖²`. Converges faster and dramatically improves dynamic range — rendering stark high-contrast anime lighting correctly. **NoobAI-XL V-Pred** uses this.

**Rectified Flow Matching (RFM):** Deterministic neural ODE transporting mass along near-straight paths. Train velocity field `v_θ` to regress the constant connecting vector via least-squares; generate by ODE integration. Much fewer steps at equal fidelity. Backbone of FLUX and experimental NoobAI-XL RF conversions.

**Practical rule:** Match LoRA/training objective to the base model (eps vs v-pred vs RF). Mixing degrades output.

### 19.2 SDXL Anime Fine-Tunes

| Model | Conditioning | Key Property | Caveat |
|---|---|---|---|
| **Illustrious XL v2.0** | Danbooru tags + NL | Native 1536px; recommended base | Token dilution — front-load critical concepts |
| **NoobAI-XL V-Pred** | Danbooru tags | v-pred; superior color fidelity | Separate training workflow |
| **Animagine XL 4.0** | Structured tags | Cleanest line art; strict tag adherence | Structured prompt order required |
| **Pony Diffusion V6 XL** | Score tags | Ranked aesthetic training | Incompatible with standard LoRAs; needs rigid `score_9,...` prefix |

### 19.3 FLUX and MM-DiT

**FLUX.1 [dev]** (12B MM-DiT): joint image+text token attention via T5XXL LLM embedding. Extreme anatomical coherence (fixes SDXL finger/limb mutations) but heavy photoreal priors fight anime style. Anime fine-tunes: **Chroma** (de-distilled FLUX.1 schnell derivative), **Kaleidoscope** (FLUX.2 Klein).

**Hardware reality:** FLUX.1 runs natively on RTX 3090 Ti (24 GB) at FP16 (~12s per 1024² image at 1.7 it/s). On RTX 4080 (12 GB), GGUF Q4_0 quantization is mandatory (~8 GB total with FP8 T5XXL, ~35s per image). **Never apply LoRAs with NF4 quantization** — NF4 disrupts the attention blocks that LoRA adapts.

### 19.4 AnimateDiff for Temporal Generation

AnimateDiff inserts a motion module (pseudo-3D temporal self-attention) into a frozen spatial T2I network:

- Spatial layers: frozen (preserves base checkpoint's style/anatomy)
- Temporal layers: trained on motion from large video datasets

This model-agnostic design means any community checkpoint, LoRA, or DreamBooth model can be animated without retraining. The spatial prior is preserved; only the temporal prior is added.

**Critical anime fix:** Highly-tuned anime SDXL models alter the noise schedule. Standard AnimateDiff schedulers then mis-align temporal gradients → chaotic output. **Force `beta_schedule = linear`** and lock context length to 16 frames.

### 19.5 ASP-Relevant Generative Capabilities

**Super-resolution:** Real-ESRGAN `anime_6B` (trained on anime-specific degradations) is shared with the ASP's super-resolution stage (`animation/super_res.py`). APISR provides an alternative that inverts non-linear broadcast compression artifacts (JPEG/WebP/H.264 intra-prediction) without photoreal hallucination.

**Inbetweening:** ToonCrafter provides generative inbetweening between two keyframes — the exact operation needed for pose-gap bridge in the ASP (§12.2). ToonComposer (DiT-based) adds simultaneous inbetweening + colorization from one colored keyframe + sparse sketches.

**Outpainting:** GVS (§15.4) for extending background beyond the original canvas bounds.

---

## 20. Implementation Roadmap

### 20.1 Phase 1 — Highest Impact, Lowest Complexity

| # | Item | Section | Complexity | Expected GT-SSIM Δ |
|---|---|---|---|---|
| 1 | Pose-consistent frame selection (Overmix cel clustering + DP) | §7.5 | Medium | +0.05–0.15 |
| 2 | GNC-TLS bundle adjustment | §9.2 | Low (~100 lines) | +0.02–0.05 |
| 3 | SAM-2 as BiRefNet replacement (one bbox prompt, video propagation) | §8.2 | Low | +0.02–0.05 |
| 4 | Pre-bundle static edge rejection | §9.3 | Low (done S32) | ✅ |

**Decision gate:** If Phase 1 raises ASP-beats-naive from 14.5% → ≥35% and lowers fallback from 40.6% → ≤25%, Phase 2 is justified.

### 20.2 Phase 2 — Medium Complexity, Large Impact

| # | Item | Section | Complexity | Expected GT-SSIM Δ |
|---|---|---|---|---|
| 5 | AnimeInterp SGM + LinkTo-Anime SEA-RAFT FT | §10.2–10.3 | Medium | +0.03–0.08 |
| 6 | SemanticStitch hard seam barrier + OBJ-GSP | §11.2 | Low (cost-term mod + SAM-2) | +0.02–0.05 |
| 7 | Full Sýkora 2009 ARAP + DeepLSD collinearity | §10.5 | Medium | +0.01–0.03 |
| 8 | ProPainter background completion | §12.1 | Medium | +0.01–0.02 |

### 20.3 Phase 3 — High Complexity, Fallback-Only

| # | Item | Section | Complexity | Expected GT-SSIM Δ |
|---|---|---|---|---|
| 9 | ToonCrafter midpoint synthesis for unresolvable pose gaps | §12.2 | Medium-High | +0.05–0.15 on fallback subset |
| 10 | StabStitch++ bidirectional midplane + 2D trajectory smoothing | §11.1 | Medium | +0.03–0.10 on diagonal/horizontal pans |

**Conditional routing:**
- If Phase 1 failures concentrate in "alignment crashes": try EDM as drop-in LoFTR replacement first
- If Phase 2 residual failures concentrate in "character moves between strips": ToonCrafter (Phase 3) becomes critical
- If Phase 2 failures concentrate in "diagonal pan failed": StabStitch++ + 2D BA becomes critical

### 20.4 Phase 4 — Multi-Modal HITL Extension

| # | Item | Section | Complexity | Notes |
|---|---|---|---|---|
| A1 | Grounded SAM-2 (text → DINO → SAM-2) | §8.3 | Medium | `groundingdino` pip-installable, DINO weights ~700MB |
| A2 | Click-based segmentation refinement | §8.4 | Medium | Requires A1 |
| A3 | NL seam routing via DINO exclusion | §16.3 | Low | Requires A1 |
| B1 | COCOAnnotationBuilder | §17.2 | Medium | No prerequisite |
| B2 | Label Studio JSON export | §17.3 | Low | Requires B1 |

### 20.5 Phase 5 — Progressive Automation (Data-Gated)

| # | Item | Data Prerequisite | Notes |
|---|---|---|---|
| C1 | SAM-2 anime fine-tuning | 100+ COCO sessions | Frozen ViT-H; fine-tune mask decoder + memory |
| C2 | Pose contrastive fine-tuning | 500+ selection pairs | DWPose/ViTPose InfoNCE objective |
| C3 | PPO compositing optimization | Calibrated reward model | Requires Issue 6A (RLHF reward model) |

### 20.6 Phase 6 — Generative Extension

| # | Item | Section | Notes |
|---|---|---|---|
| G1 | See-Through character layer decomposition | §15.3 | Awaiting pip-installable release of `shitagaki-lab/see-through` |
| G2 | ControlNet/IP-Adapter generative seam blending | §12 | ComfyUI; §4.8A in new_features.md |
| G3 | PanoDiffusion 360° boundary-seamless generation | §15.3 | Available; needs anime fine-tuning |

### 20.7 Dependency Map

| Feature | Python Package | Notes |
|---|---|---|
| Video ingestion (9A/9B) | `av` (PyAV) | `pip install av`; wraps libavcodec |
| Grounded SAM-2 (A1) | `groundingdino` | `pip install groundingdino`; DINO weights ~700MB |
| SAM-2 (A1/A2) | `sam2` | ✅ already wired S79-S80 |
| COCO serialization (B1) | `pycocotools` | `pip install pycocotools` |
| ProPainter (§12.1) | `propainter` | `pip install propainter`; hook ready in `bg_complete.py` |
| AnimeSR (Stage 1) | — | No official pip package; use `basicsr` base |
| ToonCrafter (§12.2) | — | Available via `diffusion-pipe` |
| PPO (C3) | `stable-baselines3` | `pip install stable-baselines3` |
| Optuna (Bayesian search) | `optuna` | `pip install optuna` |

---

## 21. Decision Thresholds and Anti-Patterns

### 21.1 Decision Thresholds

- **Phase 1 gate:** ASP-beats-naive ≥ 35% AND fallback rate ≤ 25% → proceed to Phase 2
- **EDM substitution gate:** If Phase 1 failures concentrate in "alignment crashes" → try EDM first before deeper changes
- **ToonCrafter gate:** If Phase 2 residual failures concentrate in "character moves between strips" → ToonCrafter (Phase 3) is critical
- **StabStitch++ gate:** If Phase 2 residual failures concentrate in "diagonal pan failed" → StabStitch++ + 2D BA (Phase 3) is critical
- **SAM-2 adoption gate:** 20-clip validation against BiRefNet must show ≥ neutral result before replacing BiRefNet as default

### 21.2 Do Not Adopt

| Algorithm | Reason |
|---|---|
| **UDIS++/SRStitcher wholesale** | TPS warps and diffusion priors tuned to natural-photo parallax, not anime cel-shaded pose mismatch |
| **VidPanos for background completion** | Generative overkill; hallucination risk on flat-color fills. Use ProPainter (deterministic) |
| **JamMa over EDM** | EDM has bidirectional sub-pixel head that translation-only BA actually benefits from |
| **Alpha warp > 0.5 on RAFT flow** | Amplifies flow noise directly proportional to alpha. Catastrophic: test27 SSIM 0.709 → 0.558 |
| **Foreground bbox crop without scroll-axis awareness** | Crops 44% of image width on vertical pans; confirmed failure on test27 |
| **mpdecimate for anime telecine** | Variable animation timing makes fixed-cycle decimation inappropriate |
| **NF4 quantization with LoRAs** | NF4 disrupts the attention blocks that LoRA adapts; use GGUF instead |

### 21.3 Known Caveats

**ToonCrafter is non-deterministic.** Authors warn: "*due to the variety of generative video prior, the success rate is not guaranteed.*" LPIPS/CLIP quality-gating is mandatory on every ToonCrafter output.

**SAM-2 on anime is not officially benchmarked.** Community reports are positive for clean-outline characters, mixed for translucent effects. Treat as beta until validated on the 96-test ASP corpus.

**GNC convergence depends on μ scheduling.** Too-fast annealing → local minima. Yang et al. 2020 schedule (divide by 1.4 per outer iteration) is a safe default.

**LinkTo-Anime is 3D-rendered, not hand-drawn.** Cross-validate on ATD-12K to catch the domain gap. A model trained exclusively on LinkTo-Anime may fail on the irregular line weights and hand-drawn texture variation of real anime production.

**Overmix is GPL-3.0.** Algorithmic ideas may be freely re-implemented; the source code may not be linked into a non-GPL ASP. The animated aligner concept (§7.2) is a clean-room re-implementation.

**LSD collinearity term is NOT in original Sýkora 2009.** It is a novel extension that must be tuned on a held-out validation set. The λ_lsd ≈ 0.1 value is empirical — validate before production use.

**"SGM raises PSNR by 0.6–1.0 dB" is unverified.** The CVPR 2021 paper reports 0.34 dB overall. Dedicated SGM-only ablation on the Hard split was not located. Treat per-module attribution as needing reproduction on the ASP corpus.

---

## 22. Reference Datasets

| Dataset | Purpose | Scale | Key Properties |
|---|---|---|---|
| **ATD-12K** (AnimeInterp, CVPR 2021) | Anime interpolation ground truth | 12,000 triplets | Easy/Medium/Hard splits by motion magnitude; GT for SGM/RFR evaluation |
| **AnimeRun** (NeurIPS 2022) | 2D-styled cartoon optical correspondence | 30 clips | Full flow + region-level segment matching labels; boundary-aware evaluation |
| **LinkTo-Anime** (arXiv:2506.02733, 2025) | Cel-shaded anime character optical flow | 395 sequences; 24,230 train frames | Mixamo skeletons + toon shading; pixel-perfect GT flow + occlusion masks + skeleton annotations |
| **STD-12K** | Sketch-aware inbetweening | 30 sketch animation series | Pre-colorization interpolation; diverse artistic styles |
| **PaintBucket-Character** (CVPR 2024) | Colorization + segment ground truth | 22 Mixamo models | Anti-aliasing disabled; semantic labels per color; shading annotations |
| **Sakuga-42M** | Foundation model training | ~42M frames | Extreme non-linear motion (sakuga); professional animation quality |
| **StabStitch-D** | Video stitching trajectory evaluation | — | Used by StabStitch++ at 28.3 FPS (RTX 4090) |
| **ASP 96-test corpus** | Production benchmark | 96 tests, 55 with GT | Only ground truth corpus for anime panoramic stitching; proprietary |

---

## Appendix A — Algorithmic Synergy Map

| Upstream Algorithm | Output | Downstream Receiver | Synergy |
|---|---|---|---|
| SAM-2 Video Predictor | Temporally consistent foreground masks | Phase correlation + BGM frame selector | Background-subtracted correlation eliminates character-animation contamination |
| Pose-consistent frame selector | Frame subset sharing the same drawn cel | ARAP foreground registration | Near-identical poses → ARAP residuals drop from 22–50 lum to <8 lum → feather widening triggers → clean seam |
| GNC-TLS BA | Outlier-free translation affines | Validation Gate + PANORAMA fallback | Fewer outlier-poisoned affines → fewer BA failures → fallback rate drops |
| AnimeInterp SGM flow | Piece-wise constant flow on foreground | SEA-RAFT | SGM warm-starts RAFT with noise-free coarse field → avoids flat-region collapse |
| SAM-2 mask | Per-frame character mask | SemanticStitch hard barrier | DP seam cost 10⁶× inside mask guarantees routing around character |
| DLNR-SIQA error segments | Pixel coordinates of stitch errors | ToonCrafter or ProPainter | Error maps serve as targeted inpainting masks for localized re-synthesis |
| ToonCrafter midpoint synthesis | Hallucinated in-between frame | Seam compositing | Seam transitions to synthesized content rather than blending mismatched poses |

## Appendix B — Implementation Status (Session 119, 2026-06-15)

| Session | Item | Status |
|---|---|---|
| S6 | Hold detection (`_detect_hold_blocks`) | ✅ |
| S6 | GNC Cauchy loss in BA | ✅ |
| S6 | SLIC SGM proxy (`ASP_SGM_PROXY=1`) | ✅ |
| S8 | DINOv2 frame selection (`ASP_POSE_WINDOW_PX=80`) | ✅ |
| S8 | LSD collinearity in ARAP | ✅ |
| S9 | ToonCrafter seam synthesis (`ASP_TOONCRAFTER_SEAM=1`, default OFF) | ✅ |
| S21 | Poisson seam blend (`ASP_POISSON_SEAM=1`, default OFF) | ✅ |
| S29 | RLHF post-run quality gate | ✅ |
| S30 | Adaptive GNC f_scale in BA | ✅ |
| S31 | PANORAMA stitcher fallback | ✅ |
| S32 | Pre-bundle static edge rejection | ✅ |
| S63 | CanvasInspectorDialog | ✅ |
| S79 | SelectionReviewDialog, EdgeReviewDialog | ✅ |
| S79 | QWaitCondition/QMutex staged execution (4 checkpoints) | ✅ |
| S79–80 | SAM-2 video predictor wired (not yet default) | ✅ |
| S79 | Coverage heatmap | ✅ |
| S95–96 | SeamDiagnosticDialog | ✅ |
| S119 | `_apply_hires_keyframes` + Stage 8.8 (hybrid 4K/1080p) | ✅ |
| S119 | `bench_import.py` + StitchFeedbackTab import group | ✅ |

**Next priority items (from roadmap, not yet implemented):**
- Pose-consistent frame selection redesign (Phase 1, highest-leverage item)
- SAM-2 as default replacement for BiRefNet (Phase 1)
- AnimeInterp SGM integration (Phase 2)
- Horizontal/diagonal scroll support (Phase 3)
- Grounded SAM-2 HITL (Phase 4 A1)
- COCOAnnotationBuilder (Phase 4 B1)

---

*End of report. Total source material consolidated: 6 research documents, ~2100 lines of primary sources, covering sessions 1 through 119 of ASP development.*
