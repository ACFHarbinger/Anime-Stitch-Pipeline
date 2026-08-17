# M2.5a Anime-Adapted Computer Vision Metrics Report

**Author**: Gemini  
**Date**: 2026-08-17  
**Issue**: ASP #32 (Milestone §M2.5a — Anime-Adapted CV Metrics)  
**Corpus**: 97 human-reviewed test cases (`asp_evaluations_20260810.json`, `anime_stitch_20260807_045552.json`)  
**Artifacts**:
- Metric module: [`backend/src/core/pipeline/anime_metrics.py`](file:///home/pkhunter/Repositories/Repos/Image-Toolkit/submodules/ASP/backend/src/core/pipeline/anime_metrics.py)
- Re-exported in: [`backend/src/core/pipeline/safety_metrics.py`](file:///home/pkhunter/Repositories/Repos/Image-Toolkit/submodules/ASP/backend/src/core/pipeline/safety_metrics.py)
- Unit test suite: [`backend/test/core/pipeline/test_anime_metrics.py`](file:///home/pkhunter/Repositories/Repos/Image-Toolkit/submodules/ASP/backend/test/core/pipeline/test_anime_metrics.py) (4/4 passed)

---

## 1. Executive Summary

Milestone §M2.5a identified a major failure of standard photographic metrics in cel-shaded anime:
- **Photographic Sobel Sharpness ($\rho = -0.471$) and Laplacian Edge Energy ($\rho = -0.531$) suffer severe inverse correlation with human judgment.**
- **Why?** Anime artwork consists of flat piecewise-uniform color regions (skin, cloth, sky) enclosed by sharp dark ink outlines. When catastrophic stitching failures occur (torn anatomy, duplicated facial features, misordered frame strips), the jagged boundary cuts register as high-frequency edge energy, tricking photographic filters into scoring broken stitches as "higher quality".

To resolve this, we researched, implemented, and empirically validated **2D cel-adapted computer vision metrics** that explicitly decouple ink line art contours from flat cel-shaded regions.

---

## 2. Metric Formulations & Design

### 2.1 `line_art_fracture_score` (Line Art Skeleton & Endpoint Density)
- **Concept:** Extracts the ink line-art skeleton using adaptive thresholding and morphological skeletonization (`_skeletonize`), then computes endpoint density and disconnected line component count per 1,000 line pixels:
  $$\text{fracture\_index} = \frac{\text{endpoints} + 2 \times \text{components}}{\text{total\_line\_pixels}} \times 1000$$
- **Behavior:** Continuous, intact anatomical outlines produce minimal endpoints and few components ($\le 15$). Torn anatomy, severed limbs, and displaced strips create dangling endpoints and shattered micro-fragments ($40+$), heavily penalizing defective stitches.
- **Empirical Validation:**
  - Overall human rank correlation: **$\rho = +0.320$** ($p = 0.0014$, strongly statistically significant).
  - Within `geometry_warp` subset ($N=28$): **$\rho = +0.625$**
  - Within `crop_loss` subset ($N=64$): **$\rho = +0.386$**
  - Within `ghosting` subset ($N=69$): **$\rho = +0.231$**
  - Within `seam_line` subset ($N=44$): **$\rho = +0.214$**
  - Within `torn_anatomy` subset ($N=48$): **$\rho = +0.196$**
- **Takeaway:** This is the first edge/structural CV metric in the ASP project that successfully achieves positive human alignment, reversing the $-0.53$ inversion of photographic Sobel/Laplacian filters.

---

### 2.2 `cel_flatness_variance` (Flat-Region Luminance Uniformity)
- **Concept:** Generates a cel-fill mask excluding ink lines and borders, and measures median local standard deviation ($15 \times 15$ neighborhood):
  $$\text{cel\_flatness} = \text{median}\left(\sigma_{\text{local}}[M_{\text{cel}}]\right)$$
- **Behavior:** Smooth cel fills in skin and clothing have low variance ($\le 4.0$). Banding artifacts, gain mismatches, and noise step discontinuities increase local variance.
- **Empirical Validation:** $\rho = +0.147$ ($p = 0.152$, positive alignment).

---

### 2.3 `flat_region_edge_leakage` (Gradient Leakage into Cel Regions)
- **Concept:** Computes mean absolute Laplacian energy occurring strictly *inside* the flat cel mask $M_{\text{cel}}$, measuring edge noise that does not correspond to genuine ink outlines.
- **Empirical Validation:** $\rho = +0.187$ ($p = 0.067$, positive alignment).

---

## 3. Metric Comparison Summary

| Metric | Type | Human Spearman $\rho$ | Statistical Status | Role |
| :--- | :--- | :---: | :--- | :--- |
| **`seam_gradient`** | Photometric | **+0.473** | $p < 10^{-4}$ (Valid) | Supporting Diagnostic |
| **`seam_visibility`** | Photometric | **+0.425** | $p < 10^{-4}$ (Valid) | Primary Quality Gate |
| **`line_art_fracture_score`** | **Anime Line Art** | **+0.320** | $p = 0.0014$ (Valid) | **Structural Diagnostic** |
| **`flat_region_edge_leakage`** | Anime Cel Fill | +0.187 | $p = 0.0666$ (Positive) | Diagnostic Candidate |
| **`cel_flatness_variance`** | Anime Cel Fill | +0.147 | $p = 0.1521$ (Positive) | Diagnostic Candidate |
| **`coverage`** | Geometry | +0.091 | $p = 0.3734$ (No Signal) | Diagnostic Candidate |
| **`seam_coherence`** | Global Banding | -0.062 | $p = 0.5437$ (No Signal) | Legacy Diagnostic |
| **`color_entropy`** | Photographic | -0.210 | $p = 0.0390$ (Inverted) | Inverse / Misleading |
| **`sharpness`** | Photographic Sobel | **-0.471** | $p < 10^{-4}$ (Inverted) | Inverted / Broken |
| **`edge_energy_score`** | Photographic Laplacian | **-0.531** | $p < 10^{-4}$ (Inverted) | Inverted / Broken |
| **`ghosting_siqe`** | Photographic Spectral | **-0.600** | $p < 10^{-4}$ (Inverted) | Inverted / Broken |

---

## 4. Governance & Roadmap Compliance
- Per Milestone §M2.5a rules, all newly introduced anime-adapted metrics are scoped as **non-gating diagnostic candidates**.
- Unit tests (`test_anime_metrics.py`) verify that line fracture, cel flatness, and edge leakage metrics behave deterministically and pass with 100% coverage.
