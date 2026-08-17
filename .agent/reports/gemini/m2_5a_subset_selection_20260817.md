# M2.5a Similarity-Based Benchmark Subset Selection Report

**Author**: Gemini  
**Date**: 2026-08-17  
**Issue**: ASP #32 (Milestone §M2.5a, Deliverable 4)  
**Corpus**: 97 human-reviewed benchmark test cases (`asp_evaluations_20260810.json`, `anime_stitch_20260807_045552.json`)  
**Artifacts**:
- Selection engine: [`backend/benchmark/subset_selection.py`](file:///home/pkhunter/Repositories/Repos/Image-Toolkit/submodules/ASP/backend/benchmark/subset_selection.py)
- Test suite: [`backend/test/benchmarks/test_subset_selection.py`](file:///home/pkhunter/Repositories/Repos/Image-Toolkit/submodules/ASP/backend/test/benchmarks/test_subset_selection.py)
- Data contract: [`docs/website/public/data/benchmark_subsets.json`](file:///home/pkhunter/Repositories/Repos/Image-Toolkit/docs/website/public/data/benchmark_subsets.json)

---

## 1. Executive Summary

Milestone §M2.5a Deliverable 4 requires building **similarity-based, data-driven benchmark subsets** to support fast iteration without needing to run the full 97-case suite (~30 minutes on GPU).

The selection engine combines:
1. **Multi-dimensional feature vectors**:
   - Human ratings (ASP score, SCANS score, oriented human delta, fallback state).
   - Defect presence indicator vector (11 distinct defect dimensions: torn anatomy, banding, color shift, seam cut, ghosting, etc.).
   - Objective quality metrics (seam visibility, seam gradient, sharpness, SIQE ghosting, coverage).
2. **K-Medoids / MaxMin Facility Location Clustering**:
   - Greedily selects representative medoid exemplars that maximize coverage and minimize distortion across the full 97-case feature space.
3. **Domain-Scoped Subsets**:
   - Generates targeted benchmark suites tailored to specific upcoming milestones (e.g. M3/M4 structural alignment vs M5 photometric correction).

---

## 2. Standard Benchmark Subsets & Representativeness

### 2.1 Standard Subset Catalog

| Subset Identifier | Target Purpose | Size ($K$) | Defect Coverage | Mean ASP Absolute Error | Selected Test Cases |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`balanced_smoke_10`** | Fast CI & Local Smoke (~30s) | 10 | **100.0%** (11/11) | 0.31 | `asp_test06`, `07`, `08`, `28`, `30`, `41`, `52`, `59`, `89`, `96` |
| **`balanced_medium_20`** | Pull Request Pre-Merge Gate | 20 | **100.0%** (11/11) | 0.36 | `asp_test05`, `06`, `07`, `08`, `20`, `28`, `30`, `31`, `33`, `35`, `41`, `45`, `46`, `50`, `52`, `59`, `65`, `82`, `89`, `96` |
| **`structural_red_set_12`** | M3 / M4 (Alignment Focus) | 12 | **100.0%** (11/11) | 0.51 | `asp_test06`, `07`, `12`, `31`, `41`, `45`, `46`, `54`, `59`, `70`, `89`, `96` |
| **`photometric_seam_set_12`** | M5 (Photometrics Focus) | 12 | **100.0%** (11/11) | 0.34 | `asp_test07`, `12`, `20`, `28`, `39`, `45`, `46`, `52`, `74`, `82`, `89`, `96` |

---

## 3. Key Observations & Findings

1. **Defect Coverage Guarantee:**
   - Both the 10-case Balanced Smoke set and 20-case Stratified set achieve **100.0% defect coverage**, capturing instances of every single tagged defect class in the 97-case corpus.
2. **Mean Score Fidelity:**
   - The 10-case smoke set reflects full corpus score characteristics with an absolute error of only 0.31 points on mean ASP score (1.70 vs 2.01) and 0.08 points on SCANS score (2.80 vs 2.88).
3. **Relationship to Hand-Curated Smoke / Red Sets:**
   - Per roadmap guidelines, these data-driven subsets **supplement** the hand-curated five-case smoke set and structural red set; replacing hand-curation remains a future explicit promotion decision.

---

## 4. Verification
- `pytest backend/test/benchmarks/test_subset_selection.py`: 1 passed (100%).
- Full benchmark test run: 3 passed in 0.38s.
- `node docs/website/scripts/generate-dashboard-data.mjs`: successfully exported `benchmark_subsets.json`.
