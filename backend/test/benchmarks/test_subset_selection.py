"""Unit tests for M2.5a similarity-based benchmark subset selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SUBSET_PATH = (
    Path(__file__).resolve().parents[3]
    / "backend"
    / "benchmark"
    / "subset_selection.py"
)


def _load_mod():
    import sys
    spec = importlib.util.spec_from_file_location("subset_selection", _SUBSET_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_mod()
extract_features = _MOD.extract_features
select_k_medoids_subset = _MOD.select_k_medoids_subset
select_domain_scoped_subset = _MOD.select_domain_scoped_subset
evaluate_subset_fidelity = _MOD.evaluate_subset_fidelity
generate_all_standard_subsets = _MOD.generate_all_standard_subsets


def test_extract_features_and_select_k_medoids():
    datasets = []
    labels = {}
    for i in range(15):
        name = f"asp_test{i:02d}"
        datasets.append({
            "name": name,
            "used_fallback": i % 3 == 0,
            "metrics_asp": {
                "seam_visibility": 2.0 + i,
                "seam_gradient": 1.0 + (i % 4),
                "sharpness": 10.0 + i * 2,
                "ghosting_siqe": 15.0 + i,
                "coverage": 0.8 + (i * 0.01),
            },
            "metrics_simple": {
                "seam_visibility": 3.0,
                "seam_gradient": 2.0,
                "sharpness": 10.0,
                "ghosting_siqe": 12.0,
                "coverage": 0.9,
            },
        })
        labels[name] = {
            "reviewed": True,
            "asp": (i % 5),
            "simple": 3,
            "defects": ["torn_anatomy"] if i < 5 else ["banding"] if i < 10 else ["ghosting"],
        }

    cases = extract_features(datasets, labels)
    assert len(cases) == 15
    assert cases[0].feature_vector.shape[0] > 10

    # Test K-medoids selection
    subset = select_k_medoids_subset(cases, 5)
    assert len(subset) == 5
    subset_names = {c.name for c in subset}
    assert len(subset_names) == 5

    # Test Domain Scoped selection
    structural = select_domain_scoped_subset(cases, 3, "structural")
    assert len(structural) == 3
    assert any("torn_anatomy" in c.defects for c in structural)

    # Test Fidelity Evaluation
    fid = evaluate_subset_fidelity(cases, subset)
    assert fid["subset_size"] == 5
    assert fid["full_corpus_size"] == 15
    assert 0.0 <= fid["defect_coverage_ratio"] <= 1.0

    # Test Standard subsets generation
    all_sub = generate_all_standard_subsets(cases)
    assert "balanced_smoke_10" in all_sub["subsets"]
    assert "structural_red_set_12" in all_sub["subsets"]
