"""Unit tests for M2.5a (#32) per-defect-category and stage-attributed correlation audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_AUDIT_PATH = (
    Path(__file__).resolve().parents[3]
    / "backend"
    / "benchmark"
    / "audit_defect_correlation.py"
)


def _load_audit_mod():
    spec = importlib.util.spec_from_file_location("audit_defect_correlation", _AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_audit_mod()
compute_defect_correlation_matrix = _MOD.compute_defect_correlation_matrix
format_cli_report = _MOD.format_cli_report
classify_diagnosis = _MOD.classify_diagnosis


def test_classify_diagnosis():
    assert classify_diagnosis(0.45, 10) == "tracks_quality"
    assert classify_diagnosis(-0.35, 10) == "inverse_misleading"
    assert classify_diagnosis(0.05, 10) == "no_signal"
    assert classify_diagnosis(None, 10) == "insufficient_data"
    assert classify_diagnosis(0.9, 3) == "insufficient_data"


def test_compute_defect_correlation_matrix_synthetic():
    # Build synthetic datasets with 10 cases, 2 defects
    datasets = []
    labels = {}
    for i in range(10):
        name = f"asp_test{i:02d}"
        is_clean = i < 5
        datasets.append({
            "name": name,
            "used_fallback": False,
            "metrics_asp": {
                "sharpness": 10.0 + (5.0 if not is_clean else 0.0),  # Inverted: broken has higher sharpness
                "seam_visibility": 2.0 if is_clean else 8.0,          # Aligned: broken has higher visibility (lower delta)
                "ghosting_siqe": 15.0 + i,
                "seam_coherence": 5.0,
                "cqas": 0.5,
                "coverage": 0.9,
                "color_entropy": 1.0,
                "seam_gradient": 1.0 if is_clean else 5.0,
                "edge_energy_score": 1.0 + i,
            },
            "metrics_simple": {
                "sharpness": 10.0,
                "seam_visibility": 4.0,
                "ghosting_siqe": 10.0,
                "seam_coherence": 5.0,
                "cqas": 0.5,
                "coverage": 0.9,
                "color_entropy": 1.0,
                "seam_gradient": 2.0,
                "edge_energy_score": 1.0,
            },
        })
        labels[name] = {
            "reviewed": True,
            "asp": 4 if is_clean else 1,
            "simple": 3,
            "defects": [] if is_clean else ["torn_anatomy", "banding"],
        }

    run_data = {"datasets": datasets}
    matrix = compute_defect_correlation_matrix(run_data, labels, min_defect_samples=3)

    assert matrix["schema_version"] == 1
    assert matrix["total_reviewed_cases"] == 10
    assert "torn_anatomy" in matrix["defect_summaries"]
    assert "banding" in matrix["defect_summaries"]
    assert matrix["defect_summaries"]["torn_anatomy"]["count"] == 5

    # Check stage attribution
    assert matrix["defect_summaries"]["torn_anatomy"]["category"] == "structural"
    assert matrix["defect_summaries"]["banding"]["category"] == "photometric"

    # Check format_cli_report produces valid output
    report = format_cli_report(matrix)
    assert "M2.5a (#32)" in report
    assert "Torn Anatomy" in report
    assert "STRUCTURAL" in report
