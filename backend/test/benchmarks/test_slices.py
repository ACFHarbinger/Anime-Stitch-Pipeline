"""Unit tests for versioned benchmark development slices (M0d — issue #48)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from asp_backend_benchmark.slices import (
        CANONICAL_SLICES,
        SMOKE_SET_V1,
        STRUCTURAL_RED_SET_V1,
        DevelopmentSlice,
        export_slices_manifest,
        get_cases_for_failure_mode,
        get_slice,
        get_slice_cases,
        list_slices,
        verify_slice_coverage,
    )
except ImportError:
    from backend.benchmark.slices import (
        CANONICAL_SLICES,
        SMOKE_SET_V1,
        STRUCTURAL_RED_SET_V1,
        DevelopmentSlice,
        export_slices_manifest,
        get_cases_for_failure_mode,
        get_slice,
        get_slice_cases,
        list_slices,
        verify_slice_coverage,
    )


def test_smoke_set_cases_and_cardinality():
    smoke = get_slice("smoke_v1")
    assert len(smoke.case_ids) == 5
    assert set(smoke.case_ids) == {
        "asp_test04",
        "asp_test08",
        "asp_test09",
        "asp_test27",
        "asp_test57",
    }
    assert smoke.version == "1.0.0"


def test_smoke_set_aliases():
    assert get_slice("smoke") == SMOKE_SET_V1
    assert get_slice("SMOKE_V1") == SMOKE_SET_V1


def test_structural_red_set_coverage():
    red = get_slice("structural_red_v1")
    assert len(red.case_ids) == 14
    coverage = verify_slice_coverage("structural_red_v1")

    # M0d requirements: crop loss, torn anatomy, duplicated strips, misordering, banding, known-good, test-14 oracle
    assert "crop_loss" in coverage and len(coverage["crop_loss"]) >= 1
    assert "torn_anatomy" in coverage and len(coverage["torn_anatomy"]) >= 1
    assert "duplicated_strip" in coverage and len(coverage["duplicated_strip"]) >= 1
    assert "misordered_content" in coverage and len(coverage["misordered_content"]) >= 1
    assert "banding" in coverage and len(coverage["banding"]) >= 1
    assert "known_good" in coverage and len(coverage["known_good"]) >= 1
    assert "test14_oracle" in coverage and "asp_test14" in coverage["test14_oracle"]


def test_structural_red_set_aliases():
    assert get_slice("red_set") == STRUCTURAL_RED_SET_V1
    assert get_slice("structural_red") == STRUCTURAL_RED_SET_V1


def test_get_cases_for_failure_mode():
    crop_cases = get_cases_for_failure_mode("structural_red_v1", "crop_loss")
    assert "asp_test07" in crop_cases
    assert "asp_test97" in crop_cases

    torn_cases = get_cases_for_failure_mode("structural_red_v1", "torn_anatomy")
    assert "asp_test04" in torn_cases
    assert "asp_test06" in torn_cases
    assert "asp_test12" in torn_cases
    assert "asp_test15" in torn_cases


def test_unknown_slice_raises_key_error():
    with pytest.raises(KeyError, match="Unknown benchmark slice"):
        get_slice("non_existent_slice_name")


def test_slice_serialization_roundtrip():
    original = STRUCTURAL_RED_SET_V1
    d = original.to_dict()
    restored = DevelopmentSlice.from_dict(d)
    assert restored == original
    assert restored.name == original.name
    assert restored.case_ids == original.case_ids
    assert restored.target_failure_modes == original.target_failure_modes


def test_export_and_load_manifest(tmp_path):
    out_file = tmp_path / "test_slices.json"
    export_slices_manifest(out_file)

    assert out_file.exists()
    with open(out_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == "1.0.0"
    assert "smoke_v1" in data["slices"]
    assert "structural_red_v1" in data["slices"]
    assert len(data["slices"]["smoke_v1"]["case_ids"]) == 5
    assert len(data["slices"]["structural_red_v1"]["case_ids"]) == 14
