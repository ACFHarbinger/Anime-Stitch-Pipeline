"""
Tests for the M0 raw_asp/safe_asp/scans relabeling (issue #24).

Uses small synthetic fixtures matching the real schema shapes, not the real
97-case files -- keeps the test isolated and fast, and independent of the
real data files ever moving/changing. The real-data cross-check (43 true
composites / mean 1.326, 54 fallbacks / mean 2.556, matching the numbers
already cited in asp_change_roadmap_2026q3.md §3) was verified manually
against backend/benchmark/output/anime_stitch_20260807_045552.json and
data/benchmarks/asp_evaluations_20260810.json when this module was written;
not re-asserted here to keep this test from silently depending on those
files never moving.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

from asp_backend_evaluation.other.relabel import (  # noqa: E402
    RelabeledCase,
    relabel_corpus,
    save_relabeled,
    summarize,
)


def _write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _bench_dataset(name: str, fallback_code: int) -> dict:
    return {"name": name, "time": {"render_gate_fallback": fallback_code}}


@pytest.fixture
def fixture_paths(tmp_path):
    bench = {
        "datasets": [
            _bench_dataset("asp_test01", 0),  # true composite
            _bench_dataset("asp_test02", 1),  # composite/ghost fallback
            _bench_dataset("asp_test03", 2),  # seam_vis fallback
        ]
    }
    evaluations = {
        "asp_test01": {"asp": 3, "simple": 2, "reviewed": True},
        "asp_test02": {"asp": 4, "simple": 4, "reviewed": True},  # rated 4, but it's SCANS
        "asp_test03": {"asp": 1, "simple": 3, "reviewed": True},
    }
    bench_path = str(tmp_path / "bench.json")
    evals_path = str(tmp_path / "evals.json")
    _write_json(bench_path, bench)
    _write_json(evals_path, evaluations)
    return bench_path, evals_path


def test_true_composite_rated_identity_is_raw_asp(fixture_paths):
    bench_path, evals_path = fixture_paths
    relabeled = relabel_corpus(bench_path, evals_path)
    assert relabeled["asp_test01"].true_raw_asp_composite is True
    assert relabeled["asp_test01"].rated_identity == "raw_asp"
    assert relabeled["asp_test01"].fallback_gate == "none"


def test_fallback_rated_identity_is_scans_not_asp(fixture_paths):
    """The core point of this module: a case where the human's "asp" score
    actually rated a SCANS substitution must be labeled as such, not left
    to silently masquerade as a genuine ASP rating."""
    bench_path, evals_path = fixture_paths
    relabeled = relabel_corpus(bench_path, evals_path)
    assert relabeled["asp_test02"].true_raw_asp_composite is False
    assert relabeled["asp_test02"].rated_identity == "scans"
    assert relabeled["asp_test02"].fallback_gate == "composite_or_ghost"
    assert relabeled["asp_test03"].fallback_gate == "seam_vis"


def test_human_scores_carried_through(fixture_paths):
    bench_path, evals_path = fixture_paths
    relabeled = relabel_corpus(bench_path, evals_path)
    assert relabeled["asp_test01"].human_asp_score == 3
    assert relabeled["asp_test01"].human_simple_score == 2
    assert relabeled["asp_test01"].human_reviewed is True


def test_mismatched_case_sets_raises(tmp_path):
    bench_path = str(tmp_path / "bench.json")
    evals_path = str(tmp_path / "evals.json")
    _write_json(bench_path, {"datasets": [_bench_dataset("asp_test01", 0)]})
    _write_json(evals_path, {"asp_test01": {"asp": 3, "simple": 2}, "asp_test02": {"asp": 1, "simple": 1}})
    with pytest.raises(ValueError, match="don't match"):
        relabel_corpus(bench_path, evals_path)


def test_unrecognized_fallback_code_raises(tmp_path):
    bench_path = str(tmp_path / "bench.json")
    evals_path = str(tmp_path / "evals.json")
    _write_json(bench_path, {"datasets": [_bench_dataset("asp_test01", 99)]})
    _write_json(evals_path, {"asp_test01": {"asp": 3, "simple": 2}})
    with pytest.raises(ValueError, match="unrecognized render_gate_fallback"):
        relabel_corpus(bench_path, evals_path)


def test_summarize_matches_manual_counts(fixture_paths):
    bench_path, evals_path = fixture_paths
    relabeled = relabel_corpus(bench_path, evals_path)
    summary = summarize(relabeled)
    assert summary["total_cases"] == 3
    assert summary["true_raw_asp_composites"]["count"] == 1
    assert summary["true_raw_asp_composites"]["mean_human_asp_score"] == 3.0
    assert summary["safety_fallbacks_to_scans"]["count"] == 2
    assert summary["safety_fallbacks_to_scans"]["mean_human_asp_score"] == pytest.approx(2.5)
    assert summary["safety_fallbacks_to_scans"]["by_gate"] == {
        "composite_or_ghost": 1,
        "seam_vis": 1,
    }


def test_save_relabeled_round_trip(tmp_path, fixture_paths):
    bench_path, evals_path = fixture_paths
    relabeled = relabel_corpus(bench_path, evals_path)
    out_path = str(tmp_path / "relabeled.json")
    save_relabeled(out_path, relabeled)
    with open(out_path) as fh:
        doc = json.load(fh)
    assert set(doc.keys()) == {"asp_test01", "asp_test02", "asp_test03"}
    assert doc["asp_test02"]["rated_identity"] == "scans"


def test_relabeled_case_to_dict_is_json_safe():
    case = RelabeledCase(
        case_id="asp_test01",
        fallback_code=0,
        fallback_gate="none",
        true_raw_asp_composite=True,
        rated_identity="raw_asp",
        human_reviewed=True,
        human_asp_score=3,
        human_simple_score=2,
    )
    json.dumps(case.to_dict())  # must not raise
