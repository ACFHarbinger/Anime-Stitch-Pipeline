"""
Tests for the corpus-wide dashboard sections in bench_anime_stitch.py added
for issue #69 (roadmap Phase 11, §11.6/§11.9/§11.10):

  _log_resource(tag, store=...)        — §11.6 per-stage RSS accumulation
  _report_stage_memory_waterfall       — §11.6 waterfall report section
  detect_regressions                   — §11.9 cross-run regression detection
  _report_regression_dashboard         — §11.9 report section
  _report_experiment_comparison        — §11.10 experiment tracker report section

These exercise the report-section functions directly against synthetic
result dicts rather than running the full ASP pipeline, mirroring how
test_bench_metrics.py isolates the pure-function metric layer from the
dataset loop.
"""

from __future__ import annotations

import os
import sys

import pytest

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

try:
    from asp_backend_benchmark.bench_anime_stitch import (  # noqa: E402
        STAGE_MEMORY_ORDER,
        _log_resource,
        _report_experiment_comparison,
        _report_regression_dashboard,
        _report_stage_memory_waterfall,
        detect_regressions,
    )
except ImportError:
    from backend.benchmark.bench_anime_stitch import (  # noqa: E402
        STAGE_MEMORY_ORDER,
        _log_resource,
        _report_experiment_comparison,
        _report_regression_dashboard,
        _report_stage_memory_waterfall,
        detect_regressions,
    )

# ---------------------------------------------------------------------------
# §11.6 — stage-level memory profiling
# ---------------------------------------------------------------------------

class TestLogResourceStore:
    def test_store_none_is_a_noop(self):
        """Default call (no store) must not raise — the existing 9 call
        sites inside process_dataset don't all pass one in every branch."""
        snap = _log_resource("dataset_start")
        assert "rss_gb" in snap

    def test_store_receives_tag_in_mb(self):
        store: dict = {}
        snap = _log_resource("before_birefnet", store=store)
        assert "before_birefnet" in store
        assert store["before_birefnet"] == pytest.approx(snap["rss_gb"] * 1024, abs=20.0)

    def test_multiple_tags_accumulate_in_one_store(self):
        store: dict = {}
        _log_resource("dataset_start", store=store)
        _log_resource("dataset_end", store=store)
        assert set(store.keys()) == {"dataset_start", "dataset_end"}


class TestStageMemoryWaterfallReport:
    def _result(self, name: str, stage_memory: dict) -> dict:
        return {"name": name, "stage_memory_rss_mb": stage_memory}

    def test_no_data_renders_explanation_not_crash(self):
        lines: list = []
        _report_stage_memory_waterfall([self._result("t1", {})], "/tmp", lines)
        text = "".join(lines)
        assert "No stage_memory_rss_mb data" in text

    def test_averages_across_datasets_and_orders_by_stage_sequence(self):
        lines: list = []
        results = [
            self._result("t1", {"dataset_start": 1000.0, "dataset_end": 1200.0}),
            self._result("t2", {"dataset_start": 1100.0, "dataset_end": 1300.0}),
        ]
        _report_stage_memory_waterfall(results, "/tmp", lines)
        text = "".join(lines)
        # averaged dataset_start = 1050.0, dataset_end = 1250.0
        assert "1050.0" in text
        assert "1250.0" in text
        # dataset_start must appear before dataset_end (STAGE_MEMORY_ORDER)
        assert text.index("dataset_start") < text.index("dataset_end")

    def test_flags_largest_growth_stage(self):
        lines: list = []
        results = [
            self._result(
                "t1",
                {
                    "dataset_start": 1000.0,
                    "before_birefnet": 1010.0,
                    "after_birefnet_offload": 1500.0,  # big jump
                    "dataset_end": 1520.0,
                },
            )
        ]
        _report_stage_memory_waterfall(results, "/tmp", lines)
        text = "".join(lines)
        assert "after_birefnet_offload" in text
        assert "Largest single-stage growth" in text

    def test_stage_memory_order_matches_pipeline_call_sequence(self):
        assert STAGE_MEMORY_ORDER == (
            "dataset_start",
            "before_birefnet",
            "after_birefnet_offload",
            "before_loftr",
            "after_loftr_offload",
            "before_render_median",
            "after_render_median",
            "after_composite",
            "dataset_end",
        )


# ---------------------------------------------------------------------------
# §11.9 — cross-run regression dashboard
# ---------------------------------------------------------------------------

class TestDetectRegressions:
    def _result(self, name: str, quality: float, ghosting: float, total_sec: float) -> dict:
        return {
            "name": name,
            "metrics_asp": {"composite_quality": quality, "ghosting_siqe": ghosting},
            "time": {"total_sec": total_sec},
        }

    def test_no_baseline_match_is_skipped(self):
        current = [self._result("only_in_current", 0.8, 20.0, 10.0)]
        baseline = [self._result("different", 0.8, 20.0, 10.0)]
        assert detect_regressions(current, baseline) == []

    def test_quality_drop_over_5pct_flagged(self):
        baseline = [self._result("t1", 0.80, 20.0, 10.0)]
        current = [self._result("t1", 0.70, 20.0, 10.0)]  # -12.5%
        regressions = detect_regressions(current, baseline)
        assert len(regressions) == 1
        assert regressions[0]["name"] == "t1"
        assert "composite_quality" in regressions[0]["reasons"]

    def test_quality_drop_under_5pct_not_flagged(self):
        baseline = [self._result("t1", 0.80, 20.0, 10.0)]
        current = [self._result("t1", 0.78, 20.0, 10.0)]  # -2.5%
        regressions = detect_regressions(current, baseline)
        assert regressions == []

    def test_ghosting_increase_over_10pct_flagged(self):
        baseline = [self._result("t1", 0.80, 20.0, 10.0)]
        current = [self._result("t1", 0.80, 25.0, 10.0)]  # +25%
        regressions = detect_regressions(current, baseline)
        assert len(regressions) == 1
        assert "ghosting_siqe" in regressions[0]["reasons"]

    def test_total_time_increase_over_20pct_flagged(self):
        baseline = [self._result("t1", 0.80, 20.0, 10.0)]
        current = [self._result("t1", 0.80, 20.0, 13.0)]  # +30%
        regressions = detect_regressions(current, baseline)
        assert len(regressions) == 1
        assert "total_sec" in regressions[0]["reasons"]

    def test_multiple_dimensions_regress_at_once(self):
        baseline = [self._result("t1", 0.80, 20.0, 10.0)]
        current = [self._result("t1", 0.60, 30.0, 15.0)]
        regressions = detect_regressions(current, baseline)
        assert len(regressions) == 1
        assert set(regressions[0]["reasons"]) == {
            "composite_quality", "ghosting_siqe", "total_sec",
        }

    def test_improvement_is_not_a_regression(self):
        baseline = [self._result("t1", 0.60, 30.0, 15.0)]
        current = [self._result("t1", 0.80, 20.0, 10.0)]
        assert detect_regressions(current, baseline) == []


class TestRegressionDashboardReport:
    def test_no_baseline_renders_note(self):
        lines: list = []
        _report_regression_dashboard([{"name": "t1"}], None, lines)
        assert "no baseline" in "".join(lines).lower()

    def test_regressions_rendered_with_red_indicator(self):
        lines: list = []
        current = [
            {
                "name": "t1",
                "metrics_asp": {"composite_quality": 0.60, "ghosting_siqe": 20.0},
                "time": {"total_sec": 10.0},
            }
        ]
        baseline = [
            {
                "name": "t1",
                "metrics_asp": {"composite_quality": 0.80, "ghosting_siqe": 20.0},
                "time": {"total_sec": 10.0},
            }
        ]
        _report_regression_dashboard(current, baseline, lines)
        text = "".join(lines)
        assert "🔴" in text or "regression" in text.lower()
        assert "t1" in text

    def test_all_clean_renders_green(self):
        lines: list = []
        current = [
            {
                "name": "t1",
                "metrics_asp": {"composite_quality": 0.80, "ghosting_siqe": 20.0},
                "time": {"total_sec": 10.0},
            }
        ]
        _report_regression_dashboard(current, current, lines)
        text = "".join(lines)
        assert "🟢" in text or "no regression" in text.lower()


# ---------------------------------------------------------------------------
# §11.10 — comparative seam-configuration experiment tracker
# ---------------------------------------------------------------------------

class TestExperimentComparisonReport:
    def test_no_label_renders_note(self):
        lines: list = []
        _report_experiment_comparison([{"name": "t1"}], lines)
        text = "".join(lines).lower()
        assert "no experiment label" in text or "not set" in text

    def test_single_run_summary_rendered(self):
        lines: list = []
        results = [
            {
                "name": "t1",
                "experiment_label": "S44-seam-cache",
                "metrics_asp": {"composite_quality": 0.75},
                "time": {"total_sec": 12.0},
            },
            {
                "name": "t2",
                "experiment_label": "S44-seam-cache",
                "metrics_asp": {"composite_quality": 0.85},
                "time": {"total_sec": 14.0},
            },
        ]
        _report_experiment_comparison(results, lines)
        text = "".join(lines)
        assert "S44-seam-cache" in text
        assert "0.80" in text  # mean of 0.75/0.85
