"""
Tests for benchmark-metric flattening, N-way asset discovery, and the FiftyOne
triage payload mapping (issue #123).

Two properties matter most here:

- **Metric direction correctness.** Several of the pipeline's metric names read
  backwards from their meaning, and the old dashboard displayed none of them so
  nothing ever checked: ``seam_coherence`` is the std of per-row mean luminance
  (a *banding* proxy, lower better despite "coherence"), ``edge_energy_score``
  is a double-Sobel sharpness proxy and explicitly *not* ghosting, and
  ``ghosting_siqe`` is 0-100 with lower clean. A wrong direction here would tint
  the losing comparator as the winner in the metrics table.

- **The radar's absolute normalization.** Normalizing min-max across the
  comparators present pins the winner at 1.0 and the loser at 0.0 on every axis,
  drawing a dramatic star out of a 0.1% difference. The radar uses
  ``_compute_cqas``'s own references instead, so a radius means what the
  pipeline's aggregate quality score means by it.

The fixture below mirrors the real shape of ``bench_anime_stitch.py``'s
per-dataset result block, with the numbers from ``asp_test01`` of
``anime_stitch_20260728_013215.json``.
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

from asp_backend.evaluation.other import discovery  # noqa: E402
from asp_backend.evaluation.other import metrics_view as mv  # noqa: E402
from asp_backend.evaluation.other.schema import BoundingBox, RatingEntry  # noqa: E402
from asp_backend.evaluation.plugin import sample_fields as sf  # noqa: E402

ENTRY = {
    "name": "asp_test01",
    "time": {"birefnet_sec": 8.769, "render_sec": 13.944, "composite_sec": 20.107,
             "render_gate_fallback": 2, "total_sec": 63.89},
    "frames": {"count": 16, "source_h": 1072, "source_w": 1787},
    "phases": {"count": 3},
    "mean_post_warp_diff": 7.3926,
    "pipeline_config": {"use_birefnet": True, "renderer": "median"},
    "matching": {
        "raw_edges": 41, "filtered_edges": 36, "methods": {"unknown": 41},
        "edges": [
            {"i": 0, "j": 1, "weight": 0.55, "n_pts": 48},
            {"i": 3, "j": 5, "weight": 0.4628, "n_pts": 30},
            {"i": 4, "j": 7, "weight": 0.8788, "n_pts": 60},
        ],
    },
    "alignment": {
        "affines": [{"frame": i, "tx": 0.0, "ty": 967.55 - 62.0 * i} for i in range(6)],
        "dy_steps": [-62.0, -62.0, -300.0, -62.0, -62.0],
        "dx_steps": [0.0, 0.0, -4.1, 0.0, 0.0],
        "dy_cv": 0.1561, "dx_cv": 3.458,
    },
    "affine_health": {"valid": True, "reason": "ok"},
    "photometric": {"ref_lum": 93.67, "bg_lums": [144.59, 120.0, 100.0],
                    "applied_gains": [0.88, 1.0, 1.4], "frames_corrected": 14,
                    "gain_range": [0.88, 1.4]},
    "metrics_asp": {"sharpness": 119.33, "coverage": 0.9999, "seam_gradient": 7.777,
                    "color_entropy": 7.6492, "edge_energy_score": 34.5869,
                    "ghosting_siqe": 58.88, "seam_coherence": 21.64,
                    "seam_visibility": 4.79, "ghost_seam_scores": [12.0, 45.0, 70.0],
                    "ghost_seam_max": 70.0, "width": 1703, "height": 1704, "cqas": 0.5125},
    "metrics_simple": {"sharpness": 79.96, "coverage": 1.0, "seam_gradient": 6.385,
                       "color_entropy": 7.7266, "edge_energy_score": 28.9596,
                       "ghosting_siqe": 66.27, "seam_coherence": 25.16,
                       "seam_visibility": 4.02, "ghost_seam_scores": [],
                       "ghost_seam_max": None, "width": 1917, "height": 2050,
                       "cqas": 0.4711},
    "metrics_overmix": {},
    "metrics_hugin": {},
    "comparison": {"ssim": 0.6999, "psnr_db": 12.42, "verdict": "simple_better",
                   "verdict_source": "ground_truth"},
    "ground_truth": {
        "available": True,
        "metrics_asp": {"ssim_vs_gt": 0.6877, "aligned_ssim_vs_gt": 0.7259,
                        "psnr_vs_gt": 11.86, "seam_coherence": 21.64},
        "metrics_simple": {"ssim_vs_gt": 0.7077, "aligned_ssim_vs_gt": 0.7501,
                           "psnr_vs_gt": 12.02, "seam_coherence": 25.16},
        "verdict": "simple_better",
    },
    "used_fallback": True,
    "fallback_reason": "seam_vis_gate:asp=42.5_sim=4.0_limit=35.0",
    "frame_selection": {"original_count": 98, "smart_select_count": 16,
                        "spatial_dedup_count": 16, "final_count": 16,
                        "selection_mode": "phase_correlation"},
}


# ---------------------------------------------------------------------------
# Metric direction — the thing a wrong answer would silently mis-tint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric,direction", [
    ("cqas", mv.HIGHER_BETTER),
    ("sharpness", mv.HIGHER_BETTER),
    ("coverage", mv.HIGHER_BETTER),
    ("edge_energy_score", mv.HIGHER_BETTER),
    ("ghosting_siqe", mv.LOWER_BETTER),
    ("seam_visibility", mv.LOWER_BETTER),
    ("seam_coherence", mv.LOWER_BETTER),
    ("seam_gradient", mv.LOWER_BETTER),
    ("ghost_seam_max", mv.LOWER_BETTER),
    ("color_entropy", mv.NEUTRAL),
])
def test_metric_directions_match_the_pipeline_definitions(metric, direction):
    row = next(r for r in mv.cv_metric_rows(ENTRY) if r.key == metric)
    assert row.direction == direction


def test_winner_follows_direction_not_magnitude():
    rows = {r.key: r for r in mv.cv_metric_rows(ENTRY)}
    assert rows["sharpness"].best_key() == "asp"        # 119.33 > 79.96, higher better
    assert rows["ghosting_siqe"].best_key() == "asp"    # 58.88 < 66.27, lower better
    assert rows["seam_visibility"].best_key() == "simple"  # 4.02 < 4.79
    assert rows["coverage"].best_key() == "simple"      # 1.0 > 0.9999


def test_neutral_metric_has_no_winner():
    row = next(r for r in mv.cv_metric_rows(ENTRY) if r.key == "color_entropy")
    assert row.best_key() is None


def test_single_comparator_has_no_winner():
    row = mv.MetricRow("x", "X", mv.HIGHER_BETTER, "{:.2f}", {"asp": 1.0, "simple": None})
    assert row.best_key() is None


def test_missing_values_format_as_a_dash():
    rows = {r.key: r for r in mv.cv_metric_rows(ENTRY, ["asp", "simple"])}
    assert rows["ghost_seam_max"].formatted("simple") == "—"
    assert rows["ghost_seam_max"].formatted("asp") == "70.00"


def test_present_comparators_excludes_empty_metric_blocks():
    """Overmix images exist for all 97 tests but metrics_overmix is empty until
    the comparator run is merged into the results JSON (roadmap §0.3)."""
    assert mv.present_comparators(ENTRY) == ["asp", "simple"]


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def test_gt_rows_prefer_the_aligned_ssim():
    rows = {r.key: r for r in mv.gt_metric_rows(ENTRY)}
    assert rows["aligned_ssim_vs_gt"].values == {"asp": 0.7259, "simple": 0.7501}
    assert rows["aligned_ssim_vs_gt"].best_key() == "simple"


def test_gt_rows_are_empty_without_ground_truth():
    entry = dict(ENTRY, ground_truth={"available": False})
    assert mv.gt_metric_rows(entry) == []


# ---------------------------------------------------------------------------
# Radar normalization
# ---------------------------------------------------------------------------


def test_radar_uses_absolute_cqas_scales_not_min_max():
    rows = {r.key: r for r in mv.radar_rows(ENTRY)}
    # sharpness/100, clamped: 119.33 -> 1.0, 79.96 -> 0.7996. Under min-max
    # these two would have been exactly 1.0 and 0.0.
    assert rows["sharpness"].values["asp"] == pytest.approx(1.0)
    assert rows["sharpness"].values["simple"] == pytest.approx(0.7996, abs=1e-4)
    # 1 - ghosting/60, clamped at 0: both are near-terrible and stay near 0.
    assert rows["ghosting_siqe"].values["asp"] == pytest.approx(0.0187, abs=1e-3)
    assert rows["ghosting_siqe"].values["simple"] == pytest.approx(0.0)


@pytest.mark.parametrize("metric,raw,expected", [
    ("cqas", 0.5, 0.5),
    ("coverage", 1.0, 1.0),
    ("sharpness", 50.0, 0.5),
    ("sharpness", 250.0, 1.0),      # clamped
    ("ghosting_siqe", 0.0, 1.0),
    ("ghosting_siqe", 60.0, 0.0),
    ("ghosting_siqe", 120.0, 0.0),  # clamped
    ("seam_visibility", 12.5, 0.5),
    ("seam_coherence", 25.0, 0.5),
])
def test_radar_value_matches_compute_cqas_references(metric, raw, expected):
    assert mv.radar_value(metric, raw) == pytest.approx(expected, abs=1e-6)


def test_radar_value_of_none_is_none():
    assert mv.radar_value("sharpness", None) is None


def test_radar_value_of_an_unscaled_metric_is_none():
    assert mv.radar_value("color_entropy", 7.5) is None


# ---------------------------------------------------------------------------
# Headline facts and chart series
# ---------------------------------------------------------------------------


def test_headline_facts_surface_the_fallback_reason():
    facts = dict(mv.headline_facts(ENTRY))
    assert facts["Verdict"] == "simple_better"
    assert facts["Verdict source"] == "ground_truth"
    assert "fallback" in facts["Composite"]
    assert facts["Fallback reason"].startswith("seam_vis_gate")
    assert facts["Ground truth"] == "available"


def test_headline_facts_of_an_empty_entry_explains_itself():
    facts = mv.headline_facts({})
    assert facts and "no benchmark result" in facts[0][1]


def test_alignment_series_flags_the_outlier_step():
    series = mv.alignment_series(ENTRY)
    assert len(series.frames) == 6
    # -300 against a median magnitude of 62 is well past the 2x rule.
    assert series.outlier_steps == [2]
    assert series.dy_cv == pytest.approx(0.1561)


def test_alignment_series_is_none_without_affines():
    assert mv.alignment_series({"alignment": {}}) is None


def test_photometric_series_flags_gains_beyond_15_percent():
    series = mv.photometric_series(ENTRY)
    # 0.88 is 12% off (not flagged); 1.4 is 40% off (flagged); 1.0 is exact.
    assert series.flagged_frames == [2]
    assert series.frames_corrected == 14


def test_matching_summary_records_frame_gaps():
    summary = mv.matching_summary(ENTRY)
    assert (summary.raw_edges, summary.filtered_edges) == (41, 36)
    assert summary.frame_gaps == [1, 2, 3]
    assert summary.weights == [0.55, 0.4628, 0.8788]


def test_seam_ghost_series_skips_comparators_without_scores():
    series = mv.seam_ghost_series(ENTRY)
    assert list(series) == ["asp"]
    assert series["asp"] == [12.0, 45.0, 70.0]


@pytest.mark.parametrize("score,band", [(0.0, "good"), (29.9, "good"),
                                        (30.0, "warn"), (59.9, "warn"),
                                        (60.0, "bad"), (100.0, "bad")])
def test_ghost_bands_use_the_pipeline_thresholds(score, band):
    assert mv.ghost_band(score) == band


def test_timing_breakdown_is_sorted_and_drops_non_time_counters():
    stages = mv.timing_breakdown(ENTRY)
    assert stages[0][0] == "composite"
    assert [s[1] for s in stages] == sorted((s[1] for s in stages), reverse=True)
    assert all("gate fallback" not in name for name, _ in stages)
    assert all(name != "total" for name, _ in stages)


def test_frame_selection_funnel_is_ordered():
    stages = mv.frame_selection_stages(ENTRY)
    assert [name for name, _ in stages] == [
        "original", "after smart select", "after spatial dedup", "final"
    ]
    assert [count for _, count in stages] == [98, 16, 16, 16]


# ---------------------------------------------------------------------------
# N-way discovery
# ---------------------------------------------------------------------------


def _fake_corpus(
    tmp_path, name="asp_test01", with_gt=True, with_overmix=True, with_hugin=False,
    simple_suffix="opencv_stitch",
):
    import cv2
    import numpy as np

    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    img = np.full((20, 20, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(out / f"{name}_anime_stitch.png"), img)
    cv2.imwrite(str(out / f"{name}_{simple_suffix}.png"), img)
    if with_gt:
        gt = tmp_path / "ground_truth"
        gt.mkdir(exist_ok=True)
        cv2.imwrite(str(gt / f"{name}.png"), img)
    test_out = tmp_path / name / "output"
    test_out.mkdir(parents=True, exist_ok=True)
    if with_overmix:
        cv2.imwrite(str(test_out / "overmix_stitch.png"), img)
    if with_hugin:
        cv2.imwrite(str(test_out / "hugin_stitch.png"), img)
    (test_out / "plots").mkdir(exist_ok=True)
    cv2.imwrite(str(test_out / "plots" / "gains.png"), img)
    stages = test_out / "panorama_stages"
    stages.mkdir(exist_ok=True)
    for i in range(3):
        cv2.imwrite(str(stages / f"stage02_normalised_frame{i:02d}.png"), img)
    cv2.imwrite(str(stages / "stage11_fg_composite.png"), img)
    return str(tmp_path)


def test_discovery_finds_every_comparator_present(tmp_path):
    base = _fake_corpus(tmp_path, with_hugin=True)
    assets = discovery.load_test_assets(base, "asp_test01", _repo_root)
    assert assets.available() == ["asp", "simple", "overmix", "hugin", "ground_truth"]


def test_discovery_omits_absent_comparators(tmp_path):
    base = _fake_corpus(tmp_path, with_gt=False, with_overmix=False)
    assets = discovery.load_test_assets(base, "asp_test01", _repo_root)
    assert assets.available() == ["asp", "simple"]
    assert assets.gt_path is None


def test_legacy_path_properties_still_work(tmp_path):
    base = _fake_corpus(tmp_path)
    assets = discovery.load_test_assets(base, "asp_test01", _repo_root)
    assert assets.asp_path.endswith("_anime_stitch.png")
    assert assets.simple_path.endswith("_opencv_stitch.png")
    assert assets.gt_path.endswith("asp_test01.png")


def test_discovery_falls_back_to_the_pre_rename_filename(tmp_path):
    """2026-07-30: the OpenCV SCANS output was renamed from
    "_simple_stitch.png" to "_opencv_stitch.png" (issue #123 followup) so its
    name doesn't imply it's the only ASP alternative now that Overmix/Hugin
    exist. A corpus generated before the rename must still be discoverable."""
    base = _fake_corpus(tmp_path, simple_suffix="simple_stitch")
    assets = discovery.load_test_assets(base, "asp_test01", _repo_root)
    assert assets.simple_path.endswith("_simple_stitch.png")


def test_stage_renders_group_by_prefix(tmp_path):
    base = _fake_corpus(tmp_path)
    assets = discovery.load_test_assets(base, "asp_test01", _repo_root)
    groups = discovery.stage_groups(assets.stage_dir)
    assert groups["stage02_normalised"] and len(groups["stage02_normalised"]) == 3
    assert "stage11_fg_composite" in groups


def test_load_images_decodes_only_what_exists(tmp_path):
    base = _fake_corpus(tmp_path, with_gt=False)
    assets = discovery.load_test_assets(base, "asp_test01", _repo_root)
    images = discovery.load_images(assets)
    assert set(images) == {"asp", "simple", "overmix"}


def test_metrics_cache_is_keyed_per_results_file(tmp_path):
    """Two runs must not evict each other — the diagnostics tab reads both for
    its cross-run regression check."""
    results_dir = tmp_path / "backend" / "benchmark" / "output"
    results_dir.mkdir(parents=True)
    for stamp, sharpness in (("20260101_000000", 1.0), ("20260102_000000", 2.0)):
        (results_dir / f"anime_stitch_{stamp}.json").write_text(json.dumps({
            "metadata": {"timestamp": "2026-01-01T00:00:00", "total_datasets": 1},
            "datasets": [{"name": "asp_test01", "metrics_asp": {"sharpness": sharpness}}],
        }))
    files = discovery.results_files(str(tmp_path))
    assert len(files) == 2
    old = discovery.load_metrics(str(tmp_path), files[0])["asp_test01"]
    new = discovery.load_metrics(str(tmp_path), files[1])["asp_test01"]
    assert old["metrics_asp"]["sharpness"] == 1.0
    assert new["metrics_asp"]["sharpness"] == 2.0
    # Re-reading the first must still give the first.
    assert discovery.load_metrics(str(tmp_path), files[0])["asp_test01"]["metrics_asp"]["sharpness"] == 1.0


# ---------------------------------------------------------------------------
# FiftyOne payload mapping (no fiftyone import needed)
# ---------------------------------------------------------------------------


def _rated(asp=1, simple=3, **kwargs) -> RatingEntry:
    entry = RatingEntry(**kwargs)
    entry.set_score("asp", "coherence", asp)
    entry.set_score("simple", "coherence", simple)
    entry.reviewed = True
    return entry


def test_fallback_gate_is_split_off_the_reason_string():
    """97 unique reason strings with embedded numbers make a useless sidebar
    facet; the gate name gives ~6 discrete values."""
    fields = sf.run_fields(ENTRY)
    assert fields["fallback_gate"] == "seam_vis_gate"
    assert fields["fallback_reason"].startswith("seam_vis_gate:")


def test_frames_dropped_pct_is_derived():
    assert sf.run_fields(ENTRY)["frames_dropped_pct"] == pytest.approx(83.67, abs=0.01)


@pytest.mark.parametrize("asp,simple,verdict,expected", [
    (1, 3, "simple_better", False),   # human agrees
    (3, 1, "simple_better", True),    # human contradicts
    (2, 2, "comparable", False),      # tie matches comparable
    (2, 2, "asp_better", True),       # tie contradicts asp_better
    (4, 1, "asp_better", False),
])
def test_disagreement_detection(asp, simple, verdict, expected):
    entry = dict(ENTRY, comparison=dict(ENTRY["comparison"], verdict=verdict))
    assert sf.disagreement(entry, _rated(asp, simple)) is expected


def test_disagreement_is_none_when_unrated():
    assert sf.disagreement(ENTRY, None) is None
    assert sf.disagreement(ENTRY, RatingEntry()) is None


def test_tags_carry_the_triage_facets():
    tags = sf.sample_tags(ENTRY, _rated(1, 3, defects=["torn_anatomy"], preference="simple"))
    assert "verdict:simple_better" in tags
    assert "fallback" in tags and "gate:seam_vis_gate" in tags
    assert "has_gt" in tags
    assert "rated" in tags and "human_asp:1" in tags and "prefers:simple" in tags
    assert "defect:torn_anatomy" in tags


def test_defect_tags_are_emitted_even_when_unscored():
    """Regression: keeping these inside the rated branch made a push drop the
    tags of a test that had been tagged in the App but never scored."""
    entry = RatingEntry(defects=["banding"])
    tags = sf.sample_tags(ENTRY, entry)
    assert "defect:banding" in tags
    assert "unrated" in tags


def test_true_composite_is_tagged_too():
    tags = sf.sample_tags(dict(ENTRY, used_fallback=False, fallback_reason=""), None)
    assert "true_composite" in tags and "fallback" not in tags


def test_payloads_cover_every_available_comparator():
    paths = {"asp": "/a.png", "simple": "/s.png", "ground_truth": "/g.png"}
    payloads = dict(sf.build_payloads("asp_test01", ENTRY, paths, _rated()))
    assert set(payloads) == set(paths)
    assert payloads["asp"]["dataset_name"] == "asp_test01"
    assert payloads["asp"]["human_score"] == 1
    assert payloads["simple"]["human_score"] == 3
    # Ground truth is a reference and is never scored.
    assert payloads["ground_truth"]["human_score"] is None


def test_gt_metrics_only_land_on_the_two_scored_comparators():
    assert sf.image_metric_fields(ENTRY, "asp")["aligned_ssim_vs_gt"] == 0.7259
    assert "aligned_ssim_vs_gt" not in sf.image_metric_fields(ENTRY, "overmix")


def test_bbox_detections_use_fiftyone_bbox_convention():
    entry = _rated(bboxes=[
        BoundingBox(image="asp", x=0.1, y=0.2, w=0.3, h=0.4,
                    label="tear", defect="torn_anatomy", severity=3),
        BoundingBox(image="simple", x=0.5, y=0.5, w=0.1, h=0.1, defect="banding"),
    ])
    detections = sf.bbox_detections(entry, "asp")
    assert len(detections) == 1
    assert detections[0]["bounding_box"] == [0.1, 0.2, 0.3, 0.4]
    assert detections[0]["label"] == "torn_anatomy"
    assert detections[0]["severity"] == 3
    assert detections[0]["note"] == "tear"


def test_untagged_region_gets_a_placeholder_label():
    entry = _rated(bboxes=[BoundingBox(image="asp", x=0, y=0, w=0.1, h=0.1)])
    assert sf.bbox_detections(entry, "asp")[0]["label"] == "untagged"


def test_field_schema_declares_every_emitted_field():
    """Guards the FiftyOne failure this schema exists to prevent: a field absent
    from the dataset can't have None written to it ("Cannot infer an appropriate
    field type for value 'None'")."""
    declared = {name for name, _kind in sf.FIELD_SCHEMA}
    paths = {"asp": "/a.png", "ground_truth": "/g.png"}
    for _key, fields in sf.build_payloads("asp_test01", ENTRY, paths, _rated()):
        emitted = set(fields) - {"_tags", "source_path"}
        assert emitted <= declared, f"undeclared fields: {sorted(emitted - declared)}"


def test_field_schema_kinds_are_known():
    assert {kind for _name, kind in sf.FIELD_SCHEMA} <= {"float", "int", "bool", "str", "strlist"}


def test_human_fields_default_cleanly_for_an_unrated_test():
    fields = sf.human_fields(None)
    assert fields["human_rated"] is False
    assert fields["human_asp"] is None
    assert fields["human_defects"] == []
