"""
Tests for the ASP evaluation schema (issue #123).

The load-bearing property is backward compatibility: ``bench_anime_stitch.py``'s
``_load_human_evaluations()`` reads the newest ``asp_evaluations_*.json`` and does
only ``.get("asp")`` / ``.get("simple")`` on each entry to drive its
one-directional coherence veto. Every richer field is additive, so a file written
by this schema has to stay readable by that consumer, and a file written by the
*old* tool has to stay readable by this schema.

Also covers defect 5 from #123: the old tool inferred "rated" from dict
membership, and since it persisted an entry for any test merely *visited*, an
all-null entry permanently excluded that test from the next session's queue.
``is_rated()`` is now a predicate over the scores.
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

from backend.benchmark.evaluation.constants.schema import (  # noqa: E402
    DEFECT_KEYS,
    DIM_COHERENCE,
    DIMENSION_KEYS,
    SCORABLE_KEYS,
)
from backend.benchmark.evaluation.other.schema import (  # noqa: E402
    BoundingBox,
    Edge,
    EdgePoint,
    RatingEntry,
    load_evaluations,
    rated_names,
    save_evaluations,
)

LEGACY_DOC = {
    "asp_test01": {
        "asp": 3,
        "simple": 2,
        "notes": "seam at 40%",
        "bboxes": [{"image": "asp", "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.1, "label": "tear"}],
        "edges": [{
            "a": {"image": "asp", "x": 0.1, "y": 0.2},
            "b": {"image": "ground_truth", "x": 0.11, "y": 0.2},
            "label": "off by 8px",
        }],
    },
    # Exactly what the old tool wrote for a test that was opened but not rated.
    "asp_test02": {"asp": None, "simple": None, "notes": "", "bboxes": [], "edges": []},
}


@pytest.fixture()
def legacy_file(tmp_path):
    path = tmp_path / "asp_evaluations_20260101.json"
    path.write_text(json.dumps(LEGACY_DOC))
    return str(path)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_reads_a_file_written_by_the_old_tool(legacy_file):
    evaluations = load_evaluations(legacy_file)
    entry = evaluations["asp_test01"]
    assert (entry.asp, entry.simple) == (3, 2)
    assert entry.notes == "seam at 40%"
    assert len(entry.bboxes) == 1 and len(entry.edges) == 1
    # Fields the old schema never had must default, not raise.
    assert entry.bboxes[0].defect == "" and entry.bboxes[0].severity is None
    assert entry.dimensions.get("asp", {}).get(DIM_COHERENCE) == 3


def test_bench_contract_keys_survive_a_round_trip(legacy_file, tmp_path):
    evaluations = load_evaluations(legacy_file)
    entry = evaluations["asp_test01"]
    entry.set_score("asp", "sharpness", 4)
    entry.preference = "simple"
    entry.confidence = 3
    entry.defects = ["seam_line"]
    entry.reviewed = True

    out = str(tmp_path / "out.json")
    save_evaluations(out, evaluations)
    with open(out) as f:
        raw = json.loads(f.read())["asp_test01"]

    # This is the whole contract bench_anime_stitch.py depends on.
    assert raw.get("asp") == 3
    assert raw.get("simple") == 2
    for key in ("notes", "bboxes", "edges"):
        assert key in raw
    assert load_evaluations(out)["asp_test01"].to_dict() == raw


def test_a_plain_pass_writes_no_additive_noise(tmp_path):
    """A 0-4-only pass must produce a file as small as the old tool's, so the
    additive fields never become a diff burden on the common path."""
    entry = RatingEntry()
    entry.set_score("asp", DIM_COHERENCE, 4)
    entry.set_score("simple", DIM_COHERENCE, 2)
    out = str(tmp_path / "out.json")
    save_evaluations(out, {"asp_test01": entry})
    with open(out) as f:
        raw = json.loads(f.read())["asp_test01"]

    # The old schema's five keys, plus the mirrored coherence block. No
    # updated_at: only EvaluationSession.commit() stamps that, via touch().
    assert set(raw) == {"asp", "simple", "notes", "bboxes", "edges", "dimensions"}
    assert "preference" not in raw and "defects" not in raw and "confidence" not in raw


# ---------------------------------------------------------------------------
# defect 5 — "rated" must be a property of the scores, not of file membership
# ---------------------------------------------------------------------------


def test_null_entry_reads_as_unrated(legacy_file):
    """The poisoned-entry case that was live in data/benchmarks."""
    evaluations = load_evaluations(legacy_file)
    poisoned = evaluations["asp_test02"]
    assert poisoned.is_rated() is False
    assert poisoned.reviewed is False
    assert rated_names(evaluations) == ["asp_test01"]


def test_partially_rated_entry_is_not_rated():
    entry = RatingEntry()
    entry.set_score("asp", DIM_COHERENCE, 4)
    assert entry.is_rated() is False
    entry.set_score("simple", DIM_COHERENCE, 1)
    assert entry.is_rated() is True


def test_legacy_reviewed_is_inferred_from_scores():
    assert RatingEntry.from_dict({"asp": 2, "simple": 2}).reviewed is True
    assert RatingEntry.from_dict({"asp": None, "simple": None}).reviewed is False


# ---------------------------------------------------------------------------
# Score mirroring and validation
# ---------------------------------------------------------------------------


def test_coherence_mirrors_the_top_level_fields():
    entry = RatingEntry()
    entry.set_score("asp", DIM_COHERENCE, 1)
    entry.set_score("simple", DIM_COHERENCE, 4)
    assert (entry.asp, entry.simple) == (1, 4)
    assert entry.score("asp") == 1 and entry.score("simple") == 4
    # Clearing has to clear the mirror too, or the file keeps a stale score.
    entry.set_score("asp", DIM_COHERENCE, None)
    assert entry.asp is None and entry.score("asp") is None


def test_non_primary_comparators_score_without_a_top_level_field():
    entry = RatingEntry()
    entry.set_score("overmix", DIM_COHERENCE, 3)
    assert entry.score("overmix") == 3
    assert entry.asp is None and entry.simple is None


@pytest.mark.parametrize("bad", [99, -1, "x", 4.7, None, {}])
def test_out_of_range_scores_clamp_to_none_instead_of_crashing(bad):
    entry = RatingEntry.from_dict({"asp": bad, "simple": bad})
    assert entry.asp is None or entry.asp in range(0, 5)
    assert entry.simple is None or entry.simple in range(0, 5)


def test_unknown_images_and_dimensions_are_dropped_on_load():
    entry = RatingEntry.from_dict({
        "dimensions": {"not_an_image": {"coherence": 3}, "asp": {"nope": 2, "sharpness": 1}},
    })
    assert "not_an_image" not in entry.dimensions
    assert entry.dimensions["asp"] == {"sharpness": 1}


def test_every_scorable_image_and_dimension_is_addressable():
    entry = RatingEntry()
    for image in SCORABLE_KEYS:
        for dimension in DIMENSION_KEYS:
            entry.set_score(image, dimension, 2)
            assert entry.score(image, dimension) == 2


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


def test_bbox_defect_and_severity_round_trip(tmp_path):
    entry = RatingEntry(bboxes=[BoundingBox(
        image="asp", x=0.1, y=0.2, w=0.3, h=0.4,
        label="tear", defect="torn_anatomy", severity=3,
    )])
    out = str(tmp_path / "out.json")
    save_evaluations(out, {"t": entry})
    box = load_evaluations(out)["t"].bboxes[0]
    assert (box.defect, box.severity, box.label) == ("torn_anatomy", 3, "tear")
    assert box.defect in DEFECT_KEYS


def test_edge_round_trip(tmp_path):
    entry = RatingEntry(edges=[Edge(
        points=[
            EdgePoint(image="asp", x=0.2, y=0.3),
            EdgePoint(image="simple", x=0.25, y=0.3),
        ],
        label="shifted right",
    )])
    out = str(tmp_path / "out.json")
    save_evaluations(out, {"t": entry})
    edge = load_evaluations(out)["t"].edges[0]
    assert [p.image for p in edge.points] == ["asp", "simple"]
    assert edge.label == "shifted right"


def test_edge_chain_of_three_or_more_endpoints_round_trips(tmp_path):
    """The concrete case the followup feedback asked for: a link spanning
    more than two images (e.g. ASP, Overmix, and ground truth)."""
    entry = RatingEntry(edges=[Edge(
        points=[
            EdgePoint(image="asp", x=0.2, y=0.3),
            EdgePoint(image="overmix", x=0.22, y=0.31),
            EdgePoint(image="ground_truth", x=0.19, y=0.29),
        ],
        label="same seam, three ways",
    )])
    out = str(tmp_path / "out.json")
    save_evaluations(out, {"t": entry})
    edge = load_evaluations(out)["t"].edges[0]
    assert [p.image for p in edge.points] == ["asp", "overmix", "ground_truth"]


def test_edge_region_endpoint_round_trips(tmp_path):
    entry = RatingEntry(edges=[Edge(
        points=[
            EdgePoint(image="asp", x=0.1, y=0.1, w=0.2, h=0.15),
            EdgePoint(image="simple", x=0.12, y=0.1),
        ],
        label="torn region vs clean point",
    )])
    out = str(tmp_path / "out.json")
    save_evaluations(out, {"t": entry})
    edge = load_evaluations(out)["t"].edges[0]
    assert edge.points[0].is_region is True
    assert edge.points[1].is_region is False


def test_legacy_two_point_edge_shape_still_loads(tmp_path):
    """Pre-2026-07-30 files store {"a": ..., "b": ...} instead of "points"."""
    path = str(tmp_path / "legacy_edge.json")
    with open(path, "w") as fh:
        json.dump({
            "t": {
                "asp": None, "simple": None, "notes": "", "bboxes": [],
                "edges": [{
                    "a": {"image": "asp", "x": 0.2, "y": 0.3},
                    "b": {"image": "simple", "x": 0.25, "y": 0.3},
                    "label": "shifted right",
                }],
            },
        }, fh)
    edge = load_evaluations(path)["t"].edges[0]
    assert [p.image for p in edge.points] == ["asp", "simple"]
    assert edge.label == "shifted right"
    assert edge.points[0].is_region is False


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    out = str(tmp_path / "out.json")
    save_evaluations(out, {"t": RatingEntry(asp=1, simple=1)})
    assert os.path.exists(out)
    assert not os.path.exists(f"{out}.tmp")


def test_missing_file_loads_as_empty(tmp_path):
    assert load_evaluations(str(tmp_path / "nope.json")) == {}
