"""Unit tests for pipeline trace failure context recording (§5.15 Option D)."""

import datetime

from gui.src.helpers.animation._progress_pipeline import _STAGE_LABELS


def test_stage_labels_indexing():
    assert len(_STAGE_LABELS) == 13
    assert _STAGE_LABELS[0] == "Loading and trimming frames"
    assert _STAGE_LABELS[12] == "Saving output"


def test_trace_failure_recording_schema():
    trace = {
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "output_path": "output.png",
        "frames_input": 10,
        "edges_found": 9,
        "canvas_size": [1080, 1920],
        "fallback_used": True,
        "failures": [],
        "stage_timings": [],
        "success": True,
        "error": None,
    }

    # Simulate recording stage failure
    err = ValueError("Alignment failed due to insufficient inliers")
    trace["failures"].append({
        "stage": 6,
        "label": _STAGE_LABELS[5],
        "exception_type": type(err).__name__,
        "message": str(err),
        "fallback_used": "SCANS",
    })

    assert len(trace["failures"]) == 1
    f = trace["failures"][0]
    assert f["stage"] == 6
    assert f["label"] == "Bundle adjustment"
    assert f["exception_type"] == "ValueError"
    assert "insufficient inliers" in f["message"]
    assert f["fallback_used"] == "SCANS"
