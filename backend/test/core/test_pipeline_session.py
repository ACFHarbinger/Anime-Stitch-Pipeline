"""M1a: PipelineSession / stage protocol — no GPU, no pixel path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from asp_backend.core.pipeline.session import (
    HitlCheckpoint,
    PipelineSession,
    PipelineStage,
    ResultIdentity,
    json_safe,
    snapshot_pipeline_config,
)
from asp_backend import AnimeStitchPipeline
from asp_gui.helpers._progress_pipeline import _ProgressPipeline
from backend.src.errors import PipelineError
from PIL import Image


def _host(**overrides):
    defaults = {
        "use_basic": True,
        "use_birefnet": False,
        "use_loftr": True,
        "use_efficient_loftr": True,
        "use_aliked": False,
        "use_roma": False,
        "use_sea_raft": False,
        "use_jamma": False,
        "use_ecc": True,
        "renderer": "median",
        "composite_fg": True,
        "bands": 5,
        "edge_crop": 30,
        "motion_model": "translation",
        "kwargs": {"stitch_net_ckpt": "", "pause_hook": lambda e, d: {}},
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestSnapshotAndJsonSafe:
    def test_snapshot_copies_flags_and_strips_pause_hook(self):
        snap = snapshot_pipeline_config(_host())
        assert snap["renderer"] == "median"
        assert snap["motion_model"] == "translation"
        assert snap["kwargs"] == {"stitch_net_ckpt": ""}
        assert "pause_hook" not in snap["kwargs"]

    def test_json_safe_replaces_ndarray_with_metadata(self):
        arr = np.zeros((4, 6, 3), dtype=np.uint8)
        encoded = json_safe({"mask": arr, "n": 2})
        assert encoded["n"] == 2
        assert encoded["mask"]["__array__"] is True
        assert encoded["mask"]["shape"] == [4, 6, 3]
        assert "uint8" in encoded["mask"]["dtype"]


class TestPipelineSession:
    def test_mark_preserves_order_and_digest_is_stable(self):
        session = PipelineSession.create(
            ["a.png", "b.png"],
            "out.png",
            config={"renderer": "median"},
        )
        session.mark(PipelineStage.LOAD, n=2)
        session.mark(PipelineStage.NORMALISE, width=64, height=64)
        session.record_artifact("output_path", "out.png")
        session.finish(success=True, identity=ResultIdentity.RAW_ASP)

        assert session.stage_names() == ["load", "normalise"]
        first = session.digest()
        # timestamps must not affect the digest
        session.stages[0].started_at += 10
        session.stages[0].duration_s = 99.0
        assert session.digest() == first

        other = PipelineSession.create(
            ["a.png", "b.png"],
            "out.png",
            config={"renderer": "median"},
        )
        other.mark(PipelineStage.LOAD, n=2)
        other.mark(PipelineStage.NORMALISE, width=64, height=64)
        other.record_artifact("output_path", "out.png")
        other.finish(success=True, identity=ResultIdentity.RAW_ASP)
        assert other.digest() == first

    def test_fallback_changes_identity_and_digest(self):
        session = PipelineSession.create(["a.png", "b.png"], "out.png", config={})
        session.mark(PipelineStage.MATCH, n_edges=0)
        session.record_fallback(ResultIdentity.SCANS, "no_valid_edges")
        session.finish(success=True, identity=ResultIdentity.SCANS)
        assert session.identity == "scans"
        assert session.fallbacks[0]["reason"] == "no_valid_edges"
        assert session.digest() != PipelineSession.create(
            ["a.png", "b.png"], "out.png", config={}
        ).digest()

    def test_safe_fallback_keeps_algorithm_out_of_result_identity(self):
        session = PipelineSession.create(["a.png", "b.png"], "out.png", config={})
        session.record_fallback(
            ResultIdentity.SAFE_ASP,
            "affine_validation_failed",
            algorithm="panorama",
        )
        session.record_artifact("safe_asp_path", "out.png")
        session.finish(success=True, identity=ResultIdentity.SAFE_ASP)

        assert session.identity == ResultIdentity.SAFE_ASP
        assert session.fallbacks[0]["identity"] == ResultIdentity.SAFE_ASP
        assert session.fallbacks[0]["algorithm"] == "panorama"
        assert "safe_asp_path" in session.artifacts

    def test_mark_dropped_paths_sets_reason(self):
        session = PipelineSession.create(
            ["a.png", "b.png", "c.png"], "out.png", config={}
        )
        session.init_frame_provenance(["a.png", "b.png", "c.png"])
        session.mark_dropped_paths(["a.png", "c.png"], "near_static")
        by_path = {row["path"]: row for row in session.frame_provenance}
        assert by_path["a.png"]["kept"] is True
        assert by_path["b.png"]["kept"] is False
        assert by_path["b.png"]["drop_reason"] == "near_static"
        session.mark_dropped_paths(["a.png"], "spatial_dedup")
        assert by_path["c.png"]["drop_reason"] == "spatial_dedup"

    def test_finish_publishes_m2_observability_envelope(self):
        session = PipelineSession.create(["a.png", "b.png"], "out.png", config={})
        session.note_geometry(PipelineStage.NORMALISE, width=64, height=48, n_frames=2)
        session.note_frame_provenance(
            [{"index": 0, "path": "a.png", "kept": True, "drop_reason": None}]
        )
        session.note_pose_provenance(
            [{"frame": 0, "tx": 0.0, "ty": 10.0, "source": "bundle_adjust"}]
        )
        session.note_gain_telemetry({"n_clamped": 1, "mean_residual": 0.12})
        session.note_seam_feasibility(
            {"attempted": True, "feasible": True, "seam_crops": "drop-me"}
        )
        session.record_fallback(ResultIdentity.SCANS, "coverage_gate")
        session.finish(success=True, identity=ResultIdentity.SCANS)

        assert session.fallback_reason == "coverage_gate"
        obs = session.artifacts["observability"]
        assert obs["fallback_reason"] == "coverage_gate"
        assert obs["geometry"][0]["width"] == 64
        assert obs["frame_provenance"][0]["path"] == "a.png"
        assert obs["pose_provenance"][0]["ty"] == 10.0
        assert obs["gain"]["n_clamped"] == 1
        assert obs["seam"]["feasible"] is True
        assert "seam_crops" not in obs["seam"]
        assert session.as_dict()["fallback_reason"] == "coverage_gate"

    def test_pause_is_noop_without_hook(self):
        session = PipelineSession.create(["a.png"], "out.png", config={})
        assert session.pause(HitlCheckpoint.FRAMES, {"paths": ["a.png"]}) == {}
        assert session.hitl_overrides == {}

    def test_pause_records_overrides_without_applying_them(self):
        seen: list[tuple[str, dict]] = []

        def hook(event: str, data: dict) -> dict:
            seen.append((event, data))
            return {"frame_override": ["b.png"]}

        session = PipelineSession.create(
            ["a.png", "b.png"], "out.png", config={}, pause_hook=hook
        )
        override = session.pause(HitlCheckpoint.FRAMES, {"paths": ["a.png", "b.png"]})
        assert override == {"frame_override": ["b.png"]}
        assert seen[0][0] == "frames"
        assert session.hitl_overrides["frames"]["frame_override"] == ["b.png"]
        assert session.inputs.image_paths == ["a.png", "b.png"]

    def test_stages_emit_host_telemetry_spans_when_enabled(self, tmp_path, monkeypatch):
        pytest.importorskip("backend.src.core.telemetry")
        from backend.src.core import telemetry

        monkeypatch.setattr(telemetry, "TELEMETRY_DIR", tmp_path)
        previous = telemetry.is_enabled()
        telemetry.set_enabled(True)
        try:
            session = PipelineSession.create(["a.png"], "out.png", config={})
            session.mark(PipelineStage.LOAD, n=1)
            session.start_stage(PipelineStage.MATCH)
            session.complete_stage(PipelineStage.MATCH, n_edges=3)
            session.finish(success=True, identity=ResultIdentity.RAW_ASP)
        finally:
            telemetry.close()
            telemetry.set_enabled(previous)

        files = list(tmp_path.glob("telemetry-*.jsonl"))
        assert files, "PipelineSession must write host telemetry when enabled"
        import json

        events = [json.loads(line) for line in files[0].read_text().splitlines() if line]
        names = [e["event"] for e in events]
        assert "stage.load.start" in names
        assert "stage.load.end" in names
        assert "stage.match.start" in names
        assert "stage.match.end" in names
        load_start = next(e for e in events if e["event"] == "stage.load.start")
        load_end = next(e for e in events if e["event"] == "stage.load.end")
        assert load_start["span_id"] == load_end["span_id"]
        assert load_start["category"] == "asp"


class TestHitlEventParity:
    def test_checkpoint_names_match_gui_pause_events(self):
        # Locked to _ProgressPipeline._hitl_pause first-arg literals.
        assert HitlCheckpoint.FRAMES == "frames"
        assert HitlCheckpoint.MASKS == "masks"
        assert HitlCheckpoint.EDGES == "edges"
        assert HitlCheckpoint.CANVAS == "canvas"
        assert HitlCheckpoint.RENDER == "render"
        assert HitlCheckpoint.BOUNDARIES == "boundaries"
        assert HitlCheckpoint.SEAMS == "seams"
        assert HitlCheckpoint.COMPOSITE == "composite"
        assert HitlCheckpoint.OUTPUT == "output"

    def test_gui_progress_pipeline_still_subclasses_without_session_override(self):
        # M1a must not rewrite the GUI fork. Inheritance remains the old
        # override-run() shape until M1c.
        assert issubclass(_ProgressPipeline, AnimeStitchPipeline)
        assert _ProgressPipeline.run is not AnimeStitchPipeline.run


class TestCanonicalRunBookkeeping:
    def test_too_few_frames_still_attaches_a_session(self, tmp_path):
        frame = tmp_path / "only.png"
        Image.new("RGB", (32, 32), (200, 180, 160)).save(frame)
        out = tmp_path / "out.png"
        pipe = AnimeStitchPipeline(
            use_basic=False,
            use_birefnet=False,
            use_loftr=False,
            use_efficient_loftr=False,
            use_aliked=False,
            use_roma=False,
            use_sea_raft=False,
            use_ecc=False,
            composite_fg=False,
        )
        with pytest.raises(PipelineError, match="at least 2"):
            pipe.run([str(frame)], str(out))
        assert pipe.last_session is not None
        assert pipe.last_session.success is False
        assert pipe.last_session.inputs.image_paths == [str(frame)]
        assert pipe.last_session.config["use_basic"] is False
