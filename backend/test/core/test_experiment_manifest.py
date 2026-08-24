"""M0b: experiment manifest + OTel-shaped telemetry — no GPU, no pixel path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from asp_backend.core.pipeline.manifest import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    compare_traces,
    hash_file,
)
from asp_backend.core.pipeline.session import (
    PipelineSession,
    PipelineStage,
    ResultIdentity,
)
from asp_backend.core.pipeline.telemetry import (
    CANONICAL_METRICS,
    METRIC_GAIN_CLAMP_RESIDUAL,
    METRIC_SEAM_CUT_ENERGY,
    METRIC_STAGE_DURATION_MS,
    NullTelemetrySink,
    OtlpJsonlSink,
    RerunSink,
    ResourceTracker,
    sample_rss_bytes,
    sample_vram_bytes,
    sink_from_env,
)

_SRC = Path(__file__).resolve().parents[2] / "src" / "core" / "pipeline"


def _session(tmp_paths: list[str] | None = None, **kwargs) -> PipelineSession:
    paths = tmp_paths or ["a.png", "b.png"]
    return PipelineSession.create(paths, "out.png", config={"renderer": "median"}, **kwargs)


def _run_pair(session: PipelineSession) -> None:
    session.mark(PipelineStage.LOAD, n=2)
    session.mark(PipelineStage.NORMALISE, width=64, height=64)
    session.record_artifact("output_path", "out.png")
    session.finish(success=True, identity=ResultIdentity.RAW_ASP)


class TestManifest:
    def test_manifest_carries_locked_fields(self, tmp_path):
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)
        session = _session([str(frame)])
        _run_pair(session)
        payload = session.experiment_manifest()
        assert payload["schema"] == SCHEMA_NAME
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["git"]["commit"]
        assert payload["profile"]
        assert payload["config"]["renderer"] == "median"
        assert "effective_env" in payload
        assert "reproducibility" in payload
        assert "torch_threads" in payload["reproducibility"]
        assert "cudnn_deterministic" in payload["reproducibility"]
        assert "OMP_NUM_THREADS" in payload["reproducibility"]["environment"]
        assert "model_versions" in payload
        assert payload["inputs"]["hashes"][str(frame)] == hash_file(frame)
        assert payload["trace"]["digest"] == session.digest()
        assert payload["outputs"]["identity"] == ResultIdentity.RAW_ASP
        assert "peak_rss_bytes" in payload["resources"]
        assert payload["resources"]["peak_rss_bytes"] >= 0

    def test_two_runs_same_manifest_are_equivalent(self):
        a = _session()
        b = _session()
        _run_pair(a)
        _run_pair(b)
        diff = compare_traces(a, b)
        assert a.digest() == b.digest()
        assert diff.equivalent
        assert diff.divergent_stages == []

    def test_duration_only_is_not_nondeterminism(self):
        a = _session()
        b = _session()
        _run_pair(a)
        _run_pair(b)
        a.stages[0].duration_s = 0.01
        b.stages[0].duration_s = 9.99
        diff = compare_traces(a, b)
        assert diff.equivalent
        assert "load" in diff.timing_only
        assert diff.divergent_stages == []

    def test_note_divergence_is_identified(self):
        a = _session()
        b = _session()
        a.mark(PipelineStage.MATCH, n_edges=4)
        b.mark(PipelineStage.MATCH, n_edges=0)
        b.record_fallback(ResultIdentity.SCANS, "no_valid_edges")
        a.finish(success=True, identity=ResultIdentity.RAW_ASP)
        b.finish(success=True, identity=ResultIdentity.SCANS)
        diff = compare_traces(a, b)
        assert not diff.equivalent
        assert "match" in diff.divergent_stages
        assert a.digest() != b.digest()

    def test_missing_input_files_do_not_raise(self):
        session = _session(["missing-a.png"])
        assert session.input_hashes["missing-a.png"] is None
        payload = session.experiment_manifest()
        assert payload["inputs"]["hashes"]["missing-a.png"] is None

    def test_deterministic_request_is_recorded(self, monkeypatch):
        monkeypatch.setenv("ASP_DETERMINISTIC", "1")
        monkeypatch.setenv("ASP_REPRO_SEED", "19")
        session = _session()
        repro = session.reproducibility
        assert repro["requested"] is True
        assert repro["seed"] == 19
        assert repro["opencv_rng_seed"] == 19
        assert repro["torch_rng_seed"] == 19


class TestResources:
    def test_rss_samples_without_cuda(self):
        rss = sample_rss_bytes()
        assert rss > 0
        vram = sample_vram_bytes()
        assert vram is None or vram >= 0
        tracker = ResourceTracker()
        tracker.sample()
        assert tracker.peak_rss_bytes > 0
        # CPU-only CI must not pretend it saw VRAM.
        if vram is None:
            assert tracker.peak_vram_bytes is None


class TestOtlpSink:
    def test_jsonl_span_and_canonical_metrics(self, tmp_path):
        path = tmp_path / "run.jsonl"
        sink = OtlpJsonlSink(path)
        session = _session(telemetry=sink)
        session.mark(PipelineStage.LOAD, n=1)
        session.note_gain_telemetry({"mean_residual": 0.25, "n_clamped": 1})
        session.note_seam_feasibility({"attempted": True, "cut_energy": 12.5})
        session.finish(success=True, identity=ResultIdentity.RAW_ASP)
        lines = [json.loads(line) for line in path.read_text().splitlines() if line]
        assert lines
        spans = [row["span"] for row in lines if "span" in row]
        metrics = [row["metric"] for row in lines if "metric" in row]
        assert spans
        assert spans[0]["name"] == "asp.stage.load"
        assert "traceId" in spans[0]
        assert "spanId" in spans[0]
        assert spans[0]["status"]["code"] == 1
        names = {row["name"] for row in metrics}
        assert METRIC_STAGE_DURATION_MS in names
        assert METRIC_GAIN_CLAMP_RESIDUAL in names
        assert METRIC_SEAM_CUT_ENERGY in names
        # VRAM metric is omitted on CPU-only hosts.
        assert names <= set(CANONICAL_METRICS)
        for row in lines:
            assert row["instrumentationScope"]["name"] == "asp.pipeline"
            assert row["resource"]["attributes"]["service.name"] == "asp"

    def test_null_sink_is_default_and_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ASP_TELEMETRY", raising=False)
        monkeypatch.delenv("ASP_RERUN", raising=False)
        session = _session()
        assert isinstance(session.telemetry, NullTelemetrySink)
        _run_pair(session)
        assert list(tmp_path.iterdir()) == []

    def test_env_path_selects_file_sink(self, tmp_path, monkeypatch):
        path = tmp_path / "env.jsonl"
        monkeypatch.setenv("ASP_TELEMETRY", str(path))
        monkeypatch.delenv("ASP_RERUN", raising=False)
        sink = sink_from_env()
        session = _session(telemetry=sink)
        session.mark(PipelineStage.SAVE)
        session.finish(success=True, identity=ResultIdentity.RAW_ASP)
        assert path.is_file()
        assert "asp.stage.save" in path.read_text()

    def test_rerun_sink_noops_without_sdk(self):
        sink = RerunSink()
        # Either the extra is installed (still must not raise) or it is a no-op.
        sink.span_end("asp.stage.load", duration_ms=1.0)
        sink.metric(METRIC_STAGE_DURATION_MS, 1.0, unit="ms")
        sink.close()


class TestImportBoundary:
    def test_run_and_session_do_not_import_otel_or_rerun(self):
        session_src = (_SRC / "session.py").read_text(encoding="utf-8")
        run_src = (_SRC / "run_stage.py").read_text(encoding="utf-8")
        for src in (session_src, run_src):
            assert "opentelemetry" not in src
            assert "import rerun" not in src
            assert "from rerun" not in src

    def test_session_import_does_not_load_optional_backends(self):
        assert "opentelemetry" not in sys.modules
        # rerun may be installed in a desktop env; importing session must not
        # have pulled it in unless some other test already did.
        import asp_backend.core.pipeline.session as session_mod

        assert session_mod.PipelineSession is PipelineSession
