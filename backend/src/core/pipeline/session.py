"""Shared ASP stage protocol and ``PipelineSession`` (M1a).

This is the extraction-only contract that M1b (benchmark) and M1c (GUI)
will later adapt onto. It records inputs, a frozen config snapshot, an
ordered stage trace, JSON-safe artifacts, and optional HITL pause hooks.

It must not change pixels. ``AnimeStitchPipeline.run()`` creates a session
and writes bookkeeping next to existing log lines; image operations stay
on their current path. Pause hooks are stored and callable, but the
canonical runner does not insert new HITL checkpoints here (that is M1c).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

PauseHook = Callable[[str, dict[str, Any]], dict[str, Any]]


class PipelineStage(StrEnum):
    """Canonical stage ids shared by CLI, benchmark, and GUI adapters.

    String values are stable identifiers for traces and future parity
    digests. They are *not* the GUI's 1-based progress index — that
    numbering disagrees with ``run()`` and will be mapped in M1c.
    """

    LOAD = "load"
    NORMALISE = "normalise"
    PHOTOMETRIC_BASIC = "photometric_basic"
    MASK = "mask"
    PHOTOMETRIC_BG = "photometric_bg"
    DEDUP = "dedup"
    MATCH = "match"
    SPATIAL_DEDUP = "spatial_dedup"
    PHASE = "phase"
    FILTER_EDGES = "filter_edges"
    BUNDLE_ADJUST = "bundle_adjust"
    AFFINE_VALIDATE = "affine_validate"
    REFINE = "refine"
    HIRES = "hires"
    CANVAS = "canvas"
    ALIGN_GATES = "align_gates"
    RENDER = "render"
    COVERAGE_GATE = "coverage_gate"
    COMPOSITE = "composite"
    CONTENT_TRIM = "content_trim"
    CROP = "crop"
    INPAINT = "inpaint"
    SAVE = "save"


class HitlCheckpoint(StrEnum):
    """Event names already used by ``_ProgressPipeline._hitl_pause``."""

    FRAMES = "frames"
    MASKS = "masks"
    EDGES = "edges"
    CANVAS = "canvas"
    RENDER = "render"
    BOUNDARIES = "boundaries"
    SEAMS = "seams"
    COMPOSITE = "composite"
    OUTPUT = "output"


class ResultIdentity(StrEnum):
    """Which compositor identity produced the file at ``output_path``."""

    RAW_ASP = "raw_asp"
    SAFE_ASP = "safe_asp"
    SCANS = "scans"


_CONFIG_KEYS = (
    "use_basic",
    "use_birefnet",
    "use_loftr",
    "use_efficient_loftr",
    "use_aliked",
    "use_roma",
    "use_sea_raft",
    "use_jamma",
    "use_ecc",
    "renderer",
    "composite_fg",
    "bands",
    "edge_crop",
    "motion_model",
)


def snapshot_pipeline_config(host: Any) -> dict[str, Any]:
    """Copy image-affecting flags off an ``AnimeStitchPipeline``-like host."""
    snap: dict[str, Any] = {key: getattr(host, key, None) for key in _CONFIG_KEYS}
    extra = getattr(host, "kwargs", None)
    if isinstance(extra, Mapping):
        snap["kwargs"] = {
            key: value for key, value in extra.items() if key != "pause_hook"
        }
    return snap


def json_safe(value: Any) -> Any:
    """Reduce a value to JSON-serialisable metadata (never raw image bytes)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return {
            "__array__": True,
            "shape": list(shape),
            "dtype": str(dtype),
        }
    return str(value)


@dataclass
class StageRecord:
    """One completed or in-flight stage in a session trace."""

    name: str
    started_at: float
    finished_at: float | None = None
    duration_s: float | None = None
    skipped: bool = False
    fallback: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def close(self, **notes: Any) -> None:
        self.notes.update(notes)
        if "skipped" in notes:
            self.skipped = bool(notes["skipped"])
        if "fallback" in notes:
            self.fallback = notes["fallback"]
        self.finished_at = time.perf_counter()
        self.duration_s = round(self.finished_at - self.started_at, 6)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "skipped": self.skipped,
            "fallback": self.fallback,
            "notes": json_safe(self.notes),
        }


@dataclass
class PipelineInputs:
    image_paths: list[str]
    output_path: str
    hires_keyframes: dict[int, str] | None = None


@dataclass
class PipelineSession:
    """Bookkeeping object shared by the three ASP entry points after M1.

    Image arrays stay on the caller. This object only stores paths, config,
    stage names, JSON-safe artifact metadata, and pause-hook traffic.
    """

    inputs: PipelineInputs
    config: dict[str, Any]
    pause_hook: PauseHook | None = None
    stages: list[StageRecord] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    hitl_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    identity: str | None = None
    success: bool | None = None
    error: str | None = None
    fallback_reason: str | None = None
    geometry: list[dict[str, Any]] = field(default_factory=list)
    frame_provenance: list[dict[str, Any]] = field(default_factory=list)
    pose_provenance: list[dict[str, Any]] = field(default_factory=list)
    gain_telemetry: dict[str, Any] = field(default_factory=dict)
    seam_feasibility: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)
    finished_at: float | None = None
    _open: StageRecord | None = field(default=None, init=False, repr=False)

    @classmethod
    def create(
        cls,
        image_paths: list[str],
        output_path: str,
        hires_keyframes: dict[int, str] | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        host: Any | None = None,
        pause_hook: PauseHook | None = None,
    ) -> PipelineSession:
        snap = dict(config) if config is not None else (
            snapshot_pipeline_config(host) if host is not None else {}
        )
        return cls(
            inputs=PipelineInputs(
                image_paths=list(image_paths),
                output_path=output_path,
                hires_keyframes=dict(hires_keyframes) if hires_keyframes else None,
            ),
            config=snap,
            pause_hook=pause_hook,
        )

    def start_stage(self, stage: PipelineStage | str, **notes: Any) -> StageRecord:
        if self._open is not None:
            self._open.close()
            self._open = None
        record = StageRecord(
            name=str(stage),
            started_at=time.perf_counter(),
            notes=dict(notes),
        )
        self.stages.append(record)
        self._open = record
        return record

    def complete_stage(
        self,
        stage: PipelineStage | str | None = None,
        **notes: Any,
    ) -> StageRecord:
        record = self._open
        expected = str(stage) if stage is not None else None
        if record is None or (expected is not None and record.name != expected):
            record = self.start_stage(stage or "unknown")
        record.close(**notes)
        self._open = None
        return record

    def mark(self, stage: PipelineStage | str, **notes: Any) -> StageRecord:
        """Record a stage as an instantaneous (already-finished) event."""
        self.start_stage(stage)
        return self.complete_stage(stage, **notes)

    def record_artifact(self, key: str, value: Any) -> None:
        self.artifacts[key] = json_safe(value)

    def note_geometry(
        self,
        stage: PipelineStage | str,
        *,
        width: int,
        height: int,
        n_frames: int | None = None,
    ) -> None:
        """M2: per-stage image geometry (never pixels)."""
        entry: dict[str, Any] = {
            "stage": str(stage),
            "width": int(width),
            "height": int(height),
        }
        if n_frames is not None:
            entry["n_frames"] = int(n_frames)
        self.geometry.append(entry)

    def note_frame_provenance(self, rows: list[Mapping[str, Any]]) -> None:
        self.frame_provenance = [dict(row) for row in rows]

    def note_pose_provenance(self, rows: list[Mapping[str, Any]]) -> None:
        self.pose_provenance = [dict(row) for row in rows]

    def note_gain_telemetry(self, payload: Mapping[str, Any]) -> None:
        self.gain_telemetry = dict(payload)

    def note_seam_feasibility(self, payload: Mapping[str, Any]) -> None:
        # Drop image crops if a caller forwards seam_meta_out wholesale.
        clean = {
            key: value
            for key, value in payload.items()
            if key != "seam_crops"
        }
        self.seam_feasibility = clean

    def observability(self) -> dict[str, Any]:
        return {
            "geometry": list(self.geometry),
            "frame_provenance": list(self.frame_provenance),
            "pose_provenance": list(self.pose_provenance),
            "gain": dict(self.gain_telemetry),
            "seam": dict(self.seam_feasibility),
            "fallback_reason": self.fallback_reason,
        }

    def record_fallback(
        self,
        identity: ResultIdentity,
        reason: str,
        **notes: Any,
    ) -> None:
        """Record a non-raw result using one of the canonical identities."""
        entry = {"identity": identity, "reason": reason, **json_safe(notes)}
        self.fallbacks.append(entry)
        self.fallback_reason = reason
        if self._open is not None:
            self._open.fallback = identity
            self._open.notes.setdefault("fallback_reason", reason)
        self.identity = identity

    def pause(self, event: HitlCheckpoint | str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke the optional HITL hook; default is a no-op ``{}``.

        Override application is deliberately not done here. The GUI fork
        still owns override semantics until M1c routes them through this
        method.
        """
        payload = data or {}
        if self.pause_hook is None:
            return {}
        override = self.pause_hook(str(event), payload) or {}
        if override:
            self.hitl_overrides[str(event)] = json_safe(override)
        return override

    def finish(
        self,
        *,
        success: bool,
        identity: str | None = None,
        error: str | None = None,
    ) -> None:
        if self._open is not None:
            self._open.close()
            self._open = None
        if identity is not None:
            self.identity = identity
        if self.fallback_reason is None and self.fallbacks:
            self.fallback_reason = str(self.fallbacks[-1].get("reason") or "") or None
        self.record_artifact("observability", self.observability())
        self.success = success
        self.error = error
        self.finished_at = time.perf_counter()

    def stage_names(self) -> list[str]:
        return [record.name for record in self.stages]

    def as_dict(self) -> dict[str, Any]:
        return {
            "inputs": {
                "image_paths": list(self.inputs.image_paths),
                "output_path": self.inputs.output_path,
                "hires_keyframes": self.inputs.hires_keyframes,
            },
            "config": json_safe(self.config),
            "stages": [record.as_dict() for record in self.stages],
            "artifacts": dict(self.artifacts),
            "hitl_overrides": dict(self.hitl_overrides),
            "fallbacks": list(self.fallbacks),
            "fallback_reason": self.fallback_reason,
            "identity": self.identity,
            "success": self.success,
            "error": self.error,
        }

    def digest(self) -> str:
        """Stable hex digest of stage names, identity, fallbacks, artifact keys.

        Omits timestamps and durations so two identical pixel paths compare
        equal. Used by the M1c headless parity suite.
        """
        payload = {
            "stages": self.stage_names(),
            "identity": self.identity,
            "fallbacks": [
                {"identity": item.get("identity"), "reason": item.get("reason")}
                for item in self.fallbacks
            ],
            "artifact_keys": sorted(self.artifacts),
            "config": json_safe(self.config),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


__all__ = [
    "HitlCheckpoint",
    "PauseHook",
    "PipelineInputs",
    "PipelineSession",
    "PipelineStage",
    "ResultIdentity",
    "StageRecord",
    "json_safe",
    "snapshot_pipeline_config",
]
