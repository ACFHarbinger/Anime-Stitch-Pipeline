"""M0b emission API: ``TelemetrySink`` + local OTel-shaped JSONL.

Canonical ``run()`` and ``session.py`` must not import ``opentelemetry`` or
``rerun``. This module is the only place those optional backends are
touched, and only behind lazy imports.

Metric names are locked in ``asp_change_roadmap_2026q3.md`` §19.3.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TextIO

METRIC_STAGE_DURATION_MS = "asp.stage.duration_ms"
METRIC_VRAM_PEAK_BYTES = "asp.vram.peak_bytes"
METRIC_GAIN_CLAMP_RESIDUAL = "asp.gain.clamp_residual"
METRIC_SEAM_CUT_ENERGY = "asp.seam.cut_energy"

CANONICAL_METRICS = (
    METRIC_STAGE_DURATION_MS,
    METRIC_VRAM_PEAK_BYTES,
    METRIC_GAIN_CLAMP_RESIDUAL,
    METRIC_SEAM_CUT_ENERGY,
)

_SERVICE = {"service.name": "asp", "service.version": "0.1.0"}


class TelemetrySink(Protocol):
    """One emission surface. Implementations must never raise into the pipeline."""

    def span_start(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> str | None: ...

    def span_end(
        self,
        name: str,
        *,
        span_id: str | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None: ...

    def metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "1",
        attributes: Mapping[str, Any] | None = None,
    ) -> None: ...

    def close(self) -> None: ...


class NullTelemetrySink:
    """Default no-op. Headless CI and ``laptop_balanced`` stay here."""

    def span_start(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> str | None:
        return None

    def span_end(
        self,
        name: str,
        *,
        span_id: str | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        return None

    def metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "1",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        return None

    def close(self) -> None:
        return None


def _otel_attrs(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    if not mapping:
        return {}
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        if value is None or isinstance(value, (bool, int, float, str)):
            out[str(key)] = value
        else:
            out[str(key)] = str(value)
    return out


def _now_unix_nano() -> int:
    return time.time_ns()


def _new_ids() -> tuple[str, str]:
    # 16-byte trace / 8-byte span, hex — OTLP JSON wire shape.
    return uuid.uuid4().hex, uuid.uuid4().hex[:16]


class OtlpJsonlSink:
    """OTLP-shaped JSONL to a file or stdout. No OpenTelemetry SDK required."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        stream: TextIO | None = None,
        resource: Mapping[str, Any] | None = None,
    ) -> None:
        self._path = str(path) if path is not None else None
        self._stream = stream
        self._owned: TextIO | None = None
        self._resource = {**_SERVICE, **dict(resource or {})}
        self._open: dict[str, dict[str, Any]] = {}
        self._trace_id, _ = _new_ids()

    def _writer(self) -> TextIO:
        if self._stream is not None:
            return self._stream
        if self._owned is None:
            if not self._path:
                self._stream = sys.stdout
                return self._stream
            parent = os.path.dirname(self._path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._owned = open(self._path, "a", encoding="utf-8")
        return self._owned

    def _emit(self, payload: dict[str, Any]) -> None:
        try:
            self._writer().write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._writer().flush()
        except Exception:
            return

    def _envelope(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "resource": {"attributes": dict(self._resource)},
            "instrumentationScope": {"name": "asp.pipeline", "version": "0.1.0"},
            **body,
        }

    def span_start(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> str | None:
        try:
            _trace, span_id = self._trace_id, _new_ids()[1]
            self._open[span_id] = {
                "name": name,
                "spanId": span_id,
                "traceId": _trace,
                "startTimeUnixNano": _now_unix_nano(),
                "attributes": _otel_attrs(attributes),
            }
            return span_id
        except Exception:
            return None

    def span_end(
        self,
        name: str,
        *,
        span_id: str | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            started = self._open.pop(span_id, None) if span_id else None
            start_ns = (
                started["startTimeUnixNano"]
                if started
                else _now_unix_nano()
            )
            end_ns = _now_unix_nano()
            attrs = dict(started["attributes"] if started else {})
            attrs.update(_otel_attrs(attributes))
            if duration_ms is not None:
                attrs["asp.stage.duration_ms"] = duration_ms
            status = (
                {"code": 2, "message": error}
                if error
                else {"code": 1}
            )
            self._emit(
                self._envelope(
                    {
                        "span": {
                            "name": started["name"] if started else name,
                            "traceId": started["traceId"] if started else self._trace_id,
                            "spanId": (started["spanId"] if started else span_id)
                            or _new_ids()[1],
                            "kind": 1,
                            "startTimeUnixNano": start_ns,
                            "endTimeUnixNano": end_ns,
                            "attributes": attrs,
                            "status": status,
                        }
                    }
                )
            )
            if duration_ms is not None:
                self.metric(
                    METRIC_STAGE_DURATION_MS,
                    duration_ms,
                    unit="ms",
                    attributes={"asp.stage": name.removeprefix("asp.stage.")},
                )
        except Exception:
            return

    def metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "1",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            self._emit(
                self._envelope(
                    {
                        "metric": {
                            "name": name,
                            "unit": unit,
                            "gauge": {"asDouble": float(value)},
                            "attributes": _otel_attrs(attributes),
                        }
                    }
                )
            )
        except Exception:
            return

    def close(self) -> None:
        if self._owned is not None:
            try:
                self._owned.flush()
                self._owned.close()
            except Exception:
                pass
            self._owned = None


class CompositeTelemetrySink:
    """Fan-out. One failure does not skip the rest."""

    def __init__(self, sinks: Sequence[TelemetrySink]) -> None:
        self.sinks = list(sinks)

    def span_start(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> str | None:
        span_id: str | None = None
        for sink in self.sinks:
            try:
                got = sink.span_start(name, attributes=attributes)
            except Exception:
                continue
            if span_id is None:
                span_id = got
        return span_id

    def span_end(
        self,
        name: str,
        *,
        span_id: str | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        for sink in self.sinks:
            try:
                sink.span_end(
                    name,
                    span_id=span_id,
                    duration_ms=duration_ms,
                    error=error,
                    attributes=attributes,
                )
            except Exception:
                continue

    def metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "1",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        for sink in self.sinks:
            try:
                sink.metric(name, value, unit=unit, attributes=attributes)
            except Exception:
                continue

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                continue


class RerunSink:
    """Opt-in ``desktop_quality`` sidecar. No-ops if ``rerun-sdk`` is absent.

    2D stage scalars only. Do not log ``Transform3D`` / pinhole cameras —
    ASP's BA is a 2D affine chain, not a reconstruct.
    """

    def __init__(self, application_id: str = "asp") -> None:
        self._rr: Any | None = None
        try:
            import rerun as rr  # type: ignore[import-not-found]

            rr.init(application_id, spawn=False)
            self._rr = rr
        except Exception:
            self._rr = None

    def span_start(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> str | None:
        return None

    def span_end(
        self,
        name: str,
        *,
        span_id: str | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if self._rr is None or duration_ms is None:
            return
        try:
            self._rr.log(f"stages/{name}", self._rr.Scalars(float(duration_ms)))
        except Exception:
            return

    def metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "1",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if self._rr is None:
            return
        try:
            self._rr.log(f"metrics/{name}", self._rr.Scalars(float(value)))
        except Exception:
            return

    def close(self) -> None:
        return None


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on", "stdout"}


def sink_from_env() -> TelemetrySink:
    """Build the process sink from ``ASP_TELEMETRY`` / ``ASP_RERUN``.

    * unset / ``0`` / ``false`` → ``NullTelemetrySink`` (CI default)
    * ``1`` / ``stdout`` → JSONL on stdout
    * any other non-empty value → JSONL append at that path
    * ``ASP_RERUN=1`` additionally fans out to ``RerunSink`` (no-op without the extra)
    """
    mode = os.environ.get("ASP_TELEMETRY", "").strip()
    sinks: list[TelemetrySink] = []
    if _truthy(mode) or mode.lower() == "stdout":
        sinks.append(OtlpJsonlSink(stream=sys.stdout))
    elif mode and mode.lower() not in {"0", "false", "no", "off"}:
        sinks.append(OtlpJsonlSink(path=mode))
    rerun_flag = os.environ.get("ASP_RERUN", "").strip()
    if rerun_flag and rerun_flag.lower() not in {"0", "false", "no", "off"}:
        sinks.append(RerunSink())
    if not sinks:
        return NullTelemetrySink()
    if len(sinks) == 1:
        return sinks[0]
    return CompositeTelemetrySink(sinks)


def sample_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/statm", encoding="utf-8") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def sample_vram_bytes() -> int | None:
    """Peak allocated CUDA bytes, or ``None`` when there is no GPU.

    Reads ``max_memory_allocated`` only. Does not synchronize the device.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


@dataclass
class ResourceTracker:
    """Running peak RSS / VRAM. VRAM stays ``None`` on CPU-only CI."""

    peak_rss_bytes: int = 0
    peak_vram_bytes: int | None = None

    def sample(self) -> dict[str, int | None]:
        rss = sample_rss_bytes()
        vram = sample_vram_bytes()
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        if vram is not None:
            current = self.peak_vram_bytes
            if current is None or vram > current:
                self.peak_vram_bytes = vram
        return {"rss_bytes": rss, "vram_bytes": vram}

    def as_dict(self) -> dict[str, int | None]:
        return {
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_vram_bytes": self.peak_vram_bytes,
        }


_NULL = NullTelemetrySink()


def default_sink() -> TelemetrySink:
    return _NULL


__all__ = [
    "CANONICAL_METRICS",
    "METRIC_GAIN_CLAMP_RESIDUAL",
    "METRIC_SEAM_CUT_ENERGY",
    "METRIC_STAGE_DURATION_MS",
    "METRIC_VRAM_PEAK_BYTES",
    "CompositeTelemetrySink",
    "NullTelemetrySink",
    "OtlpJsonlSink",
    "ResourceTracker",
    "RerunSink",
    "TelemetrySink",
    "default_sink",
    "sample_rss_bytes",
    "sample_vram_bytes",
    "sink_from_env",
]
