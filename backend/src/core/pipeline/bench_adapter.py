"""M1b: thin benchmark adapter over ``AnimeStitchPipeline.run()``.

Frame selection and the SCANS comparator stay in the benchmark script.
This module owns the ASP stitch + Safe ASP policy so the published
``panorama.png`` is the same compositor the product calls.

``ASP_BENCH_LEGACY=1`` keeps the pre-M1b inline orchestrator for A/B.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from asp_backend.core.pipeline.manager import AnimeStitchPipeline
from asp_backend.core.pipeline.safety_policy import SafeAspPolicy, default_benchmark_policy
from asp_backend.core.pipeline.session import (
    PipelineSession,
    ResultIdentity,
    snapshot_pipeline_config,
)
from asp_backend.core.pipeline._frame_utils import _sort_frames_by_index


def bench_legacy_enabled() -> bool:
    return os.environ.get("ASP_BENCH_LEGACY", "0") == "1"


@dataclass
class CanonicalStitchResult:
    """Raw / Safe / SCANS paths after one adapter call."""

    raw_asp_path: str
    safe_asp_path: str
    scans_path: str | None
    used_fallback: bool
    fallback_reason: str | None
    identity: str
    session: PipelineSession
    pipeline: AnimeStitchPipeline
    raw_asp_available: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def _affines_from_session(session: PipelineSession) -> list[np.ndarray]:
    raw = session.artifacts.get("affines")
    if not isinstance(raw, list):
        return []
    out: list[np.ndarray] = []
    for item in raw:
        try:
            out.append(np.asarray(item, dtype=np.float32))
        except (TypeError, ValueError):
            continue
    return out


def run_canonical_asp(
    frames_paths: list[str],
    *,
    raw_asp_path: str,
    safe_asp_path: str,
    scans_path: str | None,
    policy: SafeAspPolicy | None = None,
    pipeline: AnimeStitchPipeline | None = None,
    renderer: str = "median",
) -> CanonicalStitchResult:
    """Stitch with the product runner, then apply the injectable Safe ASP policy.

    ``run()`` writes to a staging file. ``raw_asp_path`` is created only
    when that output is a true Raw ASP composite. Internal SCANS/PANORAMA
    fallbacks never occupy a ``raw_asp`` filename (Chat/Codex M1b review).
    ``safe_asp_path`` is the published artifact.
    """
    paths = _sort_frames_by_index(list(frames_paths))
    if len(paths) < 2:
        raise ValueError("Need at least 2 frames for the canonical adapter.")

    pipe = pipeline or AnimeStitchPipeline(renderer=renderer)
    session = PipelineSession.create(
        paths,
        raw_asp_path,
        config=snapshot_pipeline_config(pipe),
    )
    raw_dir = os.path.dirname(os.path.abspath(raw_asp_path)) or "."
    safe_dir = os.path.dirname(os.path.abspath(safe_asp_path)) or "."
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(safe_dir, exist_ok=True)
    work_path = os.path.join(raw_dir, "run_output.png")

    pipe.run(paths, work_path, session=session)
    session = pipe.last_session or session

    scans_img = cv2.imread(scans_path) if scans_path and os.path.isfile(scans_path) else None
    work_img = cv2.imread(work_path) if os.path.isfile(work_path) else None
    identity = str(session.identity or ResultIdentity.RAW_ASP)
    raw_available = identity == ResultIdentity.RAW_ASP and work_img is not None
    if raw_available:
        shutil.copy2(work_path, raw_asp_path)
        session.record_artifact("raw_asp_path", raw_asp_path)
        session.record_artifact("raw_asp_available", True)
        raw_img = work_img
    else:
        if os.path.isfile(raw_asp_path):
            os.remove(raw_asp_path)
        session.record_artifact("raw_asp_available", False)
        raw_img = None

    used_fallback = identity != ResultIdentity.RAW_ASP
    fallback_reason = None
    if session.fallbacks:
        fallback_reason = str(session.fallbacks[-1].get("reason") or "")

    policy = policy or default_benchmark_policy()

    # Only a true Raw ASP composite is eligible for the output-safety policy.
    # Internal run() fallbacks already chose SCANS/panorama.
    if raw_available and raw_img is not None and scans_img is not None:
        affines = _affines_from_session(session)
        for decision in (
            policy.evaluate_composite(raw_img, scans_img, affines or None),
            policy.evaluate_ghost(raw_img, scans_img),
            policy.evaluate_seam_vis(raw_img, scans_img),
        ):
            if decision.log_line:
                print(decision.log_line)
            if not decision.accept:
                if decision.fail_log_line:
                    print(decision.fail_log_line)
                policy.apply_to_session(session, decision)
                shutil.copy2(scans_path, safe_asp_path)
                session.record_artifact("safe_asp_path", safe_asp_path)
                session.finish(success=True, identity=ResultIdentity.SAFE_ASP)
                return CanonicalStitchResult(
                    raw_asp_path=raw_asp_path,
                    safe_asp_path=safe_asp_path,
                    scans_path=scans_path,
                    used_fallback=True,
                    fallback_reason=decision.reason,
                    identity=ResultIdentity.SAFE_ASP,
                    session=session,
                    pipeline=pipe,
                    raw_asp_available=True,
                    extra={"gate": decision.name},
                )

    if os.path.isfile(work_path):
        shutil.copy2(work_path, safe_asp_path)
    session.record_artifact("safe_asp_path", safe_asp_path)
    return CanonicalStitchResult(
        raw_asp_path=raw_asp_path if raw_available else "",
        safe_asp_path=safe_asp_path,
        scans_path=scans_path,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        identity=identity,
        session=session,
        pipeline=pipe,
        raw_asp_available=raw_available,
    )


__all__ = [
    "CanonicalStitchResult",
    "bench_legacy_enabled",
    "run_canonical_asp",
]
