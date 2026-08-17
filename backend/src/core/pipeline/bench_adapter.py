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
from asp_backend.core.pipeline.safety_policy import (
    SafeAspPolicy,
    default_benchmark_policy,
    product_safe_asp_policy,
    safe_asp_counterfactual,
)
from asp_backend.core.pipeline.session import (
    PipelineSession,
    ResultIdentity,
    snapshot_pipeline_config,
)
from asp_backend.core.pipeline._frame_utils import _sort_frames_by_index


def bench_legacy_enabled() -> bool:
    return os.environ.get("ASP_BENCH_LEGACY", "0") == "1"


def bench_ungated_enabled() -> bool:
    """#30: keep Raw ASP as the published result; still record Safe/SCANS."""
    return os.environ.get("ASP_BENCH_UNGATED", "0") == "1"


# Run-internal floors that can replace a would-be Raw ASP composite with
# SCANS before a file exists. Forced (not setdefault) so inherited shell
# values cannot leak into a documented ungated baseline. Safe ASP
# Composite/Ghost/SeamVis knobs are *not* set here — those are evaluated
# as a frozen product-default counterfactual and never applied to the
# published file. Geometric no-edge / affine-invalid fallbacks stay:
# there is no raw composite to keep in those cases.
_UNGATED_RUN_ENV = {
    "ASP_ALIGN_GATE_DX": "9999",
    "ASP_COV_MIN_MULTI_PCT": "0",
}


def apply_ungated_gate_env() -> dict[str, str]:
    """Force run-internal ungated knobs. Returns the effective config."""
    if not bench_ungated_enabled():
        return {}
    for key, value in _UNGATED_RUN_ENV.items():
        os.environ[key] = value
    return dict(_UNGATED_RUN_ENV)


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

    ungated = bench_ungated_enabled()
    ungated_config = apply_ungated_gate_env() if ungated else {}

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

    # Ungated counterfactual must use frozen product defaults, not
    # inherited ASP_GATE_* from the caller's shell.
    policy = policy or (
        product_safe_asp_policy() if ungated else default_benchmark_policy()
    )
    affines = _affines_from_session(session)
    decisions = (
        policy.evaluate_all(raw_img, scans_img, affines or None)
        if raw_img is not None
        else []
    )
    for decision in decisions:
        if decision.log_line:
            print(decision.log_line)
    counterfactual = safe_asp_counterfactual(
        decisions,
        policy,
        raw_available=raw_available,
        scans_available=scans_img is not None,
    )
    extra: dict[str, Any] = {
        "safe_asp_counterfactual": counterfactual,
        "ungated_gate_config": ungated_config or None,
        "observability": session.observability(),
    }
    session.record_artifact("safe_asp_counterfactual", counterfactual)
    if ungated_config:
        session.record_artifact("ungated_gate_config", ungated_config)

    # Gated path: first rejecting gate still publishes SCANS as Safe ASP.
    # Ungated (#30): never replace the published file; the counterfactual
    # is the Safe ASP record.
    first_reject = next((d for d in decisions if not d.accept), None)
    if (
        not ungated
        and raw_available
        and raw_img is not None
        and scans_img is not None
        and first_reject is not None
        and scans_path
    ):
        if first_reject.fail_log_line:
            print(first_reject.fail_log_line)
        policy.apply_to_session(session, first_reject)
        shutil.copy2(scans_path, safe_asp_path)
        session.record_artifact("safe_asp_path", safe_asp_path)
        session.finish(success=True, identity=ResultIdentity.SAFE_ASP)
        extra["gate"] = first_reject.name
        extra["policy_would_reject"] = first_reject.reason
        extra["observability"] = session.observability()
        return CanonicalStitchResult(
            raw_asp_path=raw_asp_path,
            safe_asp_path=safe_asp_path,
            scans_path=scans_path,
            used_fallback=True,
            fallback_reason=first_reject.reason,
            identity=ResultIdentity.SAFE_ASP,
            session=session,
            pipeline=pipe,
            raw_asp_available=True,
            extra=extra,
        )

    if ungated and first_reject is not None:
        print(
            "  [Ungated] Safe ASP policy would reject "
            f"({first_reject.reason}); keeping Raw ASP as the published baseline."
        )
        extra["policy_would_reject"] = first_reject.reason
        extra["gate"] = first_reject.name

    if os.path.isfile(work_path):
        shutil.copy2(work_path, safe_asp_path)
    session.record_artifact("safe_asp_path", safe_asp_path)
    extra["observability"] = session.observability()
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
        extra=extra,
    )


__all__ = [
    "CanonicalStitchResult",
    "apply_ungated_gate_env",
    "bench_legacy_enabled",
    "bench_ungated_enabled",
    "run_canonical_asp",
]
