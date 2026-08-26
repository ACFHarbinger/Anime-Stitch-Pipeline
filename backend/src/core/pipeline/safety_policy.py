"""Injectable Safe ASP output policy (M1b).

Owns CompositeGate / GhostGate / SeamVisGate — the three SCANS-relative
checks that currently live only in ``bench_anime_stitch.py``. Canonical
``AnimeStitchPipeline.run()`` does not call this yet (M2). The benchmark
adapter does; production will inject the same object later.

Thresholds and reason strings are copied from the 2026-08 benchmark path
so existing 97-case fallback labels stay comparable.

Quirk preserved on purpose: CompositeGate never reads SCANS strip-banding
(the bench left ``_scans_sb_ref = 0.0``), so the sb limit is the hard floor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .safety_metrics import (
    ghosting_score_v2,
    seam_coherence,
    seam_visibility_score,
    strip_banding_score,
)
from .session import PipelineSession, ResultIdentity


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class GateDecision:
    """One gate's accept/reject plus the strings the bench already prints."""

    name: str
    accept: bool
    skipped: bool = False
    reason: str | None = None
    runtime_message: str | None = None
    log_line: str | None = None
    fail_log_line: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    fallback_code: int = 0  # bench timings["render_gate_fallback"]
    # Chat/Codex M2 design: GhostGate may record scores without rejecting.
    # ``telemetry_only_inverse_validated`` means the signal is kept for
    # traces/reports and must not drive Safe ASP selection.
    status: str | None = None


@dataclass
class SafeAspPolicy:
    """SCANS-relative output-safety policy. Default-off on the product path."""

    composite_sc_floor: float = 38.0
    composite_sb_floor: float = 35.0
    composite_scans_mult: float = 2.0
    ghost_ratio: float = 2.0
    ghost_floor: float = 40.0
    seam_vis_ratio: float = 3.0
    seam_vis_floor: float = 35.0
    registration_gate_enabled: bool = False
    uncertain_result_policy: str = "prompt"
    # Default-off M2 candidates. Do not flip until the promotion ladder
    # (five-case → red set → 97) reports no human-worse selection.
    ghost_telemetry_only: bool = False
    composite_sb_telemetry_only: bool = False
    composite_sc_telemetry_only: bool = False

    @classmethod
    def from_environ(cls) -> SafeAspPolicy:
        """Same env knobs the benchmark already documents."""
        both = os.environ.get("ASP_COMPOSITE_TELEMETRY_ONLY", "0") == "1"
        registration_gate_enabled = os.environ.get("ASP_REGISTRATION_GATE_ENABLED", "0") == "1"
        return cls(
            composite_sc_floor=_env_float("ASP_GATE_SC", 38.0),
            composite_sb_floor=_env_float("ASP_GATE_SB", 35.0),
            ghost_ratio=_env_float("ASP_GATE_GHOST", 2.0),
            ghost_floor=_env_float("ASP_GATE_GHOST_FLOOR", 40.0),
            seam_vis_ratio=_env_float("ASP_GATE_SEAM_VIS", 3.0),
            seam_vis_floor=_env_float("ASP_GATE_SEAM_VIS_FLOOR", 35.0),
            registration_gate_enabled=registration_gate_enabled,
            uncertain_result_policy=os.environ.get("ASP_UNCERTAIN_RESULT_POLICY", "prompt"),
            ghost_telemetry_only=os.environ.get("ASP_GHOST_TELEMETRY_ONLY", "0") == "1",
            composite_sb_telemetry_only=both
            or os.environ.get("ASP_COMPOSITE_SB_TELEMETRY_ONLY", "0") == "1",
            composite_sc_telemetry_only=both
            or os.environ.get("ASP_COMPOSITE_SC_TELEMETRY_ONLY", "0") == "1",
        )

    def evaluate_composite(
        self,
        asp_img: np.ndarray,
        scans_img: np.ndarray | None,
        affines: list[np.ndarray] | None,
    ) -> GateDecision:
        asp_sc = seam_coherence(asp_img)
        asp_sb = strip_banding_score(asp_img, affines)
        scans_sc = seam_coherence(scans_img) if scans_img is not None else 0.0
        # Preserved bench quirk: SCANS strip-banding is not measured here.
        scans_sb = 0.0
        sc_limit = max(self.composite_sc_floor, scans_sc * self.composite_scans_mult)
        sb_limit = max(self.composite_sb_floor, scans_sb * self.composite_scans_mult)
        log_line = (
            f"  [CompositeGate] asp sc={asp_sc:.1f} sb={asp_sb:.1f}  "
            f"scans sc={scans_sc:.1f} sb={scans_sb:.1f}  "
            f"limits sc<{sc_limit:.1f} sb<{sb_limit:.1f}"
        )
        sc_fail = asp_sc > sc_limit
        sb_fail = asp_sb > sb_limit
        sc_rejects = sc_fail and not self.composite_sc_telemetry_only
        sb_rejects = sb_fail and not self.composite_sb_telemetry_only
        telem_bits = []
        if self.composite_sb_telemetry_only:
            telem_bits.append("sb")
        if self.composite_sc_telemetry_only:
            telem_bits.append("sc")
        status = (
            "telemetry_only_inverse_validated" if telem_bits else None
        )
        if telem_bits:
            log_line += f"  [telemetry_only:{'+'.join(telem_bits)}]"
        scores = {
            "asp_sc": asp_sc,
            "asp_sb": asp_sb,
            "scans_sc": scans_sc,
            "sc_limit": sc_limit,
            "sb_limit": sb_limit,
            "would_reject_sc": 1.0 if sc_fail else 0.0,
            "would_reject_sb": 1.0 if sb_fail else 0.0,
        }
        if not (sc_rejects or sb_rejects):
            which = "sc" if sc_fail else "sb" if sb_fail else None
            reason = (
                None
                if which is None
                else (
                    f"composite_gate_{which}:asp_sc={asp_sc:.1f}_limit={sc_limit:.1f},"
                    f"asp_sb={asp_sb:.1f}_limit={sb_limit:.1f}"
                )
            )
            return GateDecision(
                name="composite",
                accept=True,
                reason=reason if status else None,
                log_line=log_line,
                scores=scores,
                status=status,
            )
        which = "sc" if sc_rejects else "sb"
        reason = (
            f"composite_gate_{which}:asp_sc={asp_sc:.1f}_limit={sc_limit:.1f},"
            f"asp_sb={asp_sb:.1f}_limit={sb_limit:.1f}"
        )
        return GateDecision(
            name="composite",
            accept=False,
            reason=reason,
            runtime_message=(
                f"Composite quality gate: asp sc={asp_sc:.1f} (limit={sc_limit:.1f}), "
                f"asp sb={asp_sb:.1f} (limit={sb_limit:.1f})"
            ),
            log_line=log_line,
            fail_log_line=(
                f"  [CompositeGate] FAILED "
                f"(asp sc={asp_sc:.1f}>{sc_limit:.1f} or "
                f"asp sb={asp_sb:.1f}>{sb_limit:.1f}) → SCANS fallback."
            ),
            scores=scores,
            fallback_code=1,
            status=status,
        )

    def ghost_limit(self, sim_g: float) -> float:
        return max(self.ghost_floor, self.ghost_ratio * max(sim_g, 1.0))

    def evaluate_ghost(
        self,
        asp_img: np.ndarray,
        scans_img: np.ndarray | None,
    ) -> GateDecision:
        telemetry = (
            "telemetry_only_inverse_validated" if self.ghost_telemetry_only else None
        )
        if scans_img is None or self.ghost_ratio >= 90:
            return GateDecision(
                name="ghost", accept=True, skipped=True, status=telemetry
            )
        asp_g = ghosting_score_v2(asp_img)
        sim_g = ghosting_score_v2(scans_img)
        ratio = asp_g / max(sim_g, 1.0)
        limit = max(self.ghost_floor, self.ghost_ratio * max(sim_g, 1.0))
        would_reject = asp_g > limit
        log_line = (
            f"  [GhostGate/siqe] asp_ghost={asp_g:.1f}  "
            f"sim_ghost={sim_g:.1f}  "
            f"ratio={ratio:.2f}"
        )
        if telemetry:
            log_line += f"  [{telemetry}]"
        scores = {
            "asp_ghost": asp_g,
            "sim_ghost": sim_g,
            "limit": limit,
            "would_reject": 1.0 if would_reject else 0.0,
        }
        reason = (
            f"ghost_gate_siqe:asp={asp_g:.1f}_sim={sim_g:.1f}_limit={limit:.1f}"
            if would_reject
            else None
        )
        # Candidate policy: never reject. Do not substitute SeamVis here.
        if telemetry or not would_reject:
            return GateDecision(
                name="ghost",
                accept=True,
                reason=reason if telemetry else None,
                log_line=log_line,
                scores=scores,
                status=telemetry,
            )
        return GateDecision(
            name="ghost",
            accept=False,
            reason=reason,
            runtime_message=(
                f"Ghosting gate (siqe): asp_ghost={asp_g:.1f}, ratio={ratio:.2f}"
            ),
            log_line=log_line,
            fail_log_line=(
                f"  [GhostGate/siqe] FAILED "
                f"(asp={asp_g:.1f} > limit={limit:.1f} "
                f"[floor={self.ghost_floor:.0f}, {self.ghost_ratio:.1f}× "
                f"sim={sim_g:.1f}]) → SCANS fallback."
            ),
            scores=scores,
            fallback_code=1,
        )

    def evaluate_seam_vis(
        self,
        asp_img: np.ndarray,
        scans_img: np.ndarray | None,
    ) -> GateDecision:
        if scans_img is None or self.seam_vis_ratio >= 90:
            return GateDecision(name="seam_vis", accept=True, skipped=True)
        asp_sv = seam_visibility_score(asp_img)
        sim_sv = seam_visibility_score(scans_img)
        ratio = asp_sv / max(sim_sv, 1.0)
        limit = max(self.seam_vis_floor, self.seam_vis_ratio * max(sim_sv, 1.0))
        log_line = (
            f"  [SeamVisGate] asp_sv={asp_sv:.1f}  "
            f"sim_sv={sim_sv:.1f}  "
            f"ratio={ratio:.2f}"
        )
        if asp_sv <= limit:
            return GateDecision(
                name="seam_vis",
                accept=True,
                log_line=log_line,
                scores={
                    "asp_sv": asp_sv,
                    "sim_sv": sim_sv,
                    "limit": limit,
                },
            )
        reason = (
            f"seam_vis_gate:asp={asp_sv:.1f}_sim={sim_sv:.1f}_limit={limit:.1f}"
        )
        return GateDecision(
            name="seam_vis",
            accept=False,
            reason=reason,
            runtime_message=(
                f"SeamVis gate: asp_sv={asp_sv:.1f}, ratio={ratio:.2f}"
            ),
            log_line=log_line,
            fail_log_line=(
                f"  [SeamVisGate] FAILED "
                f"(asp={asp_sv:.1f} > limit={limit:.1f} "
                f"[floor={self.seam_vis_floor:.0f}, "
                f"{self.seam_vis_ratio:.1f}× sim={sim_sv:.1f}]) "
                f"→ SCANS fallback."
            ),
            scores={
                "asp_sv": asp_sv,
                "sim_sv": sim_sv,
                "limit": limit,
            },
            fallback_code=2,
        )

    def apply_to_session(
        self,
        session: PipelineSession | None,
        decision: GateDecision,
    ) -> None:
        if session is None or decision.accept:
            return
        session.record_fallback(
            ResultIdentity.SAFE_ASP,
            decision.reason or decision.name,
            gate=decision.name,
            algorithm="scans",
        )
        session.record_artifact("safe_asp_selected", "scans")

    def evaluate_registration_risk(self, telemetry: dict[str, Any] | None, affine_health: dict[str, Any] | None = None, crop_coverage: float | None = None) -> GateDecision:
        from .registration_gate import RegistrationRiskGate
        return RegistrationRiskGate().evaluate(telemetry, affine_health, crop_coverage)

    def evaluate_all(
        self,
        asp_img: np.ndarray,
        scans_img: np.ndarray | None,
        affines: list[np.ndarray] | None,
        *, telemetry: dict[str, Any] | None = None,
        affine_health: dict[str, Any] | None = None,
        crop_coverage: float | None = None,
    ) -> list[GateDecision]:
        decisions = [
            self.evaluate_composite(asp_img, scans_img, affines),
            self.evaluate_ghost(asp_img, scans_img),
            self.evaluate_seam_vis(asp_img, scans_img),
        ]
        if self.registration_gate_enabled and telemetry is not None:
            decisions.insert(0, self.evaluate_registration_risk(telemetry, affine_health, crop_coverage))
        return decisions

    def snapshot(self) -> dict[str, float | str]:
        return {
            "composite_sc_floor": self.composite_sc_floor,
            "composite_sb_floor": self.composite_sb_floor,
            "composite_scans_mult": self.composite_scans_mult,
            "ghost_ratio": self.ghost_ratio,
            "ghost_floor": self.ghost_floor,
            "seam_vis_ratio": self.seam_vis_ratio,
            "seam_vis_floor": self.seam_vis_floor,
            "registration_gate_enabled": 1.0 if self.registration_gate_enabled else 0.0,
            "uncertain_result_policy": self.uncertain_result_policy,
            "ghost_telemetry_only": 1.0 if self.ghost_telemetry_only else 0.0,
            "composite_sb_telemetry_only": (
                1.0 if self.composite_sb_telemetry_only else 0.0
            ),
            "composite_sc_telemetry_only": (
                1.0 if self.composite_sc_telemetry_only else 0.0
            ),
        }


def product_safe_asp_policy() -> SafeAspPolicy:
    """Frozen shipped defaults. Ignores inherited ``ASP_GATE_*`` env.

    ``ghost_telemetry_only=True`` (2026-08-17, Harbinger ACK): the
    promotion-ladder replay (five-case / structural red set / all 97,
    `.agent/reports/grok/m2_ghostgate_telemetry_screen_20260817.md`) showed
    zero Safe ASP identity changes in either direction, because GhostGate's
    `ghosting_score_v2` signal never actually rejects on this corpus — there
    is no historic GhostGate-only fallback to regress. Its score is still
    recorded (``GateDecision.status == "telemetry_only_inverse_validated"``)
    but no longer drives selection. CompositeGate's `sb`/`sc` telemetry-only
    candidates remain default-off/rejecting — unlike GhostGate, demoting `sb`
    was shown to change 26 historic identities with no raw-composite ground
    truth to confirm the flip is safe (same report).
    """
    return SafeAspPolicy(ghost_telemetry_only=True)


def default_benchmark_policy() -> SafeAspPolicy:
    return SafeAspPolicy.from_environ()


def safe_asp_counterfactual(
    decisions: list[GateDecision],
    policy: SafeAspPolicy,
    *,
    raw_available: bool,
    scans_available: bool,
) -> dict[str, Any]:
    """Typed per-case Safe ASP what-if, independent of what was published."""
    first_reject = next((d for d in decisions if not d.accept), None)
    uncertain = next((d for d in decisions if d.status == "uncertain"), None)
    if not raw_available:
        would_select = None
        unavailable = "raw_unavailable"
    elif not scans_available:
        would_select = "raw_asp"
        unavailable = "no_scans"
    elif first_reject is not None:
        would_select = "scans"
        unavailable = None
    elif uncertain is not None:
        would_select = policy.uncertain_result_policy
        unavailable = None
    else:
        would_select = "raw_asp"
        unavailable = None
    return {
        "would_select": would_select,
        "gate": None if first_reject is None else first_reject.name,
        "reason": None if first_reject is None else first_reject.reason,
        "unavailable": unavailable,
        "policy": policy.snapshot(),
        "decisions": [
            {
                "name": d.name,
                "accept": d.accept,
                "skipped": d.skipped,
                "reason": d.reason,
                "status": d.status,
                "scores": dict(d.scores),
            }
            for d in decisions
        ],
    }


__all__ = [
    "GateDecision",
    "SafeAspPolicy",
    "default_benchmark_policy",
    "product_safe_asp_policy",
    "safe_asp_counterfactual",
]
