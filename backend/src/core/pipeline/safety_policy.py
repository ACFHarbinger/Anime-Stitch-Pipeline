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

    @classmethod
    def from_environ(cls) -> SafeAspPolicy:
        """Same env knobs the benchmark already documents."""
        return cls(
            composite_sc_floor=_env_float("ASP_GATE_SC", 38.0),
            composite_sb_floor=_env_float("ASP_GATE_SB", 35.0),
            ghost_ratio=_env_float("ASP_GATE_GHOST", 2.0),
            ghost_floor=_env_float("ASP_GATE_GHOST_FLOOR", 40.0),
            seam_vis_ratio=_env_float("ASP_GATE_SEAM_VIS", 3.0),
            seam_vis_floor=_env_float("ASP_GATE_SEAM_VIS_FLOOR", 35.0),
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
        failed = asp_sc > sc_limit or asp_sb > sb_limit
        if not failed:
            return GateDecision(
                name="composite",
                accept=True,
                log_line=log_line,
                scores={
                    "asp_sc": asp_sc,
                    "asp_sb": asp_sb,
                    "scans_sc": scans_sc,
                    "sc_limit": sc_limit,
                    "sb_limit": sb_limit,
                },
            )
        which = "sc" if asp_sc > sc_limit else "sb"
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
            scores={
                "asp_sc": asp_sc,
                "asp_sb": asp_sb,
                "scans_sc": scans_sc,
                "sc_limit": sc_limit,
                "sb_limit": sb_limit,
            },
            fallback_code=1,
        )

    def evaluate_ghost(
        self,
        asp_img: np.ndarray,
        scans_img: np.ndarray | None,
    ) -> GateDecision:
        if scans_img is None or self.ghost_ratio >= 90:
            return GateDecision(name="ghost", accept=True, skipped=True)
        asp_g = ghosting_score_v2(asp_img)
        sim_g = ghosting_score_v2(scans_img)
        ratio = asp_g / max(sim_g, 1.0)
        limit = max(self.ghost_floor, self.ghost_ratio * max(sim_g, 1.0))
        log_line = (
            f"  [GhostGate/siqe] asp_ghost={asp_g:.1f}  "
            f"sim_ghost={sim_g:.1f}  "
            f"ratio={ratio:.2f}"
        )
        if asp_g <= limit:
            return GateDecision(
                name="ghost",
                accept=True,
                log_line=log_line,
                scores={
                    "asp_ghost": asp_g,
                    "sim_ghost": sim_g,
                    "limit": limit,
                },
            )
        reason = (
            f"ghost_gate_siqe:asp={asp_g:.1f}_sim={sim_g:.1f}_limit={limit:.1f}"
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
            scores={
                "asp_ghost": asp_g,
                "sim_ghost": sim_g,
                "limit": limit,
            },
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


def default_benchmark_policy() -> SafeAspPolicy:
    return SafeAspPolicy.from_environ()


__all__ = [
    "GateDecision",
    "SafeAspPolicy",
    "default_benchmark_policy",
]
