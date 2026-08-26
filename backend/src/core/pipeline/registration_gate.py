"""Default-off M2 registration-risk classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .safety_policy import GateDecision


class RiskLevel(str, Enum):
    LOW_RISK = "low_risk"
    UNCERTAIN = "uncertain"
    HIGH_RISK = "high_risk"


@dataclass(frozen=True)
class RegistrationThresholds:
    max_ba_residual_rms: float = 80.0
    max_cycle_error_rms: float = 300.0
    min_raw_edges: int = 10
    uncertain_ba_residual_rms: float = 45.0
    uncertain_cycle_error_rms: float = 150.0


class RegistrationRiskGate:
    """Classify calibrated BA/cycle/edge evidence without rendering changes."""
    def __init__(self, thresholds: RegistrationThresholds | None = None) -> None:
        self.thresholds = thresholds or RegistrationThresholds()

    def evaluate(self, telemetry: dict[str, Any] | None, affine_health: dict[str, Any] | None = None, crop_coverage: float | None = None) -> GateDecision:
        telemetry = telemetry or {}
        matching, alignment = telemetry.get("matching", {}), telemetry.get("alignment", {})
        raw = telemetry.get("raw_edges", matching.get("raw_edges", 0))
        filtered = telemetry.get("filtered_edges", matching.get("filtered_edges", 0))
        ba = telemetry.get("ba_residual_rms", alignment.get("ba_residual_rms"))
        cycle = telemetry.get("cycle_error_rms", alignment.get("cycle_error_rms"))
        scores = {"raw_edges": float(raw or 0), "filtered_edges": float(filtered or 0), "ba_residual_rms": float(ba) if ba is not None else -1.0, "cycle_error_rms": float(cycle) if cycle is not None else -1.0, "crop_coverage": float(crop_coverage) if crop_coverage is not None else 1.0}
        def reject(reason: str) -> GateDecision:
            return GateDecision("registration_risk", False, reason=reason, runtime_message=f"Registration risk gate FAILED ({reason})", scores=scores, status=RiskLevel.HIGH_RISK.value, fallback_code=3)
        reason = str((affine_health or {}).get("reason") or "")
        deferred_min_gap = affine_health is not None and not affine_health.get("valid", True) and reason.startswith("min_gap=")
        if affine_health is not None and not affine_health.get("valid", True) and not deferred_min_gap:
            return reject("registration_gate:affine_health_invalid")
        if raw <= self.thresholds.min_raw_edges:
            return reject(f"registration_gate:insufficient_edges(raw={raw}<=min={self.thresholds.min_raw_edges})")
        if ba is None or ba < 0:
            return reject("registration_gate:missing_ba_residual")
        if ba > self.thresholds.max_ba_residual_rms:
            return reject(f"registration_gate:ba_rms_exceeded({ba:.1f}>{self.thresholds.max_ba_residual_rms:.1f})")
        if cycle is not None and cycle > self.thresholds.max_cycle_error_rms:
            return reject(f"registration_gate:cycle_error_exceeded({cycle:.1f}>{self.thresholds.max_cycle_error_rms:.1f})")
        marginal = deferred_min_gap or ba > self.thresholds.uncertain_ba_residual_rms or (cycle is not None and cycle > self.thresholds.uncertain_cycle_error_rms)
        if marginal:
            return GateDecision("registration_risk", True, reason="registration_gate:uncertain", scores=scores, status=RiskLevel.UNCERTAIN.value)
        return GateDecision("registration_risk", True, scores=scores, status=RiskLevel.LOW_RISK.value)


__all__ = ["RegistrationRiskGate", "RegistrationThresholds", "RiskLevel"]
