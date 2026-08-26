from asp_backend.core.pipeline.registration_gate import RegistrationRiskGate, RiskLevel


def test_low_risk_and_uncertain_review_states():
    gate = RegistrationRiskGate()
    low = gate.evaluate({"raw_edges": 20, "filtered_edges": 18, "ba_residual_rms": 10.0, "cycle_error_rms": 20.0})
    uncertain = gate.evaluate({"raw_edges": 20, "filtered_edges": 18, "ba_residual_rms": 60.0, "cycle_error_rms": 20.0})
    assert low.status == RiskLevel.LOW_RISK.value
    assert uncertain.accept and uncertain.status == RiskLevel.UNCERTAIN.value


def test_hard_failure_and_min_gap_deferral():
    gate = RegistrationRiskGate()
    rejected = gate.evaluate({"raw_edges": 20, "ba_residual_rms": 90.0})
    deferred = gate.evaluate({"raw_edges": 20, "ba_residual_rms": 10.0}, {"valid": False, "reason": "min_gap=9 < 20"})
    assert not rejected.accept and rejected.status == RiskLevel.HIGH_RISK.value
    assert deferred.accept and deferred.status == RiskLevel.UNCERTAIN.value
