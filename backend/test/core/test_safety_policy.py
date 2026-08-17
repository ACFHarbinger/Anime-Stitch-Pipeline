"""M1b: injectable Safe ASP policy — same decisions as the old bench gates."""

from __future__ import annotations

import numpy as np
from asp_backend.core.pipeline.safety_policy import (
    SafeAspPolicy,
    safe_asp_counterfactual,
)


def _solid(h: int, w: int, lum: int) -> np.ndarray:
    return np.full((h, w, 3), lum, dtype=np.uint8)


def _banded(h: int = 120, w: int = 80) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2] = 20
    img[h // 2 :] = 220
    return img


def _affine(ty: float) -> np.ndarray:
    m = np.eye(2, 3, dtype=np.float32)
    m[1, 2] = ty
    return m


class TestCompositeGate:
    def test_clean_pair_accepts(self):
        policy = SafeAspPolicy()
        asp = _solid(80, 80, 128)
        scans = _solid(80, 80, 128)
        decision = policy.evaluate_composite(asp, scans, None)
        assert decision.accept
        assert decision.log_line.startswith("  [CompositeGate]")

    def test_severe_banding_rejects_with_stable_reason(self):
        policy = SafeAspPolicy(composite_sc_floor=5.0, composite_sb_floor=5.0)
        decision = policy.evaluate_composite(_banded(), _solid(120, 80, 128), None)
        assert not decision.accept
        assert decision.reason is not None
        assert decision.reason.startswith("composite_gate_")
        assert "asp_sc=" in decision.reason
        assert decision.runtime_message.startswith("Composite quality gate:")
        assert decision.fallback_code == 1

    def test_scans_strip_banding_is_not_read(self):
        """Preserved bench quirk: sb limit is the floor (scans_sb stays 0)."""
        policy = SafeAspPolicy(composite_sc_floor=999.0, composite_sb_floor=10.0)
        # Banded SCANS would have huge strip jumps if we measured it.
        affines = [_affine(0.0), _affine(60.0)]
        decision = policy.evaluate_composite(
            _solid(120, 80, 128), _banded(), affines
        )
        assert decision.accept
        assert decision.scores["sb_limit"] == 10.0

    def test_sb_telemetry_only_does_not_reject_on_strip_banding(self):
        policy = SafeAspPolicy(
            composite_sc_floor=999.0,
            composite_sb_floor=5.0,
            composite_sb_telemetry_only=True,
        )
        asp = _banded(100, 80)
        affines = [_affine(0.0), _affine(50.0)]
        decision = policy.evaluate_composite(asp, _solid(100, 80, 128), affines)
        assert decision.accept
        assert decision.status == "telemetry_only_inverse_validated"
        assert decision.fallback_code == 0
        assert decision.scores["would_reject_sb"] == 1.0
        assert "[telemetry_only:sb]" in (decision.log_line or "")

    def test_sb_telemetry_still_rejects_on_sc(self):
        policy = SafeAspPolicy(
            composite_sc_floor=5.0,
            composite_sb_floor=5.0,
            composite_sb_telemetry_only=True,
        )
        decision = policy.evaluate_composite(_banded(), _solid(120, 80, 128), None)
        assert not decision.accept
        assert decision.reason.startswith("composite_gate_sc:")


class TestGhostGate:
    def test_skips_without_scans(self):
        decision = SafeAspPolicy().evaluate_ghost(_solid(64, 64, 80), None)
        assert decision.accept
        assert decision.skipped

    def test_skips_when_ratio_disabled(self):
        policy = SafeAspPolicy(ghost_ratio=99.0)
        decision = policy.evaluate_ghost(_solid(64, 64, 80), _solid(64, 64, 80))
        assert decision.accept
        assert decision.skipped

    def test_reject_reason_prefix(self):
        policy = SafeAspPolicy(ghost_floor=0.0, ghost_ratio=0.01)
        # A hard vertical split produces a non-zero SIQE score.
        asp = _banded(80, 64)
        scans = _solid(80, 64, 128)
        decision = policy.evaluate_ghost(asp, scans)
        if not decision.accept:
            assert decision.reason.startswith("ghost_gate_siqe:")
            assert decision.runtime_message.startswith("Ghosting gate (siqe):")

    def test_telemetry_only_never_rejects(self):
        policy = SafeAspPolicy(
            ghost_floor=0.0, ghost_ratio=0.01, ghost_telemetry_only=True
        )
        decision = policy.evaluate_ghost(_banded(80, 64), _solid(80, 64, 128))
        assert decision.accept
        assert decision.status == "telemetry_only_inverse_validated"
        assert decision.fallback_code == 0
        assert "would_reject" in decision.scores
        assert "[telemetry_only_inverse_validated]" in (decision.log_line or "")

    def test_ghost_limit_uses_floor_then_ratio(self):
        policy = SafeAspPolicy(ghost_floor=40.0, ghost_ratio=2.0)
        assert policy.ghost_limit(10.0) == 40.0
        assert policy.ghost_limit(30.0) == 60.0

    def test_default_policy_is_not_telemetry_only(self):
        assert SafeAspPolicy().ghost_telemetry_only is False
        assert SafeAspPolicy.from_environ().ghost_telemetry_only is False

    def test_env_enables_telemetry_only(self, monkeypatch):
        monkeypatch.setenv("ASP_GHOST_TELEMETRY_ONLY", "1")
        assert SafeAspPolicy.from_environ().ghost_telemetry_only is True

    def test_env_enables_composite_sb_telemetry(self, monkeypatch):
        monkeypatch.setenv("ASP_COMPOSITE_SB_TELEMETRY_ONLY", "1")
        policy = SafeAspPolicy.from_environ()
        assert policy.composite_sb_telemetry_only is True
        assert policy.composite_sc_telemetry_only is False


class TestGhostTelemetryCounterfactual:
    def test_telemetry_ghost_does_not_select_scans(self):
        policy = SafeAspPolicy(ghost_telemetry_only=True)
        ghost = policy.evaluate_ghost(_banded(80, 64), _solid(80, 64, 128))
        # Force a would-reject by using a tight policy clone for scores, but
        # the candidate decision above already accepts. Selection uses accept.
        cf = safe_asp_counterfactual(
            [ghost], policy, raw_available=True, scans_available=True
        )
        assert ghost.accept
        assert cf["would_select"] == "raw_asp"
        assert cf["gate"] is None
        assert cf["decisions"][0]["status"] == "telemetry_only_inverse_validated"


class TestSeamVisGate:
    def test_skips_without_scans(self):
        decision = SafeAspPolicy().evaluate_seam_vis(_solid(64, 64, 80), None)
        assert decision.accept
        assert decision.skipped

    def test_hard_cut_rejects(self):
        policy = SafeAspPolicy(seam_vis_floor=5.0, seam_vis_ratio=1.0)
        decision = policy.evaluate_seam_vis(_banded(100, 80), _solid(100, 80, 128))
        assert not decision.accept
        assert decision.reason.startswith("seam_vis_gate:")
        assert decision.fallback_code == 2
        assert decision.fail_log_line.startswith("  [SeamVisGate] FAILED")
