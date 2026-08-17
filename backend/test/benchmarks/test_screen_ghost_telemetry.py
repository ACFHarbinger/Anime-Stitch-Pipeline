"""Offline GhostGate telemetry-only screen does not change Safe ASP identity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SRC = (
    Path(__file__).resolve().parents[3]
    / "backend"
    / "benchmark"
    / "screen_ghost_telemetry.py"
)
_spec = importlib.util.spec_from_file_location("screen_ghost_telemetry", _SRC)
assert _spec and _spec.loader
screen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(screen)


def _ds(name, *, asp_g, sim_g, reason=None, fallback=False, **metrics):
    ma = {"ghosting_siqe": asp_g, "seam_coherence": 10.0, "seam_visibility": 4.0}
    ms = {"ghosting_siqe": sim_g, "seam_coherence": 10.0, "seam_visibility": 3.0}
    ma.update(metrics.get("ma", {}))
    return {
        "name": name,
        "used_fallback": fallback,
        "fallback_reason": reason,
        "metrics_asp": ma,
        "metrics_simple": ms,
    }


def test_zero_identity_changes_when_ghost_never_fires(tmp_path: Path):
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {
                "datasets": [
                    _ds("asp_test04", asp_g=20, sim_g=20),
                    _ds(
                        "asp_test08",
                        asp_g=80,
                        sim_g=70,
                        reason="seam_vis_gate:asp=40_sim=2_limit=35",
                        fallback=True,
                    ),
                    _ds("asp_test96", asp_g=59, sim_g=80),
                ]
            }
        )
    )
    result = screen.screen(run)
    assert result["selection_changes"] == 0
    assert result["historic_ghost_only_fallbacks"] == 0


def test_ghost_only_reject_keeps_scans_under_candidate(tmp_path: Path):
    """If GhostGate were first reject, telemetry-only may keep Raw ASP."""
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {
                "datasets": [
                    _ds("asp_test99", asp_g=90, sim_g=10),  # 90 > max(40, 20)
                ]
            }
        )
    )
    result = screen.screen(run, names=None)
    assert result["historic_ghost_only_fallbacks"] == 1
    assert result["selection_changes"] == 1
    assert result["rows"][0]["baseline_select"] == "scans"
    assert result["rows"][0]["candidate_select"] == "raw_asp"
