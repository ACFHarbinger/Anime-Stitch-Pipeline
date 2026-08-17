"""SeamVis threshold sweep: discriminating feasibility."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SRC = (
    Path(__file__).resolve().parents[3]
    / "backend"
    / "benchmark"
    / "screen_seamvis_threshold.py"
)
_spec = importlib.util.spec_from_file_location("screen_seamvis_threshold", _SRC)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_infeasible_when_a_catastrophe_is_below_known_good_and_ratio_conflicts():
    # Mirrors 14/15 (sv~12) vs 96 (sv 32): catching the low-sv catastrophe
    # requires a low floor/ratio that also rejects the known-good.
    rows = [
        {"name": "asp_test15", "asp_sv": 12.55, "sim_sv": 3.07},
        {"name": "asp_test96", "asp_sv": 32.2, "sim_sv": 4.89},
    ]
    result = mod.sweep(rows, cats=("asp_test15",), goods=("asp_test96",))
    assert result["known_good_sv_above_all_cats"] is True
    assert result["n_feasible"] == 0


def test_feasible_when_cats_outrank_known_good():
    rows = [
        {"name": "asp_test06", "asp_sv": 40.0, "sim_sv": 2.0},
        {"name": "asp_test96", "asp_sv": 10.0, "sim_sv": 5.0},
    ]
    result = mod.sweep(rows, cats=("asp_test06",), goods=("asp_test96",))
    assert result["n_feasible"] > 0
