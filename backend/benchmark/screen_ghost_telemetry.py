"""Offline promotion-ladder screen for GhostGate telemetry-only.

Replays Composite / Ghost / SeamVis on a saved run JSON. Does not stitch and
does not flip defaults. Historic 2026-08-07 has **zero** GhostGate rejects
(max asp/limit = 0.70), so the candidate cannot change Safe ASP selection on
that corpus.

Five-case screen includes:
  asp_test04 / 08 / 27 — standard smoke / hard cases
  asp_test38 — closest true-composite to a GhostGate fire (ratio 0.668)
  asp_test96 — known-good true composite (human 3 vs 1)

There is no historic GhostGate-only fallback in the 97-case set to include.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FIVE_CASE_SCREEN = (
    "asp_test04",
    "asp_test08",
    "asp_test27",
    "asp_test38",
    "asp_test96",
)
STRUCTURAL_RED_SET = (
    "asp_test04",
    "asp_test06",
    "asp_test07",
    "asp_test12",
    "asp_test14",
    "asp_test15",
    "asp_test96",
)

GHOST_FLOOR = 40.0
GHOST_RATIO = 2.0
SC_FLOOR = 38.0
SB_FLOOR = 35.0
SC_MULT = 2.0
SV_FLOOR = 35.0
SV_RATIO = 3.0


def _ghost_fail(ma: dict[str, Any], ms: dict[str, Any]) -> bool:
    asp_g, sim_g = ma.get("ghosting_siqe"), ms.get("ghosting_siqe")
    if asp_g is None or sim_g is None:
        return False
    return float(asp_g) > max(GHOST_FLOOR, GHOST_RATIO * max(float(sim_g), 1.0))


def _composite_fail(ds: dict[str, Any], ma: dict[str, Any], ms: dict[str, Any]) -> bool:
    reason = ds.get("fallback_reason") or ""
    if reason.startswith("composite_gate"):
        return True
    asp_sc, sim_sc = ma.get("seam_coherence"), ms.get("seam_coherence")
    if asp_sc is None:
        return False
    limit = max(SC_FLOOR, float(sim_sc or 0.0) * SC_MULT)
    asp_sb = ma.get("strip_banding_score")
    sc_fail = float(asp_sc) > limit
    sb_fail = asp_sb is not None and float(asp_sb) > SB_FLOOR
    return sc_fail or sb_fail


def _seam_fail(ds: dict[str, Any], ma: dict[str, Any], ms: dict[str, Any]) -> bool:
    reason = ds.get("fallback_reason") or ""
    if reason.startswith("seam_vis_gate"):
        return True
    asp_sv, sim_sv = ma.get("seam_visibility"), ms.get("seam_visibility")
    if asp_sv is None or sim_sv is None:
        return False
    return float(asp_sv) > max(SV_FLOOR, SV_RATIO * max(float(sim_sv), 1.0))


def first_reject(
    ds: dict[str, Any], *, ghost_telemetry_only: bool
) -> str | None:
    ma, ms = ds.get("metrics_asp") or {}, ds.get("metrics_simple") or {}
    if _composite_fail(ds, ma, ms):
        return "composite"
    if not ghost_telemetry_only and _ghost_fail(ma, ms):
        return "ghost"
    if _seam_fail(ds, ma, ms):
        return "seam_vis"
    return None


def screen(
    run_path: Path,
    names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    doc = json.loads(run_path.read_text(encoding="utf-8"))
    want = set(names) if names is not None else None
    rows = []
    ghost_only = 0
    changed = 0
    for ds in doc.get("datasets") or []:
        name = ds.get("name")
        if want is not None and name not in want:
            continue
        base = first_reject(ds, ghost_telemetry_only=False)
        cand = first_reject(ds, ghost_telemetry_only=True)
        ghost = _ghost_fail(ds.get("metrics_asp") or {}, ds.get("metrics_simple") or {})
        if ghost and base == "ghost":
            ghost_only += 1
        base_sel = "scans" if base else "raw_asp"
        cand_sel = "scans" if cand else "raw_asp"
        identity_changed = base_sel != cand_sel
        if identity_changed:
            changed += 1
        rows.append(
            {
                "name": name,
                "baseline_gate": base,
                "candidate_gate": cand,
                "baseline_select": base_sel,
                "candidate_select": cand_sel,
                "ghost_would_reject": ghost,
                "used_fallback": bool(ds.get("used_fallback")),
                "changed": identity_changed,
            }
        )
    return {
        "n": len(rows),
        "selection_changes": changed,
        "historic_ghost_only_fallbacks": ghost_only,
        "rows": rows,
    }


def _print(result: dict[str, Any], title: str) -> None:
    print(f"{title}: n={result['n']} changes={result['selection_changes']} "
          f"historic_ghost_only={result['historic_ghost_only_fallbacks']}")
    for row in result["rows"]:
        mark = " CHANGED" if row["changed"] else ""
        print(
            f"  {row['name']:14s} base={str(row['baseline_gate']):10s} "
            f"cand={str(row['candidate_gate']):10s} "
            f"ghost_would_reject={row['ghost_would_reject']}{mark}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument(
        "--set",
        choices=("five", "red", "all"),
        default="five",
    )
    args = ap.parse_args()
    names = {
        "five": FIVE_CASE_SCREEN,
        "red": STRUCTURAL_RED_SET,
        "all": None,
    }[args.set]
    result = screen(args.run, names)
    _print(result, f"GhostGate telemetry-only screen ({args.set})")
    if result["selection_changes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
