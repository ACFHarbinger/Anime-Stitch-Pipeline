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


def _composite_parts(
    ds: dict[str, Any], ma: dict[str, Any], ms: dict[str, Any]
) -> tuple[bool, bool]:
    """Return (sc_fail, sb_fail) from recorded reason or saved metrics."""
    reason = ds.get("fallback_reason") or ""
    sc_fail = reason.startswith("composite_gate_sc")
    sb_fail = reason.startswith("composite_gate_sb")
    asp_sc, sim_sc = ma.get("seam_coherence"), ms.get("seam_coherence")
    if asp_sc is not None:
        limit = max(SC_FLOOR, float(sim_sc or 0.0) * SC_MULT)
        sc_fail = sc_fail or float(asp_sc) > limit
    asp_sb = ma.get("strip_banding_score")
    if asp_sb is not None:
        sb_fail = sb_fail or float(asp_sb) > SB_FLOOR
    return sc_fail, sb_fail


def _composite_fail(
    ds: dict[str, Any],
    ma: dict[str, Any],
    ms: dict[str, Any],
    *,
    sb_telemetry_only: bool = False,
    sc_telemetry_only: bool = False,
) -> bool:
    sc_fail, sb_fail = _composite_parts(ds, ma, ms)
    return (sc_fail and not sc_telemetry_only) or (sb_fail and not sb_telemetry_only)


def _seam_fail(ds: dict[str, Any], ma: dict[str, Any], ms: dict[str, Any]) -> bool:
    reason = ds.get("fallback_reason") or ""
    if reason.startswith("seam_vis_gate"):
        return True
    asp_sv, sim_sv = ma.get("seam_visibility"), ms.get("seam_visibility")
    if asp_sv is None or sim_sv is None:
        return False
    return float(asp_sv) > max(SV_FLOOR, SV_RATIO * max(float(sim_sv), 1.0))


def first_reject(
    ds: dict[str, Any],
    *,
    ghost_telemetry_only: bool,
    sb_telemetry_only: bool = False,
    sc_telemetry_only: bool = False,
) -> str | None:
    ma, ms = ds.get("metrics_asp") or {}, ds.get("metrics_simple") or {}
    if _composite_fail(
        ds,
        ma,
        ms,
        sb_telemetry_only=sb_telemetry_only,
        sc_telemetry_only=sc_telemetry_only,
    ):
        return "composite"
    if not ghost_telemetry_only and _ghost_fail(ma, ms):
        return "ghost"
    if _seam_fail(ds, ma, ms):
        return "seam_vis"
    return None


KNOWN_GOOD = ("asp_test96",)
CATASTROPHES = (
    "asp_test04",
    "asp_test06",
    "asp_test07",
    "asp_test12",
    "asp_test14",
    "asp_test15",
)


def screen(
    run_path: Path,
    names: tuple[str, ...] | None = None,
    *,
    ghost_telemetry_only: bool = True,
    sb_telemetry_only: bool = False,
    sc_telemetry_only: bool = False,
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
        base = first_reject(
            ds,
            ghost_telemetry_only=False,
            sb_telemetry_only=False,
            sc_telemetry_only=False,
        )
        cand = first_reject(
            ds,
            ghost_telemetry_only=ghost_telemetry_only,
            sb_telemetry_only=sb_telemetry_only,
            sc_telemetry_only=sc_telemetry_only,
        )
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


def discriminating(run_path: Path, *, candidate: dict[str, bool] | None = None) -> dict[str, Any]:
    """M2 exit: Raw ASP on a known-good AND SCANS on known catastrophes."""
    flags = candidate or {
        "ghost_telemetry_only": True,
        "sb_telemetry_only": True,
        "sc_telemetry_only": False,
    }
    result = screen(run_path, STRUCTURAL_RED_SET, **flags)
    by_name = {row["name"]: row for row in result["rows"]}
    good = [by_name[n] for n in KNOWN_GOOD if n in by_name]
    cats = [by_name[n] for n in CATASTROPHES if n in by_name]
    good_raw = [row for row in good if row["candidate_select"] == "raw_asp"]
    cat_scans = [row for row in cats if row["candidate_select"] == "scans"]
    return {
        "known_good_raw": [row["name"] for row in good_raw],
        "catastrophe_scans": [row["name"] for row in cat_scans],
        "catastrophe_raw": [
            row["name"] for row in cats if row["candidate_select"] == "raw_asp"
        ],
        "passes": bool(good_raw) and len(cat_scans) == len(cats) and bool(cats),
        "rows": result["rows"],
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
    ap.add_argument("--sb-telemetry", action="store_true")
    ap.add_argument("--sc-telemetry", action="store_true")
    ap.add_argument("--no-ghost-telemetry", action="store_true")
    ap.add_argument(
        "--discriminating",
        action="store_true",
        help="check M2 exit: known-good Raw ASP + catastrophe SCANS",
    )
    args = ap.parse_args()
    flags = {
        "ghost_telemetry_only": not args.no_ghost_telemetry,
        "sb_telemetry_only": args.sb_telemetry,
        "sc_telemetry_only": args.sc_telemetry,
    }
    if args.discriminating:
        verdict = discriminating(args.run, candidate=flags)
        print(
            f"discriminating: pass={verdict['passes']} "
            f"known_good_raw={verdict['known_good_raw']} "
            f"catastrophe_scans={verdict['catastrophe_scans']} "
            f"catastrophe_raw={verdict['catastrophe_raw']}"
        )
        for row in verdict["rows"]:
            print(
                f"  {row['name']:14s} cand={row['candidate_select']:8s} "
                f"gate={row['candidate_gate']}"
            )
        raise SystemExit(0 if verdict["passes"] else 3)
    names = {
        "five": FIVE_CASE_SCREEN,
        "red": STRUCTURAL_RED_SET,
        "all": None,
    }[args.set]
    result = screen(args.run, names, **flags)
    _print(result, f"Safe ASP candidate screen ({args.set})")
    if result["selection_changes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
