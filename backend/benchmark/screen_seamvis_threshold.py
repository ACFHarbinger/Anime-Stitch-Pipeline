"""Can SeamVis be retuned to pass the M2 discriminating exit?

Sweeps (floor, ratio) on a saved run JSON. A pair is feasible only if it
rejects every catastrophe and keeps every known-good.

On the 2026-08-07 corpus this is impossible: known-good asp_test96 has
higher ``seam_visibility`` (32.2) than every catastrophe (max 29.58).
Any threshold that catches the red-set failures also rejects the known-good.

Usage::

    .venv/bin/python backend/benchmark/screen_seamvis_threshold.py \\
        --run backend/benchmark/output/anime_stitch_20260807_045552.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CATASTROPHES = (
    "asp_test04",
    "asp_test06",
    "asp_test07",
    "asp_test12",
    "asp_test14",
    "asp_test15",
)
KNOWN_GOOD = ("asp_test96",)

DEFAULT_FLOORS = [i * 0.5 for i in range(0, 81)]
DEFAULT_RATIOS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def _rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for ds in doc.get("datasets") or []:
        ma, ms = ds.get("metrics_asp") or {}, ds.get("metrics_simple") or {}
        asp_sv, sim_sv = ma.get("seam_visibility"), ms.get("seam_visibility")
        if asp_sv is None or sim_sv is None:
            continue
        out.append(
            {
                "name": ds["name"],
                "asp_sv": float(asp_sv),
                "sim_sv": float(sim_sv),
            }
        )
    return out


def fires(row: dict[str, Any], floor: float, ratio: float) -> bool:
    return row["asp_sv"] > max(floor, ratio * max(row["sim_sv"], 1.0))


def sweep(
    rows: list[dict[str, Any]],
    *,
    cats: tuple[str, ...] = CATASTROPHES,
    goods: tuple[str, ...] = KNOWN_GOOD,
    floors: list[float] | None = None,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
) -> dict[str, Any]:
    by_name = {r["name"]: r for r in rows}
    cat_rows = [by_name[n] for n in cats if n in by_name]
    good_rows = [by_name[n] for n in goods if n in by_name]
    floors = floors if floors is not None else DEFAULT_FLOORS
    feasible: list[tuple[float, float]] = []
    for floor in floors:
        for ratio in ratios:
            if cat_rows and all(fires(r, floor, ratio) for r in cat_rows):
                if good_rows and all(not fires(r, floor, ratio) for r in good_rows):
                    feasible.append((floor, ratio))
    cat_max = max((r["asp_sv"] for r in cat_rows), default=None)
    good_min = min((r["asp_sv"] for r in good_rows), default=None)
    return {
        "n_catastrophes": len(cat_rows),
        "n_known_good": len(good_rows),
        "catastrophe_max_sv": cat_max,
        "known_good_min_sv": good_min,
        "known_good_sv_above_all_cats": (
            cat_max is not None and good_min is not None and good_min > cat_max
        ),
        "feasible_pairs": feasible,
        "n_feasible": len(feasible),
        "red_set": [
            {
                "name": r["name"],
                "asp_sv": r["asp_sv"],
                "sim_sv": r["sim_sv"],
                "role": "catastrophe" if r["name"] in cats else "known_good",
            }
            for r in cat_rows + good_rows
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    doc = json.loads(args.run.read_text(encoding="utf-8"))
    result = sweep(_rows(doc))
    print("SeamVis threshold sweep (catch all catastrophes, keep known-good)")
    print(
        f"  catastrophe max asp_sv={result['catastrophe_max_sv']}  "
        f"known-good min asp_sv={result['known_good_min_sv']}"
    )
    print(
        f"  known_good_sv_above_all_cats="
        f"{result['known_good_sv_above_all_cats']}  "
        f"feasible_pairs={result['n_feasible']}"
    )
    # Low-sv catastrophes vs a higher-sv known-good cannot share one
    # max(floor, ratio * sim) cut. Print the binding cases.
    if result["n_feasible"] == 0 and result["red_set"]:
        cats = [r for r in result["red_set"] if r["role"] == "catastrophe"]
        if cats:
            lowest = min(cats, key=lambda r: r["asp_sv"])
            print(
                f"  binding catastrophe: {lowest['name']} "
                f"asp_sv={lowest['asp_sv']:.2f}"
            )
    for row in result["red_set"]:
        print(
            f"  {row['name']:14s} {row['role']:12s} "
            f"asp_sv={row['asp_sv']:.2f} sim_sv={row['sim_sv']:.2f}"
        )
    if result["n_feasible"] == 0:
        print(
            "INFEASIBLE: no (floor, ratio) pair is discriminating on this "
            "red set. Retuning SeamVis cannot pass the M2 exit."
        )
        raise SystemExit(4)


if __name__ == "__main__":
    main()
