"""
Full-corpus triage report for the ASP Phase-4 fallback classes (roadmap
docs/moon/roadmaps/asp.md Phase 4, GitHub issue #29).

The 2026-07-27 18-test sample (.agent/cache/asp_phase4_fallback_class_analysis_2026-07-27.md)
found seam_vis_gate is not homogeneous -- some failures are pose-blend
driven, others are purely photometric (registration is fine, the defect is
banding/exposure) -- by cross-referencing seam_visibility against the
existing mean_post_warp_diff metric. Two candidate one-line dispatch rules
(a mean_post_warp_diff threshold, and pair/frame count) were each tried and
each falls short on a real counter-example. The roadmap's own conclusion:
"not a matter of finding the right one-line rule ... a full per-test
measurement pass is what's actually needed" before this class can be
meaningfully converted into real wins.

This script is that measurement pass, extended to the full 97-test corpus
and to composite_gate_sb/sc as well as seam_vis_gate: it parses every
fallback test's fallback_reason (which already encodes the gate's asp/sim/
limit values) out of an existing bench_anime_stitch.py JSON checkpoint,
cross-references mean_post_warp_diff and matching.filtered_edges (pair
count), and tabulates everything sorted by margin-over-limit (closest to
passing first) -- a triage aid for a human policy decision per test, not a
new automatic gate or dispatch rule. No pipeline behavior changes.

Usage:
  python3 backend/benchmark/triage_fallback_classes.py [--json PATH] [--out PATH]
"""

import argparse
import json
import os
import re
from typing import Any

_DEFAULT_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output",
    "anime_stitch_20260728_013215.json",
)


def _parse_fallback_reason(reason: str) -> dict[str, Any]:
    """Parses e.g. 'seam_vis_gate:asp=42.5_sim=4.0_limit=35.0' or
    'composite_gate_sb:asp_sc=23.8_limit=38.0,asp_sb=37.3_limit=35.0'
    into {"gate": ..., "values": {name: float}}."""
    gate, _, rest = reason.partition(":")
    values: dict[str, float] = {}
    for part in re.split(r"[,_](?=[a-z])", rest):
        m = re.match(r"([a-z_]+)=(-?[\d.]+)", part)
        if m:
            values[m.group(1)] = float(m.group(2))
    return {"gate": gate, "values": values}


def _photometric_lean(seam_visibility: float | None, post_warp_diff: float | None) -> str:
    """Qualitative bucketing only -- deliberately not a numeric threshold
    rule (both candidates tried in the 18-test sample failed on real
    counter-examples; see module docstring). For human triage context, not
    an automatic dispatch decision."""
    if seam_visibility is None or post_warp_diff is None:
        return "unknown"
    if post_warp_diff < 10:
        return "photometric-leaning (low post_warp_diff)"
    if post_warp_diff >= 30:
        return "pose-blend-leaning (high post_warp_diff)"
    return "mixed/moderate"


def _row_for(d: dict[str, Any]) -> dict[str, Any] | None:
    reason = d.get("fallback_reason") or ""
    if not reason:
        return None
    parsed = _parse_fallback_reason(reason)
    gate = parsed["gate"]
    values = parsed["values"]
    post_warp_diff = d.get("mean_post_warp_diff")
    pair_count = (d.get("matching") or {}).get("filtered_edges")
    frame_count = (d.get("frames") or {}).get("count")
    phase_count = (d.get("phases") or {}).get("count")

    if gate == "seam_vis_gate":
        asp_val = values.get("asp")
        limit = values.get("limit")
        margin = (asp_val - limit) if (asp_val is not None and limit is not None) else None
        lean = _photometric_lean(asp_val, post_warp_diff)
    elif gate in ("composite_gate_sb", "composite_gate_sc"):
        # Whichever of sb/sc actually exceeded its limit is the binding
        # constraint -- report both, margin against the binding one.
        # _parse_fallback_reason()'s generic key=value split isn't used here:
        # both the asp_sc and asp_sb segments use "limit" as their key name,
        # so a plain dict merge would silently drop one of them. Extract both
        # explicitly from the raw string instead.
        m_sc = re.search(r"asp_sc=(-?[\d.]+)_limit=(-?[\d.]+)", reason)
        m_sb = re.search(r"asp_sb=(-?[\d.]+)_limit=(-?[\d.]+)", reason)
        sc, sc_limit = (float(m_sc.group(1)), float(m_sc.group(2))) if m_sc else (None, None)
        sb, sb_limit = (float(m_sb.group(1)), float(m_sb.group(2))) if m_sb else (None, None)
        sc_margin = (sc - sc_limit) if (sc is not None and sc_limit is not None) else None
        sb_margin = (sb - sb_limit) if (sb is not None and sb_limit is not None) else None
        margin = sc_margin if gate == "composite_gate_sc" else sb_margin
        asp_val = sc if gate == "composite_gate_sc" else sb
        limit = sc_limit if gate == "composite_gate_sc" else sb_limit
        lean = f"sc={sc}/{sc_limit} sb={sb}/{sb_limit}"
    else:
        asp_val = limit = margin = None
        lean = gate

    return {
        "test": d.get("name"),
        "gate": gate,
        "asp_value": asp_val,
        "limit": limit,
        "margin_over_limit": margin,
        "mean_post_warp_diff": post_warp_diff,
        "pair_count": pair_count,
        "frame_count": frame_count,
        "phase_count": phase_count,
        "lean": lean,
    }


def build_triage(json_path: str) -> dict[str, list[dict[str, Any]]]:
    with open(json_path) as fh:
        data = json.load(fh)

    by_gate: dict[str, list[dict[str, Any]]] = {}
    for d in data["datasets"]:
        if not d.get("used_fallback"):
            continue
        row = _row_for(d)
        if row is None:
            continue
        by_gate.setdefault(row["gate"], []).append(row)

    for rows in by_gate.values():
        rows.sort(key=lambda r: (r["margin_over_limit"] is None, r["margin_over_limit"]))

    return by_gate


def render_markdown(by_gate: dict[str, list[dict[str, Any]]], json_path: str) -> str:
    total = sum(len(v) for v in by_gate.values())
    lines = [
        "# ASP Phase 4 — Full-Corpus Fallback Triage",
        "",
        f"Generated from `{os.path.basename(json_path)}` "
        f"({total} fallback tests across {len(by_gate)} gate classes). "
        "Sorted by margin-over-limit (closest to passing first) within each "
        "class -- a triage aid for per-test policy review, not an automatic "
        "dispatch rule (see module docstring for why).",
        "",
    ]

    for gate in sorted(by_gate.keys(), key=lambda g: -len(by_gate[g])):
        rows = by_gate[gate]
        lines.append(f"## `{gate}` ({len(rows)} tests)")
        lines.append("")
        if gate == "seam_vis_gate":
            lines.append(
                "| test | seam_visibility | limit | margin | mean_post_warp_diff | "
                "pair_count | phases | likely root cause |"
            )
            lines.append("|------|------:|------:|------:|------:|------:|------:|------|")
            for r in rows:
                lines.append(
                    f"| {r['test']} | {r['asp_value']} | {r['limit']} | "
                    f"{_fmt(r['margin_over_limit'])} | {_fmt(r['mean_post_warp_diff'])} | "
                    f"{r['pair_count']} | {r['phase_count']} | {r['lean']} |"
                )
        else:
            lines.append(
                "| test | binding value | limit | margin | mean_post_warp_diff | "
                "pair_count | phases | sc/sb detail |"
            )
            lines.append("|------|------:|------:|------:|------:|------:|------:|------|")
            for r in rows:
                lines.append(
                    f"| {r['test']} | {r['asp_value']} | {r['limit']} | "
                    f"{_fmt(r['margin_over_limit'])} | {_fmt(r['mean_post_warp_diff'])} | "
                    f"{r['pair_count']} | {r['phase_count']} | {r['lean']} |"
                )
        lines.append("")

    return "\n".join(lines)


def _fmt(v: float | None) -> str:
    return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default=_DEFAULT_JSON, help="bench_anime_stitch.py JSON checkpoint to triage")
    parser.add_argument("--out", default=None, help="Output markdown path (default: print to stdout)")
    args = parser.parse_args()

    by_gate = build_triage(args.json)
    report = render_markdown(by_gate, args.json)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report)
        print(f"Wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
