"""M0 relabeling: cross-reference the saved 2026-08-07 full-corpus benchmark
run against the completed human evaluations to produce explicit
raw_asp/safe_asp/scans identities per case (issue #24).

**Why this exists:** the "asp" score in `asp_evaluations_20260810.json` is
ambiguous — for 54/97 cases, the image actually shown to and rated by the
human was a SCANS substitution (a benchmark-only safety gate fired), not a
genuine ASP composite, but the evaluation file has no field recording that.
Averaging those 54 scores in with the 43 true-composite scores as one "ASP"
population overstates real ASP quality (see
`asp_change_roadmap_2026q3.md` §3). This module makes the distinction
explicit without touching the original human-authored evaluation data —
relabeling is additive, cross-referenced from data that already exists.

**No new GPU run required** (`asp_change_roadmap_2026q3.md` M0 exit
criterion) — this reads the already-saved
`anime_stitch_20260807_045552.json` benchmark run and the already-completed
`asp_evaluations_20260810.json` human ratings, both already on disk.
"""

from __future__ import annotations

import dataclasses
import json
import os

# fallback_code -> which gate fired, per safety_policy.py's fallback_code
# convention (composite/ghost both write 1; seam_vis writes 2). The saved
# benchmark JSON only stores the numeric code, not which of composite/ghost
# fired when the code is 1 -- that distinction wasn't persisted at run time,
# so it can't be recovered here. Label honestly as "composite_or_ghost".
_FALLBACK_LABELS = {
    0: "none",
    1: "composite_or_ghost",
    2: "seam_vis",
}


@dataclasses.dataclass
class RelabeledCase:
    """One case's raw_asp/safe_asp/scans identity, derived from the saved
    benchmark run's ``render_gate_fallback`` code, joined against the human
    rating for that case. Does not duplicate the full rating -- references
    it by ``case_id``, callers should still load the original evaluation
    for scores/notes/defects."""

    case_id: str
    fallback_code: int  # 0/1/2, straight from the saved benchmark run
    fallback_gate: str  # _FALLBACK_LABELS[fallback_code]
    true_raw_asp_composite: bool  # fallback_code == 0
    # What the human's "asp" score in asp_evaluations_20260810.json actually
    # rated: the genuine raw ASP compositor output, or a SCANS substitution
    # mislabeled as "asp" at rating time.
    rated_identity: str  # "raw_asp" or "scans" (via safe_asp fallback)
    human_reviewed: bool
    human_asp_score: int | None
    human_simple_score: int | None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def relabel_corpus(
    bench_run_path: str, evaluations_path: str
) -> dict[str, RelabeledCase]:
    """Join the saved benchmark run's fallback codes against the human
    evaluation file. Raises if any case is present in one file but not the
    other -- a silent partial join would be worse than a loud failure here,
    since this data feeds a product-quality claim."""
    with open(bench_run_path) as fh:
        bench = json.load(fh)
    with open(evaluations_path) as fh:
        evaluations = json.load(fh)

    bench_by_name = {d["name"]: d for d in bench["datasets"]}
    bench_names = set(bench_by_name)
    eval_names = set(evaluations)
    only_in_bench = bench_names - eval_names
    only_in_evals = eval_names - bench_names
    if only_in_bench or only_in_evals:
        raise ValueError(
            "relabel_corpus: bench/evaluation case sets don't match -- "
            f"only in bench run: {sorted(only_in_bench)}, "
            f"only in evaluations: {sorted(only_in_evals)}"
        )

    result: dict[str, RelabeledCase] = {}
    for name, dataset in bench_by_name.items():
        fallback_code = dataset.get("time", {}).get("render_gate_fallback")
        if fallback_code not in _FALLBACK_LABELS:
            raise ValueError(
                f"relabel_corpus: {name} has unrecognized render_gate_fallback "
                f"{fallback_code!r} -- extend _FALLBACK_LABELS or investigate "
                "before trusting this case's relabel."
            )
        is_true_composite = fallback_code == 0
        entry = evaluations[name]
        result[name] = RelabeledCase(
            case_id=name,
            fallback_code=fallback_code,
            fallback_gate=_FALLBACK_LABELS[fallback_code],
            true_raw_asp_composite=is_true_composite,
            rated_identity="raw_asp" if is_true_composite else "scans",
            human_reviewed=bool(entry.get("reviewed", False)),
            human_asp_score=entry.get("asp"),
            human_simple_score=entry.get("simple"),
        )
    return result


def summarize(relabeled: dict[str, RelabeledCase]) -> dict:
    """Aggregate stats matching the ones already cited in
    asp_change_roadmap_2026q3.md §3 (43 true composites / 54 fallbacks),
    computed from data instead of copied by hand -- a live check that the
    saved run still matches what the roadmap claims."""
    true_composites = [c for c in relabeled.values() if c.true_raw_asp_composite]
    fallbacks = [c for c in relabeled.values() if not c.true_raw_asp_composite]

    def _mean_score(cases: list[RelabeledCase]) -> float | None:
        scores = [c.human_asp_score for c in cases if c.human_asp_score is not None]
        return sum(scores) / len(scores) if scores else None

    return {
        "total_cases": len(relabeled),
        "true_raw_asp_composites": {
            "count": len(true_composites),
            "mean_human_asp_score": _mean_score(true_composites),
        },
        "safety_fallbacks_to_scans": {
            "count": len(fallbacks),
            "mean_human_asp_score": _mean_score(fallbacks),
            "by_gate": {
                gate: sum(1 for c in fallbacks if c.fallback_gate == gate)
                for gate in set(c.fallback_gate for c in fallbacks)
            },
        },
    }


def save_relabeled(path: str, relabeled: dict[str, RelabeledCase]) -> None:
    doc = {name: entry.to_dict() for name, entry in relabeled.items()}
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    os.replace(tmp_path, path)
