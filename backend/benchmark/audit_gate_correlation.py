"""M2 gate-signal audit: Spearman correlation of no-reference metrics vs human judgment.

Roadmap §5 M2 requires auditing the existing Safe ASP gate signals against the
completed human labels, and demoting/removing signals that are inversely
correlated with human quality. Grounds that requirement in a reproducible
computation instead of a one-off, unlinked number.

For each reviewed test case, correlates (ASP metric − SCANS metric) against
(human ASP score − human SCANS score), after re-orienting each metric's delta
so that positive always means "the metric says ASP got better" (per that
metric's own higher/lower-is-better design intent). A well-behaved no-reference
metric should then have rho > 0; rho < 0 means the metric moves the wrong way
relative to human preference and should not gate/demote toward SCANS on its
own value.

``strip_banding_score`` was trimmed from ``_compute_all_metrics`` in 2026-07
and is therefore missing from historical run JSONs even though CompositeGate
still uses it. Pass ``--recompute-missing`` (and optionally ``--images-root``)
to fill it from saved panoramas + ``alignment.affines``. SCANS/simple has no
affines, so its score is 0.0 — the same CompositeGate quirk as
``scans_sb = 0.0``.

Usage::

    .venv/bin/python backend/benchmark/audit_gate_correlation.py \\
        --run backend/benchmark/output/anime_stitch_20260807_045552.json \\
        --labels data/benchmarks/asp_evaluations_20260810.json

    .venv/bin/python backend/benchmark/audit_gate_correlation.py \\
        --run backend/benchmark/output/anime_stitch_20260807_045552.json \\
        --labels data/benchmarks/asp_evaluations_20260810.json \\
        --recompute-missing
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

ASP_ROOT = Path(__file__).resolve().parents[2]

METRIC_KEYS = [
    "sharpness", "edge_energy_score", "ghosting_siqe", "seam_coherence",
    "seam_visibility", "cqas", "coverage", "color_entropy", "seam_gradient",
    "strip_banding_score",
]

# Higher raw value = better output, per each metric's own docstring in
# safety_metrics.py / bench_anime_stitch.py. Lower-is-better metrics get their
# delta negated before correlating.
HIGHER_IS_BETTER = {
    "sharpness": True,
    "edge_energy_score": True,
    "ghosting_siqe": False,   # higher = more periodic ghosting (worse)
    "seam_coherence": False,  # higher = more row-luminance variance/banding (worse)
    "seam_visibility": False,  # higher = harder seam cut (worse)
    "cqas": True,              # already a normalized goodness aggregate
    "coverage": True,
    "color_entropy": True,    # not a clear quality direction; reference only
    "seam_gradient": False,   # higher = more discontinuity at seam rows (worse)
    "strip_banding_score": False,  # higher = larger inter-strip luminance jump
}


def affines_from_alignment(dataset: dict[str, Any]) -> list[Any]:
    """Rebuild 2x3 matrices; ``strip_banding_score`` only reads ``ty`` (``m[1,2]``)."""
    import numpy as np

    recs = (dataset.get("alignment") or {}).get("affines") or []
    out = []
    for rec in recs:
        m = np.eye(2, 3, dtype=np.float32)
        m[0, 0] = float(rec.get("a", 1.0))
        m[0, 1] = float(rec.get("b", 0.0))
        m[0, 2] = float(rec.get("tx", 0.0))
        m[1, 2] = float(rec.get("ty", 0.0))
        out.append(m)
    return out


def resolve_panorama(
    dataset: dict[str, Any],
    *,
    images_root: Path | None,
    kind: str,
    prefer_raw: bool,
) -> Path | None:
    """Locate the ASP or SCANS panorama on disk.

    ``kind`` is ``asp`` or ``simple``. For ASP, prefer ``*_raw_asp.png`` when
    ``prefer_raw`` is set so CompositeGate is scored on the compositor output
    rather than a later Safe ASP / SCANS publish.
    """
    name = dataset["name"]
    json_key = "anime_path" if kind == "asp" else "simple_path"
    candidates: list[Path] = []
    if kind == "asp" and prefer_raw:
        if images_root is not None:
            candidates.append(images_root / f"{name}_raw_asp.png")
        stage_dir = (dataset.get("paths") or {}).get("stage_dir")
        if stage_dir:
            candidates.append(ASP_ROOT / stage_dir / "raw_asp.png")
        candidates.append(ASP_ROOT / "dump" / "output" / f"{name}_raw_asp.png")
    if images_root is not None:
        suffix = "anime_stitch" if kind == "asp" else "opencv_stitch"
        candidates.append(images_root / f"{name}_{suffix}.png")
        if kind == "simple":
            candidates.append(images_root / f"{name}_simple_stitch.png")
    rel = dataset.get(json_key)
    if rel:
        candidates.append(ASP_ROOT / rel)
        if images_root is not None:
            candidates.append(images_root / Path(rel).name)
    for path in candidates:
        if path.is_file():
            return path
    return None


def image_mtime_date(path: Path) -> date:
    return datetime.fromtimestamp(path.stat().st_mtime).date()


def recompute_strip_banding(
    dataset: dict[str, Any],
    *,
    images_root: Path | None,
    prefer_raw: bool,
    max_image_date: date | None,
) -> tuple[float | None, date | None, str]:
    """Return (asp_score, image_date, source_tag).

    SCANS strip-banding is defined as 0.0 (CompositeGate never reads it).
    """
    import importlib.util

    import cv2

    spec = importlib.util.spec_from_file_location(
        "asp_safety_metrics",
        ASP_ROOT / "backend" / "src" / "core" / "pipeline" / "safety_metrics.py",
    )
    if spec is None or spec.loader is None:
        return None, None, "no_safety_metrics"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    strip_banding_score = mod.strip_banding_score

    path = resolve_panorama(
        dataset, images_root=images_root, kind="asp", prefer_raw=prefer_raw
    )
    if path is None:
        return None, None, "missing_image"
    vintage = image_mtime_date(path)
    if max_image_date is not None and vintage > max_image_date:
        return None, vintage, "image_too_new"
    affines = affines_from_alignment(dataset)
    if len(affines) < 2:
        return 0.0, vintage, str(path)
    img = cv2.imread(str(path))
    if img is None:
        return None, vintage, "unreadable"
    return float(strip_banding_score(img, affines)), vintage, str(path)


def audit(
    run_path: Path,
    labels_path: Path,
    *,
    recompute_missing: bool = False,
    images_root: Path | None = None,
    prefer_raw: bool = True,
    max_image_date: date | None = None,
) -> list[tuple[str, float, float, int]]:
    runs = json.loads(run_path.read_text())["datasets"]
    human = json.loads(labels_path.read_text())

    human_delta: list[float] = []
    used_fallback: list[bool] = []
    metric_delta: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    skipped = 0
    sb_from_json = 0
    sb_recomputed = 0
    sb_vintage: Counter[str] = Counter()
    sb_skipped_new = 0
    sb_missing = 0

    for d in runs:
        h = human.get(d["name"])
        ma, ms = dict(d.get("metrics_asp") or {}), dict(d.get("metrics_simple") or {})
        if not h or not h.get("reviewed") or not ma or not ms:
            skipped += 1
            continue

        if "strip_banding_score" in ma:
            sb_from_json += 1
        elif recompute_missing:
            score, vintage, tag = recompute_strip_banding(
                d,
                images_root=images_root,
                prefer_raw=prefer_raw,
                max_image_date=max_image_date,
            )
            if score is None:
                if tag == "image_too_new":
                    sb_skipped_new += 1
                else:
                    sb_missing += 1
            else:
                ma["strip_banding_score"] = score
                ms.setdefault("strip_banding_score", 0.0)
                sb_recomputed += 1
                if vintage is not None:
                    sb_vintage[vintage.isoformat()] += 1
        else:
            sb_missing += 1

        human_delta.append(h["asp"] - h["simple"])
        used_fallback.append(bool(d.get("used_fallback")))
        for k in METRIC_KEYS:
            va, vs = ma.get(k), ms.get(k)
            metric_delta[k].append(
                va - vs if va is not None and vs is not None else float("nan")
            )

    print(f"n_used={len(human_delta)} skipped={skipped}")
    print(
        "strip_banding_score sources: "
        f"json={sb_from_json} recomputed={sb_recomputed} "
        f"missing={sb_missing} skipped_too_new={sb_skipped_new}"
    )
    if sb_vintage:
        print("recomputed image mtimes:", dict(sorted(sb_vintage.items())))

    results = []
    for k in METRIC_KEYS:
        pairs = [(a, b) for a, b in zip(metric_delta[k], human_delta) if not math.isnan(a)]
        if len(pairs) < 5:
            print(f"{k:20s} n={len(pairs)}  (insufficient pairs; skipped)")
            continue
        sign = 1.0 if HIGHER_IS_BETTER[k] else -1.0
        rho, p = spearmanr([sign * a for a, _ in pairs], [b for _, b in pairs])
        results.append((k, rho, p, len(pairs)))

    sb_pairs = [
        (delta, hd, fb)
        for delta, hd, fb in zip(
            metric_delta["strip_banding_score"], human_delta, used_fallback
        )
        if not math.isnan(delta)
    ]
    if len(sb_pairs) >= 5:
        print("strip_banding_score fallback split (oriented, lower-is-better):")
        for label, pred in (
            ("all", lambda fb: True),
            ("true_composite", lambda fb: not fb),
            ("fallback", lambda fb: fb),
        ):
            sub = [(delta, hd) for delta, hd, fb in sb_pairs if pred(fb)]
            if len(sub) < 5:
                print(f"  {label:16s} n={len(sub)} (skip)")
                continue
            rho, p = spearmanr([-a for a, _ in sub], [b for _, b in sub])
            print(f"  {label:16s} rho={rho:+.3f} p={p:.4f} n={len(sub)}")

    results.sort(key=lambda r: r[1])
    return results


def _parse_date(text: str | None) -> date | None:
    if not text:
        return None
    return date.fromisoformat(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        type=Path,
        required=True,
        help="benchmark run JSON with metrics_asp/metrics_simple",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="asp_evaluations_*.json human-label file",
    )
    ap.add_argument(
        "--recompute-missing",
        action="store_true",
        help="fill strip_banding_score from panoramas + alignment.affines",
    )
    ap.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="directory of *_anime_stitch.png / *_raw_asp.png (default: ASP dump/output)",
    )
    ap.add_argument(
        "--no-prefer-raw",
        action="store_true",
        help="score published anime_stitch.png even when raw_asp.png exists",
    )
    ap.add_argument(
        "--max-image-date",
        type=str,
        default=None,
        help="ignore recomputed images newer than YYYY-MM-DD (mtime)",
    )
    args = ap.parse_args()

    images_root = args.images_root
    if args.recompute_missing and images_root is None:
        images_root = ASP_ROOT / "dump" / "output"

    results = audit(
        args.run,
        args.labels,
        recompute_missing=args.recompute_missing,
        images_root=images_root,
        prefer_raw=not args.no_prefer_raw,
        max_image_date=_parse_date(args.max_image_date),
    )
    print()
    print("Sign convention: delta re-oriented so positive = 'metric says ASP got better'.")
    print("rho > 0 = metric agrees with human preference direction (as intended).")
    print("rho < 0 = metric is INVERSELY correlated with human preference (misleading).")
    print()
    for k, rho, p, n in results:
        flag = (
            "  <-- INVERSE / MISLEADING vs human judgment" if rho < -0.2
            else "  (works as intended)" if rho > 0.2
            else "  (no clear signal)"
        )
        print(f"{k:20s} rho={rho:+.3f} p={p:.4f} n={n}{flag}")


if __name__ == "__main__":
    main()
