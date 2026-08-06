#!/usr/bin/env python3
"""
ASP Phase 0.1 — human coherence evaluation tool.

The ASP roadmap's objective ("at least as good as the OpenCV SCANS simple
stitch, as judged by a human") is defined against structural-coherence
evaluations, not any automated metric — no automated metric currently measures
whether a composite has torn anatomy, duplicated strips, or misordered content
(research/ASP_Critical_Evaluation_2026-07-08.md).

Two surfaces share one source of truth (issue #123):

**inspector** (default) — the PySide6 per-test window. Shows every comparator a
test has (ASP / Simple / Overmix / Hugin / ground truth) with locked zoom+pan,
a pixel-value probe, live comparison maps (difference / SSIM / false-colour /
swipe / blend / checkerboard / edge overlay), the per-test benchmark metrics and
pipeline diagnostics, the benchmark's own plots and stage renders,
region/link annotation, and a keyboard-first scoring form.

**triage** — the optional FiftyOne surface. One group per test, one slice per
comparator, every metric and human judgment as a filterable field, plus defect
tags and saved views for corpus-level questions ("metric says asp_better but
the human preferred Simple"). Needs the `benchmark-eval` extra.

Rating scale (0-4):
    4 = keepable — no visible structural defects
    3 = minor flaw — small artifact, still clearly usable
    2 = flawed but parses — visible defect, but anatomy/content still reads
    1 = mostly broken — hard to parse, major tearing/duplication
    0 = incoherent — torn anatomy, duplicated strips, or misordered content

Usage:
    just asp-benchmark-assess          # the inspector
    just asp-triage                    # build the FiftyOne dataset + open the App
    # or directly:
    uv run python backend/controllers/bench_eval_dispatch.py
        [--surface {inspector,triage,ingest,sync}] [--data-dir DIR] [--out PATH]
        [--results PATH] [--redo] [--start-at DATASET]
        [--default-view {display,pixel}] [--theme {dark,light}]
        [--dataset-name NAME] [--port N] [--sync-direction {pull,push}]

Output: {out}/asp_evaluations_<YYYYMMDD>.json. The original schema keys are
written unchanged — bench_anime_stitch.py's _load_human_evaluations() only ever
reads asp/simple — with the richer per-dimension scores, pairwise preference,
defect tags and annotations as purely additive keys:
    {"asp_test04": {"asp": 4, "simple": 2, "notes": "", "bboxes": [...],
                    "edges": [...], "dimensions": {...}, "preference": "asp",
                    "confidence": 3, "defects": [...], "reviewed": true}, ...}
Saved after every change — quitting the window never loses progress. A test
that is merely *visited* is never written, so it stays in the queue next
session.
"""

from __future__ import annotations

import datetime
import os
import sys


def _bootstrap_repo_root() -> str:
    """Walk up from this file to the repo root (the directory holding the
    workspace-root ``pyproject.toml``, identified by ``[tool.uv.workspace]``)
    and put it on ``sys.path``. Needed because this module is invoked as a
    direct script (``uv run python backend/controllers/bench_eval_dispatch.py``,
    see tools/benchmark/justfile), which sets ``sys.path[0]`` to this file's
    own directory rather than the repo root -- the ``backend.*`` absolute
    imports below would otherwise fail with
    ``ModuleNotFoundError: No module named 'backend'``. A plain "any
    pyproject.toml" check stops one level too early now that ``backend/``
    has its own workspace-member ``pyproject.toml``."""
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        candidate = os.path.join(d, "pyproject.toml")
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                if "[tool.uv.workspace]" in f.read():
                    if d not in sys.path:
                        sys.path.insert(0, d)
                    return d
        d = os.path.dirname(d)
    raise RuntimeError(f"Could not locate repo root (pyproject.toml) above {__file__}")


_bootstrap_repo_root()

from backend.benchmark.evaluation.constants.user_interface import (  # noqa: E402
    DISPLAY_PIXEL,
    DISPLAY_RAW,
)
from backend.benchmark.evaluation.other.discovery import repo_root_from  # noqa: E402
from backend.benchmark.evaluation.other.settings import load_settings  # noqa: E402


def _default_out_path(repo_root: str) -> str:
    # The Settings dialog's persisted directory (issue #123 followup item 9)
    # wins over the repo's own data/benchmarks/ once the user has set one —
    # --out still overrides both, same precedence as --theme below.
    settings = load_settings()
    evaluations_dir = settings.out_dir or os.path.join(repo_root, "data", "benchmarks")
    os.makedirs(evaluations_dir, exist_ok=True)
    today = datetime.datetime.now().strftime("%Y%m%d")
    return os.path.join(evaluations_dir, f"asp_evaluations_{today}.json")


def _resolve_out_path(args, repo_root: str) -> str:
    out_path = args.out or _default_out_path(repo_root)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    return out_path


def build_dashboard(args):
    """Constructs (but does not show/exec) the inspector window — split out so a
    smoke test can build it under an offscreen QPA without an event loop."""
    from backend.benchmark.evaluation.ui.main_window import InspectorWindow

    repo_root = repo_root_from(__file__)
    default_view = DISPLAY_PIXEL if args.default_view == "pixel" else DISPLAY_RAW
    window = InspectorWindow(
        base_dir=args.data_dir,
        out_path=_resolve_out_path(args, repo_root),
        redo=args.redo,
        repo_root=repo_root,
        default_display_mode=default_view,
        results_path=getattr(args, "results", None),
        theme=getattr(args, "theme", None),
    )
    start_at = getattr(args, "start_at", None)
    if start_at:
        if start_at in window.session.order:
            window.open_dataset(start_at)
        else:
            print(f"warning: --start-at {start_at!r} is not in the discovered corpus; ignoring.")
    return window


def _run_inspector(args) -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    window = build_dashboard(args)
    if window.total_datasets == 0:
        print(f"No *_anime_stitch.png outputs found under {args.data_dir}/output/ — run the benchmark first.")
        return
    remaining = len(window.todo)
    if remaining:
        print(f"{remaining}/{window.total_datasets} dataset(s) still to rate. Output: {window.out_path}")
    else:
        print(f"All {window.total_datasets} dataset(s) already rated in {window.out_path}. "
              "Opening anyway — use the queue sidebar to revisit any of them.")
    window.show()
    app.exec()
    rated = sum(1 for entry in window.evaluations.values() if entry.is_rated())
    print(f"\nSaved {rated} rated evaluation(s) to {window.out_path}.")


def _run_triage(args, launch_app: bool) -> None:
    from backend.benchmark.evaluation.plugin import ingest, preflight

    check = preflight.check(require_db=True)
    if not check.ok:
        print(check.message())
        raise SystemExit(1)

    repo_root = repo_root_from(__file__)
    out_path = _resolve_out_path(args, repo_root)
    kwargs = {}
    if args.dataset_name:
        kwargs["dataset_name"] = args.dataset_name
    result = ingest.build_dataset(
        base_dir=args.data_dir,
        repo_root=repo_root,
        evaluations_path=out_path if os.path.exists(out_path) else None,
        results_path=args.results,
        **kwargs,
    )
    print(
        f"Built '{result.dataset_name}': {result.groups} test(s), {result.samples} sample(s) "
        f"across slices {result.slices}; {result.rated} already rated."
    )
    if not launch_app:
        return
    # The plugin's operators read this to default the evaluations path.
    os.environ.setdefault("ASP_EVALUATIONS_PATH", out_path)
    print("Opening the FiftyOne App — Ctrl+C to stop.")
    ingest.launch(result.dataset_name, port=args.port, wait=True)


def _run_sync(args) -> None:
    from backend.benchmark.evaluation.plugin import preflight, sync

    check = preflight.check(require_db=True)
    if not check.ok:
        print(check.message())
        raise SystemExit(1)

    out_path = _resolve_out_path(args, repo_root_from(__file__))
    if args.sync_direction == "push":
        print(f"Pushed the evaluations file onto {sync.push(out_path, args.dataset_name)} sample(s).")
    else:
        print(sync.pull(out_path, args.dataset_name).summary())


def evaluate_benchmark_outputs() -> None:
    from backend.controllers.cli.bench_eval_args import build_parser

    args = build_parser(__doc__).parse_args()
    if args.surface == "inspector":
        _run_inspector(args)
    elif args.surface == "triage":
        _run_triage(args, launch_app=True)
    elif args.surface == "ingest":
        _run_triage(args, launch_app=False)
    else:
        _run_sync(args)


if __name__ == "__main__":
    evaluate_benchmark_outputs()
