"""Argument parser for `backend/controllers/bench_eval_dispatch.py`."""

from __future__ import annotations

import argparse
import os


def build_parser(doc: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--surface", choices=["inspector", "triage", "ingest", "sync"], default="inspector",
        help="inspector: the PySide6 per-test window (default). triage: build the "
             "FiftyOne dataset and open the App. ingest: build the dataset and exit. "
             "sync: round-trip human judgment between the dataset and the JSON.",
    )
    parser.add_argument(
        "--data-dir", default=os.path.expanduser("~/Downloads/Data/Dump"),
        help="Root data directory containing asp_testXX subdirectories and output/",
    )
    parser.add_argument(
        "--out", default=None,
        help="Evaluations JSON path. Defaults to the Settings dialog's persisted save "
             "directory if one is set, else data/benchmarks/asp_evaluations_<today>.json "
             "(resumes an existing file for today if present); pass an existing file's "
             "path explicitly to resume a specific prior session.",
    )
    parser.add_argument(
        "--results", default=None,
        help="A specific backend/benchmark/output/anime_stitch_*.json to read metrics "
             "from. Defaults to the most recent run.",
    )
    parser.add_argument(
        "--redo", action="store_true",
        help="Re-walk datasets that already have an evaluation (default: start at the "
             "first unrated one). Every dataset is reachable from the queue sidebar "
             "regardless of this flag.",
    )
    parser.add_argument(
        "--start-at", default=None, metavar="DATASET",
        help="Open directly on this dataset (e.g. asp_test27). Used by the FiftyOne "
             "plugin's 'open inspector' handoff.",
    )
    parser.add_argument(
        "--default-view", choices=["display", "pixel"], default="display",
        help="Which display mode the image panels start in (default: display). "
             "'pixel' overlays the per-pixel grid + RGB values once zoomed in far enough.",
    )
    parser.add_argument(
        "--theme", choices=["dark", "light"], default=None,
        help="Chrome theme for this run, overriding whatever the Settings dialog last "
             "persisted (default: use the persisted theme, or dark on first run).",
    )
    parser.add_argument(
        "--dataset-name", default=None,
        help="FiftyOne dataset name for the triage surfaces (default: "
             "asp_benchmark_evaluation).",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port for the FiftyOne App (--surface triage).",
    )
    parser.add_argument(
        "--sync-direction", choices=["pull", "push"], default="pull",
        help="--surface sync: pull folds App-side defect tags into the JSON; push "
             "refreshes the dataset's human fields/tags/regions from the JSON.",
    )
    return parser
