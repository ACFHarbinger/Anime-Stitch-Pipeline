"""Merge disjoint per-range benchmark JSON files into one corpus document.

The post-M1 ungated 97-run was executed as separate ``--range`` invocations.
Each write replaced ``anime_stitch_<timestamp>.json`` with only that range,
so no single post-M1 metrics file existed for the M2 audit.

Later files win on a duplicate ``name``. Summary counts are rebuilt from the
union; per-run metadata is preserved under ``metadata.source_runs``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CONSOLIDATED_NAME = "anime_stitch_latest_consolidated.json"


def load_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_run_docs(
    docs: list[dict[str, Any]],
    *,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    by_name: dict[str, dict[str, Any]] = {}
    source_meta: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        meta = dict(doc.get("metadata") or {})
        if sources and i < len(sources):
            meta["path"] = sources[i]
        source_meta.append(meta)
        for ds in doc.get("datasets") or []:
            name = ds.get("name")
            if not name:
                continue
            by_name[name] = ds
    datasets = sorted(by_name.values(), key=lambda d: str(d.get("name")))
    fallback = sum(1 for d in datasets if d.get("used_fallback"))
    return {
        "metadata": {
            "suite_name": "Anime Stitch Pipeline",
            "timestamp": datetime.now().isoformat(),
            "total_datasets": len(datasets),
            "format_version": "1.0",
            "merged": True,
            "source_runs": source_meta,
        },
        "summary": {
            "total_datasets": len(datasets),
            "datasets_fallback": fallback,
            "datasets_passed": len(datasets) - fallback,
        },
        "datasets": datasets,
    }


def discover_run_files(directory: Path) -> list[Path]:
    files = []
    for path in sorted(directory.glob("anime_stitch_*.json")):
        if path.name.endswith("_consolidated.json") or path.name.startswith(
            "anime_stitch_latest"
        ):
            continue
        files.append(path)
    return files


def merge_paths(paths: list[Path], out_path: Path) -> Path:
    docs = [load_run(p) for p in paths]
    merged = merge_run_docs(docs, sources=[str(p) for p in paths])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return out_path


def maybe_write_consolidated(
    new_run_path: Path,
    *,
    output_dir: Path | None = None,
) -> Path | None:
    """After a range write, union every sibling run into the stable file."""
    directory = output_dir or OUTPUT_DIR
    files = discover_run_files(directory)
    if new_run_path not in files and new_run_path.is_file():
        files.append(new_run_path)
    if len(files) < 2:
        return None
    out = directory / CONSOLIDATED_NAME
    merge_paths(files, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=None,
        help="JSON run files (default: all anime_stitch_*.json in output/)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR / CONSOLIDATED_NAME,
        help="destination path",
    )
    args = ap.parse_args()
    paths = args.inputs or discover_run_files(OUTPUT_DIR)
    if not paths:
        raise SystemExit("no benchmark JSON files to merge")
    dest = merge_paths(paths, args.out)
    n = len(json.loads(dest.read_text())["datasets"])
    print(f"merged {len(paths)} run(s) → {dest} ({n} datasets)")


if __name__ == "__main__":
    main()
