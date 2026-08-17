"""Union disjoint range JSONs by dataset name."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SRC = (
    Path(__file__).resolve().parents[3]
    / "backend"
    / "benchmark"
    / "merge_run_json.py"
)
_spec = importlib.util.spec_from_file_location("merge_run_json", _SRC)
assert _spec and _spec.loader
merge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge)


def _doc(*names: str, extra: dict | None = None) -> dict:
    datasets = [{"name": n, "used_fallback": False, **(extra or {})} for n in names]
    return {"metadata": {"total_datasets": len(datasets)}, "datasets": datasets}


def test_later_file_wins_on_duplicate_name():
    merged = merge.merge_run_docs(
        [
            _doc("asp_test01", extra={"mark": "old"}),
            _doc("asp_test01", "asp_test02", extra={"mark": "new"}),
        ]
    )
    by_name = {d["name"]: d for d in merged["datasets"]}
    assert set(by_name) == {"asp_test01", "asp_test02"}
    assert by_name["asp_test01"]["mark"] == "new"
    assert merged["metadata"]["merged"] is True
    assert merged["summary"]["total_datasets"] == 2


def test_merge_paths_writes_union(tmp_path: Path):
    a = tmp_path / "anime_stitch_a.json"
    b = tmp_path / "anime_stitch_b.json"
    a.write_text(json.dumps(_doc("asp_test01")))
    b.write_text(json.dumps(_doc("asp_test02", "asp_test03")))
    out = tmp_path / "anime_stitch_latest_consolidated.json"
    merge.merge_paths([a, b], out)
    names = {d["name"] for d in json.loads(out.read_text())["datasets"]}
    assert names == {"asp_test01", "asp_test02", "asp_test03"}


def test_maybe_write_consolidated_needs_two_files(tmp_path: Path):
    one = tmp_path / "anime_stitch_20260816_1.json"
    one.write_text(json.dumps(_doc("asp_test01")))
    assert merge.maybe_write_consolidated(one, output_dir=tmp_path) is None
    two = tmp_path / "anime_stitch_20260816_2.json"
    two.write_text(json.dumps(_doc("asp_test02")))
    dest = merge.maybe_write_consolidated(two, output_dir=tmp_path)
    assert dest is not None
    assert dest.name == "anime_stitch_latest_consolidated.json"
    names = {d["name"] for d in json.loads(dest.read_text())["datasets"]}
    assert names == {"asp_test01", "asp_test02"}
