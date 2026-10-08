"""Detached-runner resume must skip names already in _checkpoint.json."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "benchmark" / "bench_anime_stitch.py"
_SPEC = importlib.util.spec_from_file_location("bench_anime_stitch", _BENCH)
assert _SPEC and _SPEC.loader
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def test_checkpoint_done_names_reads_name_field(tmp_path: Path, monkeypatch) -> None:
    cp = tmp_path / "_checkpoint.json"
    cp.write_text(
        json.dumps([{"name": "asp_test02"}, {"name": "asp_test10"}, {"skip": True}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(bench, "default_checkpoint_path", lambda: str(cp))
    assert bench._checkpoint_done_names() == {"asp_test02", "asp_test10"}


def test_checkpoint_done_names_missing_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bench, "default_checkpoint_path", lambda: str(tmp_path / "nope.json"))
    assert bench._checkpoint_done_names() == set()


def test_isolated_workers_return_json_in_fresh_processes(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[4]).write_text(json.dumps({'pid': os.getpid(), 'name': sys.argv[2]}))\n"
    )
    monkeypatch.setattr(bench, "__file__", str(worker))
    monkeypatch.setattr(bench, "default_checkpoint_path", lambda: str(tmp_path / "cp.json"))
    first = bench._process_isolated("asp_test01")
    second = bench._process_isolated("asp_test02")
    assert first["name"] == "asp_test01"
    assert second["name"] == "asp_test02"
    assert first["pid"] != second["pid"]
    assert not list(tmp_path.glob("worker-*"))


def test_isolated_worker_failure_is_not_a_successful_result(tmp_path, monkeypatch):
    import subprocess

    import pytest

    worker = tmp_path / "worker.py"
    worker.write_text("raise SystemExit(9)\n")
    monkeypatch.setattr(bench, "__file__", str(worker))
    monkeypatch.setattr(bench, "default_checkpoint_path", lambda: str(tmp_path / "cp.json"))
    with pytest.raises(subprocess.CalledProcessError):
        bench._process_isolated("asp_test01")
