"""Host guard regression tests use harmless subprocesses, never the corpus."""

import json
import os
import sys

import pytest
from asp_backend_benchmark import resource_guard as guard


@pytest.fixture
def healthy():
    return dict(ram_pct=40.0, available_gib=12.0, cpu_c=65.0, gpu_c=60.0, vram_pct=20.0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("ram_pct", 80.0),
        ("available_gib", 3.9),
        ("cpu_c", 97.0),
        ("gpu_c", 86.0),
        ("vram_pct", 85.0),
    ],
)
def test_guard_limits(healthy, key, value):
    healthy[key] = value
    assert guard.danger(healthy)


def test_missing_optional_sensors(healthy):
    healthy.update(cpu_c=None, gpu_c=None, vram_pct=None)
    assert guard.danger(healthy) is None


def test_atomic_checkpoint_keeps_previous_on_serialization_failure(tmp_path):
    path = tmp_path / "checkpoint.json"
    guard.write_json_atomic(path, [{"name": "completed"}])
    with pytest.raises(TypeError):
        guard.write_json_atomic(path, object())
    assert json.loads(path.read_text()) == [{"name": "completed"}]


def test_supervisor_propagates_failure(tmp_path, monkeypatch, healthy):
    monkeypatch.setattr(guard.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(guard, "sample_resources", lambda: healthy)
    assert guard.supervise([sys.executable, "-c", "raise SystemExit(7)"], tmp_path) == 7
    assert len(list(tmp_path.glob("resources-*.jsonl"))) == 1


def test_supervisor_terminates_worker_on_pressure(tmp_path, monkeypatch, healthy):
    monkeypatch.setattr(guard.Path, "home", lambda: tmp_path)
    samples = iter([healthy, dict(healthy, ram_pct=90.0)])
    monkeypatch.setattr(guard, "sample_resources", lambda: next(samples))
    launched = []
    popen = guard.subprocess.Popen

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(guard.subprocess, "Popen", capture)
    assert guard.supervise([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path) == 2
    assert launched[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(launched[0].pid, 0)
