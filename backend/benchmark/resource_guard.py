"""Independent host protection for the benchmark's native/GPU workers."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil


def results_directory() -> Path:
    """Keep generated diagnostics outside the checkout."""
    return Path(
        os.environ.get(
            "ASP_BENCH_RESULTS_DIR", str(Path.home() / "Downloads/Data/Tests/asp-benchmark")
        )
    ).expanduser()


def write_json_atomic(path: Path, value: Any) -> None:
    """Preserve the last complete checkpoint through interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def sample_resources() -> dict[str, float | None]:
    """Read host sensors without loading CUDA in the supervisor."""
    memory = psutil.virtual_memory()
    cpu = [
        sensor.current
        for name, sensors in psutil.sensors_temperatures().items()
        if name in {"coretemp", "k10temp"}
        for sensor in sensors
    ]
    gpu_temp = gpu_used = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        rows = [[float(v) for v in line.split(",")] for line in result.stdout.strip().splitlines()]
        gpu_temp = max(row[0] for row in rows)
        gpu_used = max(100 * row[1] / row[2] for row in rows)
    except (OSError, subprocess.SubprocessError, ValueError, ZeroDivisionError):
        pass
    return {
        "time": time.time(),
        "ram_pct": memory.percent,
        "available_gib": memory.available / 2**30,
        "swap_pct": psutil.swap_memory().percent,
        "cpu_c": max(cpu, default=None),
        "gpu_c": gpu_temp,
        "vram_pct": gpu_used,
    }


def danger(sample: dict[str, float | None]) -> str | None:
    """Stop with headroom; missing optional temperature sensors are recorded."""
    limits = {
        "ram_pct": float(os.environ.get("ASP_BENCH_RAM_ABORT_PCT", "80")),
        "cpu_c": float(os.environ.get("ASP_BENCH_CPU_ABORT_C", "97")),
        "gpu_c": float(os.environ.get("ASP_BENCH_GPU_ABORT_C", "86")),
        "vram_pct": float(os.environ.get("ASP_BENCH_VRAM_ABORT_PCT", "85")),
    }
    available = sample.get("available_gib")
    if available is not None and available < 4:
        return f"only {available:.2f} GiB RAM available"
    for key, limit in limits.items():
        value = sample.get(key)
        if value is not None and value >= limit:
            return f"{key}={value:.1f} >= {limit:.1f}"
    return None


def stop_process_group(process: subprocess.Popen) -> None:
    """Reap the worker and terminate its descendants, including native helpers."""
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def supervise(command: list[str], directory: Path) -> int:
    """Run one benchmark with continuous checks and durable sensor evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    # Shared across result directories and checkouts; held outside the child.
    lock_path = Path.home() / "Downloads/Data/Tests/.asp-benchmark.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[Guard] Another supervised ASP benchmark is running.", flush=True)
            return 2
        log_path = directory / f"resources-{time.time_ns()}.jsonl"
        print(f"[Guard] Resource telemetry: {log_path}", flush=True)
        with log_path.open("a", buffering=1) as log:

            def check() -> str | None:
                sample = sample_resources()
                log.write(json.dumps(sample) + "\n")
                log.flush()
                os.fsync(log.fileno())
                return danger(sample)

            reason = check()
            if reason:
                print(f"[Guard] Refusing launch: {reason}", flush=True)
                return 2
            env = dict(os.environ, ASP_BENCH_SUPERVISED="1", PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command, env=env, start_new_session=True)
            try:
                while process.poll() is None:
                    reason = check()
                    if reason:
                        print(f"[Guard] ABORT: {reason}", flush=True)
                        return 2
                    time.sleep(1)
                return process.returncode if process.returncode >= 0 else 128 - process.returncode
            finally:
                stop_process_group(process)
