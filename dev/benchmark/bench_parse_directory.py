"""Micro-benchmark for the Python parser's throughput on its own test fixtures.

Run manually:
    python dev/benchmark/bench_parse_directory.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from input.python.parser import parse_directory  # noqa: E402

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "test" / "fixtures" / "sample_python_project"


def main() -> None:
    iterations = 200
    start = time.perf_counter()
    for _ in range(iterations):
        parse_directory(FIXTURE_ROOT)
    elapsed = time.perf_counter() - start
    print(f"parse_directory(): {elapsed / iterations * 1000:.3f} ms/op over {iterations} iterations")


if __name__ == "__main__":
    main()
