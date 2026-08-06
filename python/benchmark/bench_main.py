"""Example benchmark, run via `pytest benchmark --benchmark-only`."""

from src.main import greet


def test_greet_benchmark(benchmark) -> None:
    result = benchmark(greet, "world")
    assert result == "Hello, world!"
