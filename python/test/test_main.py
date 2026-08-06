from src.main import greet


def test_greet_returns_expected_message(sample_name: str) -> None:
    assert greet(sample_name) == "Hello, Dev-Repo-Template!"


def test_greet_default() -> None:
    assert greet("world") == "Hello, world!"
