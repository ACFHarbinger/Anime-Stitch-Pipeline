"""Example entry point for the Python module."""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def greet(name: str) -> str:
    """Return a greeting for ``name``."""
    return f"Hello, {name}!"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Dev-Repo-Template Python example")
    parser.add_argument("--name", default="world", help="Name to greet")
    args = parser.parse_args()
    print(greet(args.name))


if __name__ == "__main__":
    main()
