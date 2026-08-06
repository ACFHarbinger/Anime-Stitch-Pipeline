"""Shared pytest fixtures for the python/ module."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `import src.*` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


@pytest.fixture
def sample_name() -> str:
    return "Dev-Repo-Template"
