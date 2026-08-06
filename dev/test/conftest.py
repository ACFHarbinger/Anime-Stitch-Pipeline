"""Pytest configuration for dev/test/.

Adds dev/src/ to sys.path so tests can import `core.*`, `input.*`, and
`output.*` the same way dev/src/cli.py does, without packaging dev/src/ as
an installed distribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))
