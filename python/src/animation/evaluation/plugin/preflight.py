"""Availability checks for the FiftyOne triage surface, with actionable
diagnostics.

FiftyOne is an **optional dev extra** (issue #123): it pulls ~30 packages plus
an embedded MongoDB, and this is a dev-only benchmark tool, so the main app's
install must not carry it. That makes a clear "here's what's missing and here's
the command" message part of the feature rather than a nicety — and the second
check below is not hypothetical: on Ubuntu 26.04 (glibc 2.43) ``fiftyone-db``
installs without a bundled ``mongod`` at all, so FiftyOne imports fine and then
fails at first dataset access.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from typing import List, Optional


@dataclasses.dataclass
class Preflight:
    ok: bool
    problems: List[str]
    hints: List[str]

    def message(self) -> str:
        lines = []
        if self.problems:
            lines.append("FiftyOne triage surface is unavailable:")
            lines += [f"  - {p}" for p in self.problems]
        if self.hints:
            lines.append("")
            lines.append("To enable it:")
            lines += [f"  {h}" for h in self.hints]
        return "\n".join(lines)


_MONGO_HINTS = [
    "# Option A — point FiftyOne at a MongoDB you run (works on any distro):",
    "docker run -d --name image-toolkit-fiftyone-db -p 27017:27017 mongo:7",
    "export FIFTYONE_DATABASE_URI=mongodb://localhost:27017",
    "",
    "# Option B — install a system mongod and let FiftyOne manage it.",
    "# fiftyone-db ships prebuilt mongod only for some distros; on Ubuntu 26.04",
    "# (glibc 2.43) it installs no binary, which is why Option A is the default here.",
]


def check(require_db: bool = True) -> Preflight:
    """Report whether the triage surface can run.

    ``require_db=False`` checks only that the package imports, which is enough
    for the payload-mapping code paths that never touch the database.
    """
    problems: List[str] = []
    hints: List[str] = []

    try:
        import fiftyone  # noqa: F401
    except ImportError:
        problems.append("the `fiftyone` package is not installed")
        hints.append("uv pip install -e '.[benchmark-eval]'   # or: pip install fiftyone")
        return Preflight(ok=False, problems=problems, hints=hints)

    if not require_db:
        return Preflight(ok=True, problems=[], hints=[])

    if os.environ.get("FIFTYONE_DATABASE_URI"):
        # An explicit URI means the user has taken responsibility for the
        # database; whether it's reachable is reported at connect time with a
        # better error than anything guessable here.
        return Preflight(ok=True, problems=[], hints=[])

    if _bundled_mongod() or shutil.which("mongod"):
        return Preflight(ok=True, problems=[], hints=[])

    problems.append(
        "no `mongod` found — neither bundled with fiftyone-db nor on PATH — and "
        "FIFTYONE_DATABASE_URI is not set"
    )
    hints.extend(_MONGO_HINTS)
    return Preflight(ok=False, problems=problems, hints=hints)


def _bundled_mongod() -> Optional[str]:
    try:
        import fiftyone.db as fiftyone_db
    except ImportError:
        return None
    candidate = os.path.join(os.path.dirname(fiftyone_db.__file__), "bin", "mongod")
    return candidate if os.path.isfile(candidate) else None


def require(require_db: bool = True) -> None:
    """Raise with the full diagnostic, for callers that can't degrade."""
    result = check(require_db=require_db)
    if not result.ok:
        raise RuntimeError(result.message())
