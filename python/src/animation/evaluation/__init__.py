"""ASP coherence evaluation tooling — component package.

``backend/controllers/bench_eval_dispatch.py`` is the public entry point / CLI;
everything here is an implementation detail it assembles, split by role:

- ``other/``     — data models, dataset discovery, benchmark-metric flattening,
                   and the queue/persistence state machine. No Qt, no FiftyOne.
- ``logic/``     — pure pixel-data computation and figure building (histograms,
                   spectra, comparison maps, per-test diagnostics charts).
- ``constants/`` — the evaluation vocabulary and UI/logic tuning constants.
- ``ui/``        — the PySide6 per-test inspector (presentation only).
- ``plugin/``    — the optional FiftyOne corpus-triage surface. Doubles as the
                   FiftyOne plugin directory (it holds the ``fiftyone.yml``
                   manifest and exposes ``register()``), which is why it is named
                   for what FiftyOne needs it to be rather than for the workflow
                   it serves.

Two surfaces share one source of truth (``data/benchmarks/asp_evaluations_*.json``):
the inspector for per-test deep work (N-way locked zoom/pan, pixel probe, live
comparison maps, region/link annotation, scoring), and FiftyOne for corpus-level
triage (grid, metric filtering, tagging, saved views). See issue #123 for why the
work is split that way — FiftyOne's App has no label drawing and no pixel probe,
which is exactly the half the inspector owns.
"""

from .other.schema import (
    BoundingBox,
    Edge,
    EdgePoint,
    RatingEntry,
    load_evaluations,
    rated_names,
    save_evaluations,
)
from .other.session import EvaluationSession

__all__ = [
    "EvaluationSession",
    "RatingEntry",
    "BoundingBox",
    "Edge",
    "EdgePoint",
    "load_evaluations",
    "save_evaluations",
    "rated_names",
]


def __getattr__(name: str):
    """Lazily expose the Qt window.

    Importing this package must not pull in PySide6 — the FiftyOne surface and
    the headless tests both import the data/logic layers, and a hard Qt import
    here would make them require a display-capable environment.
    """
    if name in ("InspectorWindow", "RatingDashboard"):
        from .ui.main_window import InspectorWindow

        return InspectorWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
