"""Optional FiftyOne (Voxel51) corpus-triage surface for the ASP benchmark.

The corpus-level half of the evaluation tooling (issue #123): a grouped dataset
with one group per test and one slice per comparator, every benchmark metric and
human judgment as a filterable sample field, defect tags, and saved views for the
triage queries this workflow actually asks.

FiftyOne is an **optional dev extra** — it pulls ~30 packages plus an embedded
MongoDB, and this is a dev-only benchmark tool, so nothing here is imported by
the inspector or the shared data layer. ``preflight.check()`` reports what's
missing with the exact command to fix it.

Division of labour with the inspector, and why: FiftyOne's App has no label
drawing (annotation is delegated to CVAT/Label Studio/Labelbox), its
``PlotlyView`` exposes only ``onClick``/``onSelected``/``onDoubleClick`` with no
relayout event so Plotly's ``drawrect`` can't round-trip, and it has no pixel
probe or live compositing slider on the media. Those are exactly the inspector's
job; this surface owns the grid, filtering, tagging and saved views.

Module layout:

- ``preflight.py``     — availability checks with actionable diagnostics.
- ``sample_fields.py`` — benchmark result + evaluation -> flat sample payloads.
                         No ``fiftyone`` import, so it's testable without a DB.
- ``ingest.py``        — builds the grouped dataset and its saved views.
- ``sync.py``          — round-trips human judgment with the evaluations JSON.
- ``operators.py``     — the FiftyOne panel and operators registered below.
- ``fiftyone.yml``     — the manifest FiftyOne discovers this plugin by.

This directory is both a Python package (``animation.evaluation.plugin``)
and a FiftyOne plugin: FiftyOne finds a plugin by its ``fiftyone.yml``, imports
the directory, and calls ``register()`` on it. Hence the thin ``register`` below,
which defers the ``operators`` import so that merely importing this package never
requires ``fiftyone`` to be installed.
"""

from .preflight import Preflight, check, require

__all__ = ["Preflight", "check", "require", "register"]


def register(plugin) -> None:
    """FiftyOne plugin entry point.

    The ``operators`` import is deliberately inside the function: that module
    imports ``fiftyone.operators`` at module scope, and this package has to stay
    importable on a machine where the optional extra isn't installed.
    """
    from . import operators

    operators.register(plugin)
