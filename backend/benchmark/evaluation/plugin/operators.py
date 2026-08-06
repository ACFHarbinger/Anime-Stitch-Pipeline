"""The FiftyOne panel and operators this plugin registers.

Kept in its own module rather than in the package ``__init__`` because it imports
``fiftyone.operators`` at module scope: FiftyOne is an optional dev extra, so
``asp_backend_evaluation.plugin`` has to stay importable without it. The
package's ``register()`` defers to this module lazily.

Three registrations bridge the two surfaces (issue #123):

- ``asp_diagnostics`` — a Python Panel showing the selected test's benchmark
  facts and its per-test diagnostics as Plotly figures. Reuses the same flattened
  series as the inspector's matplotlib charts (``other/metrics_view.py``), so a
  number can never differ between the two surfaces.
- ``open_inspector`` — launches the PySide6 inspector on the currently-selected
  test. This is the handoff for everything FiftyOne structurally cannot do:
  its App has no label drawing (annotation is delegated to CVAT/Label Studio),
  no pixel probe, and no live comparison sliders.
- ``sync_evaluations`` — pulls App-side defect tagging back into the evaluations
  JSON and pushes the JSON's scores/regions back onto the samples.

Panels render through ``types.PlotlyView``, whose event surface is
``onClick``/``onSelected``/``onDoubleClick`` only — no relayout — which is why
region drawing lives in the inspector rather than being attempted here with
Plotly's ``drawrect``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, Optional

import fiftyone.operators as foo
import fiftyone.operators.types as types


def _repo_root() -> Optional[str]:
    directory = os.path.dirname(os.path.abspath(__file__))
    while directory != os.path.dirname(directory):
        if os.path.exists(os.path.join(directory, "pyproject.toml")):
            return directory
        directory = os.path.dirname(directory)
    return None


def _bootstrap() -> None:
    """Put the repo root on ``sys.path``.

    FiftyOne loads a plugin by importing its directory as a standalone module,
    *not* as part of ``asp_backend_evaluation.plugin`` — so relative
    imports out of this directory don't resolve under that loader and the shared
    data layer has to be imported absolutely. Same walk-up-to-``pyproject.toml``
    guard ``bench_eval_dispatch.py`` uses for the same reason.
    """
    root = _repo_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)


_bootstrap()

from asp_backend_evaluation.other.metrics_view import (  # noqa: E402
    RADAR_SCALES,
    radar_value,
)


def _selected_test(ctx) -> Optional[str]:
    """The dataset_name of whatever the user has selected or is viewing."""
    if getattr(ctx, "current_sample", None):
        try:
            sample = ctx.dataset[ctx.current_sample]
            return sample["dataset_name"]
        except Exception:
            pass
    if ctx.selected:
        try:
            return ctx.dataset[ctx.selected[0]]["dataset_name"]
        except Exception:
            pass
    view = ctx.view or ctx.dataset
    try:
        names = view.distinct("dataset_name")
        return names[0] if names else None
    except Exception:
        return None


def _entry_for(ctx, name: str) -> Dict:
    """The benchmark fields for one test, read back off the samples rather than
    re-reading the results JSON — the App may be pointed at a dataset built from
    a run whose JSON has since been superseded."""
    view = ctx.dataset.select_group_slices(_allow_mixed=True).match(
        _F("dataset_name") == name
    )
    if not view:
        return {}
    fields: Dict = {}
    for sample in view.limit(8):
        fields.setdefault("__by_slice__", {})[sample["comparator_key"]] = sample
        fields.update({
            "verdict": sample["verdict"],
            "verdict_source": sample["verdict_source"],
            "used_fallback": sample["used_fallback"],
            "fallback_reason": sample["fallback_reason"],
            "has_ground_truth": sample["has_ground_truth"],
            "human_asp": sample["human_asp"],
            "human_simple": sample["human_simple"],
            "human_preference": sample["human_preference"],
            "human_defects": sample["human_defects"],
            "dy_cv": sample["dy_cv"],
            "dx_cv": sample["dx_cv"],
            "total_sec": sample["total_sec"],
            "frames_count": sample["frames_count"],
            "mean_post_warp_diff": sample["mean_post_warp_diff"],
        })
    return fields


def _F(path):
    from fiftyone import ViewField

    return ViewField(path)


class ASPDiagnosticsPanel(foo.Panel):
    """Per-test facts and metric comparison for the selected group."""

    @property
    def config(self):
        return foo.PanelConfig(
            name="asp_diagnostics",
            label="ASP diagnostics",
            icon="insights",
            help_markdown=(
                "Benchmark facts and metric comparison for the selected test. "
                "Use **Open inspector** for pixel-level work, region annotation "
                "and scoring."
            ),
        )

    def on_load(self, ctx):
        ctx.panel.state.name = _selected_test(ctx)

    def on_change_current_sample(self, ctx):
        ctx.panel.state.name = _selected_test(ctx)

    def on_change_selected(self, ctx):
        ctx.panel.state.name = _selected_test(ctx)

    def open_inspector(self, ctx):
        ctx.ops.notify("Launching the inspector…")
        _spawn_inspector(ctx.panel.state.name)

    def render(self, ctx):
        panel = types.Object()
        name = ctx.panel.state.name
        if not name:
            panel.md("### No test selected\nSelect a sample to see its diagnostics.")
            return types.Property(panel, view=types.GridView())

        entry = _entry_for(ctx, name)
        by_slice = entry.pop("__by_slice__", {})
        panel.md(_facts_markdown(name, entry), name="facts")
        panel.plot("metric_bars", label="Metric comparison")
        ctx.panel.data.metric_bars = _metric_bar_figure(by_slice)
        panel.btn(
            "open_inspector",
            label="Open inspector for this test",
            on_click=self.open_inspector,
            variant="contained",
        )
        return types.Property(panel, view=types.GridView(orientation="vertical"))


def _facts_markdown(name: str, entry: Dict) -> str:
    def fmt(value, digits=3):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    human = "unrated"
    if entry.get("human_asp") is not None and entry.get("human_simple") is not None:
        human = f"ASP {entry['human_asp']} / Simple {entry['human_simple']}"
        if entry.get("human_preference"):
            human += f" (prefers {entry['human_preference']})"
    defects = entry.get("human_defects") or []
    composite = "SCANS fallback" if entry.get("used_fallback") else "true ASP composite"
    return "\n".join([
        f"### {name}",
        "",
        f"- **Verdict**: {entry.get('verdict', '—')}  _(source: {entry.get('verdict_source', '—')})_",
        f"- **Composite**: {composite}"
        + (f" — `{entry['fallback_reason']}`" if entry.get("fallback_reason") else ""),
        f"- **Ground truth**: {'available' if entry.get('has_ground_truth') else 'none'}",
        f"- **Human**: {human}",
        f"- **Defect tags**: {', '.join(defects) if defects else 'none'}",
        f"- **Frames**: {fmt(entry.get('frames_count'), 0)} · "
        f"dy_cv {fmt(entry.get('dy_cv'), 4)} · dx_cv {fmt(entry.get('dx_cv'), 4)} · "
        f"mean post-warp diff {fmt(entry.get('mean_post_warp_diff'), 2)}",
        f"- **Total time**: {fmt(entry.get('total_sec'), 1)} s",
    ])


def _metric_bar_figure(by_slice: Dict) -> Dict:
    """Grouped bars of the CQAS-scale normalized metrics, one trace per
    comparator — the Plotly twin of the inspector's radar, using the same
    absolute normalizers so the two surfaces agree."""
    labels = [label for _key, label, _ref, _dir in RADAR_SCALES]
    data = []
    for slice_name, sample in by_slice.items():
        values = []
        for metric_key, _label, _ref, _dir in RADAR_SCALES:
            try:
                raw = sample[metric_key]
            except Exception:
                raw = None
            values.append(radar_value(metric_key, raw))
        if any(v is not None for v in values):
            data.append({
                "type": "bar",
                "name": slice_name,
                "x": labels,
                "y": [0 if v is None else v for v in values],
            })
    return {
        "data": data,
        "layout": {
            "barmode": "group",
            "template": "plotly_dark",
            "yaxis": {"title": "CQAS scale (1.0 = best)", "range": [0, 1]},
            "margin": {"t": 20, "b": 60, "l": 50, "r": 20},
            "height": 320,
        },
    }


class OpenInspector(foo.Operator):
    """Launch the PySide6 inspector on the selected test."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="open_inspector",
            label="Open the Benchmark evaluation inspector",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "dataset_name",
            label="Test",
            description="Defaults to the selected sample's test",
            default=_selected_test(ctx) or "",
        )
        return types.Property(inputs)

    def execute(self, ctx):
        name = ctx.params.get("dataset_name") or _selected_test(ctx)
        pid = _spawn_inspector(name)
        return {"launched": bool(pid), "dataset_name": name, "pid": pid}


class SyncEvaluations(foo.Operator):
    """Round-trip human judgment with the evaluations JSON."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="sync_evaluations",
            label="Sync evaluations file",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.enum(
            "direction",
            ["pull", "push"],
            default="pull",
            label="Direction",
            description=(
                "pull: fold App-side defect tags into the JSON. "
                "push: refresh sample fields/tags/regions from the JSON."
            ),
        )
        inputs.str(
            "evaluations_path",
            label="Evaluations JSON",
            default=os.environ.get("ASP_EVALUATIONS_PATH", ""),
            required=True,
        )
        return types.Property(inputs)

    def execute(self, ctx):
        from asp_backend_evaluation.plugin import sync

        path = ctx.params["evaluations_path"]
        if ctx.params.get("direction") == "push":
            return {"updated_samples": sync.push(path, ctx.dataset.name)}
        report = sync.pull(path, ctx.dataset.name)
        return {"summary": report.summary()}


def _spawn_inspector(name: Optional[str]) -> Optional[int]:
    """Start the inspector as a detached process.

    Detached rather than in-process on purpose: the FiftyOne App server runs an
    asyncio event loop, and constructing a ``QApplication`` inside it would put a
    Qt event loop in the same process as the web server.
    """
    repo_root = _repo_root()
    if repo_root is None:
        return None
    command = [
        sys.executable,
        os.path.join(repo_root, "backend", "controllers", "bench_eval_dispatch.py"),
    ]
    if name:
        command += ["--start-at", name]
    process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        command, cwd=repo_root, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return process.pid


def register(plugin) -> None:
    plugin.register(ASPDiagnosticsPanel)
    plugin.register(OpenInspector)
    plugin.register(SyncEvaluations)
