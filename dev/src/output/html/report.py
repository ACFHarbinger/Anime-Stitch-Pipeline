"""Static HTML report generator — the always-available output per
moon/roadmaps/developer_tools.md D5.

Renders a self-contained HTML file (vis.js loaded from a CDN, no local
`pyvis` dependency) showing the merged code graph, colored/labeled by layer,
plus a plain-text summary of layer violations and circular-dependency
groups. Works headless — no browser or display needed to *generate* the
report, only to view it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # dev/src/

from core.cycles import find_cycles  # noqa: E402
from core.layers import Violation, find_violations  # noqa: E402
from core.model import CodeGraph  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

TEMPLATE_DIR = Path(__file__).resolve().parent
_LAYER_COLORS = {
    "logic": "#5319E7",
    "test": "#0E8A16",
    "script": "#FBCA04",
    "": "#556677",
}


def render(
    graph: CodeGraph,
    violations: list[Violation],
    cycles: list[list[str]],
    title: str = "Dependency Graph",
) -> str:
    """Render a self-contained HTML report for a code graph.

    Args:
        graph: The (layer-classified) graph to render.
        violations: Layer violations found via
            :func:`dev.src.core.layers.find_violations`.
        cycles: Circular-dependency groups found via
            :func:`dev.src.core.cycles.find_cycles`.
        title: Report title, shown in the page header/tab title.

    Returns:
        A complete HTML document as a string.
    """
    cycle_nodes: set[str] = {node_id for cycle in cycles for node_id in cycle}
    violation_ids: set[str] = {v.edge.source_id for v in violations} | {
        v.edge.target_id for v in violations
    }

    vis_nodes = [
        {
            "id": n.id,
            "label": n.qualified_name or n.id,
            "color": "#ff6b6b" if n.id in cycle_nodes else _LAYER_COLORS.get(n.layer, "#7f8c9a"),
            "title": f"{n.id}\nlayer: {n.layer or '(unclassified)'}",
        }
        for n in graph.nodes
    ]
    vis_edges = [
        {
            "from": e.source_id,
            "to": e.target_id,
            "color": {
                "color": "#ff6b6b"
                if e.source_id in violation_ids and e.target_id in violation_ids
                else "#556677"
            },
        }
        for e in graph.edges
    ]

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
    template = env.get_template("template.html.j2")
    return template.render(
        title=title,
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        violations=violations,
        cycles=cycles,
        nodes_json=json.dumps(vis_nodes),
        edges_json=json.dumps(vis_edges),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_json", type=Path, help="Path to a CodeGraph JSON file")
    parser.add_argument("-o", "--output", type=Path, default=Path("dev-graph-report.html"))
    return parser.parse_args()


def main() -> None:
    """Entry point: load a CodeGraph JSON file and write an HTML report."""
    args = parse_args()
    payload = json.loads(args.graph_json.read_text(encoding="utf-8"))
    graph = CodeGraph.from_dict(payload)
    violations = find_violations(graph, forbidden_directions=[])
    cycles = find_cycles(graph)
    html = render(graph, violations, cycles)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
