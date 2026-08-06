"""Developer-assistant tool CLI — entry point wiring input → core → output.

Per moon/roadmaps/developer_tools.md D11 (partial: wires the pieces built so
far; `just dev` recipes and CI wiring are tracked separately).

Usage:
    python -m dev.src.cli report python/src -o report.html
    python -m dev.src.cli check python/src            # CI-usable: exit 1 on violations/cycles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # dev/src/

from core.aggregate import aggregate  # noqa: E402
from core.cycles import find_cycles  # noqa: E402
from core.layers import LayerRule, classify, find_violations  # noqa: E402
from input.python.parser import parse_directory  # noqa: E402
from output.html.report import render  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "layers.yaml"


def _load_layer_config() -> tuple[list[LayerRule], list[tuple[str, str]]]:
    """Load layer rules and forbidden directions from ``dev/config/layers.yaml``.

    Returns:
        A tuple of (layer rules, forbidden ``(from, to)`` direction pairs).
    """
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rules = [LayerRule(prefix=r["prefix"], label=r["label"]) for r in payload.get("layers", [])]
    forbidden = [tuple(pair) for pair in payload.get("forbidden_directions", [])]
    return rules, forbidden


def build_graph(root: Path):
    """Parse ``root`` with every available language parser and merge the result.

    Args:
        root: Directory to analyze. Currently only Python sources are
            parsed (see moon/roadmaps/developer_tools.md D9 for the
            remaining languages); this function is the seam where
            additional ``input/<language>`` parsers plug in.

    Returns:
        A merged, layer-classified :class:`~dev.src.core.model.CodeGraph`.
    """
    graphs = []
    if any(root.rglob("*.py")):
        graphs.append(parse_directory(root))
    merged = aggregate(graphs)
    rules, _ = _load_layer_config()
    return classify(merged, rules)


def cmd_report(args: argparse.Namespace) -> int:
    """Handle the ``report`` subcommand: write a static HTML report.

    Args:
        args: Parsed CLI arguments (``root``, ``output``).

    Returns:
        Process exit code (always ``0``).
    """
    graph = build_graph(args.root)
    _, forbidden = _load_layer_config()
    violations = find_violations(graph, forbidden)
    cycles = find_cycles(graph)
    html = render(graph, violations, cycles, title=f"Dependency Graph — {args.root}")
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Handle the ``check`` subcommand: CI-usable pass/fail gate.

    Args:
        args: Parsed CLI arguments (``root``).

    Returns:
        ``0`` if no layer violations or circular dependencies were found,
        ``1`` otherwise.
    """
    graph = build_graph(args.root)
    _, forbidden = _load_layer_config()
    violations = find_violations(graph, forbidden)
    cycles = find_cycles(graph)
    for v in violations:
        print(f"VIOLATION: {v.edge.source_id} -> {v.edge.target_id} ({v.source_layer} -> {v.target_layer})")
    for cycle in cycles:
        print(f"CYCLE: {' -> '.join(cycle)}")
    if violations or cycles:
        print(f"FAILED: {len(violations)} violation(s), {len(cycles)} cycle(s)")
        return 1
    print("OK: no layer violations or circular dependencies found")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace with a bound ``func`` for dispatch.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    report_parser = subparsers.add_parser("report", help="Write a static HTML dependency report")
    report_parser.add_argument("root", type=Path)
    report_parser.add_argument("-o", "--output", type=Path, default=Path("dev-graph-report.html"))
    report_parser.set_defaults(func=cmd_report)

    check_parser = subparsers.add_parser("check", help="Exit non-zero on violations/cycles")
    check_parser.add_argument("root", type=Path)
    check_parser.set_defaults(func=cmd_check)

    return parser.parse_args()


def main() -> None:
    """Entry point: dispatch to the requested subcommand."""
    args = parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
