"""Python import-graph parser — the `input/python` half of the D1 contract.

Walks a directory tree of `.py` files, AST-parses each one, and emits a
:class:`~dev.src.core.model.CodeGraph` of module nodes and IMPORT edges.
Adapted from the AST-walk approach independently built in three of the
template's source repos (Build-Optimization/WSmart-Route
`visualize_module_graph.py`, Image-Toolkit `check_circular_imports.py`),
consolidated into the dev/src/input/python/ submodule per
moon/roadmaps/developer_tools.md D9 (Python slice).

Usage:
    python -m dev.src.input.python.parser python/src > graph.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # dev/src/

from core.model import CodeGraph, Diagnostic, Edge, EdgeKind, Node, Severity  # noqa: E402


def _module_name(file_path: Path, root: Path) -> str:
    """Derive a dotted module name from a file path relative to ``root``.

    Args:
        file_path: Absolute or root-relative path to a ``.py`` file.
        root: The scan root (added to ``sys.path`` conceptually).

    Returns:
        A dotted module name, e.g. ``app.utils.io``.
    """
    relative = file_path.relative_to(root)
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else relative.stem


def _resolve_import(
    module: str | None, level: int, importing_module: str
) -> str | None:
    """Resolve an ``ast.ImportFrom`` to a best-effort absolute dotted name.

    Args:
        module: The ``module`` attribute of the ``ImportFrom`` node (may be
            ``None`` for a bare ``from . import x``).
        level: Relative-import level (0 = absolute).
        importing_module: Dotted name of the module doing the importing.

    Returns:
        A best-effort absolute dotted module name, or ``None`` if it cannot
        be resolved (e.g. a level that walks above the scan root).
    """
    if level == 0:
        return module
    parts = importing_module.split(".")
    base_parts = parts[: -level] if level <= len(parts) else []
    if not base_parts and level > len(parts):
        return None
    prefix = ".".join(base_parts)
    if module:
        return f"{prefix}.{module}" if prefix else module
    return prefix or None


def parse_directory(root: Path) -> CodeGraph:
    """Build a :class:`CodeGraph` of Python modules and their import edges.

    Args:
        root: Directory to scan recursively for ``.py`` files.

    Returns:
        A populated :class:`CodeGraph` with one node per module and one
        IMPORT edge per resolved ``import``/``from ... import`` statement.
    """
    py_files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    module_names = {f: _module_name(f, root) for f in py_files}
    known_modules = set(module_names.values())

    nodes: list[Node] = []
    edges: list[Edge] = []
    diagnostics: list[Diagnostic] = []

    for file_path, module_name in module_names.items():
        source_id = f"python:{module_name}"
        nodes.append(
            Node(
                id=source_id,
                language="python",
                kind="module",
                qualified_name=module_name,
                file_path=str(file_path),
            )
        )
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except SyntaxError as exc:
            diagnostics.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    message=f"SyntaxError: {exc.msg}",
                    file_path=str(file_path),
                    line=exc.lineno or 0,
                )
            )
            continue

        for stmt in ast.walk(tree):
            targets: list[str] = []
            if isinstance(stmt, ast.Import):
                targets.extend(alias.name for alias in stmt.names)
            elif isinstance(stmt, ast.ImportFrom):
                base = _resolve_import(stmt.module, stmt.level, module_name)
                if base:
                    # `from base import name` may import a submodule (base.name)
                    # or an attribute of base — try the submodule form first
                    # since that's what makes it a real module-graph edge.
                    for alias in stmt.names:
                        submodule = f"{base}.{alias.name}"
                        targets.append(submodule if submodule in known_modules else base)
            for target in targets:
                # Only keep edges that land inside this scan root; external
                # (stdlib/third-party) imports are not part of this graph.
                # Exact match wins over a prefix match (e.g. "pkg.b" over "pkg").
                if target in known_modules:
                    matched = target
                else:
                    matched = next(
                        (m for m in known_modules if target.startswith(m + ".")),
                        None,
                    )
                if matched is None or matched == module_name:
                    continue
                target_id = f"python:{matched}"
                edge = Edge(source_id=source_id, target_id=target_id, kind=EdgeKind.IMPORT)
                if edge not in edges:
                    edges.append(edge)

    return CodeGraph(nodes=nodes, edges=edges, diagnostics=diagnostics, source_language="python")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory to scan for .py files")
    return parser.parse_args()


def main() -> None:
    """Entry point: parse the given directory and print a CodeGraph as JSON."""
    args = parse_args()
    graph = parse_directory(args.root.resolve())
    json.dump(graph.to_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
