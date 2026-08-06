"""
Docstring Compliance Checker.

Traverses Python files, parses them with the `ast` module, and reports
modules, classes, and functions that are missing docstrings.

Output is a Rich table grouped by file, with line numbers and a short
description of each violation.

Attributes:
    console: Rich console for output.
    _KIND_STYLE: Mapping from node type to Rich style.
    _KIND_DESC: Mapping from node type to description.
    check_path: Check a single .py file and return a list of violation dicts.
    check_docstrings_recursive: Recursively collect violations from all .py files under *directory*.
    display_results: Render the violation report as a Rich table grouped by file.

Example:
    python check_docstrings.py <path1> [path2 ...]
"""

import ast
import os
import sys
from collections import defaultdict
from typing import Dict, List

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Error: The 'rich' library is required. Install with: pip install rich")
    sys.exit(1)

console = Console()

_KIND_STYLE: Dict[str, str] = {
    "Module": "blue",
    "Class": "yellow",
    "Function": "green",
}

_KIND_DESC: Dict[str, str] = {
    "Module": "Module is missing a top-level docstring",
    "Class": "Class is missing a class-level docstring",
    "Function": "Function/method is missing a docstring",
}


def check_path(path: str) -> List[dict]:
    """Check a single .py file and return a list of violation dicts.

    Args:
        path (str): Absolute or relative path to the Python file.

    Returns:
        List[dict]: Each dict has keys ``path``, ``line``, ``type``, ``name``, ``description``.
    """
    if not path.endswith(".py"):
        return []

    try:
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source)
    except SyntaxError as exc:
        console.print(f"[bold red]Syntax error in {path}: {exc}[/bold red]")
        return []

    violations: List[dict] = []

    def _add(line: int, kind: str, name: str) -> None:
        violations.append(
            {
                "path": path,
                "line": line,
                "type": kind,
                "name": name,
                "description": _KIND_DESC[kind],
            }
        )

    if not ast.get_docstring(tree):
        _add(1, "Module", "<module>")

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name.startswith("_") and not node.name.startswith("__"):
            continue
        if not ast.get_docstring(node):
            kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
            _add(node.lineno, kind, node.name)

    return violations


def check_docstrings_recursive(directory: str) -> List[dict]:
    """Recursively collect violations from all .py files under *directory*.

    Args:
        directory (str): Root directory to walk.

    Returns:
        List[dict]: Aggregated violation list from all files.
    """
    results: List[dict] = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            results.extend(check_path(os.path.join(root, fname)))
    return results


def display_results(violations: List[dict]) -> None:
    """Render the violation report as a Rich table grouped by file.

    Args:
        violations (List[dict]): Violation dicts produced by :func:`check_path`.
    """
    if not violations:
        console.print(
            Panel(
                "[bold green]SUCCESS[/bold green] — No missing docstrings found.",
                title="Docstring Compliance Report",
                expand=False,
            )
        )
        return

    # Group by file path for section breaks
    by_file: Dict[str, List[dict]] = defaultdict(list)
    for v in violations:
        by_file[v["path"]].append(v)

    table = Table(
        title=f"Docstring Compliance Report — {len(violations)} issue(s) in {len(by_file)} file(s)",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold magenta",
        show_lines=False,
    )

    table.add_column("Location", style="dim cyan", no_wrap=True)
    table.add_column("Type", justify="center", no_wrap=True, min_width=8)
    table.add_column("Name  —  Description", ratio=1)

    first_file = True
    for _fpath, file_violations in sorted(by_file.items()):
        if not first_file:
            table.add_section()
        first_file = False

        for v in sorted(file_violations, key=lambda x: x["line"]):
            color = _KIND_STYLE.get(v["type"], "white")
            kind_cell = Text(v["type"], style=color)
            issue_cell = Text()
            issue_cell.append(v["name"], style="bold yellow")
            issue_cell.append("  —  ", style="dim")
            issue_cell.append(v["description"], style=color)
            table.add_row(
                f"{v['path']}:{v['line']}",
                kind_cell,
                issue_cell,
            )

    console.print(table)

    # Summary footer
    kind_counts: Dict[str, int] = defaultdict(int)
    for v in violations:
        kind_counts[v["type"]] += 1

    parts = []
    for kind, count in sorted(kind_counts.items()):
        color = _KIND_STYLE.get(kind, "white")
        parts.append(f"[{color}]{kind}[/{color}]: {count}")
    console.print("  " + "  |  ".join(parts))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[bold red]Usage:[/bold red] python check_docstrings.py <path1> [path2 ...]")
        sys.exit(1)

    all_violations: List[dict] = []

    with console.status("[bold green]Scanning files...[/bold green]"):
        for arg in sys.argv[1:]:
            if os.path.isfile(arg):
                all_violations.extend(check_path(arg))
            elif os.path.isdir(arg):
                all_violations.extend(check_docstrings_recursive(arg))
            else:
                console.print(f"[bold yellow]Skipping invalid path: {arg}[/bold yellow]")

    display_results(all_violations)
