"""
Measures type annotation coverage in Python source files.
For every function and method, classifies its signature as:
  - Full   : all parameters annotated AND return type present
  - Partial : some (but not all) annotations present
  - None   : no annotations at all

Produces a Rich table sorted by coverage (worst offenders first by default).

Attributes:
    SKIP_DIRS (set[str]): Directories to skip during analysis.
    analyze_function: Analyze a function for type annotation coverage.
    analyze_file: Analyze a file for type annotation coverage.
    _coverage_markup: Format coverage percentage with color coding.
    main: Main function to check for type annotation coverage.

Example:
    >>> python check_type_coverage.py
    >>> python check_type_coverage.py --sort none
    >>> python check_type_coverage.py --limit 20
    >>> python check_type_coverage.py --min-funcs 5
"""

import argparse
import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

from rich.console import Console
from rich.table import Table

SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", "dist", "build"}


def analyze_function(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Tuple[int, int, bool]:
    """
    Analyze type annotation coverage in a single function.

    Args:
        node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): Function node to analyze.

    Returns:
        Tuple[int, int, bool]: Tuple containing (annotated_params, total_params, has_return_annotation).
    """
    skip = {"self", "cls"}
    params: List[ast.arg] = []
    params.extend(a for a in node.args.posonlyargs if a.arg not in skip)
    params.extend(a for a in node.args.args if a.arg not in skip)
    params.extend(a for a in node.args.kwonlyargs if a.arg not in skip)
    if node.args.vararg and node.args.vararg.arg not in skip:
        params.append(node.args.vararg)
    if node.args.kwarg and node.args.kwarg.arg not in skip:
        params.append(node.args.kwarg)

    total = len(params)
    annotated = sum(1 for p in params if p.annotation is not None)
    return annotated, total, node.returns is not None


def analyze_file(filepath: Path) -> Dict[str, int]:
    """
    Analyze type annotation coverage in a single file.

    Args:
        filepath (Path): Path to the file to analyze.

    Returns:
        Dict[str, int]: Dictionary containing coverage statistics.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return {"funcs": 0, "full": 0, "partial": 0, "none": 0}

    funcs = full = partial = none = 0

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        ann, total, has_return = analyze_function(node)
        funcs += 1

        if total == 0:
            if has_return:
                full += 1
            else:
                none += 1
        elif ann == total and has_return:
            full += 1
        elif ann == 0 and not has_return:
            none += 1
        else:
            partial += 1

    return {"funcs": funcs, "full": full, "partial": partial, "none": none}


def _coverage_markup(numerator: int, denominator: int) -> str:
    """
    Format coverage percentage with color coding.

    Args:
        numerator (int): Number of fully annotated functions.
        denominator (int): Total number of functions.

    Returns:
        str: Formatted coverage percentage.
    """
    if denominator == 0:
        return "[dim]—[/dim]"
    pct = numerator / denominator * 100
    color = "green" if pct >= 90 else ("yellow" if pct >= 60 else "red")
    return f"[{color}]{pct:.0f}%[/{color}]"


def main() -> None:
    """
    Main entry point for the type annotation coverage checker.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Measure type annotation coverage in Python files.")
    parser.add_argument("path", nargs="?", default=".", help="Directory to scan")
    parser.add_argument(
        "--sort",
        choices=["coverage", "none", "partial", "funcs"],
        default="coverage",
        help="Sort column (coverage = worst-first)",
    )
    parser.add_argument("--limit", type=int, default=40, help="Max files to display")
    parser.add_argument("--min-funcs", type=int, default=2, help="Minimum functions for a file to appear")
    args = parser.parse_args()

    console = Console()
    rows: List[Dict] = []
    with console.status("[bold green]Analysing type annotations...", spinner="dots"):
        for root, dirs, files in os.walk(args.path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = Path(root) / fname
                stats = analyze_file(fpath)
                if stats["funcs"] < args.min_funcs:
                    continue
                cov = stats["full"] / stats["funcs"]
                rows.append(
                    {
                        "path": os.path.relpath(str(fpath), args.path),
                        "cov": cov,
                        **stats,
                    }
                )

    if args.sort == "coverage":
        rows.sort(key=lambda r: r["cov"])
    else:
        rows.sort(key=lambda r: r[args.sort], reverse=True)

    totals = {k: sum(r[k] for r in rows) for k in ("funcs", "full", "partial", "none")}
    overall_pct = totals["full"] / totals["funcs"] if totals["funcs"] else 1.0

    table = Table(
        title=f"Type Annotation Coverage  —  overall: {overall_pct:.0%} fully annotated",
        title_style="bold magenta",
        show_footer=True,
    )
    table.add_column("File", style="cyan", footer="TOTALS")
    table.add_column("Funcs", justify="right", footer=str(totals["funcs"]))
    table.add_column("Full ✓", justify="right", style="green", footer=str(totals["full"]))
    table.add_column("Partial ⚠", justify="right", style="yellow", footer=str(totals["partial"]))
    table.add_column("None ✗", justify="right", style="red", footer=str(totals["none"]))
    table.add_column("Coverage", justify="right", footer=_coverage_markup(totals["full"], totals["funcs"]))

    for r in rows[: args.limit]:
        table.add_row(
            r["path"],
            str(r["funcs"]),
            str(r["full"]),
            str(r["partial"]),
            str(r["none"]),
            _coverage_markup(r["full"], r["funcs"]),
        )

    console.print(table)
    if len(rows) > args.limit:
        console.print(f"[dim]... and {len(rows) - args.limit} more files.[/dim]")


if __name__ == "__main__":
    main()
