"""
Finds all relative imports (from .module import ...) in a Python codebase.

Attributes:
    RICH_AVAILABLE (bool): Whether the rich library is available.
    SKIP_DIRS (set[str]): Directories to skip during analysis.
    RED (str): ANSI escape code for red text.
    YELLOW (str): ANSI escape code for yellow text.
    GREEN (str): ANSI escape code for green text.
    CYAN (str): ANSI escape code for cyan text.
    DIM (str): ANSI escape code for dim text.
    RESET (str): ANSI escape code to reset text formatting.
    format_relative_import: Format the import string.
    analyze_file: Analyze a file for relative imports.
    print_stats_table: Print statistics table.
    main: Main function to find relative imports.

Example:
    >>> python check_relative_imports.py
    >>> python check_relative_imports.py --stats
    >>> python check_relative_imports.py --fail-on-found
"""

import argparse
import ast
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", "dist", "build"}

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


def format_relative_import(node: ast.ImportFrom) -> str:
    """
    Reconstruct the import string as it would appear in source.

    Args:
        node: The import node to format.

    Returns:
        Formatted import string.
    """
    dots = "." * (node.level or 0)
    module = node.module or ""
    names = ", ".join(alias.name if alias.asname is None else f"{alias.name} as {alias.asname}" for alias in node.names)
    return f"from {dots}{module} import {names}"


def analyze_file(filepath: Path) -> List[Tuple[int, int, str]]:
    """
    Return all relative imports in filepath as (lineno, level, formatted_import).

    Args:
        filepath: The path to the file to analyze.

    Returns:
        List of tuples containing the line number, level, and formatted import string.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    results: List[Tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.level or 0) >= 1:
            results.append((node.lineno, node.level or 0, format_relative_import(node)))

    return sorted(results, key=lambda x: x[0])


def print_stats_table(all_results: Dict[str, List[Tuple[int, int, str]]], target_root: Path) -> None:
    """
    Print a Rich table summarising relative import counts per top-level package.

    Args:
        all_results: Dictionary mapping file paths to lists of relative imports.
        target_root: The root directory of the codebase.

    Returns:
        None
    """
    if not RICH_AVAILABLE:
        return

    pkg_counts: Dict[str, int] = defaultdict(int)
    level_totals: Dict[int, int] = defaultdict(int)

    for filepath_str, results in all_results.items():
        rel = os.path.relpath(filepath_str, str(target_root))
        top = rel.split(os.sep)[0]
        pkg_counts[top] += len(results)
        for _, level, _ in results:
            level_totals[level] += 1

    total = sum(pkg_counts.values())
    console = Console()

    pkg_table = Table(title="Relative Import Summary by Package", title_style="bold magenta")
    pkg_table.add_column("Package / Directory", style="cyan")
    pkg_table.add_column("Relative Imports", justify="right", style="yellow")
    pkg_table.add_column("Share", justify="right")

    for pkg, count in sorted(pkg_counts.items(), key=lambda x: -x[1]):
        pct = f"{count / total * 100:.1f}%" if total else "0%"
        pkg_table.add_row(pkg, str(count), pct)

    pkg_table.add_section()
    pkg_table.add_row("[bold]TOTAL[/bold]", f"[bold]{total}[/bold]", "")
    console.print(pkg_table)

    level_table = Table(title="Relative Import Summary by Depth", title_style="bold magenta")
    level_table.add_column("Dots", justify="center", style="cyan")
    level_table.add_column("Meaning", style="dim")
    level_table.add_column("Count", justify="right", style="yellow")

    meanings = {1: "same package", 2: "parent package", 3: "grandparent"}
    for level in sorted(level_totals):
        dots = "." * level
        meaning = meanings.get(level, f"{level} levels up")
        level_table.add_row(dots, meaning, str(level_totals[level]))

    console.print(level_table)


def main() -> None:
    """
    Main entry point for the relative import checker.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Find all relative imports (from .module import ...) in a Python codebase."
    )
    parser.add_argument("directory", help="Root directory to scan")
    parser.add_argument("-e", "--exclude", nargs="+", default=[], help="Directory names to skip")
    parser.add_argument(
        "--min-level",
        type=int,
        default=1,
        metavar="N",
        help="Only report imports with at least N leading dots (default: 1 = all relative)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary tables grouped by package and import depth",
    )
    parser.add_argument(
        "--exclude-same-package",
        action="store_true",
        help="Skip single-dot imports (from . or from .module) that stay within the same package",
    )
    parser.add_argument(
        "--fail-on-found",
        action="store_true",
        help="Exit with code 1 if any relative imports are found (useful in CI)",
    )
    args = parser.parse_args()

    target_root = Path(args.directory)
    if not target_root.is_dir():
        print(f"Error: Directory '{target_root}' does not exist.")
        sys.exit(1)

    effective_min_level = max(args.min_level, 2) if args.exclude_same_package else args.min_level
    exclude: Set[str] = set(args.exclude) | SKIP_DIRS
    label = f"min-level={effective_min_level}" + (" [same-package excluded]" if args.exclude_same_package else "")
    print(f"Scanning '{target_root}'  ({label})...")
    print("=" * 60)

    files_found = 0
    total_imports = 0
    all_results: Dict[str, List[Tuple[int, int, str]]] = {}

    for root, dirs, files in os.walk(target_root):
        dirs[:] = [d for d in dirs if d not in exclude]
        for filename in sorted(files):
            if not filename.endswith(".py"):
                continue
            filepath = Path(root) / filename
            raw = analyze_file(filepath)
            results = [(ln, lvl, imp) for ln, lvl, imp in raw if lvl >= effective_min_level]
            if not results:
                continue

            all_results[str(filepath)] = results
            files_found += 1
            rel_path = os.path.relpath(str(filepath), str(target_root))
            print(f"\n{CYAN}📄 {rel_path}{RESET}")
            for lineno, level, import_str in results:
                dots = "." * level
                depth_color = YELLOW if level == 1 else RED
                print(f"   Line {lineno:<4} | {depth_color}{dots}{RESET} {import_str}")
                total_imports += 1

    print("\n" + "=" * 60)
    if total_imports == 0:
        print(f"{GREEN}✓  No relative imports found.{RESET}")
    else:
        print(f"Found {total_imports} relative import(s) across {files_found} file(s).")

    if args.stats and all_results:
        print()
        print_stats_table(all_results, target_root)

    if args.fail_on_found and total_imports > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
