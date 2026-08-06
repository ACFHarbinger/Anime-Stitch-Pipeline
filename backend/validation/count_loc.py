"""
Attributes:
    get_docstring_lines: Get docstring lines.
    analyze_file: Analyze file.
    group_by_directory: Aggregate file stats into buckets by the first `depth` directory components.
    main: Main function to count lines of code.

Example:
    >>> python logic/src/utils/validation/count_loc.py logic/src
"""

import argparse
import ast
import os
from collections import defaultdict
from typing import Dict, List, Set

from rich.console import Console
from rich.table import Table


def get_docstring_lines(source: str) -> Set[int]:
    """Get docstring lines.

    Args:
    source (str): Description of source.

    Returns:
        Any: Description of return value.
    """
    doc_lines: Set[int] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return doc_lines
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            const_node = node.body[0].value
            if isinstance(const_node.value, str):
                start = node.body[0].lineno
                end = getattr(node.body[0], "end_lineno", start)
                for i in range(start, end + 1):
                    doc_lines.add(i)
    return doc_lines


def analyze_file(filepath: str) -> Dict[str, int]:
    """Analyze file.

    Args:
    filepath (str): Description of filepath.

    Returns:
        Any: Description of return value.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
            lines = source.splitlines()
    except (UnicodeDecodeError, PermissionError):
        return {"code": 0, "comment": 0, "docstring": 0, "total": 0}

    docstring_lines_set = get_docstring_lines(source)
    n_code, n_comments, n_docs = 0, 0, 0

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if i in docstring_lines_set:
            n_docs += 1
        elif stripped.startswith("#"):
            n_comments += 1
        else:
            n_code += 1

    return {"code": n_code, "comment": n_comments, "docstring": n_docs, "total": n_code + n_comments + n_docs}


def group_by_directory(file_data: List[Dict], depth: int) -> List[Dict]:
    """
    Aggregate file stats into buckets by the first `depth` directory components.

    Args:
        file_data (List[Dict]): List of file data.
        depth (int): Number of directory levels to group by.

    Returns:
        List[Dict]: List of grouped data.
    """
    groups: Dict[str, Dict[str, int]] = defaultdict(lambda: {"code": 0, "comment": 0, "docstring": 0, "total": 0})
    for d in file_data:
        parts = d["path"].replace("\\", "/").split("/")
        key = "/".join(parts[:depth]) if len(parts) > depth else d["path"]
        for metric in ("code", "comment", "docstring", "total"):
            groups[key][metric] += d[metric]
    return [{"path": k, **v} for k, v in sorted(groups.items())]


def load_exceptions(filepath: str) -> Set[str]:
    """Load exception file paths from the given filepath.

    Args:
        filepath (str): Path to the exceptions text file.

    Returns:
        Set[str]: Set of normalized exception file paths.
    """
    exceptions: Set[str] = set()
    if not os.path.isfile(filepath):
        return exceptions
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                exceptions.add(os.path.normpath(stripped))
    return exceptions


def generate_markdown_report(display_data: List[Dict], file_data: List[Dict], limit: int, sort_key: str) -> str:
    """Generate a Markdown formatted report table.

    Args:
        display_data (List[Dict]): Table data to present.
        file_data (List[Dict]): Full set of file data for totals.
        limit (int): Row limit.
        sort_key (str): Key sorted on.

    Returns:
        str: Markdown document string.
    """
    lines = [
        f"# Codebase Line-Count Analysis Report",
        "",
        f"*Sorted by: {sort_key.upper()}*",
        "",
        "| Path | Code | Comments | Docstrings | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in display_data[:limit]:
        lines.append(f"| `{d['path']}` | {d['code']} | {d['comment']} | {d['docstring']} | {d['total']} |")
    lines.append(
        f"| **TOTALS** | **{sum(d['code'] for d in file_data)}** | "
        f"**{sum(d['comment'] for d in file_data)}** | "
        f"**{sum(d['docstring'] for d in file_data)}** | "
        f"**{sum(d['total'] for d in file_data)}** |"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Main."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="*", default=["."])
    parser.add_argument("--sort", choices=["code", "comment", "docstring", "total"], default="total")
    parser.add_argument("--limit", type=int, default=30, help="Number of files to show")
    parser.add_argument(
        "--group-by-dir",
        type=int,
        default=0,
        metavar="N",
        help="Aggregate stats by first N directory levels instead of per-file",
    )
    parser.add_argument(
        "--max-code-lines",
        type=int,
        default=500,
        help="Maximum allowed code lines per file before triggering a violation",
    )
    parser.add_argument(
        "--fail-over",
        action="store_true",
        help="Exit with non-zero exit code if any file exceeds max-code-lines",
    )
    parser.add_argument(
        "--exceptions-file",
        type=str,
        default="docs/loc_exceptions.txt",
        help="Path to file listing grandfathered file-size exception paths",
    )
    parser.add_argument(
        "--markdown-out",
        type=str,
        default=None,
        help="Optional path to output a Markdown report document",
    )
    args = parser.parse_args()

    console = Console()
    file_data: List[Dict] = []
    skip_dirs = {".git", "__pycache__", "venv", ".venv", "node_modules"}

    search_paths = args.path if isinstance(args.path, list) else [args.path]

    with console.status("[bold green]Analyzing codebase...", spinner="dots"):
        for search_path in search_paths:
            for root, dirs, files in os.walk(search_path):
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                for file in files:
                    if file.endswith(".py"):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, ".")
                        metrics = analyze_file(full_path)
                        if metrics["total"] > 0:
                            file_data.append({"path": rel_path, "full_path": full_path, **metrics})

    display_data = group_by_directory(file_data, args.group_by_dir) if args.group_by_dir > 0 else file_data
    display_data.sort(key=lambda x: x[args.sort], reverse=True)

    view_label = f"grouped by top-{args.group_by_dir} dir" if args.group_by_dir > 0 else "per file"
    table = Table(
        title=f"Codebase Analysis — sorted by {args.sort.upper()}  ({view_label})",
        title_style="bold magenta",
        show_footer=True,
    )

    table.add_column("Path", style="cyan", no_wrap=False, footer="TOTALS")
    table.add_column("Code", justify="right", style="green", footer=str(sum(d["code"] for d in file_data)))
    table.add_column("Comments", justify="right", style="yellow", footer=str(sum(d["comment"] for d in file_data)))
    table.add_column("Docstrings", justify="right", style="blue", footer=str(sum(d["docstring"] for d in file_data)))
    table.add_column("Total", justify="right", style="bold white", footer=str(sum(d["total"] for d in file_data)))

    for d in display_data[: args.limit]:
        table.add_row(d["path"], str(d["code"]), str(d["comment"]), str(d["docstring"]), str(d["total"]))

    console.print(table)
    if len(display_data) > args.limit:
        console.print(f"[dim]... and {len(display_data) - args.limit} more entries.[/dim]")

    if args.markdown_out:
        md_content = generate_markdown_report(display_data, file_data, args.limit, args.sort)
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown_out)), exist_ok=True)
        with open(args.markdown_out, "w", encoding="utf-8") as f:
            f.write(md_content)
        console.print(f"[bold green]Markdown report written to {args.markdown_out}[/bold green]")

    # Check for line count violations
    exceptions = load_exceptions(args.exceptions_file)
    violations: List[Dict] = []
    for item in file_data:
        norm_path = os.path.normpath(item["path"])
        if item["code"] > args.max_code_lines:
            if norm_path not in exceptions and not any(norm_path.endswith(exc) for exc in exceptions):
                violations.append(item)

    if violations:
        console.print(
            f"\n[bold red]FAILURE: Found {len(violations)} file(s) exceeding {args.max_code_lines} code lines limit:[/bold red]"
        )
        for v in violations:
            console.print(f"  [red]• {v['path']} ({v['code']} code lines)[/red]")
        if args.fail_over:
            sys.exit(1)
    elif args.fail_over:
        console.print(
            f"\n[bold green]SUCCESS: All files satisfy max code lines limit ({args.max_code_lines}) or are grandfathered in exceptions.[/bold green]"
        )


if __name__ == "__main__":
    import sys
    main()

