"""
Attributes:
    get_docstring_lines: Get docstring lines.
    analyze_file: Analyze file.
    print_tree: Print tree.
    main: Main function to count lines of code.

Example:
    >>> python logic/src/utils/validation/tree_loc.py logic/src
"""

import argparse
import ast
import os
from typing import Set, Tuple


def get_docstring_lines(source: str) -> Set[int]:
    """
    Parses the source code AST to find all lines that are part of docstrings.

    Args:
        source (str): Source code.
    Returns:
        Set[int]: Set of line numbers that are part of docstrings.
    """
    doc_lines: Set[int] = set()
    try:
        tree = ast.parse(source)
    except Exception:
        return doc_lines

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            start = node.body[0].lineno
            end = getattr(node.body[0], "end_lineno", start)
            for i in range(start, end + 1):
                doc_lines.add(i)
    return doc_lines


def analyze_file(filepath: str) -> Tuple[int, int, int]:
    """
    Analyze a file and return the number of code lines, comment lines, and docstring lines.

    Args:
        filepath (str): File path.
    Returns:
        Tuple[int, int, int]: (number of code lines, number of comment lines, number of docstring lines).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
            lines = source.splitlines()
    except Exception:
        return 0, 0, 0

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
    return n_code, n_comments, n_docs


def print_tree(directory, prefix="", is_last=True, skip_dirs=None) -> None:
    """
    Print tree.

    Args:
        directory (Any): Description of directory.
        prefix (Any): Description of prefix.
        is_last (Any): Description of is_last.
        skip_dirs (Any): Description of skip_dirs.
    """
    if skip_dirs is None:
        skip_dirs = {".git", "__pycache__", "venv", ".venv", "node_modules"}

    # Prepare visual markers
    marker = "└── " if is_last else "├── "
    name = os.path.basename(directory) or directory

    # We don't calculate stats for folders themselves here, just print the name
    print(f"{prefix}{marker}{name}/")

    # Update prefix for children
    new_prefix = prefix + ("    " if is_last else "│   ")

    # Get all items, filter ignored ones
    try:
        items = sorted([i for i in os.listdir(directory) if i not in skip_dirs])
    except PermissionError:
        return

    # Separate directories and files
    dirs = [i for i in items if os.path.isdir(os.path.join(directory, i))]
    files = [i for i in items if i.endswith(".py") and os.path.isfile(os.path.join(directory, i))]

    entries = dirs + files

    for i, entry in enumerate(entries):
        full_path = os.path.join(directory, entry)
        last_entry = i == len(entries) - 1

        if os.path.isdir(full_path):
            print_tree(full_path, new_prefix, last_entry, skip_dirs)
        else:
            c, m, d = analyze_file(full_path)
            entry_marker = "└── " if last_entry else "├── "
            # Format the output to keep the columns somewhat aligned
            stats = f" [Code: {c:>4} | Cmnt: {m:>3} | Doc: {d:>3}]"
            print(f"{new_prefix}{entry_marker}{entry:<30}{stats}")


def main() -> None:
    """
    Main function.

    Args:
        path (str): Path to the root directory.
    """
    parser = argparse.ArgumentParser(description="Output Python LoC in a tree format.")
    parser.add_argument("path", nargs="?", default=".", help="Root directory")
    args = parser.parse_args()

    print(f"{'Structure':<40} | {'Statistics'}")
    print("-" * 80)
    print_tree(args.path)


if __name__ == "__main__":
    main()
