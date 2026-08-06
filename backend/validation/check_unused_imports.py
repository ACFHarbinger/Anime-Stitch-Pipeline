"""
Finds unused imports in a Python codebase using AST analysis.
Produces a Rich table of unused imports per file.

Attributes:
    SKIP_DIRS (set[str]): Directories to skip during analysis.
    analyze_file: Analyze a file for unused imports.
    main: Main function to check for unused imports.

Example:
    python -m src.utils.validation.check_unused_imports src/policies
"""

import argparse
import ast
import os
import sys
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
RESET = "\033[0m"


class UsageVisitor(ast.NodeVisitor):
    """
    AST visitor to collect all used names in a module.

    Attributes:
        used_names (set[str]): The set of used names.
    """

    def __init__(self) -> None:
        """
        Initialize the visitor.

        Returns:
            None
        """
        self.used_names: Set[str] = set()

    def visit_Name(self, node: ast.Name):
        """
        Visit a name node.

        Args:
            node: The name node to visit.
        """
        if isinstance(node.ctx, (ast.Load, ast.Del)):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """
        Visit an attribute node.

        Args:
            node: The attribute node to visit.
        """
        # In 'os.path.join', 'os' is a Name, but we also want to be careful.
        # ast.walk or generic_visit will find the base Name.
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        """
        Visit a constant node.

        Args:
            node: The constant node to visit.
        """
        # Strings in type hints like 'List["MyClass"]'
        if isinstance(node.value, str):
            # This is a bit naive but helps with some type hint strings
            self.used_names.add(node.value)
        self.generic_visit(node)


def get_factory_line_ranges(tree: ast.AST) -> List[Tuple[int, int]]:
    """
    Identifies the start and end lines of all classes containing 'Factory'.

    Args:
        tree: The AST node to extract imports from.

    Returns:
        A list of tuples, where each tuple contains the start and end line numbers of a Factory class.
    """
    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and "Factory" in node.name:
            start = node.lineno
            # end_lineno is Python 3.8+. Fallback to walking children if not available.
            if hasattr(node, "end_lineno"):
                end = node.end_lineno
            else:
                end = max((n.lineno for n in ast.walk(node) if hasattr(n, "lineno")), default=start)
            ranges.append((start, end))
    return ranges


def analyze_file(filepath: Path, ignore_factories: bool = False) -> List[Tuple[int, str]]:  # noqa: C901
    """
    Return all unused imports in filepath as (lineno, name).

    Args:
        filepath: The path to the file to analyze.
        ignore_factories: Whether to ignore imports inside Factory classes.

    Returns:
        A list of tuples, where each tuple contains the line number and name of an unused import.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    # 1. Collect all imports
    # name -> (lineno, node)
    imports: Dict[str, int] = {}

    # Track __all__ to consider those names "used"
    defined_in_all: Set[str] = set()

    factory_ranges = get_factory_line_ranges(tree) if ignore_factories else []

    for node in ast.walk(tree):
        # Skip nested imports if they are inside a Factory class
        if (
            hasattr(node, "lineno")
            and ignore_factories
            and any(start <= node.lineno <= end for start, end in factory_ranges)
        ):
            # We still want to walk inside if it's NOT an import node,
            # but actually we only want to ignore the IMPORT nodes.
            # If we skip here, we might miss usages inside the factory.
            # So let's only skip if it IS an import node.
            pass

        # Handle Import (e.g., import os, sys)
        if isinstance(node, ast.Import):
            if (
                hasattr(node, "lineno")
                and ignore_factories
                and any(start <= node.lineno <= end for start, end in factory_ranges)
            ):
                continue
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                imports[name] = node.lineno

        # Handle ImportFrom (e.g., from math import sqrt)
        elif isinstance(node, ast.ImportFrom):
            if (
                hasattr(node, "lineno")
                and ignore_factories
                and any(start <= node.lineno <= end for start, end in factory_ranges)
            ):
                continue
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name == "*":
                    # We can't easily track star imports with AST alone
                    continue
                name = alias.asname or alias.name
                imports[name] = node.lineno

        # Handle __all__ = ["name1", "name2"]
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__all__"
                    and isinstance(node.value, (ast.List, ast.Tuple, ast.Set))
                ):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            defined_in_all.add(elt.value)

    # 2. Collect all usages
    visitor = UsageVisitor()
    visitor.visit(tree)
    used_names = visitor.used_names | defined_in_all

    # 3. Identify unused imports
    unused = []
    for name, lineno in imports.items():
        if name not in used_names:
            unused.append((lineno, name))

    return sorted(unused)


def main() -> None:  # noqa: C901
    """
    Main function to find unused imports in a Python codebase.

    This function:
    1. Parses command-line arguments for the target directory/file and exclusions.
    2. Initializes Rich console and table for formatted output (if Rich is available).
    3. Walks the target directory (or analyzes a single file if specified).
    4. Calls `analyze_file` for each Python file to find unused imports.
    5. Collects and sorts all unused imports.
    6. Prints the results in a formatted table (Rich) or plain text (fallback).
    """
    parser = argparse.ArgumentParser(description="Find unused imports in a Python codebase.")
    parser.add_argument("target", type=str, help="Directory or file to analyze.")
    parser.add_argument("--exclude", type=str, nargs="+", help="Sub-directories to exclude.")
    parser.add_argument("--ignore_factories", action="store_true", help="Ignore nested imports inside Factory classes.")
    args = parser.parse_args()

    target_path = Path(args.target).resolve()
    exclude_paths = [target_path / e for e in (args.exclude or [])]

    if not target_path.exists():
        print(f"{RED}Error: Target path '{target_path}' does not exist.{RESET}")
        sys.exit(1)

    all_unused: Dict[str, List[Tuple[int, str]]] = {}
    files_analyzed = 0

    if target_path.is_file():
        if target_path.suffix == ".py":
            unused = analyze_file(target_path, ignore_factories=args.ignore_factories)
            if unused:
                all_unused[str(target_path)] = unused
            files_analyzed = 1
    else:
        for root, dirs, files in os.walk(target_path):
            current_root = Path(root)
            if any(current_root.name == skip or current_root == ex for skip in SKIP_DIRS for ex in exclude_paths):
                dirs[:] = []  # skip this directory
                continue

            for file in files:
                if file.endswith(".py"):
                    filepath = current_root / file
                    if any(filepath.is_relative_to(ex) for ex in exclude_paths):
                        continue

                    unused = analyze_file(filepath, ignore_factories=args.ignore_factories)
                    if unused:
                        rel_path = filepath.relative_to(target_path)
                        all_unused[str(rel_path)] = unused
                    files_analyzed += 1

    if not all_unused:
        print(f"{GREEN}No unused imports found in {files_analyzed} files.{RESET}")
        return

    if RICH_AVAILABLE:
        console = Console()
        table = Table(title=f"Unused Imports Found ({len(all_unused)} files)")
        table.add_column("File", style="cyan")
        table.add_column("Line", style="yellow")
        table.add_column("Import Name", style="red")
        for filepath, unused_list in sorted(all_unused.items()):
            for lineno, name in unused_list:
                table.add_row(str(filepath), str(lineno), name)

        console.print(table)
    else:
        print(f"\n{YELLOW}Unused Imports Found ({len(all_unused)} files):{RESET}")
        for filepath, unused_list in sorted(all_unused.items()):
            print(f"{CYAN}{filepath}:{RESET}")
            for lineno, name in unused_list:
                print(f"  {YELLOW}Line {lineno:4}:{RESET} {RED}{name}{RESET}")


if __name__ == "__main__":
    main()
