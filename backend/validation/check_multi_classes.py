"""
Attributes:
    RED: ANSI escape code for red text.
    CYAN: ANSI escape code for cyan text.
    YELLOW: ANSI escape code for yellow text.
    GREEN: ANSI escape code for green text.
    RESET: ANSI escape code for resetting text color.
    get_top_level_classes: Get top level classes.
    main: Main function to check for multiple top-level classes.

Example:
    >>> python logic/src/utils/validation/check_multi_classes.py logic/src --exclude logic/src/utils/validation
"""

import argparse
import ast
import os
import sys

# Terminal Colors
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"


def get_top_level_classes(filepath):
    """Get top level classes.

    Args:
    filepath (Any): Description of filepath.

    Returns:
        Any: Description of return value.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    except (SyntaxError, UnicodeDecodeError):
        return []


def main():
    """Main."""
    parser = argparse.ArgumentParser(description="Find files with multiple top-level class definitions.")
    parser.add_argument("path", nargs="?", default=".", help="Directory to scan")
    parser.add_argument("--ignore-private", action="store_true", help="Ignore classes starting with _")
    parser.add_argument("--exclude", action="append", help="Relative path(s) to exclude")
    args = parser.parse_args()

    # Normalize exclude paths to handle trailing slashes or different separators
    excluded_paths = set()
    if args.exclude:
        for p in args.exclude:
            excluded_paths.add(os.path.normpath(p))

    # Internal hidden dirs to always skip
    internal_skip = {".git", "__pycache__", "venv", ".venv", "env", "node_modules", "dist", "build"}

    print(f"{CYAN}Scanning for files with multiple top-level classes...{RESET}")
    if excluded_paths:
        print(f"Excluding paths: {YELLOW}{', '.join(excluded_paths)}{RESET}")
    print(f"{'File':<60} | {'Count':<5} | {'Classes Found'}")
    print("-" * 100)

    found_violations = False
    base_path = os.path.normpath(args.path)

    for root, dirs, files in os.walk(base_path):
        # 1. Calculate the relative path from the scan root to the current folder
        rel_root = os.path.relpath(root, base_path)
        if rel_root == ".":
            rel_root = ""

        # 2. Filter directories
        # We check both the name (for hidden folders) AND the relative path (for specific exclusions)
        dirs[:] = [
            d
            for d in dirs
            if d not in internal_skip and os.path.normpath(os.path.join(rel_root, d)) not in excluded_paths
        ]

        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                classes = get_top_level_classes(full_path)

                if args.ignore_private:
                    classes = [c for c in classes if not c.startswith("_")]

                if len(classes) > 1:
                    found_violations = True
                    display_path = os.path.relpath(full_path, base_path)

                    class_list_str = ", ".join(classes)
                    if len(class_list_str) > 35:
                        class_list_str = class_list_str[:32] + "..."

                    print(f"{display_path:<60} | {RED}{len(classes):<5}{RESET} | {YELLOW}{class_list_str}{RESET}")

    print("-" * 100)
    if found_violations:
        sys.exit(1)
    else:
        print(f"{GREEN}Clean! All scanned files contain at most one top-level class.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
