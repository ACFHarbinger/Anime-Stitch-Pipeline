"""
Static analysis tool that checks whether all concrete classes that inherit
from ABC-based or Protocol-based interfaces implement every required abstract
method.

Works entirely on AST — no imports are executed. Reports every concrete class
that is missing one or more interface methods, grouped by interface.

Attributes:
    SKIP_DIRS: Set[str] = {".git", "__pycache__", "venv", ".venv", "node_modules", "dist", "build"}
    RED: str = "\033[91m"
    YELLOW: str = "\033[93m"
    GREEN: str = "\033[92m"
    CYAN: str = "\033[96m"
    BOLD: str = "\033[1m"
    DIM: str = "\033[2m"
    RESET: str = "\033[0m"
    collect_python_files: Collect all Python files in a directory and its subdirectories.
    base_name: Get the name of a base class from an AST expression.
    has_decorator: Check if a function has a specific decorator.
    parse_file: Return a list of class descriptors from filepath.
    build_registry: Map class name → list of class descriptors.
    get_required_abstract_methods: Recursively collect abstract methods required by a class and all its bases.
    get_implemented_methods: Recursively collect all non-abstract methods in a class and its ancestors.
    check_compliance: Return a list of violation descriptors for concrete classes with missing methods.
    main: Main function to check for interface compliance.

Example:
    >>> python check_interface_compliance.py --root src --exclude tests
"""

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", "dist", "build"}

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def collect_python_files(root: Path, exclude: Set[str]) -> List[Path]:
    """
    Collect all Python files in a directory and its subdirectories.

    Args:
        root: Root directory to scan.
        exclude: Set of directory names to exclude.

    Returns:
        List of file paths.
    """
    files: List[Path] = []
    for dirpath, dirs, filenames in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d not in exclude]
        for fname in filenames:
            if fname.endswith(".py"):
                files.append(Path(dirpath) / fname)
    return files


def base_name(base_node: ast.expr) -> str:
    """
    Get the name of a base class from an AST expression.

    Args:
        base_node: AST expression representing the base class.

    Returns:
        Name of the base class.
    """
    if isinstance(base_node, ast.Name):
        return base_node.id
    if isinstance(base_node, ast.Attribute):
        return base_node.attr
    return ""


def has_decorator(func_node: ast.FunctionDef, name: str) -> bool:
    """
    Check if a function has a specific decorator.

    Args:
        func_node: AST function node.
        name: Name of the decorator to check for.

    Returns:
        True if the function has the decorator, False otherwise.
    """
    for dec in func_node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == name:
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == name:
            return True
    return False


def parse_file(filepath: Path) -> List[Dict]:
    """
    Return a list of class descriptors from filepath:
    {name, bases, abstract_methods, all_methods, is_interface, lineno, filepath}

    Args:
        filepath: Path to the file to parse.

    Returns:
        List of class descriptors.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    results = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        bases = [base_name(b) for b in node.bases]
        abstract_methods: Set[str] = set()
        concrete_methods: Set[str] = set()

        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if has_decorator(item, "abstractmethod"):  # type: ignore[arg-type]
                abstract_methods.add(item.name)
            else:
                concrete_methods.add(item.name)

        is_interface = bool(abstract_methods or any(b in ("ABC", "ABCMeta", "Protocol") for b in bases))

        results.append(
            {
                "name": node.name,
                "bases": bases,
                "abstract_methods": abstract_methods,
                "concrete_methods": concrete_methods,
                "is_interface": is_interface,
                "lineno": node.lineno,
                "filepath": filepath,
            }
        )
    return results


def build_registry(files: List[Path]) -> Dict[str, List[Dict]]:
    """
    Map class name → list of class descriptors (multiple files may define same name).

    Args:
        files: List of file paths to scan.

    Returns:
        Dictionary mapping class name to list of class descriptors.
    """
    registry: Dict[str, List[Dict]] = {}
    for fpath in files:
        for cls in parse_file(fpath):
            registry.setdefault(cls["name"], []).append(cls)
    return registry


def get_required_abstract_methods(
    class_name: str,
    registry: Dict[str, List[Dict]],
    _seen: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Recursively collect abstract methods required by a class and all its bases.

    Args:
        class_name: Name of the class to check.
        registry: Dictionary mapping class name to list of class descriptors.
        _seen: Set of class names already seen (for recursion prevention).

    Returns:
        Set of required abstract methods.
    """
    if _seen is None:
        _seen = set()
    if class_name in _seen:
        return set()
    _seen.add(class_name)

    required: Set[str] = set()
    for entry in registry.get(class_name, []):
        required.update(entry["abstract_methods"])
        for base in entry["bases"]:
            if base not in ("ABC", "ABCMeta", "Protocol", "object"):
                required.update(get_required_abstract_methods(base, registry, _seen))
    return required


def get_implemented_methods(
    class_name: str,
    registry: Dict[str, List[Dict]],
    _seen: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Recursively collect all non-abstract methods in a class and its ancestors.

    Args:
        class_name: Name of the class to check.
        registry: Dictionary mapping class name to list of class descriptors.
        _seen: Set of class names already seen (for recursion prevention).

    Returns:
        Set of implemented methods.
    """
    if _seen is None:
        _seen = set()
    if class_name in _seen:
        return set()
    _seen.add(class_name)

    implemented: Set[str] = set()
    for entry in registry.get(class_name, []):
        implemented.update(entry["concrete_methods"])
        for base in entry["bases"]:
            if base not in ("ABC", "ABCMeta", "Protocol", "object"):
                implemented.update(get_implemented_methods(base, registry, _seen))
    return implemented


def check_compliance(registry: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Return a list of violation descriptors for concrete classes with missing methods.

    Args:
        registry: Dictionary mapping class name to list of class descriptors.

    Returns:
        List of violation descriptors.
    """
    violations: List[Dict] = []

    for class_name, entries in registry.items():
        for entry in entries:
            if entry["is_interface"]:
                continue

            required: Set[str] = set()
            for base in entry["bases"]:
                required.update(get_required_abstract_methods(base, registry))

            if not required:
                continue

            implemented = get_implemented_methods(class_name, registry)
            missing = required - implemented

            if missing:
                violations.append(
                    {
                        "class": class_name,
                        "filepath": entry["filepath"],
                        "lineno": entry["lineno"],
                        "bases": entry["bases"],
                        "missing": sorted(missing),
                        "required": sorted(required),
                        "implemented": sorted(implemented & required),
                    }
                )

    return violations


def main() -> None:
    """Main function to check interface compliance."""
    parser = argparse.ArgumentParser(
        description="Check that concrete classes implement all abstract interface methods."
    )
    parser.add_argument("directory", help="Root directory to scan")
    parser.add_argument("--exclude", nargs="+", default=[], help="Directory names to skip")
    parser.add_argument(
        "--interfaces-dir",
        default="",
        metavar="PATH",
        help="Optional path to restrict interface discovery (e.g. logic/src/interfaces)",
    )
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.")
        sys.exit(1)

    exclude = set(args.exclude)
    print(f"{CYAN}Scanning '{root}'...{RESET}")
    files = collect_python_files(root, exclude)
    print(f"  {DIM}{len(files)} Python files found.{RESET}\n")

    registry = build_registry(files)

    # Identify which class names are known interfaces for the summary
    known_interfaces = {name for name, entries in registry.items() if any(e["is_interface"] for e in entries)}

    print(f"{CYAN}Checking interface compliance...{RESET}")
    print(f"  {DIM}{len(known_interfaces)} interfaces/abstract classes found.{RESET}\n")

    violations = check_compliance(registry)

    if not violations:
        print(f"{GREEN}✓  All concrete classes satisfy their interface contracts.{RESET}")
    else:
        print(f"{RED}{BOLD}✗  Found {len(violations)} compliance violation(s):{RESET}\n")
        for v in sorted(violations, key=lambda x: (str(x["filepath"]), x["class"])):
            rel = v["filepath"].as_posix()
            bases_str = ", ".join(v["bases"])
            print(f"  {YELLOW}{v['class']}{RESET}  {DIM}({rel}:{v['lineno']}){RESET}")
            print(f"    Inherits : {DIM}{bases_str}{RESET}")
            print(f"    {RED}Missing  :{RESET}")
            for m in v["missing"]:
                print(f"      {RED}✗{RESET} {m}()")
            if v["implemented"]:
                print(f"    {GREEN}Satisfied:{RESET} {', '.join(v['implemented'])}")
            print()

    concrete_count = sum(1 for entries in registry.values() for e in entries if not e["is_interface"])
    checked_count = sum(
        1
        for entries in registry.values()
        for e in entries
        if not e["is_interface"] and any(get_required_abstract_methods(b, registry) for b in e["bases"])
    )
    print(
        f"{DIM}Summary: {len(violations)} violations across {checked_count} checked classes "
        f"({concrete_count} total concrete classes, {len(known_interfaces)} interfaces).{RESET}"
    )

    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
