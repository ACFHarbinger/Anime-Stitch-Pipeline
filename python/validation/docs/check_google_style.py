"""
Google Style Docstring Validator.

Walks Python files and checks that every module, class, and function carries
a Google-style docstring with the required sections:

- **Module**: ``Attributes`` + ``Example``
- **Class**: ``Attributes``
- **Function/Method**: ``Args`` (when parameters exist), ``Returns``/``Yields``
  (when the function returns/yields a value)

Output is grouped by file and rendered in a Rich table.  The *Issue* column is
given extra width so descriptions are never truncated.

Attributes:
    console: Rich console for output.
    SECTION_ALIASES: Mapping from section aliases to normalized section names.
    _CONTEXT_LABEL: Mapping from AST node types to context labels.
    _CONTEXT_STYLE: Mapping from context labels to Rich styles.
    SKIP_DIRS: List of directories to skip.
    GoogleStyleValidator: AST visitor that collects Google-style docstring violations for one file.
    analyze_file: Analyzes a Python file for Google-style docstring violations.
    display_report: Displays the violations in a Rich table.
    main: Main entry point.

Example:
    python check_google_style.py [path]      # directory or single file (default: .)
    python check_google_style.py --help
"""

import argparse
import ast
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Union

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

SECTION_ALIASES: Dict[str, str] = {
    "ARGS": "Args",
    "ARGUMENTS": "Args",
    "PARAMETERS": "Args",
    "PARAMS": "Args",
    "ATTRIBUTES": "Attributes",
    "ATTRS": "Attributes",
    "EXAMPLE": "Example",
    "EXAMPLES": "Example",
    "RETURN": "Returns",
    "RETURNS": "Returns",
    "YIELD": "Yields",
    "YIELDS": "Yields",
    "RAISES": "Raises",
    "WARNS": "Warns",
    "NOTE": "Note",
    "NOTES": "Note",
    "TODO": "Todo",
}

_CONTEXT_LABEL: Dict[str, str] = {
    "Module": "Module",
    "ClassDef": "Class",
    "FunctionDef": "Func",
    "AsyncFunctionDef": "AsyncFunc",
    "Error": "ERROR",
}

_CONTEXT_STYLE: Dict[str, str] = {
    "Module": "blue",
    "Class": "yellow",
    "Func": "green",
    "AsyncFunc": "green",
    "ERROR": "bold red",
}

SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "env", "node_modules", "dist", "build"}


class GoogleStyleValidator(ast.NodeVisitor):
    """AST visitor that collects Google-style docstring violations for one file.

    Attributes:
        filepath (str): Path to the file being analysed.
        violations (List[dict]): Accumulated violations; each dict has keys
            ``line``, ``context``, ``name``, ``message``.
        current_class (Optional[str]): Name of the enclosing class, if any.
    """

    def __init__(self, filepath: str) -> None:
        """Initialise the validator for *filepath*.

        Args:
            filepath (str): Path to the Python source file.
        """
        self.filepath = filepath
        self.violations: List[dict] = []
        self.current_class: Optional[str] = None

    def _add(self, node: ast.AST, message: str) -> None:
        """Record a violation against *node* with the given *message*.

        Args:
            node (ast.AST): The AST node where the violation was found.
            message (str): Human-readable description of the violation.
        """
        self.violations.append(
            {
                "line": getattr(node, "lineno", 1),
                "context": type(node).__name__,
                "name": getattr(node, "name", "<module>"),
                "message": message,
            }
        )

    def _parse_sections(self, docstring: str) -> Dict[str, str]:
        """Extract named sections from a Google-style *docstring*.

        Args:
            docstring (str): Raw docstring text.

        Returns:
            Dict[str, str]: Mapping from normalised section name to its body.
        """
        if not docstring:
            return {}

        sections: Dict[str, str] = {}
        pattern = re.compile(
            r"^\s*(" + "|".join(set(SECTION_ALIASES.keys())) + r"):\s*$",
            re.MULTILINE | re.IGNORECASE,
        )
        matches = list(pattern.finditer(docstring))
        for i, match in enumerate(matches):
            key = SECTION_ALIASES.get(match.group(1).upper(), match.group(1).title())
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(docstring)
            sections[key] = docstring[start:end]
        return sections

    def _check_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], sections: Dict[str, str]) -> None:
        """Verify that every non-self parameter is documented in the Args section.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function node.
            sections (Dict[str, str]): Parsed docstring sections.
        """
        params = (
            [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
            + [a.arg for a in node.args.kwonlyargs]
            + ([node.args.vararg.arg] if node.args.vararg else [])
            + ([node.args.kwarg.arg] if node.args.kwarg else [])
        )
        if not params:
            return

        if "Args" not in sections:
            self._add(node, f"Missing 'Args' section — expected: {', '.join(params)}")
            return

        body = sections["Args"]
        missing = [p for p in params if not re.search(rf"^\s*{re.escape(p)}\s*(\(|:)", body, re.MULTILINE)]
        if missing:
            self._add(node, f"Undocumented argument(s): {', '.join(missing)}")

    def _check_returns_yields(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], sections: Dict[str, str]
    ) -> None:
        """Check that Returns/Yields sections are present when the function uses them.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function node.
            sections (Dict[str, str]): Parsed docstring sections.
        """
        has_yield = False
        has_return = False
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is not node:
                continue
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                has_yield = True
            elif isinstance(child, ast.Return) and child.value is not None:
                has_return = True

        if has_yield and "Yields" not in sections:
            self._add(node, "Function yields data but missing 'Yields' section")
        if has_return and node.name != "__init__" and "Returns" not in sections:
            self._add(node, "Function returns data but missing 'Returns' section")

    # ------------------------------------------------------------------
    # Visitor methods
    # ------------------------------------------------------------------

    def visit_Module(self, node: ast.Module) -> None:
        """Check module-level docstring for required sections.

        Args:
            node (ast.Module): The module AST node.
        """
        docstring = ast.get_docstring(node)
        if not docstring:
            self._add(node, "Missing module-level docstring")
        else:
            sections = self._parse_sections(docstring)
            if "Attributes" not in sections:
                self._add(node, "Module docstring missing 'Attributes' section")
            if "Example" not in sections:
                self._add(node, "Module docstring missing 'Example' section")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Check class docstring for required sections.

        Args:
            node (ast.ClassDef): The class AST node.
        """
        prev = self.current_class
        self.current_class = node.name
        docstring = ast.get_docstring(node)
        if not docstring:
            self._add(node, "Missing class docstring")
        else:
            sections = self._parse_sections(docstring)
            if "Attributes" not in sections:
                self._add(node, "Class docstring missing 'Attributes' section")
        self.generic_visit(node)
        self.current_class = prev

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Delegate to the shared function validator.

        Args:
            node (ast.FunctionDef): The function AST node.
        """
        self._validate_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Delegate to the shared function validator.

        Args:
            node (ast.AsyncFunctionDef): The async function AST node.
        """
        self._validate_function(node)

    def _validate_function(self, node: "Union[ast.FunctionDef, ast.AsyncFunctionDef]") -> None:
        """Core function validation: docstring presence, Args, Returns/Yields.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The function or async-function AST node.
        """
        docstring = ast.get_docstring(node)
        if not docstring:
            self._add(node, "Missing function docstring")
            return
        sections = self._parse_sections(docstring)
        self._check_args(node, sections)
        self._check_returns_yields(node, sections)


def analyze_file(filepath: str) -> List[dict]:
    """Parse *filepath* and return all Google-style violations found.

    Args:
        filepath (str): Path to the Python source file.

    Returns:
        List[dict]: Violations with keys ``line``, ``context``, ``name``, ``message``.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source)
        validator = GoogleStyleValidator(filepath)
        validator.visit(tree)
        return validator.violations
    except SyntaxError:
        return [{"line": 0, "context": "Error", "name": "File", "message": "Syntax error — file could not be parsed"}]
    except Exception as exc:  # noqa: BLE001
        return [{"line": 0, "context": "Error", "name": "File", "message": str(exc)}]


def display_report(all_violations: List[dict]) -> None:
    """Render the consolidated violation report as a Rich table.

    Violations are grouped by file and sorted by line number within each file.
    The *Issue* column uses ``min_width`` and ``ratio`` so it always gets the
    majority of the available terminal width.

    Args:
        all_violations (List[dict]): Violations with an injected ``filepath`` key.
    """
    if not all_violations:
        console.print(
            Panel(
                "[bold green]SUCCESS[/bold green] — All scanned files comply with Google Docstring standards!",
                title="Validation Complete",
                expand=False,
            )
        )
        sys.exit(0)

    # Group by file
    by_file: Dict[str, List[dict]] = defaultdict(list)
    for v in all_violations:
        by_file[v["filepath"]].append(v)

    table = Table(
        title=f"Google Style Violations — {len(all_violations)} issue(s) in {len(by_file)} file(s)",
        box=box.ROUNDED,
        header_style="bold white",
        show_lines=False,
    )

    table.add_column("Location", style="dim cyan", no_wrap=True)
    table.add_column("Type", justify="center", no_wrap=True, min_width=9)
    table.add_column("Issue", style="red", ratio=1)

    first_file = True
    for _, file_violations in sorted(by_file.items()):
        if not first_file:
            table.add_section()
        first_file = False

        for v in sorted(file_violations, key=lambda x: x["line"]):
            raw_ctx = v["context"]
            label = _CONTEXT_LABEL.get(raw_ctx, raw_ctx)
            style = _CONTEXT_STYLE.get(label, "white")
            ctx_cell = Text(label, style=style)
            location = f"{v['filepath']}:{v['line']} ({v['name']})"
            table.add_row(location, ctx_cell, v["message"])

    console.print(table)

    # Summary by context type
    counts: Dict[str, int] = defaultdict(int)
    for v in all_violations:
        label = _CONTEXT_LABEL.get(v["context"], v["context"])
        counts[label] += 1

    parts = [
        f"[{_CONTEXT_STYLE.get(k, 'white')}]{k}[/{_CONTEXT_STYLE.get(k, 'white')}]: {n}"
        for k, n in sorted(counts.items())
    ]
    console.print("\n  " + "  |  ".join(parts))
    console.print(f"\n[bold red]FAILURE:[/bold red] {len(all_violations)} issue(s) across {len(by_file)} file(s).")
    sys.exit(1)


def main() -> None:
    """Entry point: parse CLI arguments, scan files, and display the report.

    Attributes:
        parser (argparse.ArgumentParser): CLI argument parser.

    Example:
        >>> main()
    """
    parser = argparse.ArgumentParser(
        description="Strict Google Style Docstring Validator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?", default=".", help="Directory or file to scan (default: .)")
    # ADDED: new parameter to exclude specific directories
    parser.add_argument("--exclude_dir", nargs="*", default=[], help="Directories to exclude from the scan")
    args = parser.parse_args()

    console.rule("[bold cyan]Google Style Docstring Validator[/bold cyan]")
    console.print(
        "Checks: [yellow]Attributes[/yellow] (modules/classes)  "
        "[yellow]Example[/yellow] (modules)  "
        "[yellow]Args / Returns / Yields[/yellow] (functions)",
        justify="center",
        style="dim",
    )
    console.print()

    all_violations: List[dict] = []

    targets: List[tuple] = []
    if os.path.isfile(args.path):
        targets.append((os.path.dirname(args.path), [], [os.path.basename(args.path)]))
    else:
        exclude_paths = [os.path.abspath(d) for d in args.exclude_dir]
        for root, dirs, files in os.walk(args.path):
            abs_root = os.path.abspath(root)
            is_excluded = False
            for ep in exclude_paths:
                try:
                    # If relative path doesn't start with '..', abs_root is inside ep
                    if not os.path.relpath(abs_root, ep).startswith(".."):
                        is_excluded = True
                        break
                except ValueError:
                    pass  # Handle edge cases across different drives on Windows

            if is_excluded:
                dirs[:] = []  # Clear dirs to prevent os.walk from descending further
                continue

            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            targets.append((root, dirs, files))

    with console.status("[bold green]Scanning codebase...[/bold green]", spinner="dots"):
        for root, _, files in targets:
            for fname in files:
                if not fname.endswith(".py"):
                    continue

                full_path = os.path.join(root, fname) if root else fname

                if os.path.isfile(args.path):
                    display_path = os.path.basename(full_path)
                else:
                    try:
                        display_path = os.path.relpath(full_path, os.getcwd())
                    except ValueError:
                        display_path = full_path

                for v in analyze_file(full_path):
                    v["filepath"] = display_path
                    all_violations.append(v)

    display_report(all_violations)


if __name__ == "__main__":
    main()
