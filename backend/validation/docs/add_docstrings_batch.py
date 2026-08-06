#!/usr/bin/env python3
"""
Batch Docstring Adder (AST-Based)

This script automatically detects missing docstrings in Python files and
injects Google-Style docstrings. It supports type hint extraction and
handles complex multi-line function signatures.

Attributes:
    TEMPLATES: Mapping from node types to Google-Style docstring templates.
    DocstringInjector: Class for injecting docstrings into Python files.
    main: Main entry point.

Example:
    python add_docstrings_batch.py <file_or_directory> ...
"""

import argparse
import ast
import contextlib
import os
from pathlib import Path
from typing import Any, List

# --- Templates (Google Style) ---

TEMPLATES = {
    "module": '''"""{filename} module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import {module_name}
    """''',
    "class": '''"""{class_name} class.

    Attributes:
        attr (Type): Description of attribute.
    """''',
    "init": '''"""Initialize {class_name}.

        Args:
{args_section}
        """''',
    "function": '''"""{summary}.

        Args:
{args_section}

        Returns:
            {return_type}: Description of return value.
        """''',
    "function_yield": '''"""{summary}.

        Args:
{args_section}

        Yields:
            {return_type}: Description of yielded value.
        """''',
    "function_void": '''"""{summary}.

        Args:
{args_section}
        """''',
    "function_simple": '''"""{summary}."""''',
}


class DocstringInjector:
    """DocstringInjector class.

    Attributes:
        attr (Type): Description of attribute.
    """

    def __init__(self, filepath: str):
        """Initialize Class.

        Args:
            filepath (str): Description of filepath.
        """
        self.filepath = filepath
        self.lines = Path(filepath).read_text(encoding="utf-8").splitlines()
        # We process from bottom to top to avoid invalidating line numbers
        self.modifications: List[Any] = []

    def _get_indent(self, lineno: int) -> str:
        """Returns the indentation string of a specific line.

        Args:
            lineno (int): Line number.

        Returns:
            str: Indentation string.
        """
        if lineno > len(self.lines):
            return ""
        line = self.lines[lineno - 1]
        return line[: len(line) - len(line.lstrip())]

    def _format_args(self, args: List[ast.arg], base_indent: str) -> str:
        """Formats the Args section with type hints.

        Args:
            args (List[ast.arg]): List of arguments.
            base_indent (str): Base indentation string.

        Returns:
            str: Formatted Args section.
        """
        lines = []
        indent = base_indent + "    "  # Standard 4-space indent for args

        for arg in args:
            if arg.arg in ("self", "cls"):
                continue

            # Extract annotation if available
            type_name = "Any"
            if arg.annotation:
                with contextlib.suppress(AttributeError):
                    type_name = ast.unparse(arg.annotation)

            lines.append(f"{indent}{arg.arg} ({type_name}): Description of {arg.arg}.")

        return "\n".join(lines) if lines else f"{indent}None."

    def generate_docstring(self, node: Any, context: str = "") -> str:
        """Generates the appropriate docstring based on node type and content.

        Args:
            node (Any): Node to generate docstring for.
            context (str, optional): Context of the node. Defaults to "".

        Returns:
            str: Generated docstring.
        """
        if isinstance(node, ast.Module):
            filename = os.path.basename(self.filepath)
            return TEMPLATES["module"].format(filename=filename, module_name=filename.replace(".py", ""))

        if isinstance(node, ast.ClassDef):
            return TEMPLATES["class"].format(class_name=node.name)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Analyze function properties
            args = node.args.args + node.args.kwonlyargs
            if node.args.vararg:
                args.append(node.args.vararg)
            if node.args.kwarg:
                args.append(node.args.kwarg)

            has_args = any(a.arg not in ("self", "cls") for a in args)
            has_yield = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node))
            has_return = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node))

            summary = node.name.replace("_", " ").capitalize()
            indent = self._get_indent(node.lineno) + "    "
            args_section = self._format_args(args, indent)

            if node.name == "__init__":
                return TEMPLATES["init"].format(class_name=context, args_section=args_section)

            if has_yield:
                return TEMPLATES["function_yield"].format(summary=summary, args_section=args_section, return_type="Any")

            if has_return:
                # If it returns but has no args, we might skip Args section entirely?
                # For strict Google style, if args exist, they must be documented.
                if has_args:
                    return TEMPLATES["function"].format(summary=summary, args_section=args_section, return_type="Any")
                else:
                    # No args, just return
                    return f'''"""{summary}.\n\n{indent}Returns:\n{indent}    Any: Description.\n{indent}"""'''

            # Void function
            if has_args:
                return TEMPLATES["function_void"].format(summary=summary, args_section=args_section)

            return TEMPLATES["function_simple"].format(summary=summary)

        return ""

    def _find_insertion_line(self, node: Any) -> int:
        """Finds the correct line index to insert the docstring.

        Args:
            node (Any): Node to find insertion line for.

        Returns:
            int: Line index to insert docstring.
        """
        # node.lineno is the start of the definition (e.g., 'def foo():')
        # We need to find the end of the signature (the colon).
        # We iterate from node.lineno downwards.

        start = node.lineno - 1
        for i in range(start, len(self.lines)):
            line = self.lines[i].strip()
            if line.endswith(":"):
                return i + 1
        return start + 1

    def _queue_docstrings(self, tree):
        """Internal helper to queue docstrings based on node types.

        Args:
            tree (Any): Tree to queue docstrings for.
        """
        if not ast.get_docstring(tree):
            doc = self.generate_docstring(tree)
            self.modifications.append((0, doc, ""))

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not ast.get_docstring(node):
                context = ""
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
                    context = "Class"
                doc = self.generate_docstring(node, context)
                idx = self._find_insertion_line(node)
                indent = self._get_indent(node.lineno) + "    "
                self.modifications.append((idx, doc, indent))

    def scan_and_queue(self):
        """Scans the file and queues missing docstrings for insertion.

        Returns:
            None
        """
        try:
            tree = ast.parse("\n".join(self.lines))
        except SyntaxError:
            print(f"Skipping {self.filepath}: Syntax Error")
            return

        self.modifications = []
        self._queue_docstrings(tree)
        self.modifications.sort(key=lambda x: x[0], reverse=True)

    def apply(self):
        """Applies the queued modifications to the source lines.

        Returns:
            None
        """
        for idx, doc, indent in self.modifications:
            # Format lines with indentation
            doc_lines = [indent + line if line.strip() else line for line in doc.splitlines()]
            self.lines.insert(idx, "\n".join(doc_lines))

    def save(self):
        """Save.

        Returns:
            None
        """
        Path(self.filepath).write_text("\n".join(self.lines), encoding="utf-8")


def main():
    """Main."""
    parser = argparse.ArgumentParser(description="Batch Add Google Docstrings.")
    parser.add_argument("paths", nargs="+", help="Files or directories to scan.")
    args = parser.parse_args()

    for path_arg in args.paths:
        path = Path(path_arg)
        files = [path] if path.is_file() else path.rglob("*.py")

        for f in files:
            if f.name.startswith("_") and f.name != "__init__.py":
                continue
            if "venv" in f.parts or "__pycache__" in f.parts:
                continue

            print(f"Processing {f}...")
            injector = DocstringInjector(str(f))
            injector.scan_and_queue()

            if injector.modifications:
                injector.apply()
                injector.save()
                print(f"  -> Added {len(injector.modifications)} docstrings.")
            else:
                print("  -> No missing docstrings.")


if __name__ == "__main__":
    main()
