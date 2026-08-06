"""
An AST-based utility to detect embedded HTML, CSS, JavaScript, or SQL
inside Python source files.

Attributes:
    LANGUAGE_PATTERNS: Dictionary mapping language names to regex patterns.
    CYAN: ANSI escape code for cyan text.
    GREEN: ANSI escape code for green text.
    YELLOW: ANSI escape code for yellow text.
    RED: ANSI escape code for red text.
    RESET: ANSI escape code for resetting text color.
    load_patterns: Load regex patterns from external files.
    get_docstring_lines: Find all line numbers that belong to docstrings so we can ignore them.
    detect_language: Detect the language of the given text.
    EmbeddedCodeVisitor: AST visitor to detect embedded languages in Python code.
    analyze_file: Parse a file and return embedded code findings.
    main: Main function to check for embedded languages.

Example:
    >>> python check_embedded_languages.py --root src --exclude tests
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Terminal Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def load_patterns() -> Dict[str, re.Pattern]:
    """
    Dynamically load regex patterns from external files.

    Returns:
        Dictionary mapping language names to regex patterns.
    """
    patterns = {}
    base_dir = Path(__file__).parent / "target"

    try:
        # Keep HTML and JS case-insensitive as they often vary
        with open(base_dir / "pattern.html", "r", encoding="utf-8") as f:
            patterns["HTML"] = re.compile(f.read().strip(), re.IGNORECASE)

        with open(base_dir / "pattern.js", "r", encoding="utf-8") as f:
            patterns["JavaScript"] = re.compile(f.read().strip(), re.IGNORECASE)

        with open(base_dir / "pattern.css", "r", encoding="utf-8") as f:
            patterns["CSS"] = re.compile(f.read().strip(), re.IGNORECASE)

        # CHANGE: Remove re.IGNORECASE for SQL to prevent English sentence collisions
        with open(base_dir / "pattern.sql", "r", encoding="utf-8") as f:
            patterns["SQL"] = re.compile(f.read().strip(), re.DOTALL)

    except FileNotFoundError as e:
        print(f"{RED}Error loading pattern files: {e}{RESET}")
        print("Please ensure the 'target' folder exists with the language files.")
        sys.exit(1)

    return patterns


# Load them globally once
LANGUAGE_PATTERNS = load_patterns()


def get_docstring_lines(tree: ast.AST) -> Set[int]:
    """
    Finds all line numbers that belong to docstrings so we can ignore them.

    Args:
        tree: AST tree to scan.

    Returns:
        Set of line numbers that belong to docstrings.
    """
    doc_lines: Set[int] = set()
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


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Args:
        text: The text to analyze.

    Returns:
        The name of the language detected in the text, or None.
    """
    if not text.strip() or len(text) < 5:
        return ""

    for lang, pattern in LANGUAGE_PATTERNS.items():
        if pattern.search(text):
            return lang
    return ""


class EmbeddedCodeVisitor(ast.NodeVisitor):
    """
    AST visitor to detect embedded languages in Python code.

    Attributes:
        doc_lines: Set of line numbers that belong to docstrings.
        findings: List of tuples containing line number, language, and snippet.
    """

    def __init__(self, doc_lines: Set[int]):
        """
        Initialize the visitor.

        Args:
            doc_lines: Set of line numbers that belong to docstrings.
        """
        self.doc_lines = doc_lines
        self.findings: List[Tuple[int, str, str]] = []  # (lineno, language, snippet)

    def _check_and_record(self, text: str, lineno: int):
        """
        Check if the given text contains embedded code and record the finding if it does.

        Args:
            text: The text to analyze.
            lineno: The line number of the text.
        """
        if lineno in self.doc_lines:
            return

        lang = detect_language(text)
        if lang:
            # Create a short preview snippet
            snippet = text.strip().replace("\n", " ")
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            self.findings.append((lineno, lang, snippet))

    def visit_Constant(self, node: ast.Constant):
        """
        Check standard string literals.

        Args:
            node: AST node representing a constant.
        """
        if isinstance(node.value, str):
            self._check_and_record(node.value, getattr(node, "lineno", -1))
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr):
        """
        Check f-strings by reconstructing their literal parts.

        Args:
            node: AST node representing an f-string.
        """
        assembled_string = ""
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                assembled_string += part.value
            elif isinstance(part, ast.FormattedValue):
                assembled_string += "{VAR}"

        self._check_and_record(assembled_string, getattr(node, "lineno", -1))
        self.generic_visit(node)


def analyze_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """
    Parses a file and returns embedded code findings.

    Args:
        filepath: Path to the file to analyze.

    Returns:
        List of tuples containing line number, language, and snippet.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    doc_lines = get_docstring_lines(tree)
    visitor = EmbeddedCodeVisitor(doc_lines)
    visitor.visit(tree)

    return sorted(visitor.findings, key=lambda x: x[0])


def main():
    """Main function to check for embedded languages."""
    parser = argparse.ArgumentParser(description="Find embedded HTML, CSS, JS, or SQL in Python files.")
    parser.add_argument("directory", type=str, nargs="?", default=".", help="Target directory to scan")
    parser.add_argument("-e", "--exclude", nargs="+", default=[], help="Directories to exclude")
    args = parser.parse_args()

    target_root = Path(args.directory).resolve()
    if not target_root.is_dir():
        print(f"Error: Directory '{target_root}' does not exist.")
        return

    print(f"{CYAN}Scanning '{target_root}' for embedded languages...{RESET}")
    print("=" * 70)

    internal_skip = {".git", "__pycache__", "venv", ".venv", "env", "node_modules", "dist", "build", "target"}
    exclude_set = set(args.exclude) | internal_skip

    files_with_issues = 0
    total_violations = 0

    for root, dirs, files in os.walk(target_root):
        dirs[:] = [d for d in dirs if d not in exclude_set]

        for filename in files:
            if filename.endswith(".py"):
                filepath = Path(root) / filename
                findings = analyze_file(filepath)

                if findings:
                    files_with_issues += 1
                    rel_path = filepath.relative_to(target_root)
                    print(f"\n📄 {CYAN}{rel_path}{RESET}")

                    for line_no, lang, snippet in findings:
                        print(f"   Line {line_no:<4} | {RED}[{lang}]{RESET} {YELLOW}{snippet}{RESET}")
                        total_violations += 1

    print("\n" + "=" * 70)
    if total_violations > 0:
        print(f"⚠️  {RED}Found {total_violations} embedded code blocks across {files_with_issues} files.{RESET}")
    else:
        print(f"✅ {GREEN}Clean! No embedded languages detected.{RESET}")


if __name__ == "__main__":
    main()
