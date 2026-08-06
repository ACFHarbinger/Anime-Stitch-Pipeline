"""
Documentation utilities module.

Attributes:
    check_docstrings: Check a single .py file and return a list of violation dicts.
    check_docstrings_recursive: Recursively collect violations from all .py files under *directory*.
    display_results: Render the violation report as a Rich table grouped by file.
    check_google_style: Check a single .py file and return a list of violation dicts.
    display_report: Displays the violations in a Rich table.
    main: Main entry point.
    add_docstrings_batch: Add docstrings to a Python file.
    DocstringInjector: Class for injecting docstrings into Python files.
    main: Main entry point.

Example:
    uv run python backend/docs/check_docstrings.py <path1> [path2 ...]
    uv run python backend/docs/check_google_style.py <path1> [path2 ...]
    uv run python backend/docs/add_docstrings_batch.py <path1> [path2 ...]
"""
