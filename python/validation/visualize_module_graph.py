"""
Generate an interactive module-level import graph for the entire codebase.
Every Python file becomes a node; every internal import becomes a directed edge.

Key features:
  - Nodes colored by architectural layer (logic, gui, tests, etc.)
  - Cross-layer violations (e.g. logic → gui) highlighted in red
  - Condensed package view: group nodes by top-N directory levels
  - Terminal summary of violation counts and layer distribution
  - Saves an interactive pyvis HTML file for browser exploration

Attributes:
    SKIP_DIRS (Set[str]): Set of directories to exclude.
    RED (str): Red color for highlighting violations.
    YELLOW (str): Yellow color for highlighting warnings.
    GREEN (str): Green color for highlighting success.
    CYAN (str): Cyan color for highlighting information.
    BOLD (str): Bold text for highlighting.
    DIM (str): Dim text for highlighting.
    RESET (str): Reset text color.
    Network (Type[Network]): Type hint for pyvis Network class.
    DEFAULT_LAYERS (List[Tuple[str, str, str]]): Default layers for the graph.
    FORBIDDEN_DIRECTIONS (List[Tuple[str, str]]): Forbidden layer pairs.
    file_to_module: Convert a file path to a module name.
    collect_module_map: Collect a map of all modules in the codebase.
    resolve_to_module: Resolve a raw import string to a module name.
    build_graph: Build a module graph from the codebase.
    get_layer: Get the layer of a module.
    find_violations: Find architectural violations in the module graph.
    condense_to_packages: Condense the module graph to packages.
    generate_html: Generate an interactive HTML graph.
    main: Main function to generate an interactive module-level import graph.

Example:
    >>> python backend/validation/visualize_module_graph.py backend/src gui/src --html docs/module_graph.html
"""

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", "dist", "build"}

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

try:
    from pyvis.network import Network
except ImportError:
    print(f"{YELLOW}pyvis not installed — skipping HTML in generate_html. Install with: uv add pyvis{RESET}")
    Network = None

# (module prefix, display label, hex color)
DEFAULT_LAYERS: List[Tuple[str, str, str]] = [
    ("backend", "Backend", "#3498db"),
    ("gui", "GUI", "#9b59b6"),
    ("test", "Tests", "#27ae60"),
    ("script", "Scripts", "#e67e22"),
]

# Pairs (src_layer, tgt_layer) that represent architectural violations
FORBIDDEN_DIRECTIONS: List[Tuple[str, str]] = [
    ("Backend", "GUI"),
]

# Exact (src_module, tgt_module) pairs that are Backend->GUI by directory
# layout but are not real layering violations -- e.g. the application's
# composition root, which necessarily wires both layers together at
# startup. Kept as an explicit, reviewable allowlist rather than silently
# excluding a whole module from violation checking.
ALLOWED_VIOLATIONS: Set[Tuple[str, str]] = {
    ("backend.src.app", "gui.src.windows.main"),
}


def file_to_module(filepath: Path, root: Path) -> str:
    """
    Convert a file path to a module name.

    Args:
        filepath (Path): File path.
        root (Path): Root directory of the codebase.

    Returns:
        str: Module name.
    """
    rel = filepath.relative_to(root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def collect_module_map(scan_dirs: List[Path], name_root: Path, exclude: Set[str]) -> Dict[str, Path]:
    """
    Collect a map of all modules in the codebase.

    Each directory in `scan_dirs` is walked independently (so unrelated trees
    between them, e.g. everything outside `backend/src`/`gui/src`, are never
    visited), but module names are computed relative to the single shared
    `name_root` -- this is what lets an import in one scanned directory
    (e.g. `gui/src`) resolve to a module defined in another (e.g.
    `backend/src`), so cross-boundary edges actually show up in the graph.

    Args:
        scan_dirs (List[Path]): Directories to walk for `.py` files.
        name_root (Path): Common root used to compute dotted module names.
        exclude (Set[str]): Set of directories to exclude.

    Returns:
        Dict[str, Path]: Map of module names to file paths.
    """
    module_map: Dict[str, Path] = {}
    for scan_dir in scan_dirs:
        for dirpath, dirs, files in os.walk(scan_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d not in exclude]
            for fname in files:
                if fname.endswith(".py"):
                    fpath = Path(dirpath) / fname
                    module_map[file_to_module(fpath, name_root)] = fpath
    return module_map


def resolve_to_module(raw: str, level: int, current: str, known: Set[str]) -> Optional[str]:
    """
    Resolve a raw import string to a module name.

    Args:
        raw (str): Raw import string.
        level (int): Import level.
        current (str): Current module name.
        known (Set[str]): Set of known module names.

    Returns:
        Optional[str]: Resolved module name.
    """
    if level > 0:
        parts = current.split(".")
        base = parts[: max(0, len(parts) - level)]
        candidate = ".".join(base + ([raw] if raw else []))
    else:
        candidate = raw

    if candidate in known:
        return candidate
    for m in known:
        if m.startswith(candidate + "."):
            return candidate
    return None


def build_graph(scan_dirs: List[Path], name_root: Path, exclude: Set[str]) -> Dict[str, Set[str]]:
    """
    Build a module graph from the codebase.

    Args:
        scan_dirs (List[Path]): Directories to walk for `.py` files.
        name_root (Path): Common root used to compute dotted module names.
        exclude (Set[str]): Set of directories to exclude.

    Returns:
        Dict[str, Set[str]]: Module graph.
    """
    module_map = collect_module_map(scan_dirs, name_root, exclude)
    known = set(module_map.keys())
    graph: Dict[str, Set[str]] = {m: set() for m in known}
    for module, fpath in module_map.items():
        try:
            tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dep = resolve_to_module(alias.name, 0, module, known)
                    if dep and dep != module:
                        graph[module].add(dep)
            elif isinstance(node, ast.ImportFrom):
                dep = resolve_to_module(node.module or "", node.level or 0, module, known)
                if dep and dep != module:
                    graph[module].add(dep)

    return graph


def get_layer(module: str, layers: List[Tuple[str, str, str]]) -> Tuple[str, str]:
    """
    Get the layer of a module.

    Args:
        module (str): Module name.
        layers (List[Tuple[str, str, str]]): List of layers.

    Returns:
        Tuple[str, str]: Layer name and color.
    """
    for prefix, label, color in layers:
        if module.lower().startswith(prefix.lower()):
            return label, color
    return "Other", "#95a5a6"


def find_violations(
    graph: Dict[str, Set[str]],
    layers: List[Tuple[str, str, str]],
    forbidden: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """
    Find architectural violations in the module graph.

    Args:
        graph (Dict[str, Set[str]]): Module graph.
        layers (List[Tuple[str, str, str]]): List of layers.
        forbidden (List[Tuple[str, str]]): List of forbidden layer pairs.

    Returns:
        List[Tuple[str, str]]: List of violation edges.
    """
    violations: List[Tuple[str, str]] = []
    for src, targets in graph.items():
        src_layer, _ = get_layer(src, layers)
        for tgt in targets:
            if (src, tgt) in ALLOWED_VIOLATIONS:
                continue
            tgt_layer, _ = get_layer(tgt, layers)
            if (src_layer, tgt_layer) in forbidden:
                violations.append((src, tgt))
    return violations


def condense_to_packages(graph: Dict[str, Set[str]], depth: int) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """
    Collapse module nodes to their top-`depth` package prefix.

    Args:
        graph (Dict[str, Set[str]]): Module graph.
        depth (int): Depth of the graph.

    Returns:
        Tuple[Dict[str, Set[str]], Dict[str, str]]: Collapsed graph and node-to-package mapping.
    """

    def pkg(module: str) -> str:
        """
        Convert a module name to its package prefix.

        Args:
            module (str): Module name.

        Returns:
            str: Package prefix.
        """
        return ".".join(module.split(".")[:depth])

    node_to_pkg: Dict[str, str] = {m: pkg(m) for m in graph}
    pkg_graph: Dict[str, Set[str]] = {}
    for m, targets in graph.items():
        src_pkg = node_to_pkg[m]
        pkg_graph.setdefault(src_pkg, set())
        for t in targets:
            tgt_pkg = node_to_pkg.get(t, pkg(t))
            if tgt_pkg != src_pkg:
                pkg_graph[src_pkg].add(tgt_pkg)
    return pkg_graph, node_to_pkg


def generate_html(
    graph: Dict[str, Set[str]],
    layers: List[Tuple[str, str, str]],
    violation_edges: Set[Tuple[str, str]],
    output: Path,
    depth: int = 0,
) -> None:
    """
    Generate HTML visualization of the module graph.

    Args:
        graph (Dict[str, Set[str]]): Module graph.
        layers (List[Tuple[str, str, str]]): List of layers.
        violation_edges (Set[Tuple[str, str]]): Set of violation edges.
        output (Path): Output path.
        depth (int): Depth of the graph.

    Returns:
        None
    """
    display_graph = graph
    if depth > 0:
        display_graph, _ = condense_to_packages(graph, depth)

    # cdn_resources="in_line": embed vis.js assets directly in the HTML file
    # instead of pyvis's default of copying them to a `lib/` dir relative to
    # the current working directory (which collided with the unrelated
    # `lib/cmake` tracked in this repo's root).
    net = Network(height="900px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.set_options(
        '{"physics":{"barnesHut":{"gravitationalConstant":-5000,"springLength":150}},'
        '"edges":{"arrows":{"to":{"enabled":true,"scaleFactor":0.4}},'
        '"smooth":{"enabled":true,"type":"dynamic"}}}'
    )

    violation_set: Set[Tuple[str, str]] = violation_edges
    if depth > 0:

        def pkg(m: str) -> str:
            """
            Convert a module name to its package prefix.

            Args:
                m (str): Module name.

            Returns:
                str: Package prefix.
            """
            return ".".join(m.split(".")[:depth])

        violation_set = {(pkg(s), pkg(t)) for s, t in violation_edges}

    for module in display_graph:
        _, color = get_layer(module, layers)
        parts = module.split(".")
        label = ".".join(parts[-2:]) if len(parts) >= 2 else module
        net.add_node(module, label=label, title=module, color=color, size=14)

    for src, targets in display_graph.items():
        for tgt in targets:
            if tgt in display_graph:
                is_viol = (src, tgt) in violation_set
                net.add_edge(
                    src,
                    tgt,
                    color="#e74c3c" if is_viol else "#aaaaaa44",
                    width=3 if is_viol else 1,
                    title="⚠ Layer violation!" if is_viol else "",
                )

    # Legend nodes pinned off-screen
    x_pos = -900
    for _, label, color in layers:
        node_id = f"__legend_{label}"
        net.add_node(node_id, label=f"■ {label}", color=color, size=25, physics=False, x=x_pos, y=-400)
        x_pos += 200

    net.save_graph(str(output))
    print(f"{CYAN}Module graph saved → {output}{RESET}")


def main() -> None:
    """Visualise the module-level import graph of a Python codebase."""
    parser = argparse.ArgumentParser(description="Visualise the module-level import graph of a Python codebase.")
    parser.add_argument(
        "directory",
        nargs="+",
        help="Directory/directories to scan (e.g. `backend/src gui/src` to see cross-boundary edges)",
    )
    parser.add_argument(
        "--name-root",
        default=None,
        metavar="PATH",
        help="Common root used to compute dotted module names (default: nearest shared parent of `directory`)",
    )
    parser.add_argument("--exclude", nargs="+", default=[], help="Directory names to skip")
    parser.add_argument("--html", default="module_graph.html", metavar="PATH", help="Output HTML path")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML generation, print summary only")
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        metavar="N",
        help="Collapse nodes to top-N package levels (0 = file-level)",
    )
    args = parser.parse_args()

    scan_dirs = [Path(d).resolve() for d in args.directory]
    for d in scan_dirs:
        if not d.is_dir():
            print(f"Error: '{d}' is not a directory.")
            sys.exit(1)

    if args.name_root:
        name_root = Path(args.name_root).resolve()
    else:
        name_root = Path(os.path.commonpath([str(d) for d in scan_dirs]))

    exclude = set(args.exclude)
    print(f"{CYAN}Building module graph for {[str(d) for d in scan_dirs]} (names relative to '{name_root}')...{RESET}")
    graph = build_graph(scan_dirs, name_root, exclude)
    edge_count = sum(len(v) for v in graph.values())
    print(f"  {DIM}{len(graph)} modules, {edge_count} internal import edges.{RESET}\n")

    violations = find_violations(graph, DEFAULT_LAYERS, FORBIDDEN_DIRECTIONS)

    if violations:
        print(f"{RED}{BOLD}⚠  {len(violations)} cross-layer violation(s):{RESET}")
        for src, tgt in sorted(violations):
            src_lbl, _ = get_layer(src, DEFAULT_LAYERS)
            tgt_lbl, _ = get_layer(tgt, DEFAULT_LAYERS)
            print(f"  {RED}✗{RESET} [{src_lbl}] {src}  →  [{tgt_lbl}] {tgt}")
        print()
    else:
        print(f"{GREEN}✓  No cross-layer violations detected.{RESET}\n")

    # Layer distribution
    layer_counts: Dict[str, int] = {}
    for m in graph:
        label, _ = get_layer(m, DEFAULT_LAYERS)
        layer_counts[label] = layer_counts.get(label, 0) + 1
    print("Module distribution by layer:")
    for label, count in sorted(layer_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 5)
        print(f"  {label:<10} {count:>4}  {DIM}{bar}{RESET}")
    print()

    if not args.no_html:
        violation_edge_set: Set[Tuple[str, str]] = set(violations)
        generate_html(graph, DEFAULT_LAYERS, violation_edge_set, Path(args.html), depth=args.depth)

    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
