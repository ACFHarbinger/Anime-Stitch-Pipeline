"""
A tool to visually trace the origin and usages of a Python component,
outputting an interactive graph with clickable, natively nested UML-style info panels.

Attributes:
    ASTScopeVisitor: A class to visit AST nodes and capture definitions.
    DependencyGrapher: A class to trace dependencies between Python files.
    main: Main function to trace dependencies.

Example:
    >>> python logic/src/utils/validation/trace_dependencies.py logic/src/policies/route_construction/exact_and_decomposition_solvers/branch_and_cut/bc.py
"""

import ast
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import jinja2

from .ast_scope_visitor import ASTScopeVisitor

try:
    from pyvis.network import Network
except ImportError:
    print("Error: Required libraries not found.")
    print("Please run: pip install pyvis networkx")
    exit(1)




class DependencyGrapher:
    """
    A class to trace dependencies between Python files.

    Attributes:
        project_root (str): The root directory of the project.
        all_files (Set[str]): A set of all Python files in the project.
        definitions (Dict[str, Set[str]]): A dictionary mapping file paths to a set of defined names.
        imports (Dict[str, Dict[str, Tuple[str, str]]]): A dictionary mapping file paths to a dictionary of imports.
        ui_definitions (Dict[str, Dict]): A dictionary mapping file paths to a dictionary of definitions for UI.
        ui_imports_direct (Dict[str, List[str]]): A dictionary mapping file paths to a list of direct imports for UI.
        ui_imports_grouped (Dict[str, Dict[str, List[str]]]): A dictionary mapping file paths to a dictionary of grouped imports for UI.
        nodes (Set[Tuple[str, str]]): A set of nodes for the graph.
        edges (List[Tuple[str, str, str]]): A list of edges for the graph.
        jinja_env (jinja2.Environment): The Jinja2 environment for rendering the graph.
    """

    def __init__(self, project_root: str) -> None:
        """
        Initialize the DependencyGrapher.

        Args:
            project_root (str): The root directory of the project.
        """
        self.project_root = os.path.abspath(project_root)
        self.all_files: Set[str] = set()

        # Graph tracing mapping
        self.definitions: Dict[str, Set[str]] = defaultdict(set)
        self.imports: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(dict)

        # UI mapping
        self.ui_definitions: Dict[str, Dict] = {}
        self.ui_imports_direct: Dict[str, List[str]] = defaultdict(list)
        self.ui_imports_grouped: Dict[str, Dict[str, List[str]]] = defaultdict(dict)

        self.nodes: Set[Tuple[str, str]] = set()
        self.edges: List[Tuple[str, str, str]] = []

        # Load templates from the script's directory
        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "html")
        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(script_dir))

    def _python_path_to_filepath(self, module_path: str) -> Optional[str]:
        """
        Convert a Python path to a file path.

        Args:
            module_path (str): The Python path.

        Returns:
            Optional[str]: The file path.
        """
        target_suffix = module_path.replace(".", os.sep) + ".py"
        target_init_suffix = os.path.join(module_path.replace(".", os.sep), "__init__.py")

        for filepath in self.all_files:
            if filepath.endswith(target_suffix) or filepath.endswith(target_init_suffix):
                return filepath
        return None

    def scan_project(self) -> None:
        """
        Scan the project directory and extract all definitions and imports.
        """
        for root, _, files in os.walk(self.project_root):
            if any(part.startswith(".") or part in ("venv", "__pycache__", "env") for part in root.split(os.sep)):
                continue
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    self.all_files.add(filepath)

        for filepath in self.all_files:
            self._parse_file(filepath)

    def _parse_file(self, filepath: str) -> None:
        """
        Parse a file and extract its definitions and imports.

        Args:
            filepath (str): Path to the file.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=filepath)
        except (SyntaxError, UnicodeDecodeError):
            return

        visitor = ASTScopeVisitor()
        visitor.visit(tree)

        self.definitions[filepath] = visitor.flat_defs
        self.imports[filepath] = visitor.imports_graph
        self.ui_definitions[filepath] = visitor.tree
        self.ui_imports_direct[filepath] = visitor.imports_direct
        self.ui_imports_grouped[filepath] = dict(visitor.imports_grouped)

    def trace_backward(
        self, current_file: str, target_name: str, visited: Optional[Set[Tuple[str, str]]] = None
    ) -> Optional[str]:
        """
        Trace backward from the target node to find all origins.

        Args:
            current_file (str): The current file.
            target_name (str): The name of the target node.
            visited (Optional[Set[Tuple[str, str]]]): The set of visited nodes.

        Returns:
            Optional[str]: The file where the component is located.
        """
        if visited is None:
            visited = set()

        trace_key = (current_file, target_name)
        if trace_key in visited:
            return None
        visited.add(trace_key)

        self.nodes.add((current_file, "intermediate"))

        if target_name in self.definitions.get(current_file, set()):
            self.nodes.discard((current_file, "intermediate"))
            self.nodes.add((current_file, "origin"))
            return current_file

        if target_name in self.imports.get(current_file, {}):
            source_module, original_name = self.imports[current_file][target_name]
            source_file = self._python_path_to_filepath(source_module)

            if source_file:
                self.edges.append((current_file, source_file, "backward"))
                return self.trace_backward(source_file, original_name, visited)
            else:
                ext_name = f"External: {source_module}"
                self.nodes.add((ext_name, "origin"))
                self.edges.append((current_file, ext_name, "backward"))
        return None

    def trace_forward(self, target_origin_file: str, target_original_name: str) -> None:
        """
        Trace forward from the target node to find all usages.

        Args:
            target_origin_file (str): The file where the component is located.
            target_original_name (str): The name of the component (function/class).
        """
        for filepath in self.all_files:
            if filepath == target_origin_file:
                continue

            for local_alias in self.imports.get(filepath, {}):
                temp_visited = set()
                current_check = filepath
                current_name = local_alias
                path_edges = []
                found_origin = False

                while True:
                    trace_key = (current_check, current_name)
                    if trace_key in temp_visited:
                        break
                    temp_visited.add(trace_key)

                    if current_check == target_origin_file and current_name == target_original_name:
                        found_origin = True
                        break

                    if current_name in self.imports.get(current_check, {}):
                        src_mod, orig_nm = self.imports[current_check][current_name]
                        src_file = self._python_path_to_filepath(src_mod)
                        if src_file:
                            path_edges.append((src_file, current_check, "forward"))
                            current_check = src_file
                            current_name = orig_nm
                        else:
                            break
                    else:
                        break

                if found_origin:
                    self.nodes.add((filepath, "usage"))
                    self.edges.extend(path_edges)

    def _build_node_uml(self, filepath: str) -> str:
        """
        Build the UML for a node.

        Args:
            filepath (str): Path to the node.

        Returns:
            str: UML for the node.
        """
        template = self.jinja_env.get_template("node_template.html")

        if filepath.startswith("External:"):
            return template.render(is_external=True, name=filepath.replace("External: ", ""))

        display_path = os.path.relpath(filepath, self.project_root)
        defs_tree = self.ui_definitions.get(filepath, {"children": []})
        direct_imps = self.ui_imports_direct.get(filepath, [])
        grouped_imps = self.ui_imports_grouped.get(filepath, {})

        return template.render(
            is_external=False,
            filename=os.path.basename(filepath),
            display_path=display_path,
            defs_tree=defs_tree["children"],
            direct_imps=direct_imps,
            grouped_imps=grouped_imps,
            imports_limit=8,
        )

    def _build_edge_uml(self, source: str, target: str, direction: str) -> str:
        """
        Build the UML for an edge.

        Args:
            source (str): Source node.
            target (str): Target node.
            direction (str): Direction of the edge.

        Returns:
            str: UML for the edge.
        """
        template = self.jinja_env.get_template("edge_template.html")

        src_label = os.path.relpath(source, self.project_root) if not source.startswith("External:") else source
        tgt_label = os.path.relpath(target, self.project_root) if not target.startswith("External:") else target

        return template.render(direction=direction, src_label=src_label, tgt_label=tgt_label)

    def generate_graph(self, target_file: str, target_name: str) -> None:
        """
        Generate the Pyvis graph.

        Args:
            target_file (str): The file where the component is located.
            target_name (str): The name of the component (function/class).
        """
        print("Scanning project...")
        self.scan_project()
        self.nodes.add((target_file, "target"))

        print("Tracing backwards (Origin)...")
        origin_file = self.trace_backward(target_file, target_name)

        print("Tracing forwards (Usages)...")
        if origin_file:
            self.trace_forward(origin_file, target_name)
        else:
            self.trace_forward(target_file, target_name)

        self._render_pyvis(target_file)

    def _render_pyvis(self, target_node_id: str) -> None:
        """
        Render the Pyvis graph.

        Args:
            target_node_id (str): The ID of the target node.
        """
        net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white", directed=True)
        net.force_atlas_2based(gravity=-50)

        for filepath, n_type in self.nodes:
            label = (
                filepath.replace("External: ", "")
                if filepath.startswith("External:")
                else os.path.basename(filepath).replace(".py", "")
            )
            uml_content = self._build_node_uml(filepath)

            color = (
                "#ff4b4b"
                if filepath == target_node_id
                else "#4caf50"
                if n_type == "origin"
                else "#2196f3"
                if n_type == "usage"
                else "#9e9e9e"
            )
            net.add_node(filepath, label=label, color=color, size=25, title="👆 Click for UML details", uml=uml_content)

        unique_edges = set(self.edges)
        for source, target, direction in unique_edges:
            uml_content = self._build_edge_uml(source, target, direction)
            color = "#e67e22" if direction == "backward" else "#9b59b6"
            net.add_edge(source, target, color=color, arrows="to", title="👆 Click for UML details", uml=uml_content)

        output_file = "dependency_graph.html"
        net.show(output_file, notebook=False)
        self._inject_uml_panel(output_file)
        print(f"\nSuccess! Interactive graph saved to: {output_file}")

    def _inject_uml_panel(self, filepath: str) -> None:
        """
        Inject UML panel into the generated HTML file.

        Args:
            filepath (str): Path to the generated HTML file.
        """
        # 1. Read the generated HTML from Pyvis
        with open(filepath, "r", encoding="utf-8") as f:
            html_content = f.read()

        # 2. Safely resolve the path to our new template file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(script_dir, "html", "uml_panel.html")

        # 3. Read the template and inject it
        try:
            with open(template_path, "r", encoding="utf-8") as template_file:
                injection = template_file.read()

            # Replace the closing body tag with our injection + the closing body tag
            html_content = html_content.replace("</body>", injection + "\n</body>")

            # Write the final combined code back to the output file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)

        except FileNotFoundError:
            print(f"\nWarning: Could not find '{template_path}'.")
            print("The graph was generated, but the UML panel won't be interactive.")
            print("Please ensure 'uml_panel.html' is in the same directory as this script.")




