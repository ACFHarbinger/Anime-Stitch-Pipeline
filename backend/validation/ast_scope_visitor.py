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
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

try:
    from pyvis.network import Network  # noqa: F401
except ImportError:
    print("Error: Required libraries not found.")
    print("Please run: pip install pyvis networkx")
    exit(1)


class ASTScopeVisitor(ast.NodeVisitor):
    """
    Custom AST visitor to capture definitions as a nested tree and group imports.

    Attributes:
        tree (Dict[str, Union[str, List[Dict[str, List[str]]]]]): The tree structure of the AST.
        stack (List[Dict[str, Union[str, List[Dict[str, List[str]]]]]]): The stack of nodes in the AST.
        imports_grouped (Dict[str, List[str]]): A dictionary mapping import prefixes to a list of imports.
        imports_direct (List[str]): A list of direct imports.
        imports_graph (Dict[str, Tuple[str, str]]): A dictionary mapping import names to a tuple of (prefix, name).
        flat_defs (Set[str]): A set of all defined names.
        seen_in_scope (Set[Tuple[int, str]]): A set of seen names in the current scope.
    """

    def __init__(self) -> None:
        """
        Initialize the ASTScopeVisitor.
        """
        self.tree: Dict[str, Union[str, List[Dict[str, List[str]]]]] = {"name": "<module>", "children": []}
        self.stack: List[Dict[str, Union[str, List[Dict[str, List[str]]]]]] = [self.tree]

        self.imports_grouped: Dict[str, List[str]] = defaultdict(list)
        self.imports_direct: List[str] = []
        self.imports_graph: Dict[str, Tuple[str, str]] = {}

        self.flat_defs: Set[str] = set()
        self.seen_in_scope: Set[Tuple[int, str]] = set()

    def _add_node(self, name: str) -> Optional[Dict[str, List[Dict[str, List[str]]]]]:
        """
        Add a node to the graph.

        Args:
            name (str): The name of the node.

        Returns:
            Optional[Dict[str, List[Dict[str, List[str]]]]]: The node if it was added, None otherwise.
        """
        # Prevent duplicates in the exact same scope
        scope_key = id(self.stack[-1])
        sig = (scope_key, name)
        if sig not in self.seen_in_scope:
            node = {"name": name, "children": []}
            self.stack[-1]["children"].append(node)
            self.seen_in_scope.add(sig)
            return node
        return None

    def visit_ClassDef(self, node):
        """
        Visit a class definition node and add it to the graph.

        Args:
            node (ast.ClassDef): The class definition node.
        """
        new_node = self._add_node(f"class {node.name}")
        self.flat_defs.add(node.name)
        if new_node:
            self.stack.append(new_node)
            self.generic_visit(node)
            self.stack.pop()

    def visit_FunctionDef(self, node):
        """
        Visit a function definition node and add it to the graph.

        Args:
            node (ast.FunctionDef): The function definition node.
        """
        new_node = self._add_node(f"def {node.name}")
        self.flat_defs.add(node.name)
        if new_node:
            self.stack.append(new_node)
            self.generic_visit(node)
            self.stack.pop()

    def visit_AsyncFunctionDef(self, node):
        """
        Visit an async function definition node and add it to the graph.

        Args:
            node (ast.AsyncFunctionDef): The async function definition node.
        """
        new_node = self._add_node(f"async def {node.name}")
        self.flat_defs.add(node.name)
        if new_node:
            self.stack.append(new_node)
            self.generic_visit(node)
            self.stack.pop()

    def visit_Assign(self, node):
        """
        Visit an assignment node and add it to the graph.

        Args:
            node (ast.Assign): The assignment node.
        """
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._add_node(target.id)
                self.flat_defs.add(target.id)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """
        Visit an import node and add it to the graph.

        Args:
            node (ast.ImportFrom): The import node.
        """
        module = node.module or ""
        level = node.level
        prefix = "." * level + module if level > 0 else module
        for alias in node.names:
            local_name = alias.asname or alias.name
            self.imports_graph[local_name] = (prefix, alias.name)

            display_name = f"{alias.name} as {alias.asname}" if alias.asname else alias.name
            self.imports_grouped[prefix].append(display_name)
        self.generic_visit(node)

    def visit_Import(self, node):
        """
        Visit an import node and add it to the graph.

        Args:
            node (ast.Import): The import node.
        """
        for alias in node.names:
            local_name = alias.asname or alias.name
            self.imports_graph[local_name] = (alias.name, alias.name)

            display_name = f"{alias.name} as {alias.asname}" if alias.asname else alias.name
            self.imports_direct.append(display_name)
        self.generic_visit(node)






