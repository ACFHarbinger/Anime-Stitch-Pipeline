import ast
from pathlib import Path
from typing import Dict, List


def is_constant_name(name: str) -> bool:
    """Checks if a name follows the all-uppercase constant convention."""
    return name.isupper() and name.isidentifier() and not name.startswith("__")

class ConstantVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.constants = []

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and is_constant_name(target.id):
                self.constants.append({
                    "name": target.id,
                    "line": node.lineno,
                    "file": self.file_path
                })
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and is_constant_name(node.target.id):
            self.constants.append({
                "name": node.target.id,
                "line": node.lineno,
                "file": self.file_path
            })
        self.generic_visit(node)

def find_constants(root_dirs: List[str]) -> List[Dict]:
    all_constants = []
    for root_dir in root_dirs:
        for path in Path(root_dir).rglob("*.py"):
            if "__init__.py" in str(path) or "constant_checker.py" in str(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                    visitor = ConstantVisitor(str(path))
                    visitor.visit(tree)
                    all_constants.extend(visitor.constants)
            except Exception as e:
                print(f"Error parsing {path}: {e}")
    return all_constants

if __name__ == "__main__":
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]

    dirs_to_check = [
        str(project_root / "backend"),
        str(project_root / "gui")
    ]

    constants = find_constants(dirs_to_check)

    grouped = {}
    for c in constants:
        grouped.setdefault(c["file"], []).append(f"{c['name']} (L{c['line']})")

    print(f"Found {len(constants)} constant declarations:\n")
    for file, consts in sorted(grouped.items()):
        print(f"--- {file} ---")
        for c in consts:
            print(f"  {c}")
        print()
