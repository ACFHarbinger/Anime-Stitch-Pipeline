"""
A tool to visually trace the origin and usages of a Python component.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.validation.dependency_grapher import DependencyGrapher


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate an interactive dependency graph for a component.")
    parser.add_argument("project_root", help="Root directory to scan")
    parser.add_argument("target_file", help="The file where the component is located")
    parser.add_argument("target_name", help="The name of the component (function/class)")
    args = parser.parse_args()

    target_file = os.path.abspath(args.target_file)
    if not os.path.exists(target_file):
        print(f"Error: Could not find target file {target_file}")
        return

    grapher = DependencyGrapher(args.project_root)
    grapher.generate_graph(target_file, args.target_name)

if __name__ == "__main__":
    main()
