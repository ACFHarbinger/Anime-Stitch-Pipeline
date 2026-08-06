"""Multi-source CodeGraph aggregation.

Collects one :class:`~dev.src.core.model.CodeGraph` per available
``input/<language>`` parser and merges them into a single graph, per
moon/roadmaps/developer_tools.md D3. Deliberately has no per-language
knowledge — it only ever calls :meth:`CodeGraph.merge`, so dropping an
``input/<language>/`` submodule costs this module nothing.
"""

from __future__ import annotations

from core.model import CodeGraph


def aggregate(graphs: list[CodeGraph]) -> CodeGraph:
    """Merge any number of per-language graphs into one.

    Args:
        graphs: Zero or more graphs, typically one per available
            ``input/<language>`` parser that ran successfully.

    Returns:
        A single merged :class:`CodeGraph`. Returns an empty graph if
        ``graphs`` is empty, rather than raising.
    """
    merged = CodeGraph()
    for graph in graphs:
        merged = merged.merge(graph)
    return merged
