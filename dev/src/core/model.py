"""Python-side data model mirroring dev/src/input/protobuf/codegraph.proto.

Hand-written dataclasses instead of generated protobuf bindings, since this
environment has no `protoc`. Field names and the JSON shape match proto3's
canonical JSON mapping exactly, so swapping in real generated bindings later
(``google.protobuf.json_format.Parse``) is a drop-in replacement for
:func:`CodeGraph.from_dict`/:meth:`CodeGraph.to_dict`, not a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EdgeKind(str, Enum):
    """Mirrors ``Edge.Kind`` in codegraph.proto."""

    UNSPECIFIED = "KIND_UNSPECIFIED"
    IMPORT = "IMPORT"
    CALL = "CALL"
    INHERITANCE = "INHERITANCE"
    DATAFLOW = "DATAFLOW"


class Severity(str, Enum):
    """Mirrors ``Diagnostic.Severity`` in codegraph.proto."""

    UNSPECIFIED = "SEVERITY_UNSPECIFIED"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class Node:
    """A single source-code entity. Mirrors ``Node`` in codegraph.proto."""

    id: str
    language: str
    kind: str
    qualified_name: str
    file_path: str
    layer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "language": self.language,
            "kind": self.kind,
            "qualifiedName": self.qualified_name,
            "filePath": self.file_path,
            "layer": self.layer,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Node":
        return cls(
            id=payload["id"],
            language=payload["language"],
            kind=payload["kind"],
            qualified_name=payload.get("qualifiedName", payload.get("qualified_name", "")),
            file_path=payload.get("filePath", payload.get("file_path", "")),
            layer=payload.get("layer", ""),
        )


@dataclass
class Edge:
    """A directed relationship between two nodes. Mirrors ``Edge``."""

    source_id: str
    target_id: str
    kind: EdgeKind = EdgeKind.IMPORT

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceId": self.source_id,
            "targetId": self.target_id,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Edge":
        return cls(
            source_id=payload.get("sourceId", payload.get("source_id")),
            target_id=payload.get("targetId", payload.get("target_id")),
            kind=EdgeKind(payload.get("kind", EdgeKind.IMPORT.value)),
        )


@dataclass
class Diagnostic:
    """A non-fatal problem found while parsing/analyzing. Mirrors ``Diagnostic``."""

    severity: Severity
    message: str
    file_path: str = ""
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "message": self.message,
            "filePath": self.file_path,
            "line": self.line,
        }


@dataclass
class CodeGraph:
    """The graph contributed by one parser run. Mirrors ``CodeGraph``."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    source_language: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "sourceLanguage": self.source_language,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CodeGraph":
        return cls(
            nodes=[Node.from_dict(n) for n in payload.get("nodes", [])],
            edges=[Edge.from_dict(e) for e in payload.get("edges", [])],
            diagnostics=[],
            source_language=payload.get("sourceLanguage", payload.get("source_language", "")),
        )

    def merge(self, other: "CodeGraph") -> "CodeGraph":
        """Return a new :class:`CodeGraph` combining this graph with ``other``.

        Args:
            other: Another graph, typically from a different language parser.

        Returns:
            A merged graph; node/edge order is preserved, ``self`` first.
        """
        return CodeGraph(
            nodes=[*self.nodes, *other.nodes],
            edges=[*self.edges, *other.edges],
            diagnostics=[*self.diagnostics, *other.diagnostics],
            source_language=",".join(
                filter(None, [self.source_language, other.source_language])
            ),
        )
