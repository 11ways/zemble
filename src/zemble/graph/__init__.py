"""Symbol graph: extraction, workspace resolution, storage and relationship queries.

Only the model is imported eagerly: the store pulls in every extractor, and the language
catalog those extractors read needs the model first, so the rest loads on first attribute
access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from zemble.graph.model import Edge, EdgeKind, Hit, Resolution, Symbol, SymbolKind

_LAZY = {
    "GraphProvider": "zemble.graph.provider",
    "SqliteGraphProvider": "zemble.graph.provider",
    "SubtreeGraphProvider": "zemble.graph.provider",
    "open_provider": "zemble.graph.provider",
    "GraphStats": "zemble.graph.store",
    "build_graph": "zemble.graph.store",
    "graph_db_path": "zemble.graph.store",
    "graph_exists": "zemble.graph.store",
    "graph_present": "zemble.graph.store",
}


def __getattr__(name: str) -> Any:
    """Load the store and provider names on demand."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


__all__ = [
    "Edge",
    "EdgeKind",
    "GraphProvider",
    "GraphStats",
    "Hit",
    "Resolution",
    "SqliteGraphProvider",
    "SubtreeGraphProvider",
    "Symbol",
    "SymbolKind",
    "build_graph",
    "graph_db_path",
    "graph_exists",
    "graph_present",
    "open_provider",
]
