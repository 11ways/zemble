"""One payload per evidence question, shared by the daemon, the CLI and the MCP tools.

The wire shape and the in-process shape are the same object: a surface renders a
payload without caring whether a warm daemon or this process produced it.
"""

from __future__ import annotations

from typing import Any

from zemble.evidence.bundle import build_bundle
from zemble.evidence.outline import OutlineError, Signatures, outline, signatures
from zemble.graph.cli import select_symbol
from zemble.graph.model import Symbol
from zemble.graph.provider import GraphProvider
from zemble.index import ZembleIndex

#: Budget and breadth used when a caller does not state its own.
DEFAULT_BUDGET = 3000
DEFAULT_TOP_K = 20


def explain_payload(
    index: ZembleIndex, graph: GraphProvider, query: str, budget_tokens: int, top_k: int
) -> dict[str, Any]:
    """Build an evidence bundle and carry both of its renderings.

    :param index: The search index to find the primary chunks with.
    :param graph: The symbol graph to expand them through.
    :param query: The question the bundle answers.
    :param budget_tokens: The maximum estimated tokens the packed items may cost.
    :param top_k: How many search results to consider.
    :return: The bundle as data, plus its markdown rendering.
    """
    bundle = build_bundle(index, graph, query, budget_tokens, top_k=top_k)
    return {"bundle": bundle.to_dict(), "markdown": bundle.render()}


def outline_payload(graph: GraphProvider, target: str, members: str | None) -> dict[str, Any]:
    """Outline a file or a type, reporting a bad target as data rather than an exception."""
    try:
        rendered = outline(graph, target, members)
    except OutlineError as error:
        return error_payload(error.message, error.candidates)
    return {"outline": rendered.to_dict(), "text": rendered.render()}


def signatures_payload(graph: GraphProvider, symbol: str) -> dict[str, Any]:
    """Resolve a written name to one symbol and describe it, or report why it could not be."""
    chosen, candidates = select_symbol(graph.definition(symbol), symbol)
    if chosen is None:
        message = (
            f"{symbol!r} is ambiguous; pass a qualified name."
            if candidates
            else f"No symbol named {symbol!r}. {graph.coverage_note()}"
        )
        return error_payload(message, candidates)
    answer: Signatures = signatures(graph, chosen)
    return {"signatures": answer.to_dict(), "text": answer.render()}


def error_payload(message: str, candidates: list[Symbol]) -> dict[str, Any]:
    """Shape a refusal so every surface can print it, candidates included."""
    return {
        "error": message,
        "candidates": [
            {"qualified_name": symbol.qualified_name, "file_path": symbol.file_path} for symbol in candidates
        ],
    }


__all__ = [
    "DEFAULT_BUDGET",
    "DEFAULT_TOP_K",
    "error_payload",
    "explain_payload",
    "outline_payload",
    "signatures_payload",
]
