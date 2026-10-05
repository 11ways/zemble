"""Member-aware related retrieval with explicit graph evidence and file diversity."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zemble.evidence.counterparts import counterparts, declaration, source_text
from zemble.graph.model import Resolution, Symbol, SymbolKind
from zemble.graph.provider import GraphProvider
from zemble.types import SearchResult
from zemble.utils import format_results


def symbol_chunk(index: Any, symbol: Symbol) -> Any:
    """Return the chunk exposing a declaration, not its adjacent import/doc fragment."""
    chunks = index.chunks_of(symbol.file_path)
    named = [
        chunk
        for chunk in chunks
        if symbol.name in chunk.content and chunk.start_line <= symbol.end_line and chunk.end_line >= symbol.start_line
    ]
    if named:
        return min(named, key=lambda chunk: abs(chunk.start_line - symbol.start_line))
    return index.chunk_at(symbol.file_path, symbol.start_line)


@dataclass
class RelatedCandidates:
    """Request-local candidates retain provenance while deduplicating declaration chunks."""

    index: Any
    owner: Symbol | None
    candidates: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def add(self, symbol: Symbol, score: float, reason: str) -> None:
        """Record one resolved member, preserving the strongest channel and all reasons."""
        if self.owner is not None and symbol.id == self.owner.id:
            return
        chunk = symbol_chunk(self.index, symbol)
        if chunk is None:
            return
        key = (chunk.file_path, chunk.start_line)
        if key not in self.candidates or score > self.candidates[key].score:
            self.candidates[key] = SearchResult(chunk=chunk, score=score)
        self.evidence.setdefault(key, []).append(reason)


def related_payload(
    index: Any,
    graph: GraphProvider,
    root: Path,
    file: str,
    line: int,
    top_k: int = 5,
    max_snippet_lines: int | None = None,
) -> dict[str, Any]:
    """Combine explicit counterpart/callee/caller evidence with member-level semantic retrieval."""
    seed = index.chunk_at(file, line)
    if seed is None:
        from zemble.utils import describe_unresolved_location

        return {"error": describe_unresolved_location(index, file, line), "unresolved_location": True}
    owner = declaration(graph, file, line)
    collected = RelatedCandidates(index, owner)
    member, docs = source_text(root, owner) if owner is not None else (seed.content, "")
    _graph_candidates(collected, graph, root)
    _semantic_candidates(collected, graph, seed, file, member, docs, top_k)
    ordered = sorted(collected.candidates.values(), key=lambda result: -result.score)
    ordered = _rerank(index, ordered, docs + "\n" + member)
    payload = format_results(f"Members related to {file}:{line}", _diverse(ordered, top_k), max_snippet_lines)
    for result in payload["results"]:
        result["evidence"] = collected.evidence.get((result["file_path"], result["start_line"]), [])
    return payload


def _graph_candidates(collected: RelatedCandidates, graph: GraphProvider, root: Path) -> None:
    owner = collected.owner
    if owner is None:
        return
    for counterpart in counterparts(graph, root, owner):
        collected.add(counterpart.symbol, 2.0, counterpart.reason)
    for hit in graph.callees(owner.id)[:12]:
        if hit.resolution in {Resolution.EXACT, Resolution.UNIQUE_NAME}:
            collected.add(hit.symbol, 1.2, hit.reason)
    for hit in graph.callers(owner.id)[:4]:
        if hit.resolution in {Resolution.EXACT, Resolution.UNIQUE_NAME}:
            collected.add(hit.symbol, 0.75, hit.reason)


def _semantic_candidates(
    collected: RelatedCandidates, graph: GraphProvider, seed: Any, file: str, member: str, docs: str, top_k: int
) -> None:
    # Plain member text complements capsules without changing ordinary search.
    semantic = collected.index.find_related(seed, top_k=max(40, top_k * 8))
    if member:
        semantic += collected.index.search((docs + "\n" + member)[:3000], top_k=20)
    for result in semantic:
        chunk = result.chunk
        symbol = declaration(graph, chunk.file_path, chunk.start_line)
        if symbol is None or symbol.kind in {SymbolKind.PACKAGE, SymbolKind.FIELD}:
            continue
        if chunk.file_path == file and symbol.kind in {SymbolKind.CLASS, SymbolKind.INTERFACE, SymbolKind.ENUM}:
            continue
        collected.add(symbol, min(0.99, result.score), "member semantic similarity")


def _rerank(index: Any, ordered: list[SearchResult], query: str) -> list[SearchResult]:
    if os.environ.get("ZEMBLE_RELATED_RERANK", "0") != "1":
        return ordered
    reranker = index._resolve_reranker(None)
    if reranker is None:
        return ordered
    head = [result for result in ordered if result.score < 1.0][:40]
    if not head:
        return ordered
    scores = reranker.score(query[:3000], [result.chunk.content for result in head])
    ranked = [SearchResult(chunk=result.chunk, score=float(score)) for result, score in zip(head, scores, strict=True)]
    pinned = [result for result in ordered if result.score >= 1.0]
    return (
        pinned
        + sorted(ranked, key=lambda result: -result.score)
        + [result for result in ordered if result not in head + pinned]
    )


def _diverse(ordered: list[SearchResult], top_k: int) -> list[SearchResult]:
    selected = []
    files: set[str] = set()
    for result in ordered:
        if result.chunk.file_path in files:
            continue
        files.add(result.chunk.file_path)
        selected.append(result)
        if len(selected) == top_k:
            break
    return selected
