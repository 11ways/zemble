"""One payload per home question, shared by the daemon, the CLI and the MCP tool.

The wire shape and the in-process shape are the same object: a surface renders a
payload without caring whether a warm daemon or this process produced it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from zemble.graph.model import TYPE_KINDS, Resolution, Symbol, SymbolKind
from zemble.graph.provider import GraphProvider, display_name
from zemble.home.config import HomeConfig
from zemble.home.decide import DocHit, HomeAnswer, Mechanism, Similar, decide
from zemble.home.tables import load_rows, match_rows
from zemble.index import ZembleIndex
from zemble.types import ContentType, SearchResult

#: Code results considered when a caller does not state its own breadth.
DEFAULT_TOP_K = 40
#: Documentation results considered on top of them.
DOC_TOP_K = 10
#: Distinct symbols the graph is asked about.
MAX_MECHANISMS = 8
#: Locations offered as "also similar" for the single best hit.
SIMILAR_TOP_K = 3
#: Characters of a documentation chunk shown as its excerpt.
DOC_EXCERPT_CHARS = 160

_DOC_SUFFIXES = (".md", ".rst", ".txt", ".adoc")
_ANON_MARKER = "$anon@"


def home_payload(
    index: ZembleIndex,
    graph: GraphProvider,
    config: HomeConfig,
    description: str,
    top_k: int = DEFAULT_TOP_K,
    use_tables: bool = True,
) -> dict[str, Any]:
    """Answer "does this exist, and where should it live" and carry both renderings.

    :param index: The search index over the workspace.
    :param graph: The symbol graph, for consumers of the mechanisms found.
    :param config: What the workspace declared about its own modules.
    :param description: The feature someone is about to build.
    :param top_k: How many code results to weigh.
    :param use_tables: Whether declared-home tables may answer; False measures the rest.
    :return: The answer as data, plus its markdown rendering.
    """
    answer = build_answer(index, graph, config, description, top_k=top_k, use_tables=use_tables)
    return {"home": answer.to_dict(), "markdown": answer.render()}


def build_answer(
    index: ZembleIndex,
    graph: GraphProvider,
    config: HomeConfig,
    description: str,
    top_k: int = DEFAULT_TOP_K,
    use_tables: bool = True,
) -> HomeAnswer:
    """Gather the evidence for one description and weigh it into an answer."""
    results = index.search(description, top_k=top_k)
    code_hits = [result for result in results if not _is_doc(result.chunk.file_path)]
    docs = _doc_hits(index, config, description, results)
    mechanisms = _mechanisms(graph, config, code_hits)
    similar = _similar(index, config, code_hits)
    rows = match_rows(load_rows(config), description) if use_tables else []
    return decide(config, description, code_hits, mechanisms, rows, similar, docs)


def _is_doc(file_path: str) -> bool:
    """Return True for a path the index treats as prose rather than code."""
    return file_path.lower().endswith(_DOC_SUFFIXES)


def _doc_hits(
    index: ZembleIndex, config: HomeConfig, description: str, results: Sequence[SearchResult]
) -> list[DocHit]:
    """Return the documentation chunks the description matched.

    Only an index that was built over the docs lane can carry any; a code-only index
    yields nothing here rather than a second, differently-built index.
    """
    if ContentType.DOCS not in index.content:
        return []
    found = [result for result in results if _is_doc(result.chunk.file_path)]
    if len(found) < DOC_TOP_K:
        extra = index.search(description, top_k=max(DOC_TOP_K * 3, len(results)))
        seen = {(result.chunk.file_path, result.chunk.start_line) for result in found}
        for result in extra:
            key = (result.chunk.file_path, result.chunk.start_line)
            if _is_doc(result.chunk.file_path) and key not in seen:
                seen.add(key)
                found.append(result)
    found.sort(key=lambda result: -result.score)
    return [
        DocHit(
            file_path=result.chunk.file_path,
            start_line=result.chunk.start_line,
            end_line=result.chunk.end_line,
            module=config.module_of(result.chunk.file_path),
            score=result.score,
            excerpt=_excerpt(result.chunk.content),
        )
        for result in found[:DOC_TOP_K]
    ]


def _excerpt(content: str) -> str:
    """Return a one-line excerpt of a documentation chunk."""
    text = " ".join(content.split())
    return text[:DOC_EXCERPT_CHARS] + ("..." if len(text) > DOC_EXCERPT_CHARS else "")


def _mechanisms(graph: GraphProvider, config: HomeConfig, hits: Sequence[SearchResult]) -> list[Mechanism]:
    """Describe the distinct symbols behind the best code hits, consumers included."""
    mechanisms: list[Mechanism] = []
    seen: set[str] = set()
    for hit in hits:
        if len(mechanisms) >= MAX_MECHANISMS:
            break
        symbol = _anchor(graph, hit.chunk.file_path, hit.chunk.start_line, hit.chunk.end_line)
        if symbol is None or symbol.id in seen:
            continue
        seen.add(symbol.id)
        mechanisms.append(_describe(graph, config, symbol, hit.score))
    return mechanisms


def _anchor(graph: GraphProvider, file_path: str, start_line: int, end_line: int) -> Symbol | None:
    """Pick the declaration a chunk sits in: its type where there is one, else the member.

    Containment first, overlap as the fallback: a chunk that starts at the package
    statement contains its class rather than sitting inside it, and that is the common
    case for a small file.
    """
    path = file_path.replace("\\", "/")
    symbols = _usable(graph.symbols_at(path, start_line, end_line))
    if not symbols:
        touching = [
            symbol
            for symbol in graph.symbols_in_file(path)
            if symbol.start_line <= end_line and symbol.end_line >= start_line
        ]
        symbols = _usable(sorted(touching, key=lambda symbol: (symbol.end_line - symbol.start_line, symbol.start_line)))
    if not symbols:
        return None
    # The type is the mechanism; a single matched method is usually just where the words were.
    for symbol in symbols:
        if symbol.kind in TYPE_KINDS:
            return symbol
    return symbols[0]


def _usable(symbols: Sequence[Symbol]) -> list[Symbol]:
    """Drop the symbols that can never name a mechanism: packages and anonymous classes."""
    return [symbol for symbol in symbols if symbol.kind is not SymbolKind.PACKAGE and _ANON_MARKER not in symbol.name]


def _describe(graph: GraphProvider, config: HomeConfig, symbol: Symbol, score: float) -> Mechanism:
    """Build one mechanism, asking the graph who consumes it.

    A TYPE is consumed by every edge pointing at it - an import, a field declaration,
    a `new`, a subclass - and almost never by a call to the type itself, so asking
    `callers` about a class under-reports its consumers to nearly zero.
    """
    incoming = graph.references(symbol.id) if symbol.kind in TYPE_KINDS else graph.callers(symbol.id)
    callers = [hit for hit in incoming if hit.resolution in (Resolution.EXACT, Resolution.UNIQUE_NAME)]
    implementations = graph.implementations(symbol.id) if symbol.kind in TYPE_KINDS else []
    own = config.module_of(symbol.file_path)
    consumers: list[str] = []
    for hit in [*callers, *implementations]:
        module = config.module_of(hit.symbol.file_path)
        if module != own and module not in consumers:
            consumers.append(module)
    consumers.sort(key=lambda module: (config.rank(module), module))
    return Mechanism(
        label=display_name(symbol),
        kind=symbol.kind.value,
        signature=symbol.signature,
        module=own,
        file_path=symbol.file_path,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        score=score,
        consumer_modules=tuple(consumers),
        caller_count=len(callers),
        implementation_count=len(implementations),
    )


def _similar(index: ZembleIndex, config: HomeConfig, hits: Sequence[SearchResult]) -> list[Similar]:
    """Return what the single best hit resembles elsewhere in the workspace."""
    if not hits:
        return []
    related = index.find_related(hits[0], top_k=SIMILAR_TOP_K)
    return [
        Similar(
            file_path=result.chunk.file_path,
            start_line=result.chunk.start_line,
            end_line=result.chunk.end_line,
            module=config.module_of(result.chunk.file_path),
            score=result.score,
        )
        for result in related
    ]


__all__ = ["DEFAULT_TOP_K", "DOC_TOP_K", "MAX_MECHANISMS", "SIMILAR_TOP_K", "build_answer", "home_payload"]
