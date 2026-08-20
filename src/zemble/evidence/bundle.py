"""Evidence bundles: search, one-hop graph expansion, and packing under a token budget.

A bundle answers "show me what I need to understand this" in a fixed number of
tokens. Search decides where to look, the symbol graph decides what else belongs,
and the packer decides what survives the budget - degrading an item to a location
line before it will drop it, so the answer always says what exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from zemble.evidence.outline import outline_of, symbol_label
from zemble.evidence.tokens import estimate_tokens
from zemble.graph.model import CALLABLE_KINDS, TYPE_KINDS, EdgeKind, Hit, Resolution, Symbol, SymbolKind
from zemble.graph.provider import GraphProvider, display_name
from zemble.index import ZembleIndex
from zemble.types import SearchResult

PRIMARY_FILES = 5
MAX_ANCHORS = 6
MAX_CHUNKS_PER_FILE = 3
MAX_PER_TIER_PER_ANCHOR = 3
MAX_SYMBOL_LINES = 60
MAX_OUTLINE_LINES = 30
MAX_DOC_LINES = 40
MIN_TRUNCATED_LINES = 3
MAX_OMITTED = 25
# Share of the budget a tier will hold back so later tiers can at least be named.
RESERVE_FRACTION = 0.4

_ANON_MARKER = "$anon@"
_DOC_SUFFIXES = (".md", ".rst", ".txt", ".adoc")
_LANGUAGE_BY_SUFFIX = {".java": "java", ".py": "python", ".md": "markdown", ".ts": "typescript", ".js": "javascript"}


class ItemKind(str, Enum):
    """What an evidence item is, which is also what its reason will talk about."""

    CHUNK = "chunk"
    OUTLINE = "outline"
    TEST = "test"
    CALLER = "caller"
    IMPLEMENTATION = "implementation"
    SUPERTYPE = "supertype"
    CALLEE = "callee"
    DOC = "doc"
    NOTE = "note"


class Presentation(str, Enum):
    """How much of an item survived the budget."""

    CONTENT = "content"
    TRUNCATED = "truncated"
    LOCATION = "location"


# The packing order. Tier 0 is what search found; every later tier is one graph hop
# away from it, ordered by how often it turns out to be the thing you needed.
TIERS: dict[ItemKind, int] = {
    ItemKind.CHUNK: 0,
    ItemKind.OUTLINE: 1,
    ItemKind.TEST: 2,
    ItemKind.CALLER: 3,
    ItemKind.IMPLEMENTATION: 3,
    ItemKind.SUPERTYPE: 3,
    ItemKind.NOTE: 3,
    ItemKind.CALLEE: 4,
    ItemKind.DOC: 4,
}


@dataclass(frozen=True)
class BundleItem:
    """One packed piece of evidence."""

    kind: ItemKind
    file_path: str
    start_line: int
    end_line: int
    reason: str
    text: str
    tokens: int
    presentation: Presentation = Presentation.CONTENT
    tier: int = 0

    @property
    def location(self) -> str:
        """File path and line range as a string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict[str, object]:
        """Render the item as JSON-ready data."""
        return {
            "kind": self.kind.value,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "reason": self.reason,
            "text": self.text,
            "tokens": self.tokens,
            "presentation": self.presentation.value,
            "tier": self.tier,
        }


@dataclass(frozen=True)
class OmittedItem:
    """Something the bundle knows about but had no budget to show."""

    kind: ItemKind
    file_path: str
    start_line: int
    end_line: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Render the omission as JSON-ready data."""
        return {
            "kind": self.kind.value,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "reason": self.reason,
        }


@dataclass
class Bundle:
    """An ordered, budgeted set of evidence items for one query."""

    query: str
    budget_tokens: int
    items: list[BundleItem] = field(default_factory=list)
    omitted: list[OmittedItem] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """The estimated token cost of the packed items."""
        return sum(item.tokens for item in self.items)

    def files(self) -> list[str]:
        """The distinct files the bundle shows content from, in bundle order."""
        return list(dict.fromkeys(item.file_path for item in self.items))

    def render(self) -> str:
        """Render the bundle as markdown."""
        lines = [f"# Evidence for: {self.query}", "", f"{len(self.items)} item(s), ~{self.total_tokens} tokens.", ""]
        for item in self.items:
            lines.append(_render_item(item))
        if self.omitted:
            lines += ["## Not included (locations only)", ""]
            lines += [
                f"- {entry.file_path}:{entry.start_line}-{entry.end_line}  ({entry.reason})" for entry in self.omitted
            ]
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Render the bundle as JSON-ready data."""
        return {
            "query": self.query,
            "budget_tokens": self.budget_tokens,
            "total_tokens": self.total_tokens,
            "items": [item.to_dict() for item in self.items],
            "omitted": [entry.to_dict() for entry in self.omitted],
        }


@dataclass
class _Candidate:
    """An item before packing: its full text, its fallback line, and its ranking."""

    kind: ItemKind
    file_path: str
    start_line: int
    end_line: int
    reason: str
    text: str
    location_text: str
    score: float
    # Identity for deduplication: the source region an item shows, so the same lines
    # never arrive twice under two different reasons.
    key: tuple[object, ...]

    @property
    def tier(self) -> int:
        """The packing tier of this candidate's kind."""
        return TIERS[self.kind]


def _language_of(file_path: str) -> str:
    """Return the markdown fence language for a path."""
    return _LANGUAGE_BY_SUFFIX.get(Path(file_path).suffix.lower(), "")


def _render_item(item: BundleItem) -> str:
    """Render one item exactly as it is costed, so the budget matches the output."""
    header = f"## {item.location}  ({item.reason})"
    if item.presentation is Presentation.LOCATION:
        body = f"`{item.text}`" if item.text else "(location only)"
        return f"{header}\n{body}\n"
    fence = _language_of(item.file_path)
    return f"{header}\n```{fence}\n{item.text}\n```\n"


def _cost(candidate: _Candidate, text: str, presentation: Presentation) -> int:
    """Estimate what an item would cost if it were rendered with the given text."""
    probe = BundleItem(
        kind=candidate.kind,
        file_path=candidate.file_path,
        start_line=candidate.start_line,
        end_line=candidate.end_line,
        reason=candidate.reason,
        text=text,
        tokens=0,
        presentation=presentation,
        tier=candidate.tier,
    )
    return estimate_tokens(_render_item(probe))


def truncate_lines(text: str, keep: int) -> str:
    """Keep the first lines of a text and say how many were dropped."""
    lines = text.splitlines()
    if keep >= len(lines):
        return text
    dropped = len(lines) - keep
    return "\n".join([*lines[:keep], f"... (truncated, {dropped} more lines)"])


class _Sources:
    """Line-cached reader for workspace files, so one file is read at most once."""

    def __init__(self, root: Path | None) -> None:
        """Build the reader.

        :param root: The workspace root; None disables all file reads.
        """
        self.root = root
        self._cache: dict[str, list[str]] = {}

    def lines(self, file_path: str) -> list[str]:
        """Return a file's lines, or an empty list when it cannot be read."""
        if self.root is None:
            return []
        if file_path not in self._cache:
            try:
                text = (self.root / file_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            self._cache[file_path] = text.splitlines()
        return self._cache[file_path]

    def span(self, file_path: str, start_line: int, end_line: int, max_lines: int) -> str:
        """Return the source of a line span, capped to a line count."""
        lines = self.lines(file_path)
        if not lines:
            return ""
        return truncate_lines("\n".join(lines[max(0, start_line - 1) : end_line]), max_lines)


def _index_root(index: ZembleIndex) -> Path | None:
    """Return the workspace root an index was built from, if it has one.

    The index keeps its root private and a git-cloned index has none at all, so the
    bundle degrades to graph-only evidence rather than failing.
    """
    root = getattr(index, "_root", None)
    return Path(root) if root is not None else None


def _normalise(file_path: str) -> str:
    """Normalise a chunk path to the separator the graph stores."""
    return file_path.replace("\\", "/")


def _primary_files(results: Sequence[SearchResult], limit: int) -> list[str]:
    """Return the files with the most fused score behind them, best first."""
    totals: dict[str, float] = {}
    for result in results:
        path = _normalise(result.chunk.file_path)
        totals[path] = totals.get(path, 0.0) + result.score
    return [path for path, _ in sorted(totals.items(), key=lambda item: -item[1])[:limit]]


def _overlapping(graph: GraphProvider, file_path: str, start_line: int, end_line: int) -> list[Symbol]:
    """Return the declarations a chunk touches, narrowest first.

    Containment comes first because it names the declaration the chunk is inside;
    a chunk that straddles two declarations contains neither, so overlap is the
    fallback rather than the rule.
    """
    contained = graph.symbols_at(file_path, start_line, end_line)
    if contained:
        return contained
    touching = [
        symbol
        for symbol in graph.symbols_in_file(file_path)
        if symbol.start_line <= end_line and symbol.end_line >= start_line
    ]
    return sorted(touching, key=lambda symbol: (symbol.end_line - symbol.start_line, symbol.start_line))


def _anchor_of(symbols: Iterable[Symbol]) -> tuple[Symbol | None, Symbol | None]:
    """Pick the nearest enclosing callable and the type that declares it."""
    anchor: Symbol | None = None
    enclosing_type: Symbol | None = None
    for symbol in symbols:
        if _ANON_MARKER in symbol.name or symbol.kind is SymbolKind.PACKAGE:
            continue
        if anchor is None and symbol.kind in CALLABLE_KINDS:
            anchor = symbol
        if enclosing_type is None and symbol.kind in TYPE_KINDS:
            enclosing_type = symbol
    return anchor or enclosing_type, enclosing_type


def _outline_text(graph: GraphProvider, symbol: Symbol) -> str:
    """Render a type's outline, capped, for use as evidence."""
    return truncate_lines(outline_of(graph, symbol).render(), MAX_OUTLINE_LINES)


def _location_line(symbol: Symbol) -> str:
    """Return the one-line fallback for a symbol: its kind and signature."""
    return f"{symbol.kind.value} {symbol.signature or symbol.qualified_name}"


def _candidate_for_symbol(kind: ItemKind, symbol: Symbol, reason: str, text: str, score: float) -> _Candidate:
    """Build a candidate whose evidence is one symbol's declaration."""
    return _Candidate(
        kind=kind,
        file_path=symbol.file_path,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        reason=reason,
        text=text,
        location_text=symbol_label(symbol),
        score=score,
        key=(symbol.file_path, symbol.start_line, symbol.end_line),
    )


def _ordered_hits(hits: Sequence[Hit]) -> tuple[list[Hit], int]:
    """Order hits by resolution quality and count the ones too weak to show."""
    exact = [hit for hit in hits if hit.resolution is Resolution.EXACT]
    unique = [hit for hit in hits if hit.resolution is Resolution.UNIQUE_NAME]
    weak = len(hits) - len(exact) - len(unique)
    return exact + unique, weak


def _test_candidates(graph: GraphProvider, anchor: Symbol, sources: _Sources, score: float) -> list[_Candidate]:
    """Build tier 2: the tests that name or exercise the anchor."""
    hits = graph.tests_of(anchor.id)
    named = [hit for hit in hits if hit.edge_kind is EdgeKind.TESTS]
    exercising = [hit for hit in hits if hit.edge_kind is not EdgeKind.TESTS]
    candidates = []
    for hit in (named + exercising)[:MAX_PER_TIER_PER_ANCHOR]:
        text = sources.span(hit.symbol.file_path, hit.symbol.start_line, hit.symbol.end_line, MAX_SYMBOL_LINES)
        reason = f"{hit.reason}; covers {display_name(anchor)}"
        candidates.append(_candidate_for_symbol(ItemKind.TEST, hit.symbol, reason, text, score))
    return candidates


def _caller_candidates(graph: GraphProvider, anchor: Symbol, sources: _Sources, score: float) -> list[_Candidate]:
    """Build tier 3: the call sites that reach the anchor, best resolution first."""
    ordered, weak = _ordered_hits(graph.callers(anchor.id))
    candidates = []
    for hit in ordered[:MAX_PER_TIER_PER_ANCHOR]:
        text = sources.span(hit.symbol.file_path, hit.symbol.start_line, hit.symbol.end_line, MAX_SYMBOL_LINES)
        candidates.append(_candidate_for_symbol(ItemKind.CALLER, hit.symbol, hit.reason, text, score))
    if weak:
        candidates.append(
            _Candidate(
                kind=ItemKind.NOTE,
                file_path=anchor.file_path,
                start_line=anchor.start_line,
                end_line=anchor.end_line,
                reason=f"{weak} further call site(s) of {display_name(anchor)} were ambiguous or unresolved",
                text="",
                location_text=f"run `zemble graph callers <repo> {display_name(anchor)}` to list them",
                score=score,
                key=("note-callers", anchor.file_path, anchor.start_line, anchor.end_line),
            )
        )
    return candidates


def _hierarchy_candidates(graph: GraphProvider, enclosing: Symbol, score: float) -> list[_Candidate]:
    """Build tier 3 for an interface or abstract type: who implements it, what it extends."""
    if enclosing.kind is not SymbolKind.INTERFACE and "abstract" not in enclosing.modifiers:
        return []
    candidates = []
    for kind, hits in (
        (ItemKind.IMPLEMENTATION, graph.implementations(enclosing.id)),
        (ItemKind.SUPERTYPE, graph.supertypes(enclosing.id)),
    ):
        for hit in hits[:MAX_PER_TIER_PER_ANCHOR]:
            candidates.append(
                _candidate_for_symbol(kind, hit.symbol, hit.reason, _outline_text(graph, hit.symbol), score)
            )
    return candidates


def _callee_candidates(graph: GraphProvider, anchor: Symbol, sources: _Sources, score: float) -> list[_Candidate]:
    """Build tier 4: what the anchor itself calls."""
    ordered, _ = _ordered_hits(graph.callees(anchor.id))
    candidates = []
    for hit in ordered[:MAX_PER_TIER_PER_ANCHOR]:
        if hit.symbol.file_path == anchor.file_path:
            continue  # Already on screen: the anchor's own file is in the bundle.
        text = sources.span(hit.symbol.file_path, hit.symbol.start_line, hit.symbol.end_line, MAX_SYMBOL_LINES)
        candidates.append(_candidate_for_symbol(ItemKind.CALLEE, hit.symbol, hit.reason, text, score))
    return candidates


def _doc_candidates(
    results: Sequence[SearchResult], primary: Sequence[str], sources: _Sources, score: float
) -> list[_Candidate]:
    """Build tier 4 docs: prose search already returned, plus a sibling doc of a primary file."""
    candidates: list[_Candidate] = []
    for result in results:
        path = _normalise(result.chunk.file_path)
        if not path.lower().endswith(_DOC_SUFFIXES):
            continue
        candidates.append(
            _Candidate(
                kind=ItemKind.DOC,
                file_path=path,
                start_line=result.chunk.start_line,
                end_line=result.chunk.end_line,
                reason="documentation the same query matched",
                text=truncate_lines(result.chunk.content, MAX_DOC_LINES),
                location_text=path,
                score=score,
                key=(path, result.chunk.start_line, result.chunk.end_line),
            )
        )
    candidates += _sibling_docs(primary, sources, score)
    return candidates[:MAX_PER_TIER_PER_ANCHOR]


def _sibling_docs(primary: Sequence[str], sources: _Sources, score: float) -> list[_Candidate]:
    """Find a markdown file sitting next to a primary file."""
    if sources.root is None:
        return []
    candidates = []
    seen: set[str] = set()
    for path in primary:
        folder = Path(path).parent
        if str(folder) in seen:
            continue
        seen.add(str(folder))
        try:
            siblings = sorted(entry for entry in (sources.root / folder).glob("*.md") if entry.is_file())
        except OSError:
            continue
        for sibling in siblings[:1]:
            relative = str(sibling.relative_to(sources.root)).replace("\\", "/")
            text = truncate_lines("\n".join(sources.lines(relative)), MAX_DOC_LINES)
            if not text:
                continue
            candidates.append(
                _Candidate(
                    kind=ItemKind.DOC,
                    file_path=relative,
                    start_line=1,
                    end_line=len(sources.lines(relative)),
                    reason=f"documentation sitting beside {path}",
                    text=text,
                    location_text=relative,
                    score=score,
                    key=(relative, 1, 1),
                )
            )
    return candidates


def _expand(
    graph: GraphProvider, results: Sequence[SearchResult], primary: Sequence[str], sources: _Sources
) -> list[_Candidate]:
    """Run one graph hop out of every primary chunk and collect the candidates."""
    candidates: list[_Candidate] = []
    anchors: list[tuple[Symbol, Symbol | None, float]] = []
    per_file: dict[str, int] = {}
    for rank, result in enumerate(results, 1):
        path = _normalise(result.chunk.file_path)
        if path not in primary or per_file.get(path, 0) >= MAX_CHUNKS_PER_FILE:
            continue
        per_file[path] = per_file.get(path, 0) + 1
        chunk = result.chunk
        anchor, enclosing = _anchor_of(_overlapping(graph, path, chunk.start_line, chunk.end_line))
        inside = f" inside {display_name(anchor)}" if anchor is not None else ""
        candidates.append(
            _Candidate(
                kind=ItemKind.CHUNK,
                file_path=path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                reason=f"search hit #{rank}{inside}",
                text=chunk.content,
                location_text=f"{path} lines {chunk.start_line}-{chunk.end_line}",
                score=result.score,
                key=(path, chunk.start_line, chunk.end_line),
            )
        )
        if anchor is not None and len(anchors) < MAX_ANCHORS:
            anchors.append((anchor, enclosing, result.score))

    seen_anchors: set[str] = set()
    for anchor, enclosing, score in anchors:
        if anchor.id in seen_anchors:
            continue
        seen_anchors.add(anchor.id)
        if enclosing is not None:
            text = _outline_text(graph, enclosing)
            if text:
                candidates.append(
                    _candidate_for_symbol(
                        ItemKind.OUTLINE,
                        enclosing,
                        f"outline of {enclosing.name}, which declares {anchor.name}",
                        text,
                        score,
                    )
                )
            candidates += _hierarchy_candidates(graph, enclosing, score)
        candidates += _test_candidates(graph, anchor, sources, score)
        candidates += _caller_candidates(graph, anchor, sources, score)
        candidates += _callee_candidates(graph, anchor, sources, score)
    candidates += _doc_candidates(results, primary, sources, 0.0)
    return candidates


def pack(query: str, candidates: Sequence[_Candidate], budget_tokens: int) -> Bundle:
    """Pack candidates into a bundle under a token budget.

    Order is tier first, then fused score, then discovery order. An item that does
    not fit as content degrades to its location line; a chunk is truncated first.
    Anything that does not fit at all is listed as omitted.

    Each tier packs against the budget MINUS a reserve: what it would cost to name
    every later-tier candidate as a location line, capped at `RESERVE_FRACTION` of
    the budget. Without it a handful of long primary chunks eats the whole budget
    and the graph expansion - the entire point of a bundle - never appears.

    :param query: The query the bundle answers.
    :param candidates: The candidates to pack.
    :param budget_tokens: The maximum estimated tokens the packed items may cost.
    :return: The packed bundle.
    """
    bundle = Bundle(query=query, budget_tokens=budget_tokens)
    remaining = budget_tokens
    seen: set[tuple[object, ...]] = set()
    ordered = [candidate for _, candidate in sorted(enumerate(candidates), key=_pack_order)]
    floors = _tier_floors(ordered)
    cap = int(budget_tokens * RESERVE_FRACTION)
    for candidate in ordered:
        floors[candidate.tier] -= _cost(candidate, candidate.location_text, Presentation.LOCATION)
        if candidate.key in seen:
            continue
        seen.add(candidate.key)
        reserve = min(cap, sum(cost for tier, cost in floors.items() if tier > candidate.tier))
        item = _fit(candidate, max(0, remaining - reserve))
        if item is None:
            _omit(bundle, candidate)
            continue
        bundle.items.append(item)
        remaining -= item.tokens
    return bundle


def _pack_order(pair: tuple[int, _Candidate]) -> tuple[int, float, int]:
    """Sort key: tier, then fused score, then discovery order."""
    index, candidate = pair
    return candidate.tier, -candidate.score, index


def _tier_floors(candidates: Sequence[_Candidate]) -> dict[int, int]:
    """Return, per tier, what it would cost to show every candidate as a location line."""
    floors = dict.fromkeys(set(TIERS.values()), 0)
    for candidate in candidates:
        floors[candidate.tier] += _cost(candidate, candidate.location_text, Presentation.LOCATION)
    return floors


def _omit(bundle: Bundle, candidate: _Candidate) -> None:
    """Record a candidate the budget could not fit, up to the omission cap."""
    if len(bundle.omitted) >= MAX_OMITTED:
        return
    bundle.omitted.append(
        OmittedItem(
            kind=candidate.kind,
            file_path=candidate.file_path,
            start_line=candidate.start_line,
            end_line=candidate.end_line,
            reason=candidate.reason,
        )
    )


def _fit(candidate: _Candidate, remaining: int) -> BundleItem | None:
    """Return the largest form of a candidate that fits, or None if none does."""
    for text, presentation in _forms(candidate, remaining):
        cost = _cost(candidate, text, presentation)
        if cost <= remaining:
            return BundleItem(
                kind=candidate.kind,
                file_path=candidate.file_path,
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                reason=candidate.reason,
                text=text,
                tokens=cost,
                presentation=presentation,
                tier=candidate.tier,
            )
    return None


def _forms(candidate: _Candidate, remaining: int) -> list[tuple[str, Presentation]]:
    """List the forms of a candidate, largest first: full, truncated, location."""
    forms: list[tuple[str, Presentation]] = []
    if candidate.text:
        forms.append((candidate.text, Presentation.CONTENT))
        if candidate.kind is ItemKind.CHUNK:
            forms += _truncations(candidate, remaining)
    forms.append((candidate.location_text, Presentation.LOCATION))
    return forms


def _truncations(candidate: _Candidate, remaining: int) -> list[tuple[str, Presentation]]:
    """Offer the largest truncation of a chunk that fits the budget left.

    A bisection rather than a proportional guess: the truncation marker is itself a
    line, so a share-of-the-text estimate overshoots by exactly the amount that
    makes a chunk fall back to a location line instead of being trimmed.
    """
    lines = candidate.text.splitlines()
    if len(lines) <= MIN_TRUNCATED_LINES:
        return []
    low, high = MIN_TRUNCATED_LINES, len(lines) - 1
    best: str | None = None
    while low <= high:
        middle = (low + high) // 2
        text = truncate_lines(candidate.text, middle)
        if _cost(candidate, text, Presentation.TRUNCATED) <= remaining:
            best = text
            low = middle + 1
        else:
            high = middle - 1
    return [(best, Presentation.TRUNCATED)] if best is not None else []


def build_bundle(
    index: ZembleIndex,
    graph: GraphProvider,
    query: str,
    budget_tokens: int,
    top_k: int = 20,
    primary_files: int = PRIMARY_FILES,
) -> Bundle:
    """Build an evidence bundle for a query under a token budget.

    :param index: The search index to find the primary chunks with.
    :param graph: The symbol graph to expand them through.
    :param query: The question the bundle answers.
    :param budget_tokens: The maximum estimated tokens the packed items may cost.
    :param top_k: How many search results to consider.
    :param primary_files: How many of the best-scoring files count as primary.
    :return: The packed bundle.
    """
    results = index.search(query, top_k=top_k)
    bundle_sources = _Sources(_index_root(index))
    if not results:
        return Bundle(query=query, budget_tokens=budget_tokens)
    primary = _primary_files(results, primary_files)
    candidates = _expand(graph, results, primary, bundle_sources)
    bundle = pack(query, candidates, budget_tokens)
    for result in results:
        path = _normalise(result.chunk.file_path)
        if path in primary or len(bundle.omitted) >= MAX_OMITTED:
            continue
        bundle.omitted.append(
            OmittedItem(
                kind=ItemKind.CHUNK,
                file_path=path,
                start_line=result.chunk.start_line,
                end_line=result.chunk.end_line,
                reason="search hit outside the primary files",
            )
        )
    return bundle


__all__ = [
    "Bundle",
    "BundleItem",
    "ItemKind",
    "OmittedItem",
    "Presentation",
    "build_bundle",
    "pack",
    "truncate_lines",
]
