"""Signature-only views of a type or a file: the cheapest thing the graph can say.

An outline answers "what is in here" for a few hundred tokens instead of the few
thousand a full file costs. It is built from the symbol graph, never from file
text, so it carries no method bodies to trim.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from zemble.graph.model import TYPE_KINDS, Resolution, Symbol, SymbolKind
from zemble.graph.provider import GraphProvider, display_name

# Anonymous classes are named `<Type>$anon@<line>` by the extractor. They are noise in
# an outline: the reader asked what a type declares, not what it instantiates inline.
_ANON_MARKER = "$anon@"
_GLOB_CHARS = "*?["


class OutlineError(Exception):
    """Raised when an outline target names nothing, or names several things."""

    def __init__(self, message: str, candidates: list[Symbol] | None = None) -> None:
        """Build the error.

        :param message: What went wrong, phrased for a human.
        :param candidates: The competing symbols, when the target was ambiguous.
        """
        super().__init__(message)
        self.message = message
        self.candidates = candidates or []


def symbol_label(symbol: Symbol) -> str:
    """Name a symbol's kind and signature without repeating a keyword the signature already carries."""
    text = symbol.signature or symbol.name
    return text if text.split(" ", 1)[0] == symbol.kind.value else f"{symbol.kind.value} {text}"


@dataclass(frozen=True)
class OutlineEntry:
    """One line of an outline: a symbol and how deeply it is nested."""

    symbol: Symbol
    depth: int

    def render(self) -> str:
        """Render the entry as `kind signature  L<start>-<end>  [@Annotation]`."""
        symbol = self.symbol
        span = (
            f"L{symbol.start_line}"
            if symbol.end_line <= symbol.start_line
            else f"L{symbol.start_line}-{symbol.end_line}"
        )
        parts = ["  " * self.depth + symbol_label(symbol), span]
        if symbol.annotations:
            parts.append("[" + " ".join(f"@{name}" for name in symbol.annotations) + "]")
        return "  ".join(parts)


@dataclass
class Outline:
    """A package line, a set of root types, and one line per member."""

    target: str
    file_path: str
    package: str
    entries: list[OutlineEntry]
    members_filter: str | None = None

    def render(self) -> str:
        """Render the outline as plain text."""
        lines = [f"package {self.package}" if self.package else f"# {self.file_path}", ""]
        if self.package:
            lines.insert(1, f"# {self.file_path}")
        lines += [entry.render() for entry in self.entries]
        if self.members_filter and not self.entries:
            lines.append(f"(no member matches {self.members_filter!r})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Render the outline as JSON-ready data."""
        return {
            "target": self.target,
            "file_path": self.file_path,
            "package": self.package,
            "members_filter": self.members_filter,
            "entries": [
                {
                    "kind": entry.symbol.kind.value,
                    "name": entry.symbol.name,
                    "qualified_name": entry.symbol.qualified_name,
                    "signature": entry.symbol.signature,
                    "annotations": entry.symbol.annotations,
                    "modifiers": entry.symbol.modifiers,
                    "start_line": entry.symbol.start_line,
                    "end_line": entry.symbol.end_line,
                    "depth": entry.depth,
                }
                for entry in self.entries
            ],
        }


@dataclass
class Signatures:
    """A symbol's signature plus the call sites the graph resolved exactly."""

    symbol: Symbol
    callers: list[tuple[str, int, str]]
    ambiguous: int = 0

    def render(self) -> str:
        """Render the signature and its callers as plain text."""
        lines = [
            symbol_label(self.symbol),
            f"{self.symbol.file_path}:{self.symbol.start_line}",
            "",
            f"callers (exact): {len(self.callers)}",
        ]
        lines += [f"  {path}:{line}  {name}" for path, line, name in self.callers]
        if self.ambiguous:
            lines.append(f"  (+{self.ambiguous} by-name or ambiguous call site(s) not listed)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Render the signatures answer as JSON-ready data."""
        return {
            "qualified_name": self.symbol.qualified_name,
            "kind": self.symbol.kind.value,
            "signature": self.symbol.signature,
            "file_path": self.symbol.file_path,
            "start_line": self.symbol.start_line,
            "callers": [{"file_path": path, "line": line, "caller": name} for path, line, name in self.callers],
            "ambiguous_callers": self.ambiguous,
        }


def _matches(name: str, pattern: str) -> bool:
    """Match a member name against a pattern, wrapping a plain word in wildcards."""
    glob = pattern if any(char in pattern for char in _GLOB_CHARS) else f"*{pattern}*"
    return fnmatch.fnmatch(name.lower(), glob.lower())


def _looks_like_path(target: str) -> bool:
    """Return True if a target reads as a file path rather than a type name."""
    return "/" in target or "\\" in target or target.endswith(".java")


def _children(symbols: list[Symbol], container_id: str | None) -> list[Symbol]:
    """Return the symbols directly contained by a container, in declaration order."""
    found = [symbol for symbol in symbols if symbol.container_id == container_id]
    return sorted(found, key=lambda symbol: (symbol.start_line, symbol.id))


def _walk(symbols: list[Symbol], root: Symbol, depth: int, pattern: str | None) -> list[OutlineEntry]:
    """Emit a root and its members depth-first, skipping anonymous classes."""
    if _ANON_MARKER in root.name:
        return []
    entries = [OutlineEntry(root, depth)]
    for child in _children(symbols, root.id):
        if _ANON_MARKER in child.name:
            continue
        if child.kind in TYPE_KINDS:
            entries += _walk(symbols, child, depth + 1, pattern)
            continue
        if pattern is not None and not _matches(child.name, pattern):
            continue
        entries.append(OutlineEntry(child, depth + 1))
        # A member can itself hold types (a local class) or members (an enum constant body).
        for nested in _children(symbols, child.id):
            if _ANON_MARKER in nested.name:
                continue
            entries += _walk(symbols, nested, depth + 2, pattern)
    return entries


def _resolve_target(graph: GraphProvider, target: str) -> tuple[list[Symbol], list[Symbol], str]:
    """Resolve an outline target to (file symbols, root symbols, file path)."""
    if _looks_like_path(target):
        path = target.replace("\\", "/")
        symbols = graph.symbols_in_file(path)
        if not symbols:
            raise OutlineError(f"No Java symbols indexed for file {target!r}.")
        roots = [symbol for symbol in symbols if symbol.kind in TYPE_KINDS and _is_top_level(symbols, symbol)]
        return symbols, roots, path
    candidates = [symbol for symbol in graph.definition(target) if symbol.kind in TYPE_KINDS]
    if not candidates:
        raise OutlineError(f"No type named {target!r}. Pass a file path to outline a file.")
    if len({symbol.id for symbol in candidates}) > 1:
        raise OutlineError(f"{target!r} is ambiguous; pass a qualified name or a file path.", candidates)
    root = candidates[0]
    return graph.symbols_in_file(root.file_path), [root], root.file_path


def _is_top_level(symbols: list[Symbol], symbol: Symbol) -> bool:
    """Return True if a type is declared at file level rather than inside another symbol."""
    container = next((other for other in symbols if other.id == symbol.container_id), None)
    return container is None or container.kind is SymbolKind.PACKAGE


def outline(graph: GraphProvider, target: str, members: str | None = None) -> Outline:
    """Build a signature-only outline of a file or a type.

    :param graph: The symbol graph to read declarations from.
    :param target: A workspace-relative file path, or a simple or qualified type name.
    :param members: Optional name pattern; a plain word is matched as `*word*`.
    :return: The outline.
    :raises OutlineError: If the target names nothing, or names several types.
    """
    symbols, roots, path = _resolve_target(graph, target)
    return _build(target, path, symbols, roots, members)


def outline_of(graph: GraphProvider, symbol: Symbol, members: str | None = None) -> Outline:
    """Build the outline of one already-resolved type.

    Used where a symbol is in hand, so no name lookup - and no ambiguity - is involved.

    :param graph: The symbol graph to read declarations from.
    :param symbol: The type to outline.
    :param members: Optional name pattern; a plain word is matched as `*word*`.
    :return: The outline.
    """
    symbols = graph.symbols_in_file(symbol.file_path)
    return _build(symbol.qualified_name, symbol.file_path, symbols, [symbol], members)


def _build(target: str, path: str, symbols: list[Symbol], roots: list[Symbol], members: str | None) -> Outline:
    """Assemble an outline from a resolved file and its root types."""
    package = next((symbol.qualified_name for symbol in symbols if symbol.kind is SymbolKind.PACKAGE), "")
    entries: list[OutlineEntry] = []
    for root in roots:
        entries += _walk(symbols, root, 0, members)
    if members is not None:
        entries = _prune_empty_types(entries)
    return Outline(target=target, file_path=path, package=package, entries=entries, members_filter=members)


def _prune_empty_types(entries: list[OutlineEntry]) -> list[OutlineEntry]:
    """Drop type lines left with nothing under them by a member filter.

    Entries are depth-first, so a type has a descendant exactly when the next entry
    is deeper than it. Pruning one type can empty its parent, hence the fixed point.
    """
    while True:
        kept = [
            entry
            for index, entry in enumerate(entries)
            if entry.symbol.kind not in TYPE_KINDS
            or (index + 1 < len(entries) and entries[index + 1].depth > entry.depth)
        ]
        if len(kept) == len(entries):
            return kept
        entries = kept


def signatures(graph: GraphProvider, symbol: Symbol) -> Signatures:
    """Return a symbol's signature and the call sites resolved exactly.

    :param graph: The symbol graph to read call edges from.
    :param symbol: The symbol to describe.
    :return: The signature and its exact callers, with a count of the weaker ones.
    """
    hits = graph.callers(symbol.id)
    exact = [
        (hit.symbol.file_path, hit.line, display_name(hit.symbol)) for hit in hits if hit.resolution is Resolution.EXACT
    ]
    return Signatures(symbol=symbol, callers=exact, ambiguous=len(hits) - len(exact))


__all__ = [
    "Outline",
    "OutlineEntry",
    "OutlineError",
    "Signatures",
    "outline",
    "outline_of",
    "signatures",
    "symbol_label",
]
