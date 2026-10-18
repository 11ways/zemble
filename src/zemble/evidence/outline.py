"""Signature-only views of a type or a file: the cheapest thing the graph can say.

An outline answers "what is in here" for a few hundred tokens instead of the few
thousand a full file costs. It is built from the symbol graph, never from file
text, so it carries no method bodies to trim. By default it lists what code outside
the file can use: private members and every field are counted, not listed.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import PurePosixPath

from zemble.graph.model import TYPE_KINDS, Resolution, Symbol, SymbolKind
from zemble.graph.provider import GraphProvider, display_name
from zemble.graph.visibility import is_runtime_hook, symbol_visibility
from zemble.index.files import detect_language
from zemble.languages import Visibility

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
    return kind_label(symbol.kind.value, symbol.signature or symbol.name)


def kind_label(kind: str, text: str) -> str:
    """Prefix a signature or name with its kind, unless the signature already starts with that keyword."""
    return text if text.split(" ", 1)[0] == kind else f"{kind} {text}"


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


@dataclass(frozen=True)
class Omitted:
    """What a default outline counted instead of listing."""

    private: int = 0
    fields: int = 0

    def __add__(self, other: Omitted) -> Omitted:
        """Sum two counts."""
        return Omitted(self.private + other.private, self.fields + other.fields)

    def to_dict(self) -> dict[str, int]:
        """Render the counts as JSON-ready data."""
        return {"private": self.private, "fields": self.fields}


@dataclass
class Outline:
    """A package line, a set of root types, and one line per member."""

    target: str
    file_path: str
    package: str
    entries: list[OutlineEntry]
    members_filter: str | None = None
    omitted: Omitted = Omitted()

    def render(self) -> str:
        """Render the outline as plain text."""
        lines = outline_header(self.package, self.file_path) + [""]
        lines += [entry.render() for entry in self.entries]
        if self.members_filter and not self.entries:
            lines.append(no_member_line(self.members_filter))
        note = omitted_line(self.omitted.private, self.omitted.fields)
        if note:
            lines.append(note)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Render the outline as JSON-ready data."""
        data: dict[str, object] = {
            "target": self.target,
            "file_path": self.file_path,
            "package": self.package,
            "members_filter": self.members_filter,
            "omitted": self.omitted.to_dict(),
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
        note = omitted_line(self.omitted.private, self.omitted.fields)
        if note:
            data["note"] = note
        return data


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
            callers_heading(len(self.callers)),
        ]
        lines += [caller_line(path, line, name) for path, line, name in self.callers]
        if self.ambiguous:
            lines.append(ambiguous_line(self.ambiguous))
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


def outline_header(package: str, file_path: str) -> list[str]:
    """Return the lines an outline opens with: its package, when it has one, then its file."""
    return [f"package {package}", f"# {file_path}"] if package else [f"# {file_path}"]


def no_member_line(members_filter: str) -> str:
    """Say that a member filter left an outline empty."""
    return f"(no member matches {members_filter!r})"


def omitted_line(private: int, fields: int) -> str:
    """Say what a default outline counted instead of listing, and how to list it; empty when nothing was."""
    left = [f"{count} {noun}" for count, noun in ((private, "private member(s)"), (fields, "field(s)")) if count]
    if not left:
        return ""
    return f"(not listed: {' and '.join(left)}; pass `all` to list them, or `members` to find one by name)"


def callers_heading(count: int) -> str:
    """Head the exact call sites of a signatures answer."""
    return f"callers (exact): {count}"


def caller_line(path: str, line: int, name: str) -> str:
    """Name one exact call site of a signatures answer."""
    return f"  {path}:{line}  {name}"


def ambiguous_line(count: int) -> str:
    """Count the call sites a signatures answer does not list."""
    return f"  (+{count} by-name or ambiguous call site(s) not listed)"


def _matches(name: str, pattern: str) -> bool:
    """Match a member name against a pattern, wrapping a plain word in wildcards."""
    glob = pattern if any(char in pattern for char in _GLOB_CHARS) else f"*{pattern}*"
    return fnmatch.fnmatch(name.lower(), glob.lower())


def _looks_like_path(target: str) -> bool:
    """Return True if a target reads as a file path rather than a type name."""
    if "/" in target or "\\" in target:
        return True
    return "." in target and detect_language(PurePosixPath(target)) is not None


def _children(symbols: list[Symbol], container_id: str | None) -> list[Symbol]:
    """Return the symbols directly contained by a container, in declaration order."""
    found = [symbol for symbol in symbols if symbol.container_id == container_id]
    return sorted(found, key=lambda symbol: (symbol.start_line, symbol.id))


@dataclass
class _Scope:
    """What one walk lists: the file's symbols, a name pattern, and whether everything is wanted."""

    symbols: list[Symbol]
    pattern: str | None
    everything: bool

    def __post_init__(self) -> None:
        """Index the file's symbols by id, for container lookups."""
        self.by_id = {symbol.id: symbol for symbol in self.symbols}

    def container(self, symbol: Symbol) -> Symbol | None:
        """The symbol's declaring container within the file."""
        return self.by_id.get(symbol.container_id) if symbol.container_id else None

    def left_out(self, symbol: Symbol) -> Omitted | None:
        """Count a symbol and what it holds when a default outline leaves it out, else None.

        AIDEV-NOTE: only a level the language rules place as PRIVATE hides a member; UNKNOWN
        and package-private stay listed, since an outline that hides what it cannot place
        would drop real API. A runtime hook (`__init__`) is API whatever its spelling. A
        name filter is an explicit ask, so it searches every member.
        """
        if self.everything or self.pattern is not None:
            return None
        if symbol.kind is SymbolKind.FIELD:
            return Omitted(fields=1)
        if symbol_visibility(symbol, self.container) is Visibility.PRIVATE and not is_runtime_hook(symbol):
            return Omitted(private=1 + self.descendants(symbol))
        return None

    def descendants(self, symbol: Symbol) -> int:
        """Count the named symbols a symbol holds, at any depth."""
        return sum(1 + self.descendants(child) for child in _children(self.symbols, symbol.id) if not _anonymous(child))


def _anonymous(symbol: Symbol) -> bool:
    """Whether a symbol is an anonymous class, which an outline never lists."""
    return _ANON_MARKER in symbol.name


def _walk(scope: _Scope, root: Symbol, depth: int) -> tuple[list[OutlineEntry], Omitted]:
    """Emit a root and its members depth-first, skipping anonymous classes and counting what is left out."""
    if _anonymous(root):
        return [], Omitted()
    entries = [OutlineEntry(root, depth)]
    omitted = Omitted()
    for child in _children(scope.symbols, root.id):
        if _anonymous(child):
            continue
        left = scope.left_out(child)
        if left is not None:
            omitted += left
            continue
        if child.kind in TYPE_KINDS:
            found, hidden = _walk(scope, child, depth + 1)
            entries += found
            omitted += hidden
            continue
        if scope.pattern is not None and not _matches(child.name, scope.pattern):
            continue
        entries.append(OutlineEntry(child, depth + 1))
        # A member can itself hold types (a local class) or members (an enum constant body).
        for nested in _children(scope.symbols, child.id):
            found, hidden = _walk(scope, nested, depth + 2)
            entries += found
            omitted += hidden
    return entries, omitted


def _resolve_target(graph: GraphProvider, target: str) -> tuple[list[Symbol], list[Symbol], str]:
    """Resolve an outline target to (file symbols, root symbols, file path)."""
    if _looks_like_path(target):
        path = target.replace("\\", "/")
        symbols = graph.symbols_in_file(path)
        if not symbols:
            raise OutlineError(f"No symbols indexed for file {target!r}.")
        roots = [symbol for symbol in symbols if symbol.kind in TYPE_KINDS and _is_top_level(symbols, symbol)]
        return symbols, roots, path
    candidates = [symbol for symbol in graph.definition(target) if symbol.kind in TYPE_KINDS]
    if not candidates:
        raise OutlineError(f"No type or module named {target!r}. Pass a file path to outline a file.")
    if len({symbol.id for symbol in candidates}) > 1:
        raise OutlineError(f"{target!r} is ambiguous; pass a qualified name or a file path.", candidates)
    root = candidates[0]
    return graph.symbols_in_file(root.file_path), [root], root.file_path


def _is_top_level(symbols: list[Symbol], symbol: Symbol) -> bool:
    """Return True if a type is declared at file level rather than inside another symbol."""
    container = next((other for other in symbols if other.id == symbol.container_id), None)
    return container is None or container.kind is SymbolKind.PACKAGE


def outline(graph: GraphProvider, target: str, members: str | None = None, everything: bool = False) -> Outline:
    """Build a signature-only outline of a file or a type.

    :param graph: The symbol graph to read declarations from.
    :param target: A workspace-relative file path, or a simple or qualified type name.
    :param members: Optional name pattern, matched against every member; a plain word is matched as `*word*`.
    :param everything: List private members and fields too, instead of counting them.
    :return: The outline.
    :raises OutlineError: If the target names nothing, or names several types.
    """
    symbols, roots, path = _resolve_target(graph, target)
    return _build(target, path, symbols, roots, members, everything)


def outline_of(graph: GraphProvider, symbol: Symbol, members: str | None = None, everything: bool = False) -> Outline:
    """Build the outline of one already-resolved type.

    Used where a symbol is in hand, so no name lookup - and no ambiguity - is involved.

    :param graph: The symbol graph to read declarations from.
    :param symbol: The type to outline.
    :param members: Optional name pattern, matched against every member; a plain word is matched as `*word*`.
    :param everything: List private members and fields too, instead of counting them.
    :return: The outline.
    """
    symbols = graph.symbols_in_file(symbol.file_path)
    return _build(symbol.qualified_name, symbol.file_path, symbols, [symbol], members, everything)


def _build(
    target: str, path: str, symbols: list[Symbol], roots: list[Symbol], members: str | None, everything: bool
) -> Outline:
    """Assemble an outline from a resolved file and its root types."""
    package = next((symbol.qualified_name for symbol in symbols if symbol.kind is SymbolKind.PACKAGE), "")
    scope = _Scope(symbols, members, everything)
    entries: list[OutlineEntry] = []
    omitted = Omitted()
    for root in roots:
        found, hidden = _walk(scope, root, 0)
        entries += found
        omitted += hidden
    if members is not None:
        entries = _prune_empty_types(entries)
    return Outline(
        target=target, file_path=path, package=package, entries=entries, members_filter=members, omitted=omitted
    )


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
