"""Reading the markdown tables in which a workspace already declares capability homes.

A row of such a table is the strongest evidence there is: a human wrote down where
a capability lives and who consumes it. The parser stays deliberately dumb - pipe
tables, backticked module names, everything else prose - because the tables are
documentation first and a machine input second.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import log
from typing import Any

from zemble.home.config import HomeConfig, TableSpec

#: A cell separator that is not escaped as `\|` inside a code span.
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_BACKTICKED = re.compile(r"`([^`]+)`")
_SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")
_WORD = re.compile(r"[a-z0-9]+")

#: An argument list, dropped before names are read: it holds types, not the name.
_PARENS = re.compile(r"\([^()]*\)")
#: A generic parameter list, dropped for the same reason.
_GENERICS = re.compile(r"<[^<>]*>")
#: Everything that cannot be part of a symbol name separates two of them.
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9_./]+")
#: `Class` or `Class.member`, the only two shapes a declared name is read as.
_SYMBOL = re.compile(r"^[A-Z][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
#: A bare member, as written in the `Class.a/b/c` shorthand the tables use.
_MEMBER = re.compile(r"^[a-z_][A-Za-z0-9_]*$")

#: Words too common in a capability description to say anything about which row it is.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at", "be", "but", "by", "can", "for", "from", "has", "have", "how",
        "in", "into", "is", "it", "its", "must", "never", "not", "of", "on", "one", "only", "or", "over", "per",
        "so", "that", "the", "their", "them", "then", "there", "they", "this", "to", "up", "was", "what", "when",
        "where", "which", "while", "who", "will", "with", "would", "you", "your", "new", "add", "adding", "want",
        "need", "should",
    }
)  # fmt: skip

#: A row must share at least this many meaningful words with the description to match.
MIN_SHARED_WORDS = 2
#: ... and this share of the description's matchable weight.
MIN_MATCH_SCORE = 0.2


class RowMatchKind(str, Enum):
    """How a declared row's name relates to a symbol: the one home for that vocabulary."""

    #: The row names this exact symbol.
    EXACT_MEMBER = "exact_member"
    #: The row names only the type the symbol is declared in.
    BARE_TYPE = "bare_type"
    #: The row names nothing about this symbol.
    NONE = "none"


@dataclass(frozen=True)
class DeclaredRow:
    """One row of a declared-home table, as far as it could be understood."""

    capability: str
    #: Class and `Class.member` names the capability cell writes in backticks.
    symbols: tuple[str, ...]
    home_modules: tuple[str, ...]
    #: Backticked names in the home cell that are not declared modules, kept verbatim.
    home_names: tuple[str, ...]
    consumer_modules: tuple[str, ...]
    file: str
    line: int
    raw_home: str

    @property
    def title(self) -> str:
        """A short label for the row: its capability up to the first parenthesis."""
        head = self.capability.split("(")[0].strip(" .,;:")
        return head or self.capability[:60]

    def to_dict(self) -> dict[str, Any]:
        """Render the row as JSON-ready data."""
        return {
            "capability": self.capability,
            "title": self.title,
            "symbols": list(self.symbols),
            "home_modules": list(self.home_modules),
            "home_names": list(self.home_names),
            "consumer_modules": list(self.consumer_modules),
            "file": self.file,
            "line": self.line,
            "raw_home": self.raw_home,
        }


@dataclass(frozen=True)
class RowMatch:
    """A declared row the description looks like, and how much it looks like it."""

    row: DeclaredRow
    score: float
    shared: tuple[str, ...]

    @property
    def lexical_score(self) -> float:
        """The word-overlap fact alone, which never proves the row DECLARES anything."""
        return self.score

    def to_dict(self) -> dict[str, Any]:
        """Render the match as JSON-ready data."""
        return {
            "row": self.row.to_dict(),
            "score": round(self.score, 3),
            "lexical_score": round(self.lexical_score, 3),
            "shared": list(self.shared),
        }


def words(text: str) -> set[str]:
    """Return the meaningful lowercase words of a text."""
    return {word for word in _WORD.findall(text.lower()) if len(word) > 2 and word not in _STOPWORDS}


def symbol_names(text: str) -> tuple[str, ...]:
    """Extract the symbol names a text writes in backticks, in the order they appear.

    Only `Class` and `Class.member` are read as names: a backticked path, setting key
    or package (`common/holder`, `comms.channels.*`) names no symbol, and reading one
    out of it is how a row starts claiming code it never mentioned. Argument lists and
    generic parameters are dropped first, and the `Class.a/b/c` shorthand these tables
    use for a family of members expands to one name per member.
    """
    found: list[str] = []
    for span in _BACKTICKED.findall(text):
        cleaned = _strip_groups(span)
        for token in _TOKEN_SPLIT.split(cleaned):
            for name in _names_of(token):
                if name not in found:
                    found.append(name)
    return tuple(found)


def _strip_groups(span: str) -> str:
    """Remove parenthesised and generic groups, innermost first."""
    for pattern in (_PARENS, _GENERICS):
        while True:
            stripped = pattern.sub(" ", span)
            if stripped == span:
                break
            span = stripped
    # An unclosed group is the common case in a truncated cell: cut it off entirely.
    return span.split("(")[0].split("<")[0]


def _names_of(token: str) -> list[str]:
    """Read one whitespace-free token as zero or more declared names."""
    parts = [part for part in token.split("/") if part]
    if not parts:
        return []
    head = parts[0].strip(".")
    if not _SYMBOL.match(head):
        return []
    names = [head]
    if "." in head:
        owner = head.split(".")[0]
        names.extend(f"{owner}.{part}" for part in parts[1:] if _MEMBER.match(part))
    return names


def row_match_kind(declared: str, unit_name: str) -> RowMatchKind:
    """How a declared table name relates to a file-local qualified symbol name.

    Three shapes count and no more: the exact name and a `Type.member` row whose qualified
    tail the unit carries (`Outer.Texts.trimmedOrNull` is named by `Texts.trimmedOrNull`)
    are EXACT_MEMBER, a bare `Type` row naming a member declared directly in that type is
    BARE_TYPE, and everything else is NONE. A row's member name alone never matches, so
    `Other.trimmedOrNull` does not claim `Texts.trimmedOrNull`: an unrecognised shape fails
    closed rather than guessing.

    The two kinds are deliberately distinct: EXACT_MEMBER is the row DECLARING this symbol,
    while BARE_TYPE only says the row is about the class it sits in. Only the first is
    evidence that a capability already has a declared home.

    AIDEV-NOTE: `zemble.home.decide._named_rows` also relates a row's `Class.member` to
    its bare `Class`, but in the OPPOSITE direction - it indexes a declared name under
    its owning type and matches search labels by equality, while this matches a
    file-local qualified unit name by suffix. Keep the two rules in step by hand; they
    are not one predicate, and merging them would change what `home` calls strong.
    """
    if not declared or not unit_name:
        return RowMatchKind.NONE
    if declared == unit_name:
        return RowMatchKind.EXACT_MEMBER
    if "." in declared:
        return RowMatchKind.EXACT_MEMBER if unit_name.endswith(f".{declared}") else RowMatchKind.NONE
    parts = unit_name.split(".")
    return RowMatchKind.BARE_TYPE if len(parts) >= 2 and parts[-2] == declared else RowMatchKind.NONE


def row_names_symbol(declared: str, unit_name: str) -> bool:
    """Whether a declared table name names a symbol at all, of either kind.

    The bool-shaped view of `row_match_kind`, kept for callers that only ask "is this row
    about this symbol"; anything weighing the ANSWER must read the kind instead.
    """
    return row_match_kind(declared, unit_name) is not RowMatchKind.NONE


def load_rows(config: HomeConfig) -> list[DeclaredRow]:
    """Parse every declared-home table the workspace pointed at.

    A table file that cannot be read is skipped: the config names documentation, and
    missing documentation must not stop the answer.
    """
    rows: list[DeclaredRow] = []
    for spec in config.tables:
        try:
            text = (config.root / spec.file).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rows.extend(_rows_of(text, spec, config))
    return rows


def match_rows(rows: list[DeclaredRow], description: str, limit: int = 3) -> list[RowMatch]:
    """Rank declared rows against a feature description by shared vocabulary.

    Deliberately a token overlap and not an embedding: the rows are already in the
    index, and a second scoring model here would be a second thing to keep honest.
    Words are weighted by how many rows use them, because a table of capabilities is
    full of "record", "user" and "page" - sharing those says nothing, and without the
    weighting the longest row in the table matches every description.
    """
    query = words(description)
    if not query or not rows:
        return []
    weights = _weights(rows)
    # Words no row could ever match are left out of the denominator: a description
    # that happens to use rare vocabulary must not be scored as a worse match.
    total = sum(weights[word] for word in query if word in weights)
    if not total:
        return []
    matches = []
    for row in rows:
        shared = query & words(row.capability)
        if len(shared) < MIN_SHARED_WORDS:
            continue
        score = sum(weights[word] for word in shared) / total
        if score < MIN_MATCH_SCORE:
            continue
        matches.append(RowMatch(row=row, score=score, shared=tuple(sorted(shared))))
    matches.sort(key=lambda match: (-match.score, match.row.line))
    return matches[:limit]


def _weights(rows: Sequence[DeclaredRow]) -> dict[str, float]:
    """Weight every word in the table by how few of its rows use it."""
    count = len(rows)
    frequency: dict[str, int] = {}
    for row in rows:
        for word in words(row.capability):
            frequency[word] = frequency.get(word, 0) + 1
    return {word: log(1 + count / (1 + seen)) for word, seen in frequency.items()}


def _rows_of(text: str, spec: TableSpec, config: HomeConfig) -> list[DeclaredRow]:
    """Parse every table in one file that carries the declared column headers."""
    rows: list[DeclaredRow] = []
    columns: dict[str, int] | None = None
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            columns = None
            continue
        cells = _cells(stripped)
        if columns is None:
            columns = _header(cells, spec)
            continue
        if all(_SEPARATOR_CELL.match(cell.replace(" ", "")) for cell in cells if cell):
            continue
        row = _row(cells, columns, spec, config, number)
        if row is not None:
            rows.append(row)
    return rows


def _cells(line: str) -> list[str]:
    """Split a markdown table line into its cells, honouring escaped pipes."""
    parts = _CELL_SPLIT.split(line.strip())
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [part.replace("\\|", "|").strip() for part in parts]


def _header(cells: list[str], spec: TableSpec) -> dict[str, int] | None:
    """Return the column indices of a header row that carries the declared columns."""
    lowered = [cell.lower() for cell in cells]
    wanted = {"capability": spec.capability, "home": spec.home}
    if spec.consumers:
        wanted["consumers"] = spec.consumers
    found = {}
    for key, header in wanted.items():
        if header.lower() not in lowered:
            return None
        found[key] = lowered.index(header.lower())
    return found


def _row(
    cells: list[str], columns: dict[str, int], spec: TableSpec, config: HomeConfig, line: int
) -> DeclaredRow | None:
    """Parse one data row, or return None when it has too few cells to trust."""
    if max(columns.values()) >= len(cells):
        return None
    capability = cells[columns["capability"]]
    home_cell = cells[columns["home"]]
    if not capability or not home_cell:
        return None
    home_modules, home_names = _home_names(home_cell, config)
    consumer_cell = cells[columns["consumers"]] if "consumers" in columns else ""
    return DeclaredRow(
        capability=capability,
        symbols=symbol_names(capability),
        home_modules=home_modules,
        home_names=home_names,
        consumer_modules=_mentioned_modules(consumer_cell, config),
        file=spec.file,
        line=line,
        raw_home=home_cell,
    )


def _home_names(cell: str, config: HomeConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a home cell's backticked names into declared modules and everything else.

    Only backticked names count as a home: the prose around them names packages,
    classes and roles, and reading those as modules is how a table starts lying.
    """
    modules: list[str] = []
    others: list[str] = []
    for name in _BACKTICKED.findall(cell):
        module = config.known_module(name)
        if module is not None:
            if module not in modules:
                modules.append(module)
        elif name not in others:
            others.append(name)
    return tuple(modules), tuple(others)


def _mentioned_modules(cell: str, config: HomeConfig) -> tuple[str, ...]:
    """Return the declared modules a consumer cell names, backticked or in prose."""
    found: list[str] = []
    lowered = cell.lower()
    for module in config.modules:
        if module in found:
            continue
        if re.search(rf"(?<![\w-]){re.escape(module.lower())}(?![\w-])", lowered):
            found.append(module)
    return tuple(found)


__all__ = [
    "MIN_MATCH_SCORE",
    "MIN_SHARED_WORDS",
    "DeclaredRow",
    "RowMatch",
    "RowMatchKind",
    "load_rows",
    "match_rows",
    "row_match_kind",
    "row_names_symbol",
    "symbol_names",
    "words",
]
