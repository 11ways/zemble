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
from math import log
from typing import Any

from zemble.home.config import HomeConfig, TableSpec

#: A cell separator that is not escaped as `\|` inside a code span.
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_BACKTICKED = re.compile(r"`([^`]+)`")
_SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")
_WORD = re.compile(r"[a-z0-9]+")

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


@dataclass(frozen=True)
class DeclaredRow:
    """One row of a declared-home table, as far as it could be understood."""

    capability: str
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

    def to_dict(self) -> dict[str, Any]:
        """Render the match as JSON-ready data."""
        return {"row": self.row.to_dict(), "score": round(self.score, 3), "shared": list(self.shared)}


def words(text: str) -> set[str]:
    """Return the meaningful lowercase words of a text."""
    return {word for word in _WORD.findall(text.lower()) if len(word) > 2 and word not in _STOPWORDS}


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


__all__ = ["MIN_MATCH_SCORE", "MIN_SHARED_WORDS", "DeclaredRow", "RowMatch", "load_rows", "match_rows", "words"]
