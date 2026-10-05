"""Source-backed counterpart evidence shared by related, home and architectural scans."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from zemble.graph.model import Resolution, Symbol, SymbolKind
from zemble.graph.provider import GraphProvider

_LINK = re.compile(r"\{@link\s+([#A-Za-z_$][\w.$#]*)")
_CODE = re.compile(r"(?:`|\{@code\s+)([A-Z][\w$]*(?:[.#][A-Za-z_$][\w$]*)+)")


@dataclass(frozen=True)
class Counterpart:
    """A discoverable counterpart with provenance, never an assertion of behavioral equivalence."""

    symbol: Symbol
    kind: str
    reason: str


def declaration(graph: GraphProvider, file: str, line: int) -> Symbol | None:
    """Select a callable first, then the innermost named type at the actual seed."""
    symbols = graph.symbols_at(file, line, line)
    callables = [s for s in symbols if s.kind in {SymbolKind.METHOD, SymbolKind.FUNCTION, SymbolKind.CONSTRUCTOR}]
    if callables:
        return callables[0]
    return next((s for s in symbols if s.kind is not SymbolKind.PACKAGE and "$anon" not in s.name), None)


def source_text(root: Path, symbol: Symbol) -> tuple[str, str]:
    """Read one bounded member and its adjacent documentation, not its import/header capsule."""
    file = root / symbol.file_path
    if not file.is_file() or file.stat().st_size > 2_000_000:
        return "", ""
    lines = file.read_text(encoding="utf-8").splitlines()
    first = max(0, symbol.start_line - 1)
    member = "\n".join(lines[first : symbol.end_line])
    before = "\n".join(lines[max(0, first - 24) : first])
    # Only the immediately adjacent doc/comment block contributes declaration hints.
    if "*/" in before:
        opening = before.rfind("/**")
        before = before[opening:] if opening >= 0 else before
    return member, before


def counterparts(graph: GraphProvider, root: Path, symbol: Symbol, *, limit: int = 12) -> list[Counterpart]:
    """Gather documented migrations and direct delegation without a clone-size cutoff."""
    member, docs = source_text(root, symbol)
    found: list[Counterpart] = []
    seen = {symbol.id}

    def add(candidate: Symbol, kind: str, reason: str) -> None:
        if candidate.id not in seen and "$anon" not in candidate.name and len(found) < limit:
            seen.add(candidate.id)
            found.append(Counterpart(candidate, kind, reason))

    if "deprecated" in (member + docs).lower():
        for written in dict.fromkeys((*_LINK.findall(docs + member), *_CODE.findall(docs + member))):
            name = _link_name(written, symbol)
            candidates = graph.definition(name)
            # Overloads of one declared owner are candidates; unrelated owners stay ambiguous.
            if candidates and len({candidate.qualified_name for candidate in candidates}) == 1:
                for candidate in candidates[:4]:
                    add(candidate, "documented_counterpart", f"adjacent deprecation documentation names {written}")
    if symbol.kind in {SymbolKind.METHOD, SymbolKind.FUNCTION} and member.count(";") <= 3:
        for hit in graph.callees(symbol.id):
            if hit.resolution in {Resolution.EXACT, Resolution.UNIQUE_NAME}:
                add(hit.symbol, "delegation", f"thin callable delegates to {hit.symbol.qualified_name}")
    return found


def _link_name(written: str, symbol: Symbol) -> str:
    name = written.replace("#", ".")
    return symbol.qualified_name.rsplit(".", 1)[0] + name if written.startswith("#") else name


def callable_symbols(symbols: Sequence[Symbol]) -> list[Symbol]:
    """Keep callable declarations, dropping packages, fields and synthetic class scaffolding."""
    return [s for s in symbols if s.kind in {SymbolKind.METHOD, SymbolKind.FUNCTION, SymbolKind.CONSTRUCTOR}]
