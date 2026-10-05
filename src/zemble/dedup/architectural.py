"""Source-backed architectural candidates, explicitly separate from literal clone classes."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from zemble.evidence.counterparts import counterparts
from zemble.graph.model import Symbol
from zemble.graph.provider import GraphProvider


def architectural_candidates(
    root: Path, files: Sequence[str], graph: GraphProvider, limit: int = 100, min_files: int = 1
) -> dict:
    """Report documented/delegating counterparts with no token-count or flow-shape veto."""
    candidates = []
    seen = set()
    unreadable = []
    for file in files:
        symbols = graph.symbols_in_file(file)
        if not symbols:
            continue
        text = _read_source(root / file, unreadable)
        if text is None:
            continue
        # Documentary migrations are explicit architectural evidence, not literal equality.
        if "deprecated" not in text.lower():
            continue
        for symbol in symbols:
            for counterpart in counterparts(graph, root, symbol):
                identities = tuple(sorted((symbol.id, counterpart.symbol.id)))
                if identities in seen:
                    continue
                seen.add(identities)
                candidates.append(
                    {
                        "key": hashlib.sha256(repr(identities).encode()).hexdigest()[:16],
                        "kind": "architectural",
                        "evidence_kind": counterpart.kind,
                        "reason": counterpart.reason,
                        "equivalence_proven": False,
                        "members": [_member(symbol), _member(counterpart.symbol)],
                    }
                )
    candidates = [
        candidate
        for candidate in candidates
        if len({member["file_path"] for member in candidate["members"]}) >= min_files
    ]
    return {
        "candidates": candidates[:limit],
        "total_candidates": len(candidates),
        "truncated": len(candidates) > limit,
        "unreadable_files": unreadable,
        "scope": "documented migration/delegation candidates; behavioral review required",
    }


def _read_source(path: Path, unreadable: list) -> str | None:
    try:
        if path.stat().st_size > 2_000_000:
            unreadable.append({"file": str(path), "reason": "over 2 MB source limit"})
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        unreadable.append({"file": str(path), "reason": str(error)})
        return None


def _member(symbol: Symbol) -> dict:
    return {
        "file_path": symbol.file_path,
        "start_line": symbol.start_line,
        "end_line": symbol.end_line,
        "name": symbol.qualified_name,
    }
