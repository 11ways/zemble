from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from zemble.types import SearchResult

if TYPE_CHECKING:  # pragma: no cover
    from zemble.index import ZembleIndex

_GIT_URL_SCHEMES = ("https://", "http://", "ssh://", "git://", "git+ssh://", "file://")
_SCP_GIT_URL_RE = re.compile(r"^[\w.-]+@[\w.-]+:(?!/)")
DEFAULT_MODEL_NAME = "minishlab/potion-code-16M-v2"  # default Model2Vec model; specs live in zemble.embedding.registry


def is_git_url(path: str) -> bool:
    """Return True if path looks like a remote git URL rather than a local path."""
    return path.startswith(_GIT_URL_SCHEMES) or _SCP_GIT_URL_RE.match(path) is not None


def _segments(path: str) -> list[str]:
    """Split a path into its posix segments, ignoring empty ones."""
    return [segment for segment in path.replace("\\", "/").split("/") if segment and segment != "."]


def nearest_indexed_path(candidates: Iterable[str], file_path: str) -> str | None:
    """Return the indexed path that shares the longest trailing run of segments with *file_path*.

    A caller who copied a path from a tool that spoke relative to a different root has the
    file name and usually most of the directory chain right, only the head differs; the
    match therefore counts segments from the file name backwards and needs at least the
    file name itself.
    """
    wanted = _segments(file_path)
    if not wanted:
        return None
    best: tuple[int, str] | None = None
    for candidate in candidates:
        have = _segments(candidate)
        shared = 0
        while shared < len(wanted) and shared < len(have) and wanted[-1 - shared] == have[-1 - shared]:
            shared += 1
        if shared and (best is None or shared > best[0] or (shared == best[0] and candidate < best[1])):
            best = (shared, candidate)
    return best[1] if best is not None else None


def describe_unresolved_location(index: ZembleIndex, file_path: str, line: int) -> str:
    """Explain why ``file_path:line`` names no chunk, as a problem with the ARGUMENT, not the index.

    Names the nearest indexed path when the file is unknown, and the file's chunk spans when
    only the line misses, so the caller can correct the call instead of concluding the code
    is not indexed.
    """
    spans = [(chunk.start_line, chunk.end_line) for chunk in index.chunks_of(file_path)]
    if spans:
        shown = ", ".join(f"{start}-{end}" for start, end in spans[:12])
        more = f" and {len(spans) - 12} more" if len(spans) > 12 else ""
        return f"{file_path!r} is indexed, but no chunk covers line {line}; its chunks span {shown}{more}."
    hint = "Paths are relative to the repo you passed, as search and dupes print them."
    nearest = nearest_indexed_path(index.indexed_paths(), file_path)
    if nearest is None:
        return f"No indexed file matches {file_path!r}. {hint}"
    return f"No indexed file matches {file_path!r}. Did you mean {nearest!r}? {hint}"


def format_results(query: str, results: list[SearchResult], max_snippet_lines: int | None = None) -> dict[str, Any]:
    """Render results as a flat JSONable object.

    max_snippet_lines=None → full content per result.
    max_snippet_lines=0    → file path and line range only, no content.
    max_snippet_lines=N>0  → first N lines of content.
    """
    formatted = []
    for r in results:
        entry: dict[str, Any] = {
            "file_path": r.chunk.file_path,
            "start_line": r.chunk.start_line,
            "end_line": r.chunk.end_line,
            "score": r.score,
        }
        if max_snippet_lines is None:
            entry["content"] = r.chunk.content
        elif max_snippet_lines > 0:
            lines = r.chunk.content.splitlines()
            entry["content"] = "\n".join(lines[:max_snippet_lines])
        formatted.append(entry)
    return {"query": query, "results": formatted}
