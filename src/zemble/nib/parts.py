"""The part vocabulary of nib's provider protocol, version 1, and the answer a tool hands nib mode.

AIDEV-NOTE: this module is the one home of the wire names (`nib`, `parts`, `text`, `path`,
`start`, `end`, `indexed`); per-tool adapters build parts only through `text_part` and
`location_part`, so a protocol change is one edit here. It imports nothing from the MCP SDK:
tool modules the daemon also imports (`zemble.graph.mcp`) use it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from zemble.utils import is_git_url

#: The protocol version nib checks; an answer of another version is shown as plain text.
NIB_VERSION = 1

Part = dict[str, Any]
#: Builds an answer's parts from the absolute repo root and the plain payload; None means text only.
PartsBuilder = Callable[[str, Any], list[Part] | None]


class NibMode:
    """Marks a server whose tools answer for nib; tools ask `nib_answer`, never this class."""

    #: Tools whose every answer is prose, shown as one text part without any per-call opt-in.
    text_only_tools: set[str]


def nib_text_only(server: object, tool: str) -> None:
    """Declare a tool's answers prose for nib: they never show code, so one text part carries each."""
    if isinstance(server, NibMode):
        server.text_only_tools.add(tool)


@dataclass(frozen=True)
class NibAnswer:
    """A tool's plain payload plus its nib parts; None parts mean one text part holding the plain text."""

    plain: Any
    parts: list[Part] | None


def nib_answer(server: object, repo: str | None, plain: Any, build: PartsBuilder | None = None) -> Any:
    """Return `plain` as is outside nib mode, else wrap it with the parts `build` makes of it.

    A git URL repo answers plain even in nib mode: its files live in a clone nib cannot read,
    so nib shows the text as prose. `build` gets the plain payload and returns None for one
    it has no layout for (an error), which is then text only.

    :param server: The server the tool is registered on.
    :param repo: The repo the tool answered for, as resolved; None for a tool without one.
    :param plain: What the tool returns outside nib mode.
    :param build: Builds the location parts, or None for a text-only tool.
    """
    if not isinstance(server, NibMode):
        return plain
    if repo is not None and is_git_url(repo):
        return plain
    if build is None or repo is None:
        return NibAnswer(plain, None)
    return NibAnswer(plain, build(os.path.abspath(repo), plain))


def structured(parts: list[Part]) -> dict[str, Any]:
    """Wrap parts as the `structuredContent` object nib reads."""
    return {"nib": NIB_VERSION, "parts": parts}


def text_part(text: str) -> Part:
    """A prose part; it must never carry file lines."""
    return {"text": text}


def location_part(root: str, file_path: str, start: int, end: int, indexed: str | None = None) -> Part:
    """Lines `start`..`end` of a repo-relative or absolute file, with the text zemble saw there."""
    part: Part = {"path": os.path.normpath(os.path.join(root, file_path)), "start": start, "end": end}
    if indexed is not None:
        part["indexed"] = indexed
    return part


def _lines(text: str) -> list[str]:
    """Split text into lines the way a file's rows are counted: on newlines, without a final empty one."""
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()
    return lines


def _file_lines(path: str) -> list[str] | None:
    """Return a file's lines, or None when it cannot be read."""
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as handle:
            return _lines(handle.read())
    except OSError:
        return None


def source_part(root: str, file_path: str, start: int, text: str) -> Part | None:
    """Locate indexed source text that begins on line `start`, or None for empty text.

    AIDEV-NOTE: a chunk is a byte range, so its first line can start after the line's
    indentation and its last can stop short of the line's end. When the chunk still sits
    exactly there in the file, `indexed` is widened to those whole lines, which is what nib
    compares; when it does not, the chunk's own text is sent, so nib reports the index stale.
    """
    lines = _lines(text)
    if not text or not lines:
        return None
    part = location_part(root, file_path, start, start + len(lines) - 1)
    window = (_file_lines(part["path"]) or [])[start - 1 : start - 1 + len(lines)]
    part["indexed"] = "\n".join(window if _sits_in(lines, window) else lines)
    return part


def _sits_in(lines: list[str], window: list[str]) -> bool:
    """Return True if `lines` is `window` with only the first line's head and the last line's tail cut."""
    if len(window) != len(lines):
        return False
    if len(lines) == 1:
        return lines[0] in window[0]
    return window[0].endswith(lines[0]) and window[-1].startswith(lines[-1]) and window[1:-1] == lines[1:-1]


__all__ = [
    "NIB_VERSION",
    "NibAnswer",
    "NibMode",
    "Part",
    "PartsBuilder",
    "location_part",
    "nib_answer",
    "source_part",
    "structured",
    "text_part",
]
