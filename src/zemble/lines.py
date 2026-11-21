"""Line numbers of character offsets in a text, found by bisection instead of counting a prefix."""

from __future__ import annotations

from bisect import bisect_right


class LineIndex:
    """The 1-based line every character offset of one text falls on.

    Only a line feed ends a line, so a CRLF file numbers like an LF one and a lone carriage return
    ends nothing: :meth:`line_of` equals the count of line feeds before the offset plus one, for every
    offset from 0 up, past the end too.
    """

    def __init__(self, text: str) -> None:
        """Record the offset every line starts at."""
        starts = [0]
        found = text.find("\n")
        while found >= 0:
            starts.append(found + 1)
            found = text.find("\n", found + 1)
        self.starts = starts

    @property
    def line_count(self) -> int:
        """How many lines the text has; a trailing newline starts one more, empty."""
        return len(self.starts)

    def line_of(self, offset: int) -> int:
        """Return the 1-based line a non-negative character offset falls on."""
        return bisect_right(self.starts, offset)


__all__ = ["LineIndex"]
