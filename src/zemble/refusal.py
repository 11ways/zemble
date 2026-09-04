"""What every deliberate refusal is, and the one place the knob it names is declared."""

from __future__ import annotations


class Refused(RuntimeError):
    """A build was refused on purpose, by a ceiling that names itself.

    One base class so every surface - the CLI, the MCP tools and the daemon wire - can tell a
    refusal, which is the same answer in every process, from a failure, which may not be. The
    environment variable that raises the ceiling lives HERE rather than on each subclass, so a
    handler reading ``exc.knob`` off a refusal it has never heard of gets an answer instead of
    an ``AttributeError`` inside its own except block.
    """

    #: The environment variable a subclass names when a raiser does not name a narrower one.
    DEFAULT_KNOB: str = ""

    def __init__(self, message: str, knob: str | None = None) -> None:
        """Refuse, carrying the ceiling's own environment variable rather than a guess.

        :param message: The refusal text, which already names the ceiling in its own unit.
        :param knob: The environment variable that raises the ceiling this refusal hit;
            None takes the subclass's own default.
        """
        super().__init__(message)
        self.knob = knob or self.DEFAULT_KNOB


__all__ = ["Refused"]
