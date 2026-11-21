"""A graph refresh that is coming or running while a question is answered, and how an answer says so.

A graph answers from its published version while it is refreshed beside its readers, so an answer
given meanwhile can lack what the refresh brings: a method a pull added reads as "no symbol". Every
graph surface carries the refresh in its payload (`refreshing`), and one that names nothing is not
final until the refresh lands: the daemon calls it busy, which its clients wait out
(`zemble.daemon.client.answer`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, assert_never

from zemble.daemon.protocol import GraphJobState

#: The payload key every graph surface carries a pending refresh under.
REFRESHING_KEY = "refreshing"


class RefreshPhase(str, Enum):
    """Where a graph refresh is when an answer is given; its job phases are `GraphJobState`'s own members."""

    #: The root's index pass runs first; the graph refresh follows it.
    ANNOUNCED = "announced"
    #: Waiting for the daemon's single graph worker.
    QUEUED = GraphJobState.QUEUED.value
    #: A process is writing the graph.
    BUILDING = GraphJobState.BUILDING.value

    @classmethod
    def of_job(cls, state: GraphJobState) -> RefreshPhase | None:
        """The phase a graph job in *state* puts its refresh in; None once the job no longer holds a refresh."""
        match state:
            case GraphJobState.QUEUED:
                return cls.QUEUED
            case GraphJobState.BUILDING:
                return cls.BUILDING
            case GraphJobState.READY | GraphJobState.FAILED | GraphJobState.CANCELLED:
                return None
            case _:
                assert_never(state)


@dataclass(frozen=True)
class PendingRefresh:
    """A graph refresh that has not landed yet."""

    root: str
    phase: RefreshPhase
    #: When the phase began (epoch seconds), when known.
    since: float | None = None
    #: How many changed paths the refresh brings in; None when it checks the whole tree or nobody said.
    changed_paths: int | None = None
    #: Whether the daemon owns it, so a caller can wait for it; one seen only as a held writer lock cannot be.
    awaitable: bool = True

    def describe(self) -> str:
        """Word the refresh: its root, its phase and age, and what it brings in."""
        details = [self.phase.value]
        if self.since is not None:
            details[0] += f" for {max(0, round(time.time() - self.since))} s"
        if self.changed_paths is not None:
            details.append(f"{self.changed_paths} changed path(s)")
        return f"the symbol graph of {self.root} is being refreshed ({', '.join(details)})"

    def note(self) -> str:
        """Say what an answer given now is worth."""
        return f"Not final: {self.describe()}; this answer reads the graph as it was before that refresh."

    def busy_reason(self) -> str:
        """Say why an answer naming nothing waits for the refresh instead of being given."""
        return f"{self.describe()}; an answer that names nothing waits for it"

    def to_json(self) -> dict[str, Any]:
        """Render the refresh for a payload, its note included."""
        return {
            "root": self.root,
            "phase": self.phase.value,
            "since": self.since,
            "changed_paths": self.changed_paths,
            "awaitable": self.awaitable,
            "note": self.note(),
        }

    @classmethod
    def from_json(cls, data: Any) -> PendingRefresh | None:
        """Read a refresh back from a payload; None when the payload carries none."""
        if not isinstance(data, dict):
            return None
        return cls(
            root=str(data["root"]),
            phase=RefreshPhase(data["phase"]),
            since=data.get("since"),
            changed_paths=data.get("changed_paths"),
            awaitable=bool(data.get("awaitable", True)),
        )


def stamp(payload: Any, pending: PendingRefresh | None) -> Any:
    """Carry a pending refresh in a dict payload, and its note in the prose fields a surface prints as they are."""
    if pending is None or not isinstance(payload, dict):
        return payload
    stamped = {**payload, REFRESHING_KEY: pending.to_json()}
    for prose in ("text", "markdown"):
        if isinstance(stamped.get(prose), str):
            stamped[prose] = f"{stamped[prose]}\n\n({pending.note()})"
    return stamped


def names_nothing(payload: Any) -> bool:
    """Whether a graph payload found nothing: an unknown name, or a query with no result at all."""
    if not isinstance(payload, dict):
        return False
    if "error" in payload:
        return not payload.get("candidates")
    return payload.get("total") == 0


def refresh_note(payload: Any) -> str:
    """The note a payload's pending refresh carries, or empty."""
    refreshing = payload.get(REFRESHING_KEY) if isinstance(payload, dict) else None
    return str(refreshing.get("note", "")) if isinstance(refreshing, dict) else ""


__all__ = ["REFRESHING_KEY", "PendingRefresh", "RefreshPhase", "names_nothing", "refresh_note", "stamp"]
