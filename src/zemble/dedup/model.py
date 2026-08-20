"""The shapes duplication detection speaks in: units, clone classes and reports."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from zemble.graph.model import is_test_path


class CloneKind(str, Enum):
    """The three duplication kinds, from strictest to loosest."""

    EXACT = "exact"
    RENAMED = "renamed"
    LOGIC = "logic"


class Lane(str, Enum):
    """Which source sets a clone class lives in, in the order the report prints them."""

    PRODUCTION = "production"
    MIXED = "mixed"
    TEST = "test"


#: Unit kinds that own a whole declaration body, as opposed to a statement window.
BODY_KINDS = ("method", "constructor", "initializer")

#: The normalized stream a class of each kind is keyed by. Logic classes have no stream of
#: their own, so they are keyed by the exact body hash of each member.
KEY_HASH_ATTRIBUTE: dict[CloneKind, str] = {
    CloneKind.EXACT: "exact_hash",
    CloneKind.RENAMED: "renamed_hash",
    CloneKind.LOGIC: "exact_hash",
}


@dataclass(frozen=True, slots=True)
class Unit:
    """One comparable piece of Java: a whole body, or a window of statements inside one."""

    file_path: str
    start_line: int
    end_line: int
    kind: str
    name: str
    token_count: int
    exact_hash: str
    renamed_hash: str
    skeleton: tuple[str, ...]
    calls: tuple[str, ...]
    literals: tuple[str, ...]
    text: str | None = None

    @property
    def location(self) -> str:
        """File path and line range, the way the report prints it."""
        if self.start_line == self.end_line:
            return f"{self.file_path}:{self.start_line}"
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def is_body(self) -> bool:
        """Whether this unit is a whole declaration body rather than a statement window."""
        return self.kind in BODY_KINDS

    @property
    def is_test(self) -> bool:
        """Whether this unit lives in a test source set, decided by the symbol graph's own rule."""
        return is_test_path(self.file_path)


@dataclass(frozen=True, slots=True)
class PairReason:
    """Why one pair of units inside a logic class was accepted."""

    left: str
    right: str
    reason: str

    def __str__(self) -> str:
        """Render the reason the way the text report prints it."""
        return f"{self.left} ~ {self.right}: {self.reason}"


@dataclass(frozen=True, slots=True)
class CloneClass:
    """A set of units judged to be copies of one another under one kind."""

    kind: CloneKind
    members: tuple[Unit, ...]
    tokens: int
    reasons: tuple[PairReason, ...] = ()

    @property
    def files(self) -> int:
        """How many distinct files the class spans."""
        return len({member.file_path for member in self.members})

    @property
    def score(self) -> int:
        """Rank of the class: tokens x copies x files, the weighting `zenit-dev duplication` uses."""
        return self.tokens * len(self.members) * self.files

    @property
    def lane(self) -> Lane:
        """Whether every member is production code, every member is test code, or both."""
        flags = {member.is_test for member in self.members}
        if flags == {False}:
            return Lane.PRODUCTION
        if flags == {True}:
            return Lane.TEST
        return Lane.MIXED

    @property
    def key(self) -> str:
        """A stable identity for this class: its kind plus a digest of its members' streams.

        Line numbers are deliberately absent, so editing around a clone keeps its key; adding
        or removing a copy changes it, which is what makes a stale ignore entry visible.
        """
        attribute = KEY_HASH_ATTRIBUTE[self.kind]
        parts = sorted({f"{member.file_path}\0{getattr(member, attribute)}" for member in self.members})
        digest = hashlib.sha256("\n".join(parts).encode()).hexdigest()
        return f"{self.kind.value}:{digest[:12]}"

    @property
    def wire_reasons(self) -> list[str]:
        """The reasons for the wire: one entry when every pair agreed, the pairs when they did not."""
        if not self.reasons:
            return []
        verdicts = {reason.reason for reason in self.reasons}
        if len(verdicts) == 1:
            return [self.reasons[0].reason]
        return [str(reason) for reason in self.reasons]

    def to_dict(self) -> dict[str, Any]:
        """Render the class for the wire."""
        return {
            "key": self.key,
            "kind": self.kind.value,
            "lane": self.lane.value,
            "score": self.score,
            "tokens": self.tokens,
            "copies": len(self.members),
            "files": self.files,
            "reasons": self.wire_reasons,
            "members": [
                {
                    "file_path": member.file_path,
                    "start_line": member.start_line,
                    "end_line": member.end_line,
                    "kind": member.kind,
                    "name": member.name,
                    "tokens": member.token_count,
                    "is_test": member.is_test,
                }
                for member in self.members
            ],
        }


@dataclass
class DupeReport:
    """Everything one duplication run found, plus what it cost."""

    root: str
    analyzed_files: int = 0
    units: int = 0
    body_units: int = 0
    elapsed_seconds: float = 0.0
    min_tokens: int = 0
    min_statements: int = 0
    classes: list[CloneClass] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Classes a justified `.zemble/dupes.ignore` entry took out of the report.
    suppressed: list[CloneClass] = field(default_factory=list)
    #: Ignore-file violations: entries without a justification, and entries matching nothing.
    ignore_problems: list[str] = field(default_factory=list)

    def of_kind(self, kind: CloneKind) -> list[CloneClass]:
        """Return this report's classes of one kind, already ranked."""
        return [clone for clone in self.classes if clone.kind is kind]

    def section(self, lane: Lane, kind: CloneKind) -> list[CloneClass]:
        """Return this report's classes of one lane and kind, already ranked."""
        return [clone for clone in self.classes if clone.kind is kind and clone.lane is lane]

    def to_dict(self) -> dict[str, Any]:
        """Render the whole report for the wire."""
        return {
            "root": self.root,
            "analyzed_files": self.analyzed_files,
            "units": self.units,
            "body_units": self.body_units,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "min_tokens": self.min_tokens,
            "min_statements": self.min_statements,
            "notes": list(self.notes),
            "suppressed": len(self.suppressed),
            "ignore_problems": list(self.ignore_problems),
            "classes": [clone.to_dict() for clone in self.classes],
        }
