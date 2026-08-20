"""The shapes duplication detection speaks in: units, clone classes and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CloneKind(str, Enum):
    """The three duplication kinds, from strictest to loosest."""

    EXACT = "exact"
    RENAMED = "renamed"
    LOGIC = "logic"


#: Unit kinds that own a whole declaration body, as opposed to a statement window.
BODY_KINDS = ("method", "constructor", "initializer")


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


@dataclass(frozen=True, slots=True)
class CloneClass:
    """A set of units judged to be copies of one another under one kind."""

    kind: CloneKind
    members: tuple[Unit, ...]
    tokens: int
    reasons: tuple[str, ...] = ()

    @property
    def files(self) -> int:
        """How many distinct files the class spans."""
        return len({member.file_path for member in self.members})

    @property
    def score(self) -> int:
        """Rank of the class: tokens x copies x files, the weighting `zenit-dev duplication` uses."""
        return self.tokens * len(self.members) * self.files

    def to_dict(self) -> dict[str, Any]:
        """Render the class for the wire."""
        return {
            "kind": self.kind.value,
            "score": self.score,
            "tokens": self.tokens,
            "copies": len(self.members),
            "files": self.files,
            "reasons": list(self.reasons),
            "members": [
                {
                    "file_path": member.file_path,
                    "start_line": member.start_line,
                    "end_line": member.end_line,
                    "kind": member.kind,
                    "name": member.name,
                    "tokens": member.token_count,
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

    def of_kind(self, kind: CloneKind) -> list[CloneClass]:
        """Return this report's classes of one kind, already ranked."""
        return [clone for clone in self.classes if clone.kind is kind]

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
            "classes": [clone.to_dict() for clone in self.classes],
        }
