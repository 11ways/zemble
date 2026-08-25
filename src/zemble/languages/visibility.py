"""How far a declaration can be reached from, in every language zemble reads."""

from __future__ import annotations

from enum import Enum


class Visibility(str, Enum):
    """A declaration's reach; the value is both the wire spelling and the word a report prints.

    UNKNOWN is what a declaration no language rule can place gets, and every consumer must
    treat it as restricted.
    """

    PUBLIC = "public"
    PROTECTED = "protected"
    PACKAGE = "package-private"
    PRIVATE = "private"
    UNKNOWN = "unknown"

    @property
    def is_public(self) -> bool:
        """Whether this level alone allows a call from another module."""
        return self is Visibility.PUBLIC

    def phrase(self, subject: str) -> str:
        """How a report names one subject's level ("member is private", "member visibility unknown")."""
        if self is Visibility.UNKNOWN:
            return f"{subject} visibility unknown"
        return f"{subject} is {self.value}"

    def narrower(self, other: Visibility) -> Visibility:
        """The more restrictive of two levels, as folding a nested type through its parents needs.

        AIDEV-NOTE: UNKNOWN ranks below PRIVATE on purpose, so folding an unplaceable level
        through a public parent stays unknown instead of inheriting the parent's promise.
        """
        return self if _RANK[self] <= _RANK[other] else other


#: Restriction order, most restrictive first; a level without a rank raises rather than passing.
_RANK: dict[Visibility, int] = {
    Visibility.UNKNOWN: 0,
    Visibility.PRIVATE: 1,
    Visibility.PACKAGE: 2,
    Visibility.PROTECTED: 3,
    Visibility.PUBLIC: 4,
}

#: Modifier word -> the level it spells, in the order a modifier list is searched.
VISIBILITY_WORDS: tuple[tuple[str, Visibility], ...] = (
    ("private", Visibility.PRIVATE),
    ("fileprivate", Visibility.PRIVATE),
    ("protected", Visibility.PROTECTED),
    ("internal", Visibility.PACKAGE),
    ("public", Visibility.PUBLIC),
    ("pub", Visibility.PUBLIC),
    ("export", Visibility.PUBLIC),
    ("open", Visibility.PUBLIC),
)


__all__ = ["VISIBILITY_WORDS", "Visibility"]
