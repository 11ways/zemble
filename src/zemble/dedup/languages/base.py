"""What duplication detection has to know about one language before it can compare its code.

A profile is pure syntax vocabulary plus a handful of hooks; every ranking, hashing and
reporting decision downstream is language-neutral and must stay that way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from tree_sitter import Node, Parser


def node_text(source: bytes, node: Node) -> str:
    """Return a node's source text."""
    return source[node.start_byte : node.end_byte].decode("utf-8", "replace")


class Visibility(str, Enum):
    """How far a declaration can be called from, in every language duplication compares.

    The value is both the wire spelling and the word a report prints; UNKNOWN is what a unit
    no profile can place gets, and every consumer must treat it as restricted.
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


@dataclass(frozen=True, slots=True)
class Container:
    """A declaration whose body holds further members."""

    body: Node
    #: The segment this container adds to the qualified name, or None to keep the enclosing one.
    name: str | None = None
    #: How far this container itself can be reached, before it is folded through its parents.
    visibility: Visibility = Visibility.PUBLIC


# AIDEV-NOTE: eq=False keeps the profile hashable by identity; `member_kinds` is a dict, so a
# generated __eq__/__hash__ pair would raise the moment a profile lands in a set.
@dataclass(frozen=True, eq=False)
class LanguageProfile:
    """One language's answer to every question the unit extractor asks about a tree."""

    name: str
    #: File extensions this profile owns, lowercase and dotted.
    extensions: tuple[str, ...]
    #: Returns the tree-sitter parser, or None when the grammar is missing on this platform.
    parser: Callable[[], Parser | None]
    #: Node kind -> unit kind, for every declaration that owns a body worth comparing.
    member_kinds: Mapping[str, str]
    #: Node kinds whose statement children form a window.
    block_kinds: frozenset[str]
    #: Leaf node kinds whose order is a unit's control-flow skeleton.
    control_keywords: frozenset[str]
    #: Node kinds that are a literal; the walk never descends into one.
    literal_kinds: frozenset[str]
    #: Node kinds whose `name` field is an identifier the unit DECLARES.
    declared_name_fields: frozenset[str]
    #: Member kinds that are a wrapper: their named children are the real members.
    flatten_kinds: frozenset[str]
    #: Leaf texts after which an identifier is a MEMBER name and is never renamed.
    member_separators: frozenset[str]
    #: The body of one member declaration, or None when it is abstract.
    member_body: Callable[[Node], Node | None]
    #: The name segment one member declaration contributes.
    member_name: Callable[[Node, bytes], str]
    #: The nested container a member opens, or None when it opens none.
    container: Callable[[Node, bytes], Container | None]
    #: Declared identifiers a `name` field cannot express (patterns, captures, lambda params).
    declared_names_extra: Callable[[Node, bytes], list[str]]
    #: The names one node calls.
    call_names: Callable[[Node, bytes], list[str]]
    #: The declaration modifiers, reported but deliberately kept out of every hash.
    modifiers: Callable[[Node, bytes], tuple[str, ...]]
    #: How far one member declaration can be called from, its declaring container's kind included.
    visibility: Callable[[Node, bytes], Visibility]
    #: Node kinds only the hooks above name, listed so the drift test can check them too.
    hook_node_kinds: frozenset[str] = field(default_factory=frozenset)

    @property
    def node_kinds(self) -> frozenset[str]:
        """Every grammar node kind this profile names, hooks included."""
        return (
            frozenset(self.member_kinds)
            | self.block_kinds
            | self.control_keywords
            | self.literal_kinds
            | self.declared_name_fields
            | self.flatten_kinds
            | self.hook_node_kinds
        )
