"""What duplication detection has to know about one language before it can compare its code.

A profile is pure syntax vocabulary plus a handful of hooks; every ranking, hashing and
reporting decision downstream is language-neutral and must stay that way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase

from tree_sitter import Node, Parser

from zemble.languages.visibility import Visibility


def node_text(source: bytes, node: Node) -> str:
    """Return a node's source text."""
    return source[node.start_byte : node.end_byte].decode("utf-8", "replace")


class SiteKind(str, Enum):
    """The unit kinds of the sub-body and vocabulary channels: places, never whole bodies or windows."""

    #: A call chain expression.
    CHAIN = "chain"
    #: A call chain that is the whole body of its method: a one-line wrapper around the idiom.
    WRAPPER = "wrapper"
    #: Two calls on one receiver with the same first argument (`get(K)` ... `set(K, v)`).
    PAIR = "pair"
    #: A constant declaration with a literal value.
    CONSTANT = "constant"
    #: A literal used in code where a constant could have been.
    LITERAL = "literal"
    #: A value set: an enum, or a group of prefixed constants.
    SET = "set"
    #: The string labels one switch dispatches on.
    SWITCH = "switch"
    #: A regular expression literal.
    REGEX = "regex"


#: Literal node-kind words -> the typed hole a holed stream writes, first match wins; anything else is `<lit>`.
_HOLE_WORDS: tuple[tuple[str, str], ...] = (
    ("char", "<chr>"),
    ("string", "<str>"),
    ("str", "<str>"),
    ("integer", "<num>"),
    ("float", "<num>"),
    ("number", "<num>"),
    ("numeric", "<num>"),
    ("decimal", "<num>"),
    ("true", "<bool>"),
    ("false", "<bool>"),
    ("bool", "<bool>"),
    ("null", "<null>"),
    ("nil", "<null>"),
    ("none", "<null>"),
)
#: The hole a qualified constant reference (`Egress.NONE`, `MAX_SIZE`) becomes.
CONSTANT_HOLE = "<const>"


def literal_hole(node_type: str) -> str:
    """The typed hole one literal node kind becomes in a holed stream, read off the kind's own name."""
    lowered = node_type.lower()
    return next((hole for word, hole in _HOLE_WORDS if word in lowered), "<lit>")


@dataclass(frozen=True, slots=True)
class VocabularyFact:
    """One place a language declares or uses a value: a constant, a literal, a value set or a regex."""

    kind: SiteKind
    #: The declaring member or constant, qualified by its types (`ServerModel.ADMISSION_BLOCKED`).
    name: str
    #: The value, or every value of a set, in declaration order.
    values: tuple[str, ...]
    start_line: int
    end_line: int
    visibility: Visibility = Visibility.UNKNOWN
    container_visibility: Visibility = Visibility.UNKNOWN
    #: What a switch dispatches on (`column.name()`); "" for every other fact.
    detail: str = ""
    #: For a value set: the qualified declaration behind each value (`Owner.TYPE_DISK`, `Kind.DISK`).
    declares: tuple[str, ...] = ()
    #: For an enum's value set: the constants its members pass, as written (`InstanceDeviceModel.TYPE_DISK`).
    constants: tuple[str, ...] = ()


class KeyArguments:
    """Where a call takes a copy or translation key rather than a value: `*Microcopy*.of:0`.

    Each pattern is `<receiver glob>.<method>:<argument index>`; the glob matches the receiver as written (an
    unqualified call has the receiver ""), so `*Microcopy*.withArg:0` also matches a chain that starts at one.
    A key is a name in a message catalog, never a value of the code's own vocabulary.
    """

    def __init__(self, patterns: Sequence[str] = ()) -> None:
        """Parse the patterns once.

        :raises ValueError: If a pattern is not `<receiver>.<method>:<index>`.
        """
        self.patterns = tuple(patterns)
        self._rules: dict[tuple[str, int], list[str]] = {}
        for pattern in self.patterns:
            head, separator, index = pattern.rpartition(":")
            receiver, dot, method = head.rpartition(".")
            if not separator or not index.isdigit() or not method or (dot and not receiver):
                raise ValueError(f"key argument {pattern!r} is not <receiver glob>.<method>:<index>")
            self._rules.setdefault((method, int(index)), []).append(receiver)

    def matches(self, receiver: str, method: str, index: int) -> bool:
        """Whether argument `index` of `receiver.method(...)` is a key."""
        return any(fnmatchcase(receiver, glob) for glob in self._rules.get((method, index), ()))


@dataclass(frozen=True, eq=False)
class StandardLibrary:
    """What a language ships with: the calls that are the language itself, and the bodies one of them replaces.

    The holed and idiom channels read `types` and `members` to tell a language idiom (`String.valueOf(x)`,
    `Boolean.TRUE.equals(x)`) from a mechanism; the re-implementation channel reads `source` and `homes` to tell a
    helper that it is a standard call (`Objects.toString(value, "")`) instead of pointing it at a domain helper.
    """

    #: Types whose static members (and constants, `Boolean.TRUE`) are the language itself.
    types: frozenset[str]
    #: Method names of the core value types (`substring`, `equals`, `get`): instance calls of the language.
    members: frozenset[str]
    #: One compilation unit in this language holding one member per canonical body; a member is named
    #: `<home key>_<n>`, so several spellings of one home share a key.
    source: str = ""
    #: Home key -> the call that replaces every body spelled under it (`Objects.toString(value, fallback)`).
    homes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Signature:
    """What a member declares about itself beyond its body: the re-implementation channel's intent facts."""

    #: Declared parameter types, simple names without generics or annotations (`Field<?, ?>` is `Field`).
    parameters: tuple[str, ...]
    #: The declared result type the same way, "" when the member returns nothing (`void`, a constructor).
    returns: str
    #: Types that accept any value here: the language's top type and the member's own type variables.
    open_types: frozenset[str]
    #: Whether the member is declared static: callable without an instance of its type.
    static: bool
    #: The member's documentation comment as plain text, "" when it has none.
    summary: str
    #: Whether the body may read the state of an instance (`this`, an instance field or method); False only when
    #: the profile proved it does not, so an instance method that reads nothing of its own is a helper too.
    reads_instance: bool = True


@dataclass(frozen=True, eq=False)
class ShapeHooks:
    """What the holed, idiom, re-implementation and vocabulary channels read beyond the clone vocabulary.

    A profile without them still takes part in the holed and re-implementation channels through its
    literal kinds and call names; it has no idiom or vocabulary sites and no constant holes.
    """

    #: Node kinds that are one call with a receiver, a name and an argument list.
    call_kinds: frozenset[str]
    #: A call node's receiver (None for an unqualified call), its name node and its argument nodes.
    call_parts: Callable[[Node], tuple[Node | None, Node | None, list[Node]]]
    #: Whether an identifier spells a constant; a holed stream folds its whole qualified reference into one hole.
    is_constant_name: Callable[[str], bool]
    #: `Receiver.member` when a member's whole body hands its parameters, in order, to that one callable.
    forward_target: Callable[[Node, Node, bytes], str | None]
    #: Every vocabulary fact of one parsed file; a string at a key argument is never a value.
    vocabulary: Callable[[Node, bytes, KeyArguments], list[VocabularyFact]]
    #: Whether a member implements a declared contract (Java `@Override`): a role, never a utility to call.
    implements_contract: Callable[[Node, bytes], bool]
    #: A member's declared types, staticness and documentation, or None when the profile cannot read them.
    signature: Callable[[Node, bytes], Signature | None]
    #: Whether a member's whole body hands over to another constructor of its type (`this(...)`, `super(...)`).
    delegates: Callable[[Node, Node, bytes], bool]
    #: Whether a call node is made on a standard-library type or constant (`String.valueOf`, `Boolean.TRUE.equals`).
    standard_receiver: Callable[[Node, bytes], bool]
    #: What the language ships with.
    standard: StandardLibrary
    #: Node kinds only these hooks name, for the drift test.
    node_kinds: frozenset[str] = field(default_factory=frozenset)


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
    member_body: Callable[[Node, bytes], Node | None]
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
    #: Decides a node's unit kind when the node kind alone cannot (a `call` that is a `def`);
    #: None means `member_kinds` decides by node kind.
    classify: Callable[[Node, bytes], str | None] | None = None
    #: Whether a node that is neither member nor container is looked through for members
    #: (a Haskell `declarations` list, a SQL `statement`); None looks only through `flatten_kinds`.
    descend: Callable[[Node, bytes], bool] | None = None
    #: The sub-body and vocabulary hooks, or None when this language has no idiom or vocabulary sites.
    shapes: ShapeHooks | None = None

    def member_kind(self, node: Node, source: bytes) -> str | None:
        """The unit kind one node declares, or None when it declares no comparable body."""
        if self.classify is not None:
            return self.classify(node, source)
        return self.member_kinds.get(node.type)

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
            | (self.shapes.call_kinds | self.shapes.node_kinds if self.shapes is not None else frozenset())
        )
