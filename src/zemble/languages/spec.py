"""What zemble has to know about one grammar to read declarations, calls and references.

A spec is data: node kinds plus :mod:`zemble.languages.paths` expressions. The symbol graph
extractor and the duplication profiles both read it, so adding a language is adding one
spec to :mod:`zemble.languages.catalog` and nothing else. Java, Hawkeye templates and Zig's
duplication profile keep their hand-written readers because they are measured lanes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from zemble.graph.model import SymbolKind
from zemble.languages.visibility import Visibility


class Role(str, Enum):
    """What a matched declaration node contributes."""

    #: A named container of members: class, struct, interface, enum, trait, protocol, ...
    TYPE = "type"
    #: A named namespace with a body: `namespace`, `mod`, `module`, `defmodule`, ...
    MODULE = "module"
    #: A header naming the file's package, with no body of its own.
    PACKAGE = "package"
    #: A function, method or constructor.
    CALLABLE = "callable"
    #: A field, property, constant or top-level variable.
    FIELD = "field"
    #: An enum constant, variant or case.
    CONSTANT = "constant"
    #: A block that adds members to a type declared elsewhere: `impl`, `extension`.
    EXTENSION = "extension"
    #: An import, use, require or include.
    IMPORT = "import"
    #: A statement inside a type body naming a supertype (`include Comparable`, `use Trait;`).
    SUPERTYPE = "supertype"
    #: A node to ignore entirely, subtree included (a prototype, a type signature).
    SKIP = "skip"


@dataclass(frozen=True)
class Rule:
    """One declaration shape: which nodes it matches and where their parts live."""

    kinds: tuple[str, ...]
    role: Role
    #: The symbol kind a TYPE or MODULE rule declares; a CALLABLE decides by its owner.
    kind: SymbolKind | None = None
    #: Path to the declared name, or for IMPORT the imported module; None uses `default_name`.
    name: str | None = "name"
    #: Path fanning out to several declared names (a FIELD with declarators, an IMPORT's names).
    names: str | None = None
    #: Joins the `names` texts into one name (`resource.aws_instance.point`) instead of fanning out.
    name_join: str | None = None
    #: The name a nameless declaration gets (`Companion`, `module`).
    default_name: str | None = None
    #: Path to the body whose children are the members or the statements; None for a leaf.
    body: str | None = "body"
    #: The node kind the body must have, for bodies reached through a sibling step.
    body_kind: str | None = None
    #: Further paths whose children are members of a TYPE (a primary constructor's properties).
    members: tuple[str, ...] = ()
    #: Path to the parameter list.
    params: str | None = None
    #: How many leading children of the parameter list are not parameters (a lisp's own name).
    params_offset: int = 0
    #: Paths to nodes whose named children (or which themselves) name supertypes.
    supertypes: tuple[str, ...] = ()
    #: Path to the name of the type an EXTENSION or a receiver method attaches to.
    owner: str | None = None
    #: Path to the trait or protocol an EXTENSION implements.
    trait: str | None = None
    #: The first keyword token of the node -> the kind it really declares (Swift's one node).
    keyword_kinds: Mapping[str, SymbolKind] = field(default_factory=dict)
    #: The first keyword tokens the rule accepts, when the kind does not depend on it.
    keywords: frozenset[str] = frozenset()
    #: (path, allowed texts): the rule matches only when the path's text is one of these.
    when: tuple[str, frozenset[str]] | None = None
    #: (path, node kind): the rule matches only when the path resolves to a node of that kind.
    require: tuple[str, str] | None = None
    #: "type" matches only inside a type body, "top" only at module level, "any" anywhere.
    scope: str = "any"
    #: Whether every match is a constructor, regardless of its name.
    is_constructor: bool = False
    #: Path to the declared return type.
    return_type: str | None = None


@dataclass(frozen=True)
class CallRule:
    """One call shape: the node, its callee and its argument list."""

    kinds: tuple[str, ...]
    callee: str = "function"
    arguments: str | None = "arguments"
    #: How many leading children of the argument holder are not arguments (the callee itself).
    arguments_offset: int = 0
    #: Path to the receiver when the grammar keeps it beside the callee (Ruby's `call`).
    receiver: str | None = None
    #: Whether the call constructs an instance (`new Foo()`).
    is_new: bool = False
    when: tuple[str, frozenset[str]] | None = None
    require: tuple[str, str] | None = None


@dataclass(frozen=True)
class LanguageSpec:
    """Everything the generic readers need to know about one grammar."""

    language: str
    #: Languages that resolve into each other's symbols share a family (`jvm`, `c`, `js`).
    family: str
    rules: tuple[Rule, ...] = ()
    calls: tuple[CallRule, ...] = ()
    #: Node kind -> path to the declaration it wraps (decorators, exports, templates).
    wrappers: Mapping[str, str] = field(default_factory=dict)
    #: Callee node kind -> (path to the receiver, path to the member name).
    member_access: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    #: Receiver spellings that mean the enclosing instance.
    self_names: frozenset[str] = frozenset({"this", "self"})
    #: Parameter node kinds, or parameter names, that are the receiver and count for no arity.
    self_parameters: frozenset[str] = frozenset()
    #: Node kinds that are one parameter; empty means every named child of the list is one.
    parameter_kinds: frozenset[str] = frozenset()
    #: Callee node kinds beyond the plain names that still read as a name (`command_name`).
    callee_kinds: frozenset[str] = frozenset()
    #: Wrapper kinds a callee is looked through when they hold exactly one named child.
    transparent_kinds: frozenset[str] = frozenset()
    #: Callee texts that are syntax, never a call worth an edge (`def`, `let`, `if`).
    call_exclusions: frozenset[str] = frozenset()
    #: Call names that construct the receiver type (`Foo.new` in Ruby).
    constructor_call_names: frozenset[str] = frozenset()
    #: A separator a single symbol token may carry between namespace and name (`/` in Clojure).
    symbol_namespace_separator: str | None = None
    #: Keywords that make a declaration private (`defp`, `defn-`).
    private_words: frozenset[str] = frozenset()
    #: The level a member of a type gets without a visibility word, when it differs from the default.
    member_visibility: Visibility | None = None
    #: Whether a callable named like its enclosing type is that type's constructor.
    name_equal_to_type_is_constructor: bool = False
    #: Node kinds whose name descendants reference a type.
    type_ref_kinds: frozenset[str] = frozenset()
    #: Type names that are never workspace symbols.
    builtin_types: frozenset[str] = frozenset()
    #: Node kinds that are a decorator, annotation or attribute.
    annotation_kinds: frozenset[str] = frozenset()
    #: Named node kinds whose text is modifier words.
    modifier_kinds: frozenset[str] = frozenset()
    #: Bare tokens that are modifiers when they precede a declaration's name.
    modifier_words: frozenset[str] = frozenset()
    #: Node kinds whose named children are consecutive statements (duplication windows).
    block_kinds: frozenset[str] = frozenset()
    #: (node kind, path) pairs reaching identifiers a unit declares: parameters, locals, captures.
    binding_paths: tuple[tuple[str, str], ...] = ()
    #: Callable names that are constructors (`__init__`, `initialize`, `constructor`).
    constructor_names: frozenset[str] = frozenset()
    #: Whether a bare capitalised call is a constructor (`Point(1)` in Python, Kotlin, Swift).
    capitalized_call_is_new: bool = False
    #: Whether a capitalised name is exported (Go) or a leading underscore hides one (Python).
    capitalized_is_public: bool = False
    underscore_is_private: bool = False
    #: The level a declaration without a visibility word gets.
    default_visibility: Visibility = Visibility.PUBLIC
    #: File-name globs that mark a file as a test source, beside the path segments.
    test_file_patterns: tuple[str, ...] = ()
    #: How the language separates the segments of a written qualified name.
    package_separator: str = "."
    #: Leaf texts after which an identifier is a member name, never a renamed local.
    member_separators: frozenset[str] = frozenset({".", "->", "::"})

    def node_kinds(self) -> frozenset[str]:
        """Every grammar node kind the spec names, including those inside paths."""
        kinds: set[str] = set()
        for rule in self.rules:
            kinds.update(_rule_kinds(rule))
        for call in self.calls:
            kinds.update(_call_kinds(call))
        for kind, path in self.wrappers.items():
            kinds.add(kind)
            kinds.update(_path_kinds(path))
        for kind, (receiver, member) in self.member_access.items():
            kinds.add(kind)
            kinds.update(_path_kinds(receiver))
            kinds.update(_path_kinds(member))
        for kind, path in self.binding_paths:
            kinds.add(kind)
            kinds.update(_path_kinds(path))
        for group in (
            self.type_ref_kinds,
            self.annotation_kinds,
            self.modifier_kinds,
            self.block_kinds,
            self.parameter_kinds,
            self.callee_kinds,
            self.transparent_kinds,
            self.self_parameters & _NODE_LIKE,
        ):
            kinds.update(group)
        return frozenset(kinds)


def _rule_kinds(rule: Rule) -> set[str]:
    """The node kinds one declaration rule names."""
    kinds = set(rule.kinds)
    for path in (rule.name, rule.names, rule.body, rule.params, rule.owner, rule.trait, rule.return_type):
        kinds.update(_path_kinds(path))
    for path in rule.supertypes + rule.members:
        kinds.update(_path_kinds(path))
    if rule.body_kind is not None:
        kinds.add(rule.body_kind)
    kinds.update(_condition_kinds(rule.when, rule.require))
    return kinds


def _call_kinds(call: CallRule) -> set[str]:
    """The node kinds one call rule names."""
    kinds = set(call.kinds)
    for path in (call.callee, call.arguments, call.receiver):
        kinds.update(_path_kinds(path))
    kinds.update(_condition_kinds(call.when, call.require))
    return kinds


def _condition_kinds(when: tuple[str, frozenset[str]] | None, require: tuple[str, str] | None) -> set[str]:
    """The node kinds a `when` and a `require` condition name."""
    kinds: set[str] = set()
    if when is not None:
        kinds.update(_path_kinds(when[0]))
    if require is not None:
        kinds.update(_path_kinds(require[0]))
        kinds.add(require[1])
    return kinds


#: `self_parameters` mixes node kinds with parameter names; only entries spelled like a node
#: kind (`self_parameter`) are checked against the grammar.
_NODE_LIKE: frozenset[str] = frozenset({"self_parameter", "receiver_parameter", "this_parameter"})


def _path_kinds(path: str | None) -> set[str]:
    """The node kinds a path names through `@kind`, `**kind` and `field~kind` steps."""
    kinds: set[str] = set()
    if not path:
        return kinds
    for alternative in path.split("|"):
        for step in alternative.split("/"):
            if step.startswith("@"):
                kinds.add(step[1:].rstrip("*"))
            elif step.startswith("**"):
                kinds.add(step[2:])
            elif "~" in step:
                kinds.add(step.partition("~")[2])
    kinds.discard("*")
    return kinds


__all__ = ["CallRule", "LanguageSpec", "Role", "Rule"]
