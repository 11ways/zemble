"""The Zig language profile.

Zig has no class body: functions are members of the implicit file-struct, so the walk starts
at the root's own children and only descends where a `const Foo = struct { ... }` opens one.
`variable_declaration` doubles as the assignment statement node, which is why a declared name
is only taken from one that a `const`/`var` keyword leads.
"""

from __future__ import annotations

from functools import cache

from semble_grammars import LanguageNotFoundError, UnsupportedPlatformError, get_parser
from tree_sitter import Node, Parser

from zemble.dedup.languages.base import Container, LanguageProfile, node_text

_MEMBER_KINDS = {
    "function_declaration": "function",
    "test_declaration": "test",
    "comptime_declaration": "initializer",
}
#: Declarations that open a namespace of their own, always behind a `const NAME = ...`.
_CONTAINER_TYPES = frozenset({"struct_declaration", "enum_declaration", "union_declaration", "opaque_declaration"})
_DECLARATION_KEYWORDS = frozenset({"const", "var"})
_MODIFIER_KEYWORDS = frozenset({"pub", "export", "extern", "inline", "noinline", "threadlocal"})
_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "else",
        "while",
        "for",
        "switch",
        "return",
        "break",
        "continue",
        "defer",
        "errdefer",
        "try",
        "catch",
        "orelse",
        "unreachable",
        "and",
        "or",
        "comptime",
        "inline",
    }
)
_LITERAL_TYPES = frozenset({"integer", "float", "string", "character", "boolean", "null", "undefined"})


@cache
def zig_parser() -> Parser | None:
    """Return the bundled tree-sitter Zig parser, or None when it is unavailable."""
    try:
        return get_parser("zig")
    except (LanguageNotFoundError, UnsupportedPlatformError):
        return None


def _member_body(node: Node) -> Node | None:
    """A function keeps its `body` field; a test or comptime block owns a bare `block` child."""
    body = node.child_by_field_name("body")
    if body is not None:
        return body
    return next((child for child in node.named_children if child.type == "block"), None)


def _member_name(node: Node, source: bytes) -> str:
    """The name segment one member contributes: a function name, a test's title, or a marker."""
    if node.type == "comptime_declaration":
        return "<comptime>"
    if node.type == "test_declaration":
        title = next((child for child in node.named_children if child.type == "string"), None)
        return node_text(source, title).strip('"') if title is not None else "<test>"
    name = node.child_by_field_name("name")
    return node_text(source, name) if name is not None else "<anonymous>"


def _container(node: Node, source: bytes) -> Container | None:
    """`const Foo = struct { ... }` opens the namespace `Foo`; a bare struct value opens none."""
    if node.type != "variable_declaration":
        return None
    body = next((child for child in node.named_children if child.type in _CONTAINER_TYPES), None)
    if body is None:
        return None
    name = next((child for child in node.named_children if child.type == "identifier"), None)
    return Container(body=body, name=node_text(source, name) if name is not None else "<anonymous>")


def _declared_names_extra(node: Node, source: bytes) -> list[str]:
    """Locals and capture payloads; an assignment shares the declaration node and declares nothing."""
    kind = node.type
    if kind == "variable_declaration":
        leader = node.children[0] if node.child_count else None
        if leader is None or leader.type not in _DECLARATION_KEYWORDS:
            return []
        name = next((child for child in node.named_children if child.type == "identifier"), None)
        return [node_text(source, name)] if name is not None else []
    if kind == "payload":
        return [node_text(source, child) for child in node.named_children if child.type == "identifier"]
    return []


def _call_names(node: Node, source: bytes) -> list[str]:
    """The names one node calls: a plain call, a method on a value, or a `@builtin`."""
    kind = node.type
    if kind == "call_expression":
        called = node.child_by_field_name("function")
        if called is None:
            return []
        if called.type == "field_expression":
            member = called.child_by_field_name("member")
            return [node_text(source, member)] if member is not None else []
        if called.type == "identifier":
            return [node_text(source, called)]
        return []
    if kind == "builtin_function":
        name = next((child for child in node.named_children if child.type == "builtin_identifier"), None)
        return [node_text(source, name)] if name is not None else []
    return []


def _modifiers(node: Node, source: bytes) -> tuple[str, ...]:
    """The keywords in front of `fn`, `pub` chief among them."""
    return tuple(node_text(source, child) for child in node.children if child.type in _MODIFIER_KEYWORDS)


ZIG = LanguageProfile(
    name="zig",
    extensions=(".zig",),
    parser=zig_parser,
    member_kinds=_MEMBER_KINDS,
    block_kinds=frozenset({"block"}),
    control_keywords=_CONTROL_KEYWORDS,
    literal_kinds=_LITERAL_TYPES,
    declared_name_fields=frozenset({"parameter"}),
    flatten_kinds=frozenset(),
    member_separators=frozenset({"."}),
    member_body=_member_body,
    member_name=_member_name,
    container=_container,
    declared_names_extra=_declared_names_extra,
    call_names=_call_names,
    modifiers=_modifiers,
    hook_node_kinds=frozenset(
        {
            "builtin_function",
            "builtin_identifier",
            "call_expression",
            "const",
            "enum_declaration",
            "field_expression",
            "identifier",
            "opaque_declaration",
            "payload",
            "pub",
            "string",
            "struct_declaration",
            "union_declaration",
            "var",
            "variable_declaration",
        }
        | _MODIFIER_KEYWORDS
    ),
)
