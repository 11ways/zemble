"""The Java language profile.

The parser accessor is shared with the symbol graph (:func:`zemble.graph.java.java_parser`),
but nothing else is: the graph answers "what does this file declare", this module answers
"what does this file's code look like once names stop mattering".
"""

from __future__ import annotations

from tree_sitter import Node

from zemble.dedup.languages.base import Container, LanguageProfile, Visibility, node_text
from zemble.graph.java import java_parser

_CALLABLE_KINDS = {
    "method_declaration": "method",
    "constructor_declaration": "constructor",
    "compact_constructor_declaration": "constructor",
    "annotation_type_element_declaration": "method",
}
_INITIALIZER_TYPES = frozenset({"static_initializer", "block"})
_MEMBER_KINDS = {**_CALLABLE_KINDS, **dict.fromkeys(_INITIALIZER_TYPES, "initializer")}
_TYPE_DECLARATIONS = frozenset({"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"})
_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "try",
        "catch",
        "finally",
        "return",
        "throw",
        "break",
        "continue",
    }
)
_LITERAL_TYPES = frozenset(
    {
        "decimal_integer_literal",
        "hex_integer_literal",
        "octal_integer_literal",
        "binary_integer_literal",
        "decimal_floating_point_literal",
        "hex_floating_point_literal",
        "string_literal",
        "character_literal",
        "true",
        "false",
        "null_literal",
    }
)
#: Bodies whose every member and nested type is implicitly public (JLS 9.3, 9.4, 9.5, 9.6).
_IMPLICITLY_PUBLIC_BODIES = frozenset({"interface_body", "annotation_type_body"})
#: Visibility keyword -> level, in the order a modifier list is searched.
_VISIBILITY_KEYWORDS: tuple[tuple[str, Visibility], ...] = (
    ("private", Visibility.PRIVATE),
    ("protected", Visibility.PROTECTED),
    ("public", Visibility.PUBLIC),
)
_NAME_FIELD_DECLARATIONS = frozenset(
    {
        "formal_parameter",
        "spread_parameter",
        "catch_formal_parameter",
        "enhanced_for_statement",
        "resource",
        "variable_declarator",
        "type_pattern",
    }
)


def _member_body(node: Node) -> Node | None:
    """An initializer is its own body; everything else keeps the `body` field."""
    if node.type in _INITIALIZER_TYPES:
        return node
    return node.child_by_field_name("body")


def _member_name(node: Node, source: bytes) -> str:
    """The name segment one member contributes to its qualified name."""
    if node.type in _INITIALIZER_TYPES:
        return "<initializer>"
    name = node.child_by_field_name("name")
    return node_text(source, name) if name is not None else "<anonymous>"


def _container(node: Node, source: bytes) -> Container | None:
    """Type declarations open a named container; an enum constant's body keeps the enum's name."""
    if node.type in _TYPE_DECLARATIONS:
        body = node.child_by_field_name("body")
        if body is None:
            return None
        name = node.child_by_field_name("name")
        return Container(
            body=body,
            name=node_text(source, name) if name is not None else "<anonymous>",
            visibility=_visibility(node, source),
        )
    if node.type == "enum_constant":
        inner = next((child for child in node.named_children if child.type == "class_body"), None)
        return Container(body=inner, name=None, visibility=Visibility.PUBLIC) if inner is not None else None
    return None


def _declared_visibility(node: Node, source: bytes) -> Visibility:
    """The level one declaration's own keywords spell, defaulting to package-private."""
    modifiers = set(_modifiers(node, source))
    for keyword, level in _VISIBILITY_KEYWORDS:
        if keyword in modifiers:
            return level
    return Visibility.PACKAGE


def _visibility(node: Node, source: bytes) -> Visibility:
    """How far one member can be called from, the declaring body's kind included.

    AIDEV-NOTE: an interface or annotation member carries no `public` keyword and is public
    anyway; only the Java 9 `private` interface method is not, which is why the explicit
    keyword is read first and the implicit rule only fills in for a bare declaration.
    """
    declared = _declared_visibility(node, source)
    parent = node.parent
    if parent is not None and parent.type in _IMPLICITLY_PUBLIC_BODIES and declared is not Visibility.PRIVATE:
        return Visibility.PUBLIC
    return declared


def _declared_names_extra(node: Node, source: bytes) -> list[str]:
    """Pattern bindings and lambda parameters, neither of which carries a `name` field."""
    if node.type == "type_pattern":
        return [node_text(source, child) for child in node.named_children if child.type == "identifier"]
    if node.type == "lambda_expression":
        return _lambda_parameters(node, source)
    return []


def _lambda_parameters(node: Node, source: bytes) -> list[str]:
    """Collect the parameter names of a lambda, in all three spellings Java allows."""
    parameters = node.child_by_field_name("parameters")
    if parameters is None:
        return []
    if parameters.type == "identifier":
        return [node_text(source, parameters)]
    if parameters.type == "inferred_parameters":
        return [node_text(source, child) for child in parameters.named_children if child.type == "identifier"]
    return []


def _call_names(node: Node, source: bytes) -> list[str]:
    """The names one node calls: a method, a constructor, or a `this(...)`/`super(...)` chain."""
    kind = node.type
    if kind == "method_invocation":
        name = node.child_by_field_name("name")
        return [node_text(source, name)] if name is not None else []
    if kind == "object_creation_expression":
        created = node.child_by_field_name("type")
        return [node_text(source, created).rsplit(".", 1)[-1]] if created is not None else []
    if kind == "explicit_constructor_invocation":
        constructor = node.child_by_field_name("constructor")
        return [node_text(source, constructor)] if constructor is not None else []
    return []


def _modifiers(node: Node, source: bytes) -> tuple[str, ...]:
    """The keyword modifiers of a declaration; annotations are named nodes and are skipped."""
    modifiers = next((child for child in node.children if child.type == "modifiers"), None)
    if modifiers is None:
        return ()
    return tuple(node_text(source, child) for child in modifiers.children if not child.is_named)


JAVA = LanguageProfile(
    name="java",
    extensions=(".java",),
    parser=java_parser,
    member_kinds=_MEMBER_KINDS,
    block_kinds=frozenset({"block", "constructor_body", "switch_block_statement_group"}),
    control_keywords=_CONTROL_KEYWORDS,
    literal_kinds=_LITERAL_TYPES,
    declared_name_fields=_NAME_FIELD_DECLARATIONS | _TYPE_DECLARATIONS,
    flatten_kinds=frozenset({"enum_body_declarations"}),
    member_separators=frozenset({"."}),
    member_body=_member_body,
    member_name=_member_name,
    container=_container,
    declared_names_extra=_declared_names_extra,
    call_names=_call_names,
    modifiers=_modifiers,
    visibility=_visibility,
    hook_node_kinds=frozenset(
        {
            "annotation_type_body",
            "interface_body",
            "class_body",
            "enum_constant",
            "explicit_constructor_invocation",
            "identifier",
            "inferred_parameters",
            "lambda_expression",
            "method_invocation",
            "modifiers",
            "object_creation_expression",
        }
    ),
)
