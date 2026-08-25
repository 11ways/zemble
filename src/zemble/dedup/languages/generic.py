"""Duplication profiles derived from the grammar specs, one per spec-driven language.

A derived profile answers every question the unit extractor asks by reading the same
:class:`~zemble.languages.spec.LanguageSpec` the symbol graph reads, so a member the graph
declares is a member duplication compares. The token-level vocabulary (control keywords,
literals, parameter kinds) is a shared superset narrowed to what each grammar actually has.
"""

from __future__ import annotations

from functools import partial

from semble_grammars import get_language
from tree_sitter import Node, Parser

from zemble.dedup.languages.base import Container, LanguageProfile, node_text
from zemble.graph.generic import language_parser
from zemble.languages.paths import resolve, resolve_all, text
from zemble.languages.read import (
    NAME_KINDS,
    Declared,
    body_of,
    call_at,
    clean_name,
    match,
    modifiers_of,
    name_of,
    type_name,
    visibility_of,
)
from zemble.languages.spec import LanguageSpec, Role
from zemble.languages.visibility import Visibility

#: Unit kinds a derived profile emits; the Java and Zig profiles keep their own.
FUNCTION_KIND = "function"
CONSTRUCTOR_KIND = "constructor"

#: Keyword tokens whose order is a unit's control-flow skeleton, in any grammar that has them.
CONTROL_KEYWORDS: frozenset[str] = frozenset(
    {
        "if",
        "elif",
        "else",
        "elsif",
        "unless",
        "for",
        "foreach",
        "while",
        "until",
        "do",
        "loop",
        "switch",
        "match",
        "case",
        "when",
        "default",
        "try",
        "catch",
        "except",
        "finally",
        "rescue",
        "ensure",
        "return",
        "throw",
        "raise",
        "break",
        "continue",
        "yield",
        "await",
        "defer",
        "guard",
        "select",
        "go",
        "with",
        "cond",
        "receive",
        "goto",
    }
)

#: Node kinds that are a literal; the walk never descends into one.
LITERAL_KINDS: frozenset[str] = frozenset(
    {
        "string",
        "string_literal",
        "template_string",
        "interpreted_string_literal",
        "raw_string_literal",
        "rune_literal",
        "char_literal",
        "character_literal",
        "integer",
        "integer_literal",
        "int_literal",
        "float",
        "float_literal",
        "number",
        "number_literal",
        "decimal_integer_literal",
        "hex_integer_literal",
        "imaginary_literal",
        "boolean",
        "boolean_literal",
        "true",
        "false",
        "null",
        "null_literal",
        "nil",
        "none",
        "None",
        "undefined",
        "regex",
        "regex_literal",
        "heredoc_body",
        "simple_symbol",
        "symbol_literal",
        "atom",
        "char",
        "string_lit",
        "num_lit",
        "str_lit",
        "kwd_lit",
        "nil_lit",
        "bool_lit",
        "literal",
    }
)

#: Node kinds whose `name` field is an identifier the unit declares.
DECLARING_KINDS: frozenset[str] = frozenset(
    {
        "parameter",
        "formal_parameter",
        "required_parameter",
        "optional_parameter",
        "default_parameter",
        "typed_parameter",
        "typed_default_parameter",
        "parameter_declaration",
        "simple_parameter",
        "variadic_parameter",
        "keyword_parameter",
        "splat_parameter",
        "hash_splat_parameter",
        "block_parameter",
        "class_parameter",
        "variable_declarator",
        "let_declaration",
        "const_spec",
        "var_spec",
        "declaration_expression",
        "catch_declaration",
        "exception_variable",
        "for_numeric_clause",
        "lambda_parameter",
        "script_parameter",
        "formal",
        "param",
    }
)


def _named_kinds(language: str, wanted: frozenset[str], *, named: bool) -> frozenset[str]:
    """The subset of a kind superset that a grammar actually has."""
    grammar = get_language(language)
    return frozenset(kind for kind in wanted if grammar.id_for_node_kind(kind, named) is not None)


class _Reader:
    """The hooks of one derived profile, sharing one rule match per node."""

    def __init__(self, spec: LanguageSpec) -> None:
        """Bind the hooks to a spec."""
        self.spec = spec
        self._matched: dict[tuple[int, int], Declared | None] = {}

    def _declared(self, node: Node, source: bytes) -> Declared | None:
        """Match a node once per (tree, node), remembering the answer for the other hooks."""
        key = (id(source), node.id)
        if key not in self._matched:
            self._matched[key] = match(self.spec, node, source)
        return self._matched[key]

    def classify(self, node: Node, source: bytes) -> str | None:
        """The unit kind of a callable declaration, or None for anything else."""
        declared = self._declared(node, source)
        if declared is None or declared.rule.role is not Role.CALLABLE:
            return None
        _, name = name_of(self.spec, declared, source)
        if declared.rule.is_constructor or name in self.spec.constructor_names:
            return CONSTRUCTOR_KIND
        return FUNCTION_KIND

    def member_body(self, node: Node, source: bytes) -> Node | None:
        """The body of a callable, which may be the declaration itself (a lisp form)."""
        declared = self._declared(node, source)
        return body_of(declared) if declared is not None else None

    def member_name(self, node: Node, source: bytes) -> str:
        """The name segment a callable contributes, its written owner included."""
        declared = self._declared(node, source)
        if declared is None:
            return "<anonymous>"
        owner, name = name_of(self.spec, declared, source)
        name = name or "<anonymous>"
        return f"{owner}.{name}" if owner else name

    def container(self, node: Node, source: bytes) -> Container | None:
        """A type, namespace or extension block opens a container named after itself."""
        declared = self._declared(node, source)
        if declared is None or declared.rule.role not in (Role.TYPE, Role.MODULE, Role.EXTENSION):
            return None
        body = body_of(declared)
        if body is None:
            return None
        if declared.rule.role is Role.EXTENSION:
            name = type_name(text(resolve(declared.node, declared.rule.owner or "name"), source)) or "<anonymous>"
        else:
            owner, own = name_of(self.spec, declared, source)
            name = f"{owner}.{own}" if owner and own else (own or "<anonymous>")
        modifiers = modifiers_of(self.spec, declared, source)
        return Container(body=body, name=name, visibility=visibility_of(self.spec, name, modifiers, in_type=False))

    def declared_names_extra(self, node: Node, source: bytes) -> list[str]:
        """Identifiers the spec's binding paths reach from this node."""
        names: list[str] = []
        for kind, path in self.spec.binding_paths:
            if node.type != kind:
                continue
            for found in resolve_all(node, path):
                if found.type in NAME_KINDS:
                    names.append(clean_name(node_text(source, found)))
        return names

    def descend(self, node: Node, source: bytes) -> bool:
        """Look through every node that is not itself a declaration: grammars nest members freely."""
        return call_at(self.spec, node, source) is None

    def call_names(self, node: Node, source: bytes) -> list[str]:
        """The name one call node calls."""
        call = call_at(self.spec, node, source)
        return [call.name] if call is not None else []

    def modifiers(self, node: Node, source: bytes) -> tuple[str, ...]:
        """The declaration's modifier words."""
        declared = self._declared(node, source)
        return tuple(modifiers_of(self.spec, declared, source)) if declared is not None else ()

    def visibility(self, node: Node, source: bytes) -> Visibility:
        """How far the declaration reaches, from its modifiers and the language's conventions."""
        declared = self._declared(node, source)
        name = self.member_name(node, source) if declared is not None else ""
        modifiers = list(self.modifiers(node, source))
        return visibility_of(self.spec, name.rsplit(".", 1)[-1], modifiers, in_type=False)


def profile_from_spec(spec: LanguageSpec, extensions: tuple[str, ...]) -> LanguageProfile:
    """Derive the duplication profile of one spec-driven language."""
    reader = _Reader(spec)
    callable_kinds = {
        kind: (CONSTRUCTOR_KIND if rule.is_constructor else FUNCTION_KIND)
        for rule in spec.rules
        if rule.role is Role.CALLABLE
        for kind in rule.kinds
    }
    return LanguageProfile(
        name=spec.language,
        extensions=extensions,
        parser=partial(_parser, spec.language),
        member_kinds=callable_kinds,
        block_kinds=spec.block_kinds,
        control_keywords=_named_kinds(spec.language, CONTROL_KEYWORDS, named=False),
        literal_kinds=_named_kinds(spec.language, LITERAL_KINDS, named=True),
        declared_name_fields=_named_kinds(spec.language, DECLARING_KINDS, named=True),
        flatten_kinds=frozenset(),
        member_separators=spec.member_separators,
        member_body=reader.member_body,
        member_name=reader.member_name,
        container=reader.container,
        declared_names_extra=reader.declared_names_extra,
        call_names=reader.call_names,
        modifiers=reader.modifiers,
        visibility=reader.visibility,
        classify=reader.classify,
        descend=reader.descend,
        hook_node_kinds=spec.node_kinds(),
    )


def _parser(language: str) -> Parser | None:
    """The bundled parser of a language, or None when it is unavailable on this platform."""
    return language_parser(language)


__all__ = ["CONSTRUCTOR_KIND", "FUNCTION_KIND", "profile_from_spec"]
