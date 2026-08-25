"""Reading declarations and calls off a parse tree the way a :class:`LanguageSpec` says to.

Everything here is language-neutral: it evaluates spec paths and applies the handful of
conventions that hold across grammars (a name is a leaf, a keyword is an unnamed token, a
receiver text is split on a separator). The graph extractor and the duplication profiles
both build on it so the two never disagree about what a file declares.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tree_sitter import Node

from zemble.languages.paths import resolve, resolve_all, text
from zemble.languages.spec import CallRule, LanguageSpec, Rule
from zemble.languages.visibility import VISIBILITY_WORDS, Visibility

#: Node kinds that are a plain name, in every bundled grammar.
NAME_KINDS: frozenset[str] = frozenset(
    {
        "identifier",
        "type_identifier",
        "property_identifier",
        "private_property_identifier",
        "shorthand_property_identifier",
        "field_identifier",
        "simple_identifier",
        "statement_identifier",
        "namespace_identifier",
        "package_identifier",
        "name",
        "word",
        "constant",
        "sym_name",
        "symbol",
        "atom",
        "variable",
        "variable_name",
        "ident",
        "id",
        "function_name",
        "command_name",
        "class_name",
        "module_name",
        "value_name",
        "method_name",
        "constructor_name",
        "constructor",
        "type_name",
        "type_constructor",
        "label",
        "unquoted_argument",
        "tag_name",
        "attribute_name",
        "attrpath",
        "bound_identifier",
        "enum_value",
        "builtin_identifier",
        "function",
    }
)

#: Node kinds that are a name written with a path in it (`a.b.C`, `App\\Core`, `std::fmt`).
QUALIFIED_KINDS: frozenset[str] = frozenset(
    {
        "dotted_name",
        "qualified_name",
        "qualified_identifier",
        "scoped_identifier",
        "scoped_type_identifier",
        "namespace_name",
        "alias",
        "user_type",
        "generic_type",
        "generic_name",
        "named_type",
        "type",
        "module_path",
        "value_path",
        "package_name",
        "object_reference",
        "user_defined_type",
        "predefined_type",
        "primitive_type",
        "builtin_type",
        "template_type",
        "nullable_type",
        "optional_type",
        "array_type",
        "pointer_type",
        "reference_type",
        "attribute",
        "member_expression",
        "field_expression",
        "selector_expression",
        "navigation_expression",
        "member_access_expression",
        "scoped_type_identifier",
    }
)

#: Node kinds that are a string literal, whose quotes are not part of the name they carry.
STRING_KINDS: frozenset[str] = frozenset(
    {
        "string",
        "string_literal",
        "interpreted_string_literal",
        "raw_string_literal",
        "string_lit",
        "string_value",
        "string_fragment",
        "string_content",
        "quoted_attribute_value",
        "system_lib_string",
        "uri",
    }
)

#: Supertype holders keep nodes of these kinds out of the type list.
_SUPERTYPE_NOISE: frozenset[str] = frozenset(
    {"keyword_argument", "access_specifier", "comment", "line_comment", "block_comment", "modifiers", "annotation"}
)
#: Words a supertype clause writes that are syntax, not a type (an error node may hold them).
_SUPERTYPE_WORDS: frozenset[str] = frozenset(
    {"extends", "implements", "with", "is", "inherits", "public", "private", "protected"}
)
_MAX_FIRST_LINE = 120

_WHITESPACE = re.compile(r"\s+")
_SPACE_BEFORE = re.compile(r"\s+([(<:,)\].])")
_SPACE_AFTER = re.compile(r"([(<\[.])\s+")
_TYPE_TAIL = re.compile(r"[(<\[{].*$", re.DOTALL)
_NAME_PREFIX = "$@:&*#!%"
_MAX_SIGNATURE = 200
_QUOTES = "\"'`"
#: Separators a written qualified name may use, longest first so `::` wins over `:`.
_SEPARATORS = ("::", "->", "\\", ".", ":", "/")


@dataclass(frozen=True)
class Declared:
    """One declaration the reader recognised."""

    rule: Rule
    node: Node
    #: The wrapper node the declaration was found under (decorator, export), or None.
    wrapper: Node | None
    #: The keyword text that selected the rule, when the rule matched by keyword or `when`.
    keyword: str | None


@dataclass(frozen=True)
class Call:
    """One call the reader recognised."""

    name: str
    line: int
    arity: int
    receiver: str | None
    receiver_type: str | None
    is_new: bool
    #: The ids of the callee node and its ancestors below the call, which a walk must not
    #: visit again: a bareword callee inside a bracketed call is itself a call node.
    consumed_ids: tuple[int, ...] = ()


def clean_name(raw: str) -> str:
    """Strip the sigils, quotes and trailing punctuation a name token may carry."""
    value = raw.strip()
    while value and value[0] in _NAME_PREFIX:
        value = value[1:]
    value = value.strip(_QUOTES).rstrip(";:,")
    return value.strip()


def type_name(raw: str) -> str:
    """Reduce a written type to its name: generics, call parentheses and pointers dropped."""
    value = _TYPE_TAIL.sub("", raw).strip()
    value = value.replace("*", "").replace("&", "").replace("?", "").replace("!", "").strip()
    return clean_name(value)


def split_qualified(name: str) -> tuple[str | None, str]:
    """Split `a.b.c` into (`a.b`, `c`) on the last separator, or (None, name) when there is none."""
    for separator in _SEPARATORS:
        if separator in name:
            head, _, tail = name.rpartition(separator)
            if tail:
                return head, tail
    return None, name


def first_keyword(node: Node, source: bytes) -> str | None:
    """The first bare alphabetic token of a node, which is the keyword that opens it."""
    for child in node.children:
        if child.is_named:
            continue
        token = text(child, source)
        if token.isalpha():
            return token
    return None


def _matches_when(node: Node, source: bytes, when: tuple[str, frozenset[str]] | None) -> tuple[bool, str | None]:
    """Whether a `when` condition holds, and the text that satisfied it."""
    if when is None:
        return True, None
    path, allowed = when
    value = clean_name(text(resolve(node, path), source))
    return (True, value) if value in allowed else (False, None)


def _matches_require(node: Node, require: tuple[str, str] | None) -> bool:
    """Whether a `require` condition holds."""
    if require is None:
        return True
    path, kind = require
    found = resolve(node, path)
    return found is not None and found.type == kind


def _scope_allows(rule: Rule, *, in_type: bool, in_callable: bool) -> bool:
    """Whether the rule's scope admits the current owner."""
    if rule.scope == "any":
        return True
    if rule.scope == "type":
        return in_type
    if rule.scope == "top":
        return not in_type and not in_callable
    if rule.scope == "member":
        return not in_callable
    raise ValueError(f"unknown rule scope {rule.scope!r}")


def unwrap(spec: LanguageSpec, node: Node) -> tuple[Node, Node | None]:
    """Look through decorator and export wrappers, returning (declaration, wrapper)."""
    wrapper: Node | None = None
    current = node
    for _ in range(4):
        path = spec.wrappers.get(current.type)
        if path is None:
            break
        inner = resolve(current, path)
        if inner is None:
            break
        wrapper = wrapper or current
        current = inner
    return current, wrapper


def match(
    spec: LanguageSpec, node: Node, source: bytes, *, in_type: bool = False, in_callable: bool = False
) -> Declared | None:
    """Find the first rule a node satisfies, looking through wrappers first."""
    inner, wrapper = unwrap(spec, node)
    for rule in spec.rules:
        if inner.type not in rule.kinds:
            continue
        if not _scope_allows(rule, in_type=in_type, in_callable=in_callable):
            continue
        if not _matches_require(inner, rule.require):
            continue
        ok, keyword = _matches_when(inner, source, rule.when)
        if not ok:
            continue
        if rule.keyword_kinds or rule.keywords:
            keyword = first_keyword(inner, source)
            if rule.keyword_kinds and keyword not in rule.keyword_kinds:
                continue
            if rule.keywords and keyword not in rule.keywords:
                continue
        return Declared(rule=rule, node=inner, wrapper=wrapper, keyword=keyword)
    return None


def name_of(spec: LanguageSpec, declared: Declared, source: bytes) -> tuple[str | None, str]:
    """A declaration's (owner name, own name); the owner is set when the name is written scoped."""
    rule, node = declared.rule, declared.node
    if rule.names is not None and rule.name_join is not None:
        parts = [
            clean_name(text(part, source))
            for part in resolve_all(node, rule.names)
            if part.type in NAME_KINDS or part.type in STRING_KINDS or part.type in QUALIFIED_KINDS
        ]
        joined = rule.name_join.join(part for part in parts if part)
        return None, joined or (rule.default_name or "")
    if rule.name is None:
        return None, rule.default_name or ""
    name_node = resolve(node, rule.name)
    if name_node is None:
        return None, rule.default_name or ""
    access = spec.member_access.get(name_node.type)
    if access is not None:
        receiver_path, member_path = access
        owner = clean_name(text(resolve(name_node, receiver_path), source))
        member = clean_name(text(resolve(name_node, member_path), source))
        if member:
            return (type_name(owner) or None), member
    raw = clean_name(text(name_node, source))
    if name_node.type in QUALIFIED_KINDS or name_node.type not in NAME_KINDS:
        owner, own = split_qualified(type_name(raw) if name_node.type in QUALIFIED_KINDS else raw)
        return owner, own
    return None, raw


def names_of(spec: LanguageSpec, declared: Declared, source: bytes) -> list[Node]:
    """Every node a multi-name declaration names, or the single name node."""
    rule, node = declared.rule, declared.node
    if rule.names is not None:
        return resolve_all(node, rule.names)
    if rule.name is None:
        return []
    single = resolve(node, rule.name)
    return [single] if single is not None else []


def body_of(declared: Declared) -> Node | None:
    """The declaration's body node, honouring the rule's required body kind."""
    rule = declared.rule
    if rule.body is None:
        return None
    body = resolve(declared.node, rule.body)
    if body is not None and rule.body_kind is not None and body.type != rule.body_kind:
        return None
    return body


def params_of(declared: Declared) -> Node | None:
    """The declaration's parameter list node, if the rule names one."""
    if declared.rule.params is None:
        return None
    return resolve(declared.node, declared.rule.params)


def parameter_nodes(spec: LanguageSpec, declared: Declared, params: Node | None, source: bytes) -> list[Node]:
    """The parameter declarations of a callable, receiver parameters dropped."""
    if params is None:
        return []
    children = [child for child in params.named_children if "comment" not in child.type]
    if spec.parameter_kinds:
        children = [child for child in children if child.type in spec.parameter_kinds]
    children = children[declared.rule.params_offset :]
    kept: list[Node] = []
    for child in children:
        if child.type in spec.self_parameters:
            continue
        if clean_name(text(child, source)) in spec.self_parameters and child.type in NAME_KINDS:
            continue
        kept.append(child)
    return kept


def parameter_type(parameter: Node, source: bytes) -> str:
    """The written type of one parameter, or `?` when it declares none."""
    typed = parameter.child_by_field_name("type")
    if typed is None:
        return "?"
    return type_name(text(typed, source)) or "?"


def _modifier_words(spec: LanguageSpec, node: Node, source: bytes) -> list[str]:
    """The modifier words written directly on one node."""
    words: list[str] = []
    for child in node.children:
        if child.is_named:
            if child.type in spec.modifier_kinds:
                words.extend(word for word in text(child, source).split() if word.isalpha())
            continue
        token = text(child, source)
        if token in spec.modifier_words:
            words.append(token)
    return words


def modifiers_of(spec: LanguageSpec, declared: Declared, source: bytes) -> list[str]:
    """The declaration's modifier words, its wrapper's included, in written order."""
    words: list[str] = []
    if declared.wrapper is not None:
        words.extend(_modifier_words(spec, declared.wrapper, source))
    words.extend(_modifier_words(spec, declared.node, source))
    if declared.keyword is not None and declared.keyword in spec.private_words:
        words.append(declared.keyword)
    seen: list[str] = []
    for word in words:
        if word not in seen:
            seen.append(word)
    return seen


def annotations_of(spec: LanguageSpec, declared: Declared, source: bytes) -> list[str]:
    """The simple names of the decorators, annotations or attributes on a declaration.

    They may be children of the node, children of its wrapper, or the run of siblings written
    just before it (Rust attributes, Dart annotations), so all three are read.
    """
    if not spec.annotation_kinds:
        return []
    nodes: list[Node] = []
    if declared.wrapper is not None:
        nodes.extend(child for child in declared.wrapper.named_children if child.type in spec.annotation_kinds)
    nodes.extend(child for child in declared.node.named_children if child.type in spec.annotation_kinds)
    previous = declared.node.prev_named_sibling
    leading: list[Node] = []
    while previous is not None and previous.type in spec.annotation_kinds:
        leading.append(previous)
        previous = previous.prev_named_sibling
    nodes.extend(reversed(leading))
    names: list[str] = []
    for annotation in nodes:
        raw = text(annotation, source)
        raw = raw.lstrip("@#[!").rstrip("]")
        raw = _TYPE_TAIL.sub("", raw).strip()
        raw = raw.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
        if raw and raw not in names:
            names.append(raw)
    return names


def visibility_of(spec: LanguageSpec, name: str, modifiers: list[str], *, in_type: bool) -> Visibility:
    """How far a declaration reaches, from its modifier words and the language's conventions."""
    written = set(modifiers)
    for word, level in VISIBILITY_WORDS:
        if word in written:
            return level
    if written & spec.private_words:
        return Visibility.PRIVATE
    if spec.underscore_is_private and name.startswith("_"):
        return Visibility.PRIVATE
    if spec.capitalized_is_public:
        return Visibility.PUBLIC if name[:1].isupper() else Visibility.PACKAGE
    if in_type and spec.member_visibility is not None:
        return spec.member_visibility
    return spec.default_visibility


def supertypes_of(declared: Declared, source: bytes) -> list[tuple[str, int]]:
    """The (name, line) of every supertype a declaration names."""
    found: list[tuple[str, int]] = []
    for path in declared.rule.supertypes:
        for holder in resolve_all(declared.node, path):
            candidates = [holder] if _is_name(holder) else list(holder.named_children)
            for candidate in candidates:
                if candidate.type in _SUPERTYPE_NOISE:
                    continue
                if not _is_name(candidate):
                    inner = next((child for child in candidate.named_children if _is_name(child)), None)
                    if inner is None:
                        continue
                    candidate = inner
                name = type_name(text(candidate, source))
                if name and name not in _SUPERTYPE_WORDS:
                    found.append((name, candidate.start_point[0] + 1))
    return found


def _is_name(node: Node) -> bool:
    """Whether a node reads as a (possibly qualified) name."""
    return node.type in NAME_KINDS or node.type in QUALIFIED_KINDS


def signature_of(declared: Declared, body: Node | None, source: bytes, annotation_kinds: frozenset[str]) -> str:
    """Everything a declaration says before its body, annotations removed, on one line."""
    node = declared.node
    if body is node:
        # The declaration is its own body (a Fortran module, a lisp form): its first line is
        # all a signature can honestly say.
        first_line = source[node.start_byte : node.end_byte].decode("utf-8", "replace").split("\n", 1)[0]
        return _SPACE_BEFORE.sub(r"\1", " ".join(first_line.split()))[:_MAX_FIRST_LINE]
    end = body.start_byte if body is not None else node.end_byte
    pieces: list[str] = []
    for child in node.children:
        if child.start_byte >= end or child.type in annotation_kinds or "comment" in child.type:
            continue
        if child.end_byte > end:
            # The body sits inside this child (`struct { ... }` under a Go type spec): keep
            # only what the child says before it.
            pieces.append(" ".join(source[child.start_byte : end].decode("utf-8", "replace").split()))
            continue
        pieces.append(text(child, source))
    raw = _WHITESPACE.sub(" ", " ".join(piece for piece in pieces if piece)).strip()
    raw = _SPACE_BEFORE.sub(r"\1", raw)
    raw = _SPACE_AFTER.sub(r"\1", raw)
    raw = raw.rstrip(" :{=;,")
    return raw[:_MAX_SIGNATURE]


def call_at(spec: LanguageSpec, node: Node, source: bytes) -> Call | None:
    """Recognise a call node and describe it, or return None when the node is no call."""
    for rule in spec.calls:
        if node.type not in rule.kinds:
            continue
        if not _matches_require(node, rule.require):
            continue
        if not _matches_when(node, source, rule.when)[0]:
            continue
        return _describe_call(spec, rule, node, source)
    return None


def _callee_parts(
    spec: LanguageSpec, rule: CallRule, node: Node, callee: Node, source: bytes
) -> tuple[str | None, str] | None:
    """Split a callee into (receiver text, called name), or None when the node calls nothing nameable."""
    if rule.receiver is not None:
        receiver = clean_name(text(resolve(node, rule.receiver), source)) or None
        return receiver, clean_name(text(callee, source))
    if callee.type in spec.member_access:
        receiver_path, member_path = spec.member_access[callee.type]
        receiver_node = resolve(callee, receiver_path)
        member_node = resolve(callee, member_path)
        if receiver_node is not None and member_node is not None and receiver_node.id == member_node.id:
            receiver_node = None
        return clean_name(text(receiver_node, source)) or None, clean_name(text(member_node, source))
    if callee.type in NAME_KINDS or callee.type in QUALIFIED_KINDS or callee.type in spec.callee_kinds:
        head, name = split_qualified(type_name(text(callee, source)))
        return head, name
    if callee.type in STRING_KINDS:
        return None, clean_name(text(callee, source))
    return None


def _describe_call(spec: LanguageSpec, rule: CallRule, node: Node, source: bytes) -> Call | None:
    """Turn a matched call node into a :class:`Call`."""
    callee = resolve(node, rule.callee)
    if callee is None:
        return None
    while callee.type in spec.transparent_kinds and len(callee.named_children) == 1:
        callee = callee.named_children[0]
    line = node.start_point[0] + 1
    if callee.type in spec.self_names:
        # `super(...)` and `this(...)` chain to a constructor of the hierarchy.
        return Call(callee.type, line, _arity(rule, node), callee.type, None, True, _below(callee, node))
    parts = _callee_parts(spec, rule, node, callee, source)
    if parts is None:
        return None
    receiver, name = parts
    if not name or name in spec.call_exclusions:
        return None
    if spec.symbol_namespace_separator and spec.symbol_namespace_separator in name:
        receiver, name = name.rsplit(spec.symbol_namespace_separator, 1)
    name = name.lstrip(".")
    if receiver is not None and receiver in spec.self_names:
        receiver = "this"
    receiver_type = _receiver_type(receiver)
    is_new = rule.is_new
    if name in spec.constructor_call_names and receiver_type is not None:
        # `Foo.new(...)`: the receiver is the type being constructed.
        name, receiver, receiver_type, is_new = receiver_type, None, None, True
    elif spec.capitalized_call_is_new and receiver is None and name[:1].isupper():
        is_new = True
    return Call(name, line, _arity(rule, node), receiver, receiver_type, is_new, _below(callee, node))


def _receiver_type(receiver: str | None) -> str | None:
    """A receiver written as a capitalised name is taken to be a type."""
    if receiver is None or receiver == "this":
        return None
    head = type_name(receiver.lstrip("(&*")).rsplit(".", 1)[-1]
    return head if head[:1].isupper() else None


def _below(callee: Node, call: Node) -> tuple[int, ...]:
    """The ids of the callee and every ancestor of it strictly below the call node."""
    ids: list[int] = []
    current: Node | None = callee
    while current is not None and current.id != call.id:
        ids.append(current.id)
        current = current.parent
    return tuple(ids)


def _fans_out(path: str) -> bool:
    """Whether a path's last step yields a list (`@kind*`, `field+`)."""
    last = path.split("/")[-1]
    return last.endswith("+") or (last.startswith("@") and last.endswith("*"))


def _arity(rule: CallRule, node: Node) -> int:
    """Count a call's arguments, or -1 when the rule cannot see them."""
    if rule.arguments is None:
        return -1
    if _fans_out(rule.arguments):
        return len(resolve_all(node, rule.arguments))
    holder = resolve(node, rule.arguments)
    if holder is None:
        return -1
    children = [child for child in holder.named_children if "comment" not in child.type]
    return max(0, len(children) - rule.arguments_offset)


__all__ = [
    "NAME_KINDS",
    "QUALIFIED_KINDS",
    "STRING_KINDS",
    "Call",
    "Declared",
    "annotations_of",
    "body_of",
    "call_at",
    "clean_name",
    "first_keyword",
    "match",
    "modifiers_of",
    "name_of",
    "names_of",
    "parameter_nodes",
    "parameter_type",
    "params_of",
    "signature_of",
    "split_qualified",
    "supertypes_of",
    "type_name",
    "unwrap",
    "visibility_of",
]
