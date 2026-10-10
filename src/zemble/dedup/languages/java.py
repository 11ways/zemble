"""The Java language profile.

The parser accessor is shared with the symbol graph (:func:`zemble.graph.java.java_parser`),
but nothing else is: the graph answers "what does this file declare", this module answers
"what does this file's code look like once names stop mattering".
"""

from __future__ import annotations

import re
from dataclasses import replace

from tree_sitter import Node

from zemble.dedup.languages.base import (
    Container,
    LanguageProfile,
    ShapeHooks,
    Signature,
    SiteKind,
    Visibility,
    VocabularyFact,
    node_text,
)
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


def _member_body(node: Node, source: bytes) -> Node | None:
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


#: Upper snake case of at least two characters: `ID`, `STATUS_FAILED`, never a type parameter `T`.
_CONSTANT_NAME = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*")
_FIELD_DECLARATIONS = frozenset({"field_declaration", "constant_declaration"})
_ANNOTATIONS = frozenset({"annotation", "marker_annotation"})
#: JDK calls whose first string argument is a regular expression.
_REGEX_CALLS = frozenset({"compile", "matches", "replaceAll", "replaceFirst", "split"})
#: Smallest value set a switch, an enum or a constant group declares to count as a vocabulary.
_MIN_SET = 3


def _is_constant_name(text: str) -> bool:
    """Whether an identifier follows the Java constant convention."""
    return len(text) >= 2 and _CONSTANT_NAME.fullmatch(text) is not None


def _call_parts(node: Node) -> tuple[Node | None, Node | None, list[Node]]:
    """A `method_invocation`'s receiver, name and arguments."""
    arguments = node.child_by_field_name("arguments")
    values = [child for child in arguments.named_children if "comment" not in child.type] if arguments else []
    return node.child_by_field_name("object"), node.child_by_field_name("name"), values


def _forward_target(member: Node, body: Node, source: bytes) -> str | None:
    """`Type.member` when a body is one `return Type.member(p1, ..., pn);` of its own parameters, in order.

    Only a receiver spelled like a type counts: a call on a field or a parameter is delegation to an
    object, which is a design, not a facade over a static API.
    """
    statements = [child for child in body.named_children if "comment" not in child.type]
    if len(statements) != 1 or statements[0].type not in {"return_statement", "expression_statement"}:
        return None
    expression = next((child for child in statements[0].named_children if "comment" not in child.type), None)
    if expression is None or expression.type != "method_invocation":
        return None
    receiver, name, arguments = _call_parts(expression)
    if receiver is None or name is None or receiver.type != "identifier":
        return None
    owner = node_text(source, receiver)
    if not owner[:1].isupper() or _is_constant_name(owner):
        return None
    parameters = member.child_by_field_name("parameters")
    names = []
    for parameter in parameters.named_children if parameters is not None else ():
        if parameter.type != "formal_parameter":
            return None
        declared = parameter.child_by_field_name("name")
        if declared is None:
            return None
        names.append(node_text(source, declared))
    if [node_text(source, argument) for argument in arguments] != names:
        return None
    return f"{owner}.{node_text(source, name)}"


def _implements_contract(node: Node, source: bytes) -> bool:
    """Whether a member carries `@Override`."""
    modifiers = next((child for child in node.children if child.type == "modifiers"), None)
    if modifiers is None:
        return False
    return any(
        child.type == "marker_annotation"
        and node_text(source, child).replace(" ", "") in {"@Override", "@java.lang.Override"}
        for child in modifiers.named_children
    )


#: A documentation comment's markup the summary drops: delimiters, line stars, block tags that name parameters.
_DOC_MARKUP = re.compile(r"^\s*\*+", re.MULTILINE)
_DOC_INLINE = re.compile(r"\{@\w+\s+([^}]*)\}")
_DOC_PARAMETER_TAGS = re.compile(r"@(?:param|throws|exception|author|since|see)\b[^@]*")
#: Longest documentation summary kept: the intent text is a sentence or two, not the whole comment.
_SUMMARY_CHARS = 400


_NOT_A_TYPE = frozenset({"modifiers", "variable_declarator", "identifier", *_ANNOTATIONS})


def _simple_type(node: Node, source: bytes) -> str:
    """A declared type by its simple name: generics, annotations and package dropped, array brackets kept."""
    if node.type == "generic_type":
        return _simple_type(node.named_children[0], source) if node.named_children else node_text(source, node)
    if node.type == "array_type":
        element = node.child_by_field_name("element")
        dimensions = node.child_by_field_name("dimensions")
        suffix = node_text(source, dimensions) if dimensions is not None else "[]"
        return (_simple_type(element, source) if element is not None else "") + suffix.replace(" ", "")
    if node.type == "annotated_type":
        inner = [child for child in node.named_children if child.type not in _ANNOTATIONS]
        return _simple_type(inner[-1], source) if inner else node_text(source, node)
    return node_text(source, node).rsplit(".", 1)[-1]


def _parameter_type(parameter: Node, source: bytes) -> str | None:
    """One parameter's simple type; a varargs parameter is its element type plus `...`."""
    declared = parameter.child_by_field_name("type")
    if declared is not None:
        return _simple_type(declared, source)
    if parameter.type == "spread_parameter":
        element = next((child for child in parameter.named_children if child.type not in _NOT_A_TYPE), None)
        return _simple_type(element, source) + "..." if element is not None else None
    return None


def _summary(member: Node, source: bytes) -> str:
    """The member's Javadoc as one line of plain text, "" when the comment above it is not a Javadoc."""
    comment = member.prev_named_sibling
    if comment is None or comment.type != "block_comment":
        return ""
    text = node_text(source, comment)
    if not text.startswith("/**"):
        return ""
    text = text[3:].removesuffix("*/")
    text = _DOC_PARAMETER_TAGS.sub(" ", _DOC_INLINE.sub(r"\1", _DOC_MARKUP.sub(" ", text)))
    return " ".join(text.replace("@return", "returns").split())[:_SUMMARY_CHARS]


def _signature(member: Node, source: bytes) -> Signature | None:
    """A method's or constructor's declared types, staticness and Javadoc."""
    if member.type not in _CALLABLE_KINDS:
        return None
    parameters = member.child_by_field_name("parameters")
    types = []
    for parameter in parameters.named_children if parameters is not None else ():
        if parameter.type in {"formal_parameter", "spread_parameter"}:
            declared = _parameter_type(parameter, source)
            if declared is None:
                return None
            types.append(declared)
    returned = member.child_by_field_name("type")
    returns = _simple_type(returned, source) if returned is not None else ""
    variables = member.child_by_field_name("type_parameters")
    return Signature(
        parameters=tuple(types),
        returns="" if returns == "void" else returns,
        open_types=frozenset(
            node_text(source, variable.named_children[0])
            for variable in (variables.named_children if variables is not None else ())
            if variable.type == "type_parameter" and variable.named_children
        )
        | {"Object"},
        static="static" in _modifiers(member, source),
        summary=_summary(member, source),
    )


def _string_value(node: Node, source: bytes) -> str | None:
    """The content of a plain one-line string literal, or None for a text block."""
    text = node_text(source, node)
    if node.type != "string_literal" or text.startswith('"""') or len(text) < 2:
        return None
    return text[1:-1]


class _VocabularyWalk:
    """Collects one file's constants, literal uses, value sets and regex literals."""

    def __init__(self, source: bytes) -> None:
        """Start an empty walk over one file."""
        self.source = source
        self.facts: list[VocabularyFact] = []

    def visit(self, node: Node, owner: str, folded: Visibility, member: str | None) -> None:
        """Visit one node inside `owner`, whose visibility is already folded through its enclosing types."""
        kind = node.type
        if "comment" in kind or kind in _ANNOTATIONS:
            return
        if kind in _TYPE_DECLARATIONS:
            self._type(node, owner, folded)
            return
        if kind in _FIELD_DECLARATIONS and member is None:
            self._field(node, owner, folded)
            return
        if kind in _CALLABLE_KINDS:
            name = node.child_by_field_name("name")
            inner = f"{owner}.{node_text(self.source, name)}" if name is not None else owner
            for child in node.children:
                self.visit(child, owner, folded, inner)
            return
        if kind == "string_literal":
            self._literal(node, member or owner)
            return
        if kind == "switch_block":
            self._switch(node, member or owner)
        elif kind == "method_invocation":
            self._regex(node, member or owner)
        for child in node.children:
            self.visit(child, owner, folded, member)

    def _type(self, node: Node, owner: str, folded: Visibility) -> None:
        """A type declaration: its own scope, plus a value set when it is an enum."""
        name = node.child_by_field_name("name")
        inner = f"{owner}.{node_text(self.source, name)}" if owner else node_text(self.source, name or node)
        level = _visibility(node, self.source).narrower(folded)
        body = node.child_by_field_name("body")
        if body is None:
            return
        if node.type == "enum_declaration":
            self._enum(node, body, inner, level)
        constants: list[VocabularyFact] = []
        for child in body.named_children:
            if child.type == "enum_constant":
                continue  # walked by `_enum`, which names each constant's literal uses after it
            if child.type == "enum_body_declarations":
                for nested in child.named_children:
                    self._collect(nested, inner, level, constants)
            else:
                self._collect(child, inner, level, constants)
        self._groups(constants, inner, level)

    def _collect(self, node: Node, owner: str, level: Visibility, constants: list[VocabularyFact]) -> None:
        """Visit one member of a type body, keeping the constants it declares for the group sets."""
        before = len(self.facts)
        self.visit(node, owner, level, None)
        if node.type in _FIELD_DECLARATIONS:
            constants.extend(fact for fact in self.facts[before:] if fact.kind is SiteKind.CONSTANT)

    def _enum(self, node: Node, body: Node, owner: str, level: Visibility) -> None:
        """An enum's members, lower-cased, plus every string its constants pass: one value set."""
        values: list[str] = []
        for constant in body.named_children:
            if constant.type != "enum_constant":
                continue
            name = constant.child_by_field_name("name")
            if name is not None:
                values.append(node_text(self.source, name).lower())
            arguments = constant.child_by_field_name("arguments")
            if arguments is not None:
                values.extend(self._strings(arguments))
            # A constant's arguments are uses of their values too ("instance-devices" passed to a builder).
            for child in constant.children:
                self.visit(child, owner, level, f"{owner}.{node_text(self.source, name or constant)}")
        distinct = tuple(dict.fromkeys(values))
        if len(distinct) >= _MIN_SET:
            self.facts.append(_span_fact(SiteKind.SET, owner, distinct, node, level, level))

    def _strings(self, node: Node) -> list[str]:
        """Every plain string literal under a node."""
        if node.type == "string_literal":
            value = _string_value(node, self.source)
            return [value] if value is not None else []
        return [value for child in node.named_children for value in self._strings(child)]

    def _field(self, node: Node, owner: str, folded: Visibility) -> None:
        """A field: a constant when it is final (or an interface field) with one plain string value."""
        modifiers = set(_modifiers(node, self.source))
        constant = node.type == "constant_declaration" or {"static", "final"} <= modifiers
        level = _visibility(node, self.source)
        for declarator in node.children_by_field_name("declarator"):
            name, value = declarator.child_by_field_name("name"), declarator.child_by_field_name("value")
            if name is None or value is None:
                continue
            text = _string_value(value, self.source) if constant else None
            qualified = f"{owner}.{node_text(self.source, name)}"
            if text is not None:
                self.facts.append(_span_fact(SiteKind.CONSTANT, qualified, (text,), declarator, level, folded))
            else:
                self.visit(value, owner, folded, qualified)

    def _groups(self, constants: list[VocabularyFact], owner: str, level: Visibility) -> None:
        """One value set per run of constants sharing a name prefix (`STATUS_*`), or the type's unprefixed ones."""
        groups: dict[str, list[VocabularyFact]] = {}
        for fact in constants:
            simple = fact.name.rsplit(".", 1)[-1]
            prefix = simple.split("_", 1)[0] if "_" in simple else ""
            groups.setdefault(prefix, []).append(fact)
        for prefix, members in groups.items():
            values = tuple(dict.fromkeys(fact.values[0] for fact in members))
            if len(values) < _MIN_SET:
                continue
            label = f"{owner}.{prefix}_*" if prefix else f"{owner}.*"
            self.facts.append(
                VocabularyFact(
                    SiteKind.SET,
                    label,
                    values,
                    min(fact.start_line for fact in members),
                    max(fact.end_line for fact in members),
                    level,
                    level,
                )
            )

    def _literal(self, node: Node, member: str) -> None:
        """A string literal used in code."""
        value = _string_value(node, self.source)
        if value is not None:
            self.facts.append(_span_fact(SiteKind.LITERAL, member, (value,), node))

    def _switch(self, node: Node, member: str) -> None:
        """The string labels of one switch, when it has enough of them to be a value set."""
        labels: list[str] = []
        for child in node.named_children:
            for label in (part for part in child.named_children if part.type == "switch_label"):
                labels.extend(self._strings(label))
            if child.type == "switch_label":
                labels.extend(self._strings(child))
        distinct = tuple(dict.fromkeys(labels))
        if len(distinct) < _MIN_SET:
            return
        condition = node.parent.child_by_field_name("condition") if node.parent is not None else None
        selector = node_text(self.source, condition).strip() if condition is not None else ""
        if selector.startswith("(") and selector.endswith(")"):
            selector = selector[1:-1].strip()
        fact = _span_fact(SiteKind.SWITCH, member, distinct, node.parent or node)
        self.facts.append(replace(fact, detail=selector))

    def _regex(self, node: Node, member: str) -> None:
        """The first string argument of a call that takes a regular expression."""
        _receiver, name, arguments = _call_parts(node)
        if name is None or node_text(self.source, name) not in _REGEX_CALLS or not arguments:
            return
        value = _string_value(arguments[0], self.source)
        if value is not None:
            self.facts.append(_span_fact(SiteKind.REGEX, member, (value,), arguments[0]))


def _span_fact(
    kind: SiteKind,
    name: str,
    values: tuple[str, ...],
    node: Node,
    visibility: Visibility = Visibility.UNKNOWN,
    container_visibility: Visibility = Visibility.UNKNOWN,
) -> VocabularyFact:
    """A fact spanning one node's lines."""
    return VocabularyFact(
        kind, name, values, node.start_point[0] + 1, node.end_point[0] + 1, visibility, container_visibility
    )


def _vocabulary(root: Node, source: bytes) -> list[VocabularyFact]:
    """Every constant, literal use, value set and regex literal of one Java file."""
    walk = _VocabularyWalk(source)
    walk.visit(root, "", Visibility.PUBLIC, None)
    return walk.facts


_SHAPES = ShapeHooks(
    call_kinds=frozenset({"method_invocation"}),
    call_parts=_call_parts,
    is_constant_name=_is_constant_name,
    forward_target=_forward_target,
    vocabulary=_vocabulary,
    implements_contract=_implements_contract,
    signature=_signature,
    node_kinds=frozenset(
        {
            "annotated_type",
            "annotation",
            "array_type",
            "block_comment",
            "constant_declaration",
            "enum_body_declarations",
            "enum_constant",
            "expression_statement",
            "field_declaration",
            "formal_parameter",
            "generic_type",
            "marker_annotation",
            "return_statement",
            "spread_parameter",
            "string_literal",
            "switch_block",
            "switch_label",
            "type_parameter",
            "variable_declarator",
        }
    ),
)


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
    shapes=_SHAPES,
)
