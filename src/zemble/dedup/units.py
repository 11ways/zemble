"""Turning Java source into comparable units: whole bodies and statement windows.

The parser accessor is shared with the symbol graph (:func:`zemble.graph.java.java_parser`),
but nothing else is: the graph answers "what does this file declare", this module answers
"what does this file's code look like once names stop mattering".
"""

from __future__ import annotations

from hashlib import blake2b

from tree_sitter import Node

from zemble.dedup.model import Unit
from zemble.graph.java import java_parser

#: Declarations that own a body worth comparing.
_CALLABLE_KINDS = {
    "method_declaration": "method",
    "constructor_declaration": "constructor",
    "compact_constructor_declaration": "constructor",
    "annotation_type_element_declaration": "method",
}
_INITIALIZER_TYPES = frozenset({"static_initializer", "block"})
_TYPE_DECLARATIONS = frozenset({"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"})
#: Control keywords whose order is the unit's control-flow skeleton.
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


def _is_comment(node: Node) -> bool:
    """Whether a node is a comment of any dialect."""
    return "comment" in node.type


def _text(source: bytes, node: Node) -> str:
    """Return a node's source text."""
    return source[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _leaves(node: Node, source: bytes, out: list[tuple[str, str]]) -> None:
    """Append every non-comment leaf of a subtree as a (node type, text) pair."""
    if _is_comment(node):
        return
    if node.child_count == 0:
        out.append((node.type, _text(source, node)))
        return
    for child in node.children:
        _leaves(child, source, out)


def _declared_names(node: Node, source: bytes, out: list[str]) -> None:
    """Collect every identifier a unit DECLARES: locals, parameters, lambda params, local types."""
    if _is_comment(node):
        return
    kind = node.type
    if kind in _NAME_FIELD_DECLARATIONS or kind in _TYPE_DECLARATIONS:
        name = node.child_by_field_name("name")
        if name is not None:
            out.append(_text(source, name))
    if kind == "type_pattern":
        for child in node.named_children:
            if child.type == "identifier":
                out.append(_text(source, child))
    elif kind == "lambda_expression":
        _lambda_parameters(node, source, out)
    for child in node.children:
        _declared_names(child, source, out)


def _lambda_parameters(node: Node, source: bytes, out: list[str]) -> None:
    """Collect the parameter names of a lambda, in all three spellings Java allows."""
    parameters = node.child_by_field_name("parameters")
    if parameters is None:
        return
    if parameters.type == "identifier":
        out.append(_text(source, parameters))
        return
    if parameters.type == "inferred_parameters":
        for child in parameters.named_children:
            if child.type == "identifier":
                out.append(_text(source, child))


def _calls_and_literals(node: Node, source: bytes, calls: list[str], literals: list[str]) -> None:
    """Collect the called names and the literal texts of a subtree."""
    if _is_comment(node):
        return
    kind = node.type
    if kind in _LITERAL_TYPES:
        literals.append(_text(source, node))
        return
    if kind == "method_invocation":
        name = node.child_by_field_name("name")
        if name is not None:
            calls.append(_text(source, name))
    elif kind == "object_creation_expression":
        created = node.child_by_field_name("type")
        if created is not None:
            calls.append(_text(source, created).rsplit(".", 1)[-1])
    elif kind == "explicit_constructor_invocation":
        constructor = node.child_by_field_name("constructor")
        if constructor is not None:
            calls.append(_text(source, constructor))
    for child in node.children:
        _calls_and_literals(child, source, calls, literals)


def _hash(parts: list[str]) -> str:
    """Hash a token stream into a short stable hex digest."""
    digest = blake2b(digest_size=16)
    for part in parts:
        digest.update(part.encode("utf-8", "replace"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _renamed_stream(tokens: list[tuple[str, str]], declared: frozenset[str]) -> list[str]:
    """Replace declared identifiers by positional placeholders in first-seen order.

    An identifier straight after a dot is a MEMBER name, never a local, whatever it is
    spelled like: without that rule every `this.key = key;` run in every constructor
    normalizes to the same stream and constructors become one giant clone class.

    :param tokens: The unit's (node type, text) token stream.
    :param declared: The names the unit declares.
    :return: The normalized stream.
    """
    placeholders: dict[str, str] = {}
    stream: list[str] = []
    previous = ""
    for node_type, text in tokens:
        is_member = previous == "."
        previous = text
        if node_type == "identifier" and text in declared and not is_member:
            placeholder = placeholders.get(text)
            if placeholder is None:
                placeholder = f"${len(placeholders)}"
                placeholders[text] = placeholder
            stream.append(placeholder)
        else:
            stream.append(text)
    return stream


def _make_unit(
    node: Node,
    source: bytes,
    file_path: str,
    kind: str,
    name: str,
    declared: frozenset[str],
    include_text: bool,
    tokens: list[tuple[str, str]] | None = None,
    span: tuple[int, int] | None = None,
) -> Unit:
    """Build one unit from a subtree (or a pre-collected token run)."""
    if tokens is None:
        tokens = []
        _leaves(node, source, tokens)
    calls: list[str] = []
    literals: list[str] = []
    _calls_and_literals(node, source, calls, literals)
    skeleton = tuple(text for node_type, text in tokens if node_type in _CONTROL_KEYWORDS)
    start, end = span if span is not None else (node.start_point[0] + 1, node.end_point[0] + 1)
    return Unit(
        file_path=file_path,
        start_line=start,
        end_line=end,
        kind=kind,
        name=name,
        token_count=len(tokens),
        exact_hash=_hash([text for _, text in tokens]),
        renamed_hash=_hash(_renamed_stream(tokens, declared)),
        skeleton=skeleton,
        calls=tuple(sorted(set(calls))),
        literals=tuple(literals),
        text=_text(source, node) if include_text else None,
    )


def _statements(block: Node) -> list[Node]:
    """Return the statement children of a block, comments dropped."""
    return [child for child in block.named_children if not _is_comment(child)]


def _blocks(node: Node, out: list[Node]) -> None:
    """Collect every block inside a subtree, the subtree itself included."""
    if _is_comment(node):
        return
    if node.type in ("block", "constructor_body", "switch_block_statement_group"):
        out.append(node)
    for child in node.children:
        _blocks(child, out)


def _window_units(
    body: Node,
    source: bytes,
    file_path: str,
    name: str,
    declared: frozenset[str],
    min_tokens: int,
    min_statements: int,
    max_statements: int,
) -> list[Unit]:
    """Build a unit for every window of consecutive statements inside a body."""
    units: list[Unit] = []
    blocks: list[Node] = []
    _blocks(body, blocks)
    for block in blocks:
        statements = _statements(block)
        if len(statements) < min_statements:
            continue
        token_runs = []
        for statement in statements:
            run: list[tuple[str, str]] = []
            _leaves(statement, source, run)
            token_runs.append(run)
        for start in range(len(statements)):
            for length in range(min_statements, min(max_statements, len(statements) - start) + 1):
                if block is body and length == len(statements):
                    continue  # identical to the body unit itself
                window = statements[start : start + length]
                tokens = [token for run in token_runs[start : start + length] for token in run]
                if len(tokens) < min_tokens:
                    continue
                units.append(_WindowBuilder(window, source, file_path, name, declared, tokens).build())
    return units


class _WindowBuilder:
    """Builds one window unit out of a run of consecutive statements."""

    def __init__(
        self,
        statements: list[Node],
        source: bytes,
        file_path: str,
        name: str,
        declared: frozenset[str],
        tokens: list[tuple[str, str]],
    ) -> None:
        """Hold everything the window needs; :meth:`build` does the work."""
        self.statements = statements
        self.source = source
        self.file_path = file_path
        self.name = name
        self.declared = declared
        self.tokens = tokens

    def build(self) -> Unit:
        """Assemble the window's unit, merging the per-statement structural facts."""
        calls: list[str] = []
        literals: list[str] = []
        for statement in self.statements:
            _calls_and_literals(statement, self.source, calls, literals)
        skeleton = tuple(text for node_type, text in self.tokens if node_type in _CONTROL_KEYWORDS)
        return Unit(
            file_path=self.file_path,
            start_line=self.statements[0].start_point[0] + 1,
            end_line=self.statements[-1].end_point[0] + 1,
            kind="window",
            name=self.name,
            token_count=len(self.tokens),
            exact_hash=_hash([text for _, text in self.tokens]),
            renamed_hash=_hash(_renamed_stream(self.tokens, self.declared)),
            skeleton=skeleton,
            calls=tuple(sorted(set(calls))),
            literals=tuple(literals),
        )


class _UnitExtractor:
    """Walks one parsed file and emits every comparable unit in it."""

    def __init__(
        self,
        source: bytes,
        file_path: str,
        min_tokens: int,
        min_statements: int,
        windows: bool,
        include_text: bool,
        max_window_statements: int,
    ) -> None:
        """Prepare an extractor for one file."""
        self.source = source
        self.file_path = file_path
        self.min_tokens = min_tokens
        self.min_statements = min_statements
        self.windows = windows
        self.include_text = include_text
        self.max_window_statements = max_window_statements
        self.units: list[Unit] = []

    def run(self, root: Node) -> list[Unit]:
        """Extract every unit of the compilation unit."""
        for child in root.named_children:
            if child.type in _TYPE_DECLARATIONS:
                self._visit_type(child, "")
        return self.units

    def _visit_type(self, node: Node, prefix: str) -> None:
        """Walk one type declaration's body under its qualified name."""
        name_node = node.child_by_field_name("name")
        name = _text(self.source, name_node) if name_node is not None else "<anonymous>"
        qualified = f"{prefix}.{name}" if prefix else name
        body = node.child_by_field_name("body")
        if body is not None:
            self._visit_body(body, qualified)

    def _visit_body(self, body: Node, qualified: str) -> None:
        """Emit units for every member of a type body, enum declarations folded in."""
        members: list[Node] = []
        for child in body.named_children:
            if child.type == "enum_body_declarations":
                members.extend(child.named_children)
            else:
                members.append(child)
        for member in members:
            kind = _CALLABLE_KINDS.get(member.type)
            if kind is not None:
                self._emit_callable(member, qualified, kind)
            elif member.type in _INITIALIZER_TYPES:
                self._emit(member, member, f"{qualified}.<initializer>", "initializer")
            elif member.type in _TYPE_DECLARATIONS:
                self._visit_type(member, qualified)
            elif member.type == "enum_constant":
                inner = next((c for c in member.named_children if c.type == "class_body"), None)
                if inner is not None:
                    self._visit_body(inner, qualified)

    def _emit_callable(self, member: Node, qualified: str, kind: str) -> None:
        """Emit the unit of one callable, with its parameters counted as declared names."""
        body = member.child_by_field_name("body")
        if body is None:
            return
        name_node = member.child_by_field_name("name")
        name = _text(self.source, name_node) if name_node is not None else "<anonymous>"
        self._emit(member, body, f"{qualified}.{name}", kind)

    def _emit(self, declaration: Node, body: Node, name: str, kind: str) -> None:
        """Emit a body unit and, when asked, its statement windows."""
        names: list[str] = []
        _declared_names(declaration, self.source, names)
        declared = frozenset(names)
        tokens: list[tuple[str, str]] = []
        _leaves(body, self.source, tokens)
        if len(tokens) >= self.min_tokens:
            self.units.append(
                _make_unit(
                    body,
                    self.source,
                    self.file_path,
                    kind,
                    name,
                    declared,
                    self.include_text,
                    tokens=tokens,
                    span=(declaration.start_point[0] + 1, body.end_point[0] + 1),
                )
            )
        if self.windows:
            self.units.extend(
                _window_units(
                    body,
                    self.source,
                    self.file_path,
                    name,
                    declared,
                    self.min_tokens,
                    self.min_statements,
                    self.max_window_statements,
                )
            )


def extract_units(
    source: bytes,
    file_path: str,
    *,
    min_tokens: int = 30,
    min_statements: int = 6,
    windows: bool = True,
    include_text: bool = False,
    max_window_statements: int = 24,
) -> list[Unit]:
    """Extract every comparable unit from one Java source file.

    :param source: Raw file bytes.
    :param file_path: Path as the report should print it, posix style.
    :param min_tokens: Smallest token count a unit may have; below it a getter is noise.
    :param min_statements: Smallest statement window considered.
    :param windows: Whether to emit statement windows beside whole bodies.
    :param include_text: Whether to keep the source text on body units (logic mode needs it).
    :param max_window_statements: Longest statement window considered, capping the O(n^2) window set.
    :return: The units, in source order.
    :raises RuntimeError: If no Java grammar is available on this platform.
    """
    parser = java_parser()
    if parser is None:
        raise RuntimeError("No tree-sitter Java grammar available")
    tree = parser.parse(source)
    extractor = _UnitExtractor(
        source, file_path, min_tokens, min_statements, windows, include_text, max_window_statements
    )
    return extractor.run(tree.root_node)
