"""Turning source into comparable units: whole bodies and statement windows.

Every syntax fact lives in a :class:`~zemble.dedup.languages.base.LanguageProfile`; this
module only knows how to walk members, hash token streams and cut statement windows, which
is why adding a language never touches it.
"""

from __future__ import annotations

from hashlib import blake2b

from tree_sitter import Node

from zemble.dedup.languages import LanguageProfile, Visibility, node_text, profile_for
from zemble.dedup.model import Unit


def _is_comment(node: Node) -> bool:
    """Whether a node is a comment of any dialect."""
    return "comment" in node.type


def _leaves(node: Node, source: bytes, out: list[tuple[str, str]]) -> None:
    """Append every non-comment leaf of a subtree as a (node type, text) pair."""
    if _is_comment(node):
        return
    if node.child_count == 0:
        out.append((node.type, node_text(source, node)))
        return
    for child in node.children:
        _leaves(child, source, out)


def _declared_names(node: Node, source: bytes, profile: LanguageProfile, out: list[str]) -> None:
    """Collect every identifier a unit DECLARES: locals, parameters, captures, local types."""
    if _is_comment(node):
        return
    if node.type in profile.declared_name_fields:
        name = node.child_by_field_name("name")
        if name is not None:
            out.append(node_text(source, name))
    out.extend(profile.declared_names_extra(node, source))
    for child in node.children:
        _declared_names(child, source, profile, out)


def _calls_and_literals(
    node: Node, source: bytes, profile: LanguageProfile, calls: list[str], literals: list[str]
) -> None:
    """Collect the called names and the literal texts of a subtree."""
    if _is_comment(node):
        return
    if node.type in profile.literal_kinds:
        literals.append(node_text(source, node))
        return
    calls.extend(profile.call_names(node, source))
    for child in node.children:
        _calls_and_literals(child, source, profile, calls, literals)


def _hash(parts: list[str]) -> str:
    """Hash a token stream into a short stable hex digest."""
    digest = blake2b(digest_size=16)
    for part in parts:
        digest.update(part.encode("utf-8", "replace"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _renamed_stream(tokens: list[tuple[str, str]], declared: frozenset[str], profile: LanguageProfile) -> list[str]:
    """Replace declared identifiers by positional placeholders in first-seen order.

    An identifier straight after a member separator is a MEMBER name, never a local, whatever
    it is spelled like: without that rule every `this.key = key;` run in every constructor
    normalizes to the same stream and constructors become one giant clone class. Zig leans on
    the same rule for `.enumLiteral`, `error.Foo` and struct field access.

    :param tokens: The unit's (node type, text) token stream.
    :param declared: The names the unit declares.
    :param profile: The language whose separators decide what a member looks like.
    :return: The normalized stream.
    """
    placeholders: dict[str, str] = {}
    stream: list[str] = []
    previous = ""
    for node_type, text in tokens:
        is_member = previous in profile.member_separators
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
    profile: LanguageProfile,
    kind: str,
    name: str,
    declared: frozenset[str],
    include_text: bool,
    modifiers: tuple[str, ...],
    visibility: Visibility,
    container_visibility: Visibility,
    tokens: list[tuple[str, str]] | None = None,
    span: tuple[int, int] | None = None,
) -> Unit:
    """Build one unit from a subtree (or a pre-collected token run)."""
    if tokens is None:
        tokens = []
        _leaves(node, source, tokens)
    calls: list[str] = []
    literals: list[str] = []
    _calls_and_literals(node, source, profile, calls, literals)
    skeleton = tuple(text for node_type, text in tokens if node_type in profile.control_keywords)
    start, end = span if span is not None else (node.start_point[0] + 1, node.end_point[0] + 1)
    return Unit(
        file_path=file_path,
        start_line=start,
        end_line=end,
        kind=kind,
        name=name,
        token_count=len(tokens),
        exact_hash=_hash([text for _, text in tokens]),
        renamed_hash=_hash(_renamed_stream(tokens, declared, profile)),
        skeleton=skeleton,
        calls=tuple(sorted(set(calls))),
        literals=tuple(literals),
        text=node_text(source, node) if include_text else None,
        modifiers=modifiers,
        visibility=visibility,
        container_visibility=container_visibility,
    )


def _statements(block: Node) -> list[Node]:
    """Return the statement children of a block, comments dropped."""
    return [child for child in block.named_children if not _is_comment(child)]


def _blocks(node: Node, profile: LanguageProfile, out: list[Node]) -> None:
    """Collect every block inside a subtree, the subtree itself included."""
    if _is_comment(node):
        return
    if node.type in profile.block_kinds:
        out.append(node)
    for child in node.children:
        _blocks(child, profile, out)


def _window_units(
    body: Node,
    source: bytes,
    file_path: str,
    profile: LanguageProfile,
    name: str,
    declared: frozenset[str],
    container_visibility: Visibility,
    min_tokens: int,
    min_statements: int,
    max_statements: int,
) -> list[Unit]:
    """Build a unit for every window of consecutive statements inside a body."""
    units: list[Unit] = []
    blocks: list[Node] = []
    _blocks(body, profile, blocks)
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
                builder = _WindowBuilder(
                    window, source, file_path, profile, name, declared, container_visibility, tokens
                )
                units.append(builder.build())
    return units


class _WindowBuilder:
    """Builds one window unit out of a run of consecutive statements."""

    def __init__(
        self,
        statements: list[Node],
        source: bytes,
        file_path: str,
        profile: LanguageProfile,
        name: str,
        declared: frozenset[str],
        container_visibility: Visibility,
        tokens: list[tuple[str, str]],
    ) -> None:
        """Hold everything the window needs; :meth:`build` does the work."""
        self.statements = statements
        self.source = source
        self.file_path = file_path
        self.profile = profile
        self.name = name
        self.declared = declared
        self.container_visibility = container_visibility
        self.tokens = tokens

    def build(self) -> Unit:
        """Assemble the window's unit, whose own visibility is UNKNOWN: nothing can call a window."""
        calls: list[str] = []
        literals: list[str] = []
        for statement in self.statements:
            _calls_and_literals(statement, self.source, self.profile, calls, literals)
        skeleton = tuple(text for node_type, text in self.tokens if node_type in self.profile.control_keywords)
        return Unit(
            file_path=self.file_path,
            start_line=self.statements[0].start_point[0] + 1,
            end_line=self.statements[-1].end_point[0] + 1,
            kind="window",
            name=self.name,
            token_count=len(self.tokens),
            exact_hash=_hash([text for _, text in self.tokens]),
            renamed_hash=_hash(_renamed_stream(self.tokens, self.declared, self.profile)),
            skeleton=skeleton,
            calls=tuple(sorted(set(calls))),
            literals=tuple(literals),
            visibility=Visibility.UNKNOWN,
            container_visibility=self.container_visibility,
        )


class _UnitExtractor:
    """Walks one parsed file and emits every comparable unit in it."""

    def __init__(
        self,
        source: bytes,
        file_path: str,
        profile: LanguageProfile,
        min_tokens: int,
        min_statements: int,
        windows: bool,
        include_text: bool,
        max_window_statements: int,
    ) -> None:
        """Prepare an extractor for one file."""
        self.source = source
        self.file_path = file_path
        self.profile = profile
        self.min_tokens = min_tokens
        self.min_statements = min_statements
        self.windows = windows
        self.include_text = include_text
        self.max_window_statements = max_window_statements
        self.units: list[Unit] = []

    def run(self, root: Node) -> list[Unit]:
        """Extract every unit of the file, starting at its own top-level members."""
        self._visit_members(list(root.named_children), "", Visibility.PUBLIC)
        return self.units

    def _visit_members(self, nodes: list[Node], qualified: str, container_visibility: Visibility) -> None:
        """Emit units for a run of members, wrapper nodes folded in.

        The file itself is the outermost container and is PUBLIC; every nested container
        folds its own level into that with the narrower of the two, so a public class inside
        a package-private one never claims to be reachable.
        """
        members: list[Node] = []
        for node in nodes:
            if node.type in self.profile.flatten_kinds:
                members.extend(node.named_children)
            else:
                members.append(node)
        for member in members:
            kind = self.profile.member_kinds.get(member.type)
            if kind is not None:
                self._emit_member(member, qualified, kind, container_visibility)
                continue
            container = self.profile.container(member, self.source)
            if container is not None:
                inner = qualified
                if container.name is not None:
                    inner = f"{qualified}.{container.name}" if qualified else container.name
                folded = container.visibility.narrower(container_visibility)
                self._visit_members(list(container.body.named_children), inner, folded)

    def _emit_member(self, member: Node, qualified: str, kind: str, container_visibility: Visibility) -> None:
        """Emit the unit of one member declaration, its parameters counted as declared names."""
        body = self.profile.member_body(member)
        if body is None:
            return
        segment = self.profile.member_name(member, self.source)
        name = f"{qualified}.{segment}" if qualified else segment
        names: list[str] = []
        _declared_names(member, self.source, self.profile, names)
        declared = frozenset(names)
        tokens: list[tuple[str, str]] = []
        _leaves(body, self.source, tokens)
        if len(tokens) >= self.min_tokens:
            self.units.append(
                _make_unit(
                    body,
                    self.source,
                    self.file_path,
                    self.profile,
                    kind,
                    name,
                    declared,
                    self.include_text,
                    self.profile.modifiers(member, self.source),
                    self.profile.visibility(member, self.source),
                    container_visibility,
                    tokens=tokens,
                    span=(member.start_point[0] + 1, body.end_point[0] + 1),
                )
            )
        if self.windows:
            self.units.extend(
                _window_units(
                    body,
                    self.source,
                    self.file_path,
                    self.profile,
                    name,
                    declared,
                    container_visibility,
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
    """Extract every comparable unit from one source file, its language read off the path.

    :param source: Raw file bytes.
    :param file_path: Path as the report should print it, posix style.
    :param min_tokens: Smallest token count a unit may have; below it a getter is noise.
    :param min_statements: Smallest statement window considered.
    :param windows: Whether to emit statement windows beside whole bodies.
    :param include_text: Whether to keep the source text on body units (logic mode needs it).
    :param max_window_statements: Longest statement window considered, capping the O(n^2) window set.
    :return: The units, in source order.
    :raises ValueError: If no language profile claims the path's extension.
    :raises RuntimeError: If the language's grammar is unavailable on this platform.
    """
    profile = profile_for(file_path)
    if profile is None:
        raise ValueError(f"No duplication language profile for {file_path}")
    parser = profile.parser()
    if parser is None:
        raise RuntimeError(f"No tree-sitter {profile.name} grammar available")
    tree = parser.parse(source)
    extractor = _UnitExtractor(
        source, file_path, profile, min_tokens, min_statements, windows, include_text, max_window_statements
    )
    return extractor.run(tree.root_node)
