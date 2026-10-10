"""Turning source into comparable units: whole bodies and statement windows.

Every syntax fact lives in a :class:`~zemble.dedup.languages.base.LanguageProfile`; this
module only knows how to walk members, hash token streams and cut statement windows, which
is why adding a language never touches it.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from hashlib import blake2b
from typing import NamedTuple

from tree_sitter import Node

from zemble.dedup.languages import (
    CONSTANT_HOLE,
    KeyArguments,
    LanguageProfile,
    SiteKind,
    Visibility,
    VocabularyFact,
    literal_hole,
    node_text,
    profile_for,
)
from zemble.dedup.model import Unit

#: The node type a hole token carries in a holed stream; no grammar has a kind spelled like it.
HOLE = "<hole>"
#: The erased hole an idiom key writes for every typed hole and every name the site uses once.
_ERASED = "_"
#: Smallest whole body the holed and re-implementation channels compare, in tokens.
SHAPED_MIN_TOKENS = 8
#: Smallest call chain an idiom site may be, in tokens, and the calls it must make.
IDIOM_MIN_TOKENS = 8
IDIOM_MIN_CALLS = 2
#: Calls sharing one receiver and first argument beyond which the group is a builder, not a pair idiom.
_PAIR_MAX_NAMES = 3
#: Site kinds whose values are a set: keyed by their sorted values, so declaration order never moves a key.
_SET_KINDS = frozenset({SiteKind.SET, SiteKind.SWITCH})
#: Tokens a rendering glues to their left neighbour, and those it glues their right neighbour to.
_GLUE_LEFT = frozenset({".", ",", ";", ")", "]", "(", "::"})
_GLUE_RIGHT = frozenset({".", "(", "[", "::", "@"})


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


class HoledToken(NamedTuple):
    """One token of a holed stream."""

    kind: str
    text: str
    start: int
    #: What a hole stands for in the source (`"scope"`, `Egress.NONE`); "" for every other token.
    value: str = ""


def _holed_leaves(node: Node, source: bytes, profile: LanguageProfile, out: list[HoledToken]) -> None:
    """Append every non-comment leaf with its start byte, each literal as one typed hole."""
    if _is_comment(node):
        return
    if node.type in profile.literal_kinds:
        out.append(HoledToken(HOLE, literal_hole(node.type), node.start_byte, node_text(source, node)))
        return
    if node.child_count == 0:
        out.append(HoledToken(node.type, node_text(source, node), node.start_byte))
        return
    for child in node.children:
        _holed_leaves(child, source, profile, out)


def holed_tokens(node: Node, source: bytes, profile: LanguageProfile) -> list[HoledToken]:
    """A subtree's tokens with literals as typed holes and qualified constant references folded into one hole.

    A constant is an identifier the profile's shape hooks call one, not followed by a call or a
    member access; `Egress.NONE` and `NONE` both become `<const>`. A profile without shape hooks
    folds no constant.

    :param node: The subtree.
    :param source: The file's bytes.
    :param profile: The language.
    :return: The tokens, holes typed `HOLE`.
    """
    raw: list[HoledToken] = []
    _holed_leaves(node, source, profile, raw)
    hooks = profile.shapes
    if hooks is None:
        return raw
    out: list[HoledToken] = []
    for index, token in enumerate(raw):
        following = raw[index + 1].text if index + 1 < len(raw) else ""
        if token.kind != "identifier" or following in {"(", "."} or not hooks.is_constant_name(token.text):
            out.append(token)
            continue
        value = token.text
        start = token.start
        while len(out) >= 2 and out[-1].text == "." and out[-2].kind == "identifier":
            value = f"{out[-2].text}.{value}"
            start = out[-2].start
            del out[-2:]
        out.append(HoledToken(HOLE, CONSTANT_HOLE, start, value))
    return out


def render_stream(stream: list[str]) -> str:
    """Render a normalized token stream as readable code, one space between tokens unless punctuation glues them."""
    pieces: list[str] = []
    previous = ""
    for token in stream:
        if pieces and token not in _GLUE_LEFT and previous not in _GLUE_RIGHT:
            pieces.append(" ")
        pieces.append(token)
        previous = token
    return "".join(pieces)


def _erased(stream: list[str]) -> list[str]:
    """An idiom key: every typed hole and every placeholder the site uses only once becomes the one erased hole.

    A placeholder used twice stays: it is the data flow of the site (`$0.getLocales(), $0.getMessageResolver()`).
    """
    counts: dict[str, int] = {}
    for token in stream:
        if token.startswith("$"):
            counts[token] = counts.get(token, 0) + 1
    return [
        _ERASED
        if (token.startswith("<") and token.endswith(">") and token[1:-1].isalpha()) or counts.get(token, 0) == 1
        else token
        for token in stream
    ]


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


@dataclass(frozen=True, slots=True)
class ShapeRequest:
    """Which sub-body outputs one extraction produces beside the clone units."""

    #: Every body of at least `SHAPED_MIN_TOKENS` tokens with its holed shape (holed, re-implementation).
    shaped: bool = False
    #: Call-chain and paired-call sites (idiom).
    idioms: bool = False
    #: Constants, literal uses, value sets and regex literals (vocabulary).
    vocabulary: bool = False
    #: Key-argument patterns (:class:`~zemble.dedup.languages.KeyArguments`) whose strings are no values.
    copy_keys: tuple[str, ...] = ()

    @property
    def any(self) -> bool:
        """Whether anything beyond the clone units is wanted."""
        return self.shaped or self.idioms or self.vocabulary


@dataclass
class FileUnits:
    """Everything one file yields: clone units, shaped bodies, and idiom and vocabulary sites."""

    units: list[Unit] = field(default_factory=list)
    shaped: list[Unit] = field(default_factory=list)
    sites: list[Unit] = field(default_factory=list)


def _site_unit(
    file_path: str,
    kind: SiteKind,
    name: str,
    span: tuple[int, int],
    stream: list[str],
    raw: list[str],
    shape_hash: str,
    calls: tuple[str, ...] = (),
    literals: tuple[str, ...] = (),
    visibility: Visibility = Visibility.UNKNOWN,
    container_visibility: Visibility = Visibility.UNKNOWN,
) -> Unit:
    """One idiom or vocabulary site; it has no token stream of the clone kinds, so those hashes stay empty."""
    return Unit(
        file_path=file_path,
        start_line=span[0],
        end_line=span[1],
        kind=kind.value,
        name=name,
        token_count=len(stream),
        exact_hash=_hash(raw) if raw else "",
        renamed_hash=_hash(stream) if raw else "",
        skeleton=(),
        calls=calls,
        literals=literals,
        visibility=visibility,
        container_visibility=container_visibility,
        shape_hash=shape_hash,
        shape=render_stream(stream) if raw else ", ".join(literals),
    )


class _IdiomSites:
    """Cuts the call-chain and paired-call sites out of one body."""

    def __init__(
        self,
        body: Node,
        source: bytes,
        file_path: str,
        profile: LanguageProfile,
        name: str,
        declared: frozenset[str],
        holed: list[HoledToken],
    ) -> None:
        """Hold one body and its holed tokens."""
        assert profile.shapes is not None
        self.hooks = profile.shapes
        self.body = body
        self.source = source
        self.file_path = file_path
        self.profile = profile
        self.name = name
        self.declared = declared
        self.holed = holed
        self.starts = [token.start for token in holed]
        self.calls: list[Node] = []
        self.counts: dict[int, int] = {}
        self._count(body)
        #: Ids of the calls that are another call's receiver: inner links of a chain.
        self.receivers = {
            receiver.id
            for call in self.calls
            if (receiver := self.hooks.call_parts(call)[0]) is not None and receiver.type in self.hooks.call_kinds
        }

    def _count(self, node: Node) -> int:
        """Record every call node below a node, and how many calls each one's subtree holds."""
        if _is_comment(node):
            return 0
        total = sum(self._count(child) for child in node.children)
        if node.type in self.hooks.call_kinds:
            # AIDEV-NOTE: a static call on a standard-library type (`String.valueOf`, `Boolean.TRUE.equals`) is the
            # language itself; it never counts toward the calls that make a chain an idiom.
            total += 0 if self.hooks.standard_receiver(node, self.source) else 1
            self.calls.append(node)
            self.counts[node.id] = total
        return total

    def _sole_call(self) -> int | None:
        """The id of the call that IS the body (`return a.b(c);`), or None."""
        statements = [child for child in self.body.named_children if not _is_comment(child)]
        if len(statements) != 1:
            return None
        inner = [child for child in statements[0].named_children if not _is_comment(child)]
        if len(inner) == 1 and inner[0].type in self.hooks.call_kinds:
            return inner[0].id
        return None

    def sites(self) -> list[Unit]:
        """Every chain site, receiver-holed tail site and paired-call site of the body."""
        sole = self._sole_call()
        units: list[Unit] = []
        for call in self.calls:
            count = self.counts[call.id]
            low = bisect_left(self.starts, call.start_byte)
            high = bisect_left(self.starts, call.end_byte)
            if count >= IDIOM_MIN_CALLS and high - low >= IDIOM_MIN_TOKENS:
                kind = SiteKind.WRAPPER if call.id == sole else SiteKind.CHAIN
                units.append(self._chain(call, kind, self.holed[low:high]))
            receiver, _name, _arguments = self.hooks.call_parts(call)
            if receiver is None or receiver.type not in self.hooks.call_kinds:
                continue
            # The tail of a chain whose head varies: `<anything>.resolve($0.getLocales(), $0.getMessageResolver())`.
            if count - self.counts[receiver.id] >= IDIOM_MIN_CALLS:
                rest = self.holed[bisect_left(self.starts, receiver.end_byte) : high]
                tail = [HoledToken(HOLE, _ERASED, receiver.start_byte), *rest]
                if len(tail) >= IDIOM_MIN_TOKENS:
                    units.append(self._chain(call, SiteKind.CHAIN, tail, receiver))
            if call.id not in self.receivers:
                units.extend(self._suffixes(call, receiver, count, high))
        units.extend(self._pairs())
        return units

    def _suffixes(self, call: Node, receiver: Node, count: int, high: int) -> list[Unit]:
        """The longer tails of an outermost chain, `<anything>.offset(n).limit(1000).all()`, one per deeper receiver."""
        units = []
        deeper, _name, _arguments = self.hooks.call_parts(receiver)
        while deeper is not None and deeper.type in self.hooks.call_kinds:
            if count - self.counts[deeper.id] < IDIOM_MIN_CALLS:
                break
            rest = self.holed[bisect_left(self.starts, deeper.end_byte) : high]
            tail = [HoledToken(HOLE, _ERASED, deeper.start_byte), *rest]
            if len(tail) >= IDIOM_MIN_TOKENS:
                units.append(self._chain(call, SiteKind.CHAIN, tail, deeper))
            deeper, _name, _arguments = self.hooks.call_parts(deeper)
        return units

    def _chain(self, call: Node, kind: SiteKind, tokens: list[HoledToken], holed_receiver: Node | None = None) -> Unit:
        """One chain site from its holed tokens; its literals are what its holes stand for, in order.

        A tail site's calls leave out the ones its holed receiver makes.
        """
        stream = _renamed_stream([(token.kind, token.text) for token in tokens], self.declared, self.profile)
        names: list[str] = []
        _calls_and_literals(call, self.source, self.profile, names, [])
        if holed_receiver is not None:
            inside: list[str] = []
            _calls_and_literals(holed_receiver, self.source, self.profile, inside, [])
            remaining = Counter(names) - Counter(inside)
            names = list(remaining.elements())
        return _site_unit(
            self.file_path,
            kind,
            self.name,
            (call.start_point[0] + 1, call.end_point[0] + 1),
            stream,
            [token.text for token in tokens],
            _hash(["idiom", *_erased(stream)]),
            calls=tuple(sorted(set(names))),
            literals=tuple(token.value for token in tokens if token.value),
        )

    def _pairs(self) -> list[Unit]:
        """Two different calls on one receiver with the same first argument, in source order."""
        groups: dict[tuple[str, str], list[tuple[str, int, Node]]] = {}
        for call in sorted(self.calls, key=lambda node: node.start_byte):
            receiver, name, arguments = self.hooks.call_parts(call)
            if receiver is None or name is None or not arguments:
                continue
            key = (node_text(self.source, receiver), node_text(self.source, arguments[0]))
            groups.setdefault(key, []).append((node_text(self.source, name), len(arguments), call))
        units: list[Unit] = []
        for entries in groups.values():
            first: dict[str, tuple[int, Node]] = {}
            for called, arity, call in entries:
                first.setdefault(called, (arity, call))
            if not 2 <= len(first) <= _PAIR_MAX_NAMES:
                continue
            ordered = list(first.items())
            for index, (left, (left_arity, left_call)) in enumerate(ordered):
                for right, (right_arity, right_call) in ordered[index + 1 :]:
                    stream = [_pair_call(left, left_arity), "..", _pair_call(right, right_arity)]
                    units.append(
                        _site_unit(
                            self.file_path,
                            SiteKind.PAIR,
                            self.name,
                            (left_call.start_point[0] + 1, right_call.end_point[0] + 1),
                            stream,
                            [left, right],
                            _hash(["pair", left, str(left_arity), right, str(right_arity)]),
                            calls=tuple(sorted({left, right})),
                        )
                    )
        return units


def _pair_call(name: str, arity: int) -> str:
    """One half of a paired-call shape: `setAttribute(@k, _)`."""
    return f"{name}({', '.join(['@k', *[_ERASED] * (arity - 1)])})"


def _constant_names(holed: list[HoledToken], profile: LanguageProfile) -> tuple[str, ...]:
    """Every constant a body names, as written: its constant holes and the constants it calls on (`SESSIONS.remove`)."""
    hooks = profile.shapes
    if hooks is None:
        return ()
    return tuple(
        token.value if token.text == CONSTANT_HOLE else token.text
        for token in holed
        if token.text == CONSTANT_HOLE or (token.kind == "identifier" and hooks.is_constant_name(token.text))
    )


@lru_cache(maxsize=8)
def _key_arguments(patterns: tuple[str, ...]) -> KeyArguments:
    """One parsed key-argument table per pattern set, shared by every file of a run."""
    return KeyArguments(patterns)


def _vocabulary_sites(facts: list[VocabularyFact], file_path: str) -> list[Unit]:
    """The units of one file's vocabulary facts, each keyed by its kind and value (a set by its sorted values)."""
    units = []
    for fact in facts:
        values = tuple(sorted(set(fact.values))) if fact.kind in _SET_KINDS else fact.values
        unit = _site_unit(
            file_path,
            fact.kind,
            fact.name,
            (fact.start_line, fact.end_line),
            list(fact.values),
            [],
            _hash([fact.kind.value, *values]),
            literals=fact.values,
            visibility=fact.visibility,
            container_visibility=fact.container_visibility,
        )
        unit = replace(unit, declares=fact.declares, constants=fact.constants)
        units.append(replace(unit, shape=fact.detail) if fact.detail else unit)
    return units


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
        request: ShapeRequest,
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
        self.request = request
        self.result = FileUnits()
        self.units = self.result.units

    def run(self, root: Node) -> FileUnits:
        """Extract every unit of the file, starting at its own top-level members."""
        self._visit_members(list(root.named_children), "", Visibility.PUBLIC)
        if self.request.vocabulary and self.profile.shapes is not None:
            facts = self.profile.shapes.vocabulary(root, self.source, _key_arguments(self.request.copy_keys))
            self.result.sites.extend(_vocabulary_sites(facts, self.file_path))
        return self.result

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
            if _is_comment(member):
                continue
            kind = self.profile.member_kind(member, self.source)
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
                continue
            if self.profile.descend is not None and self.profile.descend(member, self.source):
                self._visit_members(list(member.named_children), qualified, container_visibility)

    def _emit_member(self, member: Node, qualified: str, kind: str, container_visibility: Visibility) -> None:
        """Emit the unit of one member declaration, its parameters counted as declared names."""
        body = self.profile.member_body(member, self.source)
        if body is None:
            return
        segment = self.profile.member_name(member, self.source)
        name = f"{qualified}.{segment}" if qualified else segment
        names: list[str] = []
        _declared_names(member, self.source, self.profile, names)
        declared = frozenset(names)
        tokens: list[tuple[str, str]] = []
        _leaves(body, self.source, tokens)
        clone = len(tokens) >= self.min_tokens
        shaped = self.request.shaped and len(tokens) >= SHAPED_MIN_TOKENS
        if clone or shaped:
            unit = _make_unit(
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
            if clone:
                self.units.append(unit)
        holed = holed_tokens(body, self.source, self.profile) if shaped or self.request.idioms else []
        if shaped:
            stream = _renamed_stream([(token.kind, token.text) for token in holed], declared, self.profile)
            hooks = self.profile.shapes
            self.result.shaped.append(
                replace(
                    unit,
                    shape_hash=_hash(stream),
                    shape=render_stream(stream),
                    forwards_to=hooks.forward_target(member, body, self.source) if hooks is not None else None,
                    implements_contract=hooks is not None and hooks.implements_contract(member, self.source),
                    signature=hooks.signature(member, self.source) if hooks is not None else None,
                    constants=_constant_names(holed, self.profile),
                    delegates=hooks is not None and hooks.delegates(member, body, self.source),
                )
            )
        if self.request.idioms and self.profile.shapes is not None:
            sites = _IdiomSites(body, self.source, self.file_path, self.profile, name, declared, holed)
            self.result.sites.extend(sites.sites())
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


def extract_file(
    source: bytes,
    file_path: str,
    *,
    min_tokens: int = 30,
    min_statements: int = 6,
    windows: bool = True,
    include_text: bool = False,
    max_window_statements: int = 24,
    shaped: bool = False,
    idioms: bool = False,
    vocabulary: bool = False,
    copy_keys: tuple[str, ...] = (),
) -> FileUnits:
    """Extract every comparable unit from one source file, its language read off the path.

    :param source: Raw file bytes.
    :param file_path: Path as the report should print it, posix style.
    :param min_tokens: Smallest token count a clone unit may have; below it a getter is noise.
    :param min_statements: Smallest statement window considered.
    :param windows: Whether to emit statement windows beside whole bodies.
    :param include_text: Whether to keep the source text on body units (logic mode needs it).
    :param max_window_statements: Longest statement window considered, capping the O(n^2) window set.
    :param shaped: Whether to emit every body of `SHAPED_MIN_TOKENS` tokens with its holed shape.
    :param idioms: Whether to emit call-chain and paired-call sites.
    :param vocabulary: Whether to emit constants, literal uses, value sets and regex literals.
    :param copy_keys: Key-argument patterns whose strings are catalog keys, never vocabulary values.
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
        source,
        file_path,
        profile,
        min_tokens,
        min_statements,
        windows,
        include_text,
        max_window_statements,
        ShapeRequest(shaped=shaped, idioms=idioms, vocabulary=vocabulary, copy_keys=tuple(copy_keys)),
    )
    return extractor.run(tree.root_node)


def extract_units(source: bytes, file_path: str, **options: object) -> list[Unit]:
    """Extract the clone units (bodies and windows) of one source file; see :func:`extract_file` for the options."""
    return extract_file(source, file_path, **options).units  # type: ignore[arg-type]
