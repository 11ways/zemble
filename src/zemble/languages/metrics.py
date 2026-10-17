"""Per-callable and per-file code measurements read off any grammar's parse tree through a :class:`ComplexitySpec`.

Cognitive complexity follows Sonar's rules: structures that break the linear flow add 1 plus
the nesting level they sit at, else and else-if branches add 1 flat, a run of one boolean
operator adds 1, a labelled jump adds 1 and so does a call to the callable itself (bare, or on
one of its own receivers: `items.size()` inside `size()` calls another object's method). Nested
lambdas and functions open a nesting level and their contents count toward the enclosing
callable, as Sonar counts them.

A declaration's doc comment is the run of comments directly above it (`doc_comment`); the file
pass reads every comment once more for the lines that read as code, the unfinished-work markers
and a generator's header (`measure_file`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from functools import cache
from typing import Any, NamedTuple

from tree_sitter import Language, Node, Query, QueryCursor, Tree

from zemble.languages.spec import ComplexitySpec


class Called(NamedTuple):
    """One call a node makes, as the extractor that emits its CALLS edge reads it."""

    name: str
    #: The receiver as written (the generic reader folds every self spelling into `this`), or None for a bare call.
    receiver: str | None


#: Reads the calls one node makes; empty for a node that is no call.
CallReader = Callable[[Node, bytes], list[Called]]


def own_receivers(self_names: frozenset[str], type_name: str | None) -> frozenset[str]:
    """The receivers a call to the callable itself may be written on: its self spellings and its type's name."""
    return self_names | {type_name} if type_name else self_names


@dataclass(frozen=True)
class CallableMetrics:
    """What one callable body measures."""

    #: Source lines the body spans.
    lines: int
    #: Named children of every statement block inside the body, comments excluded.
    statements: int
    #: Sonar cognitive complexity.
    complexity: int
    #: Deepest nesting of increment and nesting kinds; 0 for straight-line code.
    nesting: int

    @property
    def is_boilerplate(self) -> bool:
        """Whether the body is one statement without control flow: an accessor, a delegation, a constant.

        Nesting rather than complexity decides: a one-line predicate (`return a && b;`) scores a
        complexity of 1 without breaking the linear flow.
        """
        return self.statements <= 1 and self.nesting == 0


@dataclass(frozen=True)
class FileMetrics:
    """What one whole source file measures; every field is one column of the graph's `files` table."""

    #: Source lines in the file; a final line without a newline counts.
    lines: int
    #: Distinct lines a comment touches.
    comment_lines: int
    #: Comment lines outside any doc comment that read as code (`reads_as_code`).
    commented_code_lines: int = 0
    #: Comment lines holding an unfinished-work marker (`MARKER`).
    marker_lines: int = 0
    #: Bytes of the longest line.
    longest_line: int = 0
    #: Whether a comment in the first `GENERATED_HEAD_LINES` lines carries a generator's header (`GENERATED`).
    generated: bool = False

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> FileMetrics:
        """Rebuild the measurements from a row carrying every `FILE_METRIC_COLUMNS` column."""
        values = {column: row[column] for column in FILE_METRIC_COLUMNS}
        values["generated"] = bool(values["generated"])
        return cls(**values)


#: The `files` columns the measurements are stored in, named after the fields: one home for both.
FILE_METRIC_COLUMNS: tuple[str, ...] = tuple(item.name for item in fields(FileMetrics))


class DocComment(NamedTuple):
    """The documentation a declaration carries."""

    #: Lines of the comments directly above it plus its docstring's.
    lines: int
    #: The 0-based rows of those comments, which the file pass never reads as commented-out code.
    rows: frozenset[int]


class _Walker:
    """One walk over one body, accumulating the measurements."""

    def __init__(
        self,
        spec: ComplexitySpec,
        source: bytes,
        block_kinds: frozenset[str],
        own_name: str | None,
        calls: CallReader | None,
        receivers: frozenset[str],
    ) -> None:
        self.spec = spec
        self.source = source
        self.block_kinds = block_kinds
        self.own_name = own_name if spec.recursion else None
        self.calls = calls
        self.receivers = receivers
        self.labels = dict(spec.labelled_jumps)
        self.statements = 0
        self.complexity = 0
        self.nesting = 0
        #: Ids of increment nodes that continue an else branch: the else already counted them.
        self.chained: set[int] = set()
        #: Kinds `_visit` must see; any other node only passes its level on to its children.
        self.counted = (
            spec.increment_kinds
            | spec.flat_increment_kinds
            | spec.nesting_kinds
            | spec.boolean_expression_kinds
            | frozenset(self.labels)
            | block_kinds
        )

    def _text(self, node: Node) -> str:
        return self.source[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def _operator(self, node: Node) -> str | None:
        """A binary expression's boolean operator text, or None when it has none."""
        operator = node.child_by_field_name(self.spec.operator_field)
        if operator is not None:
            text = self._text(operator)
            return text if text in self.spec.boolean_operators else None
        for child in node.children[1:-1]:
            text = self._text(child)
            if text in self.spec.boolean_operators:
                return text
        return None

    def _boolean_increment(self, node: Node) -> int:
        """1 when a boolean operator starts a new run, 0 when it continues its parent's run."""
        operator = self._operator(node)
        if operator is None:
            return 0
        parent = self._outer(node)
        if (
            parent is not None
            and parent.type in self.spec.boolean_expression_kinds
            and self._operator(parent) == operator
        ):
            return 0
        return 1

    def _outer(self, node: Node) -> Node | None:
        """A node's parent, looking up through single-child transparent wrappers."""
        parent = node.parent
        while parent is not None and parent.type in self.spec.transparent_kinds and len(self._content(parent)) == 1:
            parent = parent.parent
        return parent

    def _inner(self, node: Node) -> Node:
        """A node itself, or what the single-child transparent wrappers around it hold."""
        while node.type in self.spec.transparent_kinds:
            content = self._content(node)
            if len(content) != 1:
                break
            node = content[0]
        return node

    def _is_else_token(self, node: Node) -> bool:
        """Whether a child is the else keyword, as a bare token or a named leaf (Swift)."""
        if self.spec.else_token is None or node.named_children:
            return False
        return self._text(node) == self.spec.else_token

    def _chain(self, branch: Node) -> None:
        """Mark an increment node written as an else branch as an else-if: the else counted it."""
        inner = self._inner(branch)
        if inner.type in self.spec.increment_kinds:
            self.chained.add(inner.id)

    def _content(self, node: Node) -> list[Node]:
        return [child for child in node.named_children if child.type not in self.spec.comment_kinds]

    def _own_increment(self, node: Node, kind: str, level: int, *, root: bool) -> bool:
        """Add the node's own increment and answer whether it opens a nesting level."""
        spec = self.spec
        if kind in spec.increment_kinds:
            if node.id in self.chained:
                return False
            self.complexity += 1 + level
            return True
        if kind in spec.flat_increment_kinds:
            self.complexity += 1
            content = self._content(node)
            if len(content) == 1:
                # `else if` written as an else clause holding an if: one flat increment.
                self._chain(content[0])
            return False
        if kind in spec.boolean_expression_kinds:
            self.complexity += self._boolean_increment(node)
        label = self.labels.get(kind)
        if label is not None and any(child.type == label for child in node.named_children):
            self.complexity += 1
        if self._recurses(node):
            self.complexity += 1
        return kind in spec.nesting_kinds and not root

    def _recurses(self, node: Node) -> bool:
        """Whether a node calls the callable itself, once per written call.

        Some grammars nest one call inside another of the same name (Haskell's curried
        `apply` chain, Perl's bracketed bareword), so a call directly under its own twin is
        the same call and does not count again.
        """
        calls = self.calls
        if self.own_name is None or calls is None:
            return False
        if not self._calls_itself(calls, node):
            return False
        parent = node.parent
        return parent is None or not self._calls_itself(calls, parent)

    def _calls_itself(self, calls: CallReader, node: Node) -> bool:
        """Whether one of a node's calls names the callable, bare or on one of its own receivers."""
        return any(
            call.name == self.own_name and (call.receiver is None or call.receiver in self.receivers)
            for call in calls(node, self.source)
        )

    def walk(self, body: Node) -> None:
        """Measure a body and everything below it.

        Iterative because an else-if chain nests one level of tree per branch, which a long
        generated chain would push past Python's recursion limit.
        """
        stack: list[tuple[Node, int, bool]] = [(body, 0, True)]
        counted = self.counted
        # Without recursion to read, a node of no counted kind adds nothing: a fast path that
        # skips most of a tree. Its level was already recorded when its opener was visited.
        every = self.own_name is not None and self.calls is not None
        while stack:
            node, level, root = stack.pop()
            if not every and not root and node.type not in counted:
                stack.extend([(child, level, False) for child in reversed(node.named_children)])
                continue
            stack.extend(reversed(self._visit(node, level, root=root)))

    def _visit(self, node: Node, level: int, *, root: bool) -> list[tuple[Node, int, bool]]:
        """Measure one node and answer its children with the level each is visited at."""
        spec, kind = self.spec, node.type
        if kind in self.block_kinds:
            self.statements += len(self._content(node))
        opens = self._own_increment(node, kind, level, root=root)
        inner = level + 1 if opens else level
        if inner > self.nesting:
            self.nesting = inner
        is_increment = kind in spec.increment_kinds
        if not is_increment and not (opens and spec.condition_field is not None):
            # No else token to look for and no condition to keep outside: the common, fast case.
            return [(child, inner, False) for child in node.named_children]
        after_else = False
        children: list[tuple[Node, int, bool]] = []
        for index, child in enumerate(node.children):
            if is_increment and self._is_else_token(child):
                after_else = True
                continue
            if not child.is_named:
                continue
            if after_else and child.type not in spec.comment_kinds:
                after_else = False
                self.complexity += 1
                self._chain(child)
            outer = (
                opens and spec.condition_field is not None and node.field_name_for_child(index) == spec.condition_field
            )
            children.append((child, level if outer else inner, False))
        return children


def measure_callable(
    body: Node,
    spec: ComplexitySpec,
    own_name: str | None,
    source: bytes,
    *,
    block_kinds: frozenset[str],
    calls: CallReader | None = None,
    receivers: frozenset[str] = frozenset(),
) -> CallableMetrics:
    """Measure one callable body.

    A body whose text never spells its own name cannot call itself, so its walk skips reading
    calls altogether: every call reader answers names read off the source text.

    @param own_name the callable's simple name, for recursion; None counts no recursion.
    @param block_kinds the language's statement block kinds (`LanguageSpec.block_kinds`, or a
        hand-written profile's), so statements are counted where duplication windows are cut.
    @param calls the language's call reader; without it recursion is not counted.
    @param receivers the receivers a self call is written on (`own_receivers`); a call on any
        other receiver is another object's method of the same name. A Go method's receiver
        parameter is a name the writer chose, so `s.walk()` inside `walk` is not counted.
    """
    if own_name is not None and source.find(own_name.encode(), body.start_byte, body.end_byte) < 0:
        own_name = None
    walker = _Walker(spec, source, block_kinds, own_name, calls, receivers)
    walker.walk(body)
    lines = body.end_point[0] - body.start_point[0] + 1
    return CallableMetrics(lines, walker.statements, walker.complexity, walker.nesting)


@cache
def _comment_query(language: Language, kinds: frozenset[str]) -> Query:
    """The query capturing every comment node of a grammar, compiled once per grammar."""
    return Query(language, "[" + " ".join(f"({kind})" for kind in sorted(kinds)) + "] @comment")


def _comments(tree: Tree, spec: ComplexitySpec) -> list[Node]:
    """Every comment node of a file, found by a tree-sitter query.

    The query runs in C: a Python walk over every node of every file was a fifth of a whole graph
    extraction. A query matches named nodes only, so every comment kind must be a named node,
    which `tests/test_languages.py` holds.
    """
    return QueryCursor(_comment_query(tree.language, spec.comment_kinds)).captures(tree.root_node).get("comment", [])


def _last_row(node: Node) -> int:
    """The last row a node's text is on; a line comment that swallows its newline ends at column 0 of the next."""
    last = node.end_point[0]
    return last - 1 if node.end_point[1] == 0 and last > node.start_point[0] else last


def comment_lines(tree: Tree, spec: ComplexitySpec) -> int:
    """Count the distinct source lines a comment touches in a whole file, trailing comments included."""
    lines: set[int] = set()
    for node in _comments(tree, spec):
        lines.update(range(node.start_point[0], _last_row(node) + 1))
    return len(lines)


def doc_comment(
    declaration: Node,
    spec: ComplexitySpec,
    *,
    body: Node | None = None,
    attached_kinds: frozenset[str] = frozenset(),
) -> DocComment:
    """Measure the comments written directly above a declaration, and its docstring.

    The comments are the declaration's preceding siblings, each ending on the line above the next;
    one that shares its first line with code before it trails that code and ends the run. A
    declaration that opens its parent (Python's first member opens the class block, which the
    grammar places after the comments) is looked through to the parent's siblings, and the
    attached kinds (Rust's `#[attr]` lines) are passed over without counting. A comment whose every
    line reads as code (`reads_as_code`) is disabled code, not documentation, and ends the run.

    @param declaration the declaration node, or the wrapper the spec wraps it in.
    @param body the body whose first statement may be a docstring (`ComplexitySpec.docstring_kinds`).
    @param attached_kinds sibling kinds that belong to the declaration (`LanguageSpec.annotation_kinds`).
    """
    rows: set[int] = set()
    node, top = declaration, declaration.start_point[0]
    while True:
        previous = node.prev_named_sibling
        if previous is None:
            parent = node.parent
            if parent is None or parent.start_byte != node.start_byte:
                break
            node = parent
            continue
        if _last_row(previous) < top - 1:
            break
        if previous.type in spec.comment_kinds:
            before = previous.prev_sibling
            trails = before is not None and before.type not in spec.comment_kinds
            if trails and _last_row(before) == previous.start_point[0]:
                break
            if _commented_out(previous):
                break
            rows.update(range(previous.start_point[0], _last_row(previous) + 1))
        elif previous.type not in attached_kinds:
            break
        node, top = previous, previous.start_point[0]
    return DocComment(len(rows) + _docstring_lines(body, spec), frozenset(rows))


def _docstring_lines(body: Node | None, spec: ComplexitySpec) -> int:
    """Lines of a body's docstring: its first statement when that is, or only holds, a string."""
    if body is None or not spec.docstring_kinds:
        return 0
    first = next((child for child in body.named_children if child.type not in spec.comment_kinds), None)
    if first is None:
        return 0
    inner = first.named_children
    if first.type not in spec.docstring_kinds and not (len(inner) == 1 and inner[0].type in spec.docstring_kinds):
        return 0
    return first.end_point[0] - first.start_point[0] + 1


#: Unfinished-work markers (Sonar S1134, S1135) as whole upper-case words: lower case is prose ("a hack").
MARKER = re.compile(r"\b(?:TODO|FIXME|XXX|HACK)\b")
#: What a generator writes into the header of a file it produced (`protoc`, `go generate`, Phabricator's tag).
GENERATED = re.compile(
    r"@generated|do not edit|generated by|automatically generated|autogenerated|code generated", re.IGNORECASE
)
#: How many leading lines a generator's header is looked for in.
GENERATED_HEAD_LINES = 30

#: Comment openers of every measured grammar (`//`, `/*`, `*`, `#`, `;`, `%`, `--`, `!`, `<!--`, `{-`, `(*`).
_OPENER = re.compile(r"^(?:<!--|\{-|\(\*|[/*#;%!-])+")
#: Comment closers (`*/`, `-->`, `-}`, `*)`).
_CLOSER = re.compile(r"(?:\*+/|-->|-\}|\*\))$")
#: Line endings only code writes: a statement end, a block opener or closer, an argument run on.
_CODE_ENDINGS = (";", "{", "}", "),")
#: Keywords that open a block written with a trailing colon (Python, and a C `case`'s label).
_BLOCK_KEYWORDS = frozenset({"if", "elif", "else", "for", "while", "def", "class", "try", "except", "with", "case"})
#: Characters a keyword line ending in a colon must hold to read as code, not as a prose heading ("for example:").
_CODE_CHARACTERS = frozenset("(=[.")
#: A word of prose: letters, an apostrophe, and at most one trailing punctuation mark.
_PROSE_WORD = re.compile(r"[A-Za-z']+[,.;:!?]?")
#: Consecutive prose words that make a line a sentence however it ends.
_PROSE_RUN = 4


def _uncommented(comment_line: str) -> str:
    """One line of a comment with its comment markers and surrounding blanks stripped."""
    return _CLOSER.sub("", _OPENER.sub("", comment_line.strip())).strip()


def _commented_out(comment: Node) -> bool:
    """Whether every line of a comment that says anything reads as code: disabled code is no documentation."""
    said = [line for line in (comment.text or b"").decode("utf-8", "replace").split("\n") if _uncommented(line)]
    return bool(said) and all(reads_as_code(line) for line in said)


def reads_as_code(comment_line: str) -> bool:
    """Whether one line of a comment reads as commented-out code, after its comment markers are stripped.

    A heuristic in place of Sonar S125's re-parse: the line must end the way only code ends (`;`, `{`,
    `}`, `),`), or open a keyword block with a colon and hold one of `(=[.`; it must not hold a URL,
    a javadoc inline tag (`{@link X}`) or a run of `_PROSE_RUN` prose words.
    """
    line = _uncommented(comment_line)
    if not line or "://" in line or "{@" in line:
        return False
    words = line.split()
    run = 0
    for word in words:
        run = run + 1 if _PROSE_WORD.fullmatch(word) else 0
        if run >= _PROSE_RUN:
            return False
    if line.endswith(_CODE_ENDINGS):
        return True
    keyword = re.split(r"[\s(]", line, maxsplit=1)[0]
    return (
        line.endswith(":")
        and keyword in _BLOCK_KEYWORDS
        and len(line) > len(keyword)
        and line[len(keyword)] in " ("
        and any(character in _CODE_CHARACTERS for character in line)
    )


def measure_file(
    tree: Tree, source: bytes, spec: ComplexitySpec, doc_rows: frozenset[int] = frozenset()
) -> FileMetrics:
    """Measure a whole parsed file: its lines, its comments and what they hold.

    Lines are counted from the bytes, since a grammar's root node may stop short of trailing blank lines.

    @param doc_rows the 0-based rows of every doc comment the extractor attributed (`doc_comment`), never
        read as commented-out code: an example in a docblock is documentation.
    """
    lines = source.count(b"\n") + (1 if source and not source.endswith(b"\n") else 0)
    touched: set[int] = set()
    code: set[int] = set()
    markers: set[int] = set()
    generated = False
    for node in _comments(tree, spec):
        first, last = node.start_point[0], _last_row(node)
        touched.update(range(first, last + 1))
        text = source[node.start_byte : node.end_byte].decode("utf-8", "replace")
        for row, line in enumerate(text.split("\n")[: last - first + 1], start=first):
            if MARKER.search(line):
                markers.add(row)
            if row < GENERATED_HEAD_LINES and not generated and GENERATED.search(line):
                generated = True
            if row not in doc_rows and reads_as_code(line):
                code.add(row)
    longest = max((len(line) for line in source.split(b"\n")), default=0)
    return FileMetrics(lines, len(touched), len(code), len(markers), longest, generated)


__all__ = [
    "FILE_METRIC_COLUMNS",
    "GENERATED",
    "GENERATED_HEAD_LINES",
    "MARKER",
    "CallReader",
    "CallableMetrics",
    "Called",
    "DocComment",
    "FileMetrics",
    "comment_lines",
    "doc_comment",
    "measure_callable",
    "measure_file",
    "own_receivers",
    "reads_as_code",
]
