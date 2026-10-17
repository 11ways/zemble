"""Per-callable code measurements read off any grammar's parse tree through a :class:`ComplexitySpec`.

Cognitive complexity follows Sonar's rules: structures that break the linear flow add 1 plus
the nesting level they sit at, else and else-if branches add 1 flat, a run of one boolean
operator adds 1, a labelled jump adds 1 and so does a call to the callable itself. Nested
lambdas and functions open a nesting level and their contents count toward the enclosing
callable, as Sonar counts them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tree_sitter import Node

from zemble.languages.spec import ComplexitySpec

#: Reads the names one node calls; empty for a node that is no call.
CallNames = Callable[[Node, bytes], list[str]]


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


class _Walker:
    """One walk over one body, accumulating the measurements."""

    def __init__(
        self,
        spec: ComplexitySpec,
        source: bytes,
        block_kinds: frozenset[str],
        own_name: str | None,
        call_names: CallNames | None,
    ) -> None:
        self.spec = spec
        self.source = source
        self.block_kinds = block_kinds
        self.own_name = own_name if spec.recursion else None
        self.call_names = call_names
        self.labels = dict(spec.labelled_jumps)
        self.statements = 0
        self.complexity = 0
        self.nesting = 0
        #: Ids of increment nodes that continue an else branch: the else already counted them.
        self.chained: set[int] = set()

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

    def _own_increment(self, node: Node, level: int, *, root: bool) -> bool:
        """Add the node's own increment and answer whether it opens a nesting level."""
        spec, kind = self.spec, node.type
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
        if self.own_name is None or self.call_names is None:
            return False
        if self.own_name not in self.call_names(node, self.source):
            return False
        parent = node.parent
        return parent is None or self.own_name not in self.call_names(parent, self.source)

    def walk(self, body: Node) -> None:
        """Measure a body and everything below it.

        Iterative because an else-if chain nests one level of tree per branch, which a long
        generated chain would push past Python's recursion limit.
        """
        stack: list[tuple[Node, int, bool]] = [(body, 0, True)]
        while stack:
            node, level, root = stack.pop()
            stack.extend(reversed(self._visit(node, level, root=root)))

    def _visit(self, node: Node, level: int, *, root: bool) -> list[tuple[Node, int, bool]]:
        """Measure one node and answer its children with the level each is visited at."""
        spec = self.spec
        if node.type in self.block_kinds:
            self.statements += len(self._content(node))
        opens = self._own_increment(node, level, root=root)
        inner = level + 1 if opens else level
        self.nesting = max(self.nesting, inner)
        is_increment = node.type in spec.increment_kinds
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
    call_names: CallNames | None = None,
) -> CallableMetrics:
    """Measure one callable body.

    @param own_name the callable's simple name, for recursion; None counts no recursion.
    @param block_kinds the language's statement block kinds (`LanguageSpec.block_kinds`, or a
        hand-written profile's), so statements are counted where duplication windows are cut.
    @param call_names the language's call reader; without it recursion is not counted.
    """
    walker = _Walker(spec, source, block_kinds, own_name, call_names)
    walker.walk(body)
    lines = body.end_point[0] - body.start_point[0] + 1
    return CallableMetrics(lines, walker.statements, walker.complexity, walker.nesting)


def comment_lines(root: Node, spec: ComplexitySpec) -> int:
    """Count the distinct source lines a comment touches in a whole file, trailing comments included."""
    lines: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in spec.comment_kinds:
            last = node.end_point[0]
            if node.end_point[1] == 0 and last > node.start_point[0]:
                # A line comment that swallows its newline ends at column 0 of the next line.
                last -= 1
            lines.update(range(node.start_point[0], last + 1))
            continue
        stack.extend(node.children)
    return len(lines)


__all__ = ["CallNames", "CallableMetrics", "comment_lines", "measure_callable"]
