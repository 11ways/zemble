"""A tiny path language for reaching into tree-sitter nodes, so a grammar spec is data.

A path is a `/`-separated list of steps evaluated left to right from one node; the first
step that yields nothing makes the whole path yield nothing. Alternatives are separated by
`|` and the first one that resolves wins. Steps:

    name        the child in field `name`
    name+       every child in field `name` (a list step; must be last)
    name*       follow field `name` while it exists and return the deepest node
    name~kind   follow field `name` until a node of type `kind` is reached
    @kind       the first named child of type `kind`
    @kind*      every named child of type `kind` (a list step; must be last)
    @*          every named child (a list step; must be last)
    #n          the n-th named child (negative counts from the end)
    **kind      the first descendant of type `kind`, breadth first
    <  /  >     the previous / next named sibling
    ..          the parent
    .           the node itself
"""

from __future__ import annotations

from collections import deque
from functools import cache

from tree_sitter import Node

_ALTERNATIVE = "|"
_SEPARATOR = "/"


def _named_children(node: Node) -> list[Node]:
    """The named children of a node, comments excluded."""
    return [child for child in node.named_children if "comment" not in child.type]


def _descendant(node: Node, wanted: str) -> Node | None:
    """The first descendant of a type, breadth first."""
    queue = deque(node.named_children)
    while queue:
        candidate = queue.popleft()
        if candidate.type == wanted:
            return candidate
        queue.extend(candidate.named_children)
    return None


def _by_type(node: Node, step: str) -> Node | list[Node] | None:
    """An `@kind`, `@kind*` or `@*` step."""
    wanted = step[1:]
    if wanted == "*":
        return _named_children(node)
    if wanted.endswith("*"):
        wanted = wanted[:-1]
        return [child for child in _named_children(node) if child.type == wanted]
    return next((child for child in _named_children(node) if child.type == wanted), None)


def _by_index(node: Node, step: str) -> Node | None:
    """A `#n` step."""
    children = _named_children(node)
    try:
        return children[int(step[1:])]
    except (IndexError, ValueError):
        return None


def _deepest(node: Node, field: str) -> Node | None:
    """A `field*` step: follow the field while it exists."""
    current = node
    while (following := current.child_by_field_name(field)) is not None:
        current = following
    return current if current is not node else None


def _until(node: Node, step: str) -> Node | None:
    """A `field~kind` step: follow the field until a node of the kind."""
    field, _, wanted = step.partition("~")
    current: Node | None = node
    while current is not None and current.type != wanted:
        current = current.child_by_field_name(field)
    return current


_FIXED_STEPS = {
    ".": lambda node: node,
    "..": lambda node: node.parent,
    "<": lambda node: node.prev_named_sibling,
    ">": lambda node: node.next_named_sibling,
}


def _step(node: Node, step: str) -> Node | list[Node] | None:
    """Apply one step to one node."""
    fixed = _FIXED_STEPS.get(step)
    if fixed is not None:
        return fixed(node)
    if step.startswith("**"):
        return _descendant(node, step[2:])
    if step.startswith("@"):
        return _by_type(node, step)
    if step.startswith("#"):
        return _by_index(node, step)
    if step.endswith("+"):
        return node.children_by_field_name(step[:-1])
    if step.endswith("*"):
        return _deepest(node, step[:-1])
    if "~" in step:
        return _until(node, step)
    return node.child_by_field_name(step)


@cache
def _parse(path: str) -> tuple[tuple[str, ...], ...]:
    """Split a path into its alternatives, each a tuple of steps."""
    return tuple(tuple(part.split(_SEPARATOR)) for part in path.split(_ALTERNATIVE))


def resolve(node: Node | None, path: str) -> Node | None:
    """Resolve a path to one node, or None when no alternative reaches anything."""
    if node is None:
        return None
    for steps in _parse(path):
        current: Node | None = node
        for step in steps:
            if current is None:
                break
            found = _step(current, step)
            current = found[0] if isinstance(found, list) else found
        if current is not None:
            return current
    return None


def resolve_all(node: Node | None, path: str) -> list[Node]:
    """Resolve a path to every node it reaches; a list step fans out, anything else yields one."""
    if node is None:
        return []
    for steps in _parse(path):
        current: list[Node] = [node]
        for step in steps:
            following: list[Node] = []
            for item in current:
                found = _step(item, step)
                if isinstance(found, list):
                    following.extend(found)
                elif found is not None:
                    following.append(found)
            current = following
            if not current:
                break
        if current:
            return current
    return []


def text(node: Node | None, source: bytes) -> str:
    """A node's source text with runs of whitespace collapsed, or an empty string."""
    if node is None:
        return ""
    return " ".join(source[node.start_byte : node.end_byte].decode("utf-8", "replace").split())


__all__ = ["resolve", "resolve_all", "text"]
