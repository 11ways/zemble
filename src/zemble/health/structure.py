"""Graph algorithms the structural signals share: strongly connected components and longest ancestor chains.

AIDEV-NOTE: both are iterative, never recursive: a workspace of 14k files or a deep hierarchy would
pass Python's recursion limit, and both must survive a cycle in bad graph data.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import Generic, TypeVar

Node = TypeVar("Node", bound=Hashable)


def strongly_connected(nodes: Iterable[Node], successors: Mapping[Node, Iterable[Node]]) -> list[list[Node]]:
    """Tarjan's strongly connected components, every node in exactly one; successors outside `nodes` are ignored."""
    order = list(nodes)
    tarjan = _Tarjan(set(order), successors)
    for start in order:
        if start not in tarjan.index:
            tarjan.walk(start)
    return tarjan.components


class _Tarjan(Generic[Node]):
    """The state of one Tarjan run, walked with an explicit stack of (node, successors left) frames."""

    def __init__(self, known: set[Node], successors: Mapping[Node, Iterable[Node]]) -> None:
        """Start a run over the known nodes."""
        self.known = known
        self.successors = successors
        self.index: dict[Node, int] = {}
        self.low: dict[Node, int] = {}
        self.stack: list[Node] = []
        self.on_stack: set[Node] = set()
        self.components: list[list[Node]] = []

    def _enter(self, node: Node, work: list[tuple[Node, Iterator[Node]]]) -> None:
        """Number a node, push it, and schedule its successors."""
        self.index[node] = self.low[node] = len(self.index)
        self.stack.append(node)
        self.on_stack.add(node)
        work.append((node, iter(self.successors.get(node, ()))))

    def walk(self, start: Node) -> None:
        """Collect every component reachable from an unvisited node."""
        work: list[tuple[Node, Iterator[Node]]] = []
        self._enter(start, work)
        while work:
            node, pending = work[-1]
            fresh = next((successor for successor in pending if self._unvisited(node, successor)), None)
            if fresh is not None:
                self._enter(fresh, work)
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                self.low[parent] = min(self.low[parent], self.low[node])
            if self.low[node] == self.index[node]:
                self._close(node)

    def _unvisited(self, node: Node, successor: Node) -> bool:
        """Whether a successor is still to be walked; a visited one on the stack lowers the node's link."""
        if successor not in self.known:
            return False
        if successor not in self.index:
            return True
        if successor in self.on_stack:
            self.low[node] = min(self.low[node], self.index[successor])
        return False

    def _close(self, root: Node) -> None:
        """Pop the component rooted at a node off the stack."""
        component = []
        while True:
            member = self.stack.pop()
            self.on_stack.discard(member)
            component.append(member)
            if member == root:
                break
        self.components.append(component)


def ancestor_depths(parents: Mapping[Node, Iterable[Node]]) -> dict[Node, tuple[int, tuple[Node, ...]]]:
    """Per node with parents, the length of its longest chain of ancestors and that chain, nearest first.

    A parent already on the chain being walked (a cycle in bad data) is not followed, so every chain
    is finite; a node's depth may then depend on where the walk entered its cycle.
    """
    found: dict[Node, tuple[int, tuple[Node, ...]]] = {}
    for start in parents:
        if start in found:
            continue
        on_path = {start}
        work = [(start, iter(parents.get(start, ())))]
        while work:
            node, pending = work[-1]
            advanced = False
            for parent in pending:
                if parent in on_path or parent in found or not parents.get(parent):
                    continue
                on_path.add(parent)
                work.append((parent, iter(parents.get(parent, ()))))
                advanced = True
                break
            if advanced:
                continue
            work.pop()
            on_path.discard(node)
            best: tuple[int, tuple[Node, ...]] = (0, ())
            for parent in parents.get(node, ()):
                if parent in on_path:
                    continue
                depth, chain = found.get(parent, (0, ()))
                if depth + 1 > best[0]:
                    best = (depth + 1, (parent, *chain))
            found[node] = best
    return found


__all__ = ["ancestor_depths", "strongly_connected"]
