"""Which module may depend on which, and what "may" means when nobody said.

`order` in `home.toml` ranks modules; it never granted anyone permission to depend on
anyone. This module is where dependency PERMISSION lives: a directed graph whose edges
come from the workspace's own declarations and from its Gradle build files, and one
question - `reachable(a, b)` - answered with a value that says how sure it is.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from zemble.home.gradle import GradleProject, RefKind, discover

#: Artifact suffixes a workspace publishes one module under, stripped before an artifact
#: name is read as a module name. The exact module name always wins, so a module that IS
#: called `zenit-auth-test-support` is never truncated to `zenit-auth`.
FOLD_SUFFIXES = ("-common", "-client", "-browser", "-server", "-test-support", "-test")


class DependencySource(str, Enum):
    """Where a workspace's dependency edges may come from."""

    GRADLE = "gradle"
    DECLARED = "declared"
    BOTH = "both"


class EdgeOrigin(str, Enum):
    """How one dependency edge was learned."""

    DECLARED = "declared"
    DISCOVERED = "discovered"


class Reachability(str, Enum):
    """Whether one module may depend on another, and how well that is known.

    UNKNOWN is deliberately distinct from UNREACHABLE: a workspace that says nothing
    about its dependencies must be told that, not told "no".
    """

    DIRECT = "direct"
    TRANSITIVE = "transitive"
    FORBIDDEN = "forbidden"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"

    @property
    def usable(self) -> bool:
        """Whether a mechanism in the target module may be reached from the source module."""
        return self in (Reachability.DIRECT, Reachability.TRANSITIVE)


@dataclass(frozen=True)
class DependencyEdge:
    """One module depending on another."""

    source: str
    target: str
    origin: EdgeOrigin
    #: The Gradle configuration a discovered edge was written in, "" for a declared one.
    configuration: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Render the edge as JSON-ready data."""
        return {
            "from": self.source,
            "to": self.target,
            "origin": self.origin.value,
            "configuration": self.configuration,
        }


@dataclass(frozen=True)
class DependencyGraph:
    """The modules of a workspace and the dependencies between them.

    Nodes are the modules the workspace declares; an edge means the source module may
    reach the target's code. Forbidden rules are part of the graph because they OVERRIDE
    every edge: a workspace that both builds and refuses a dependency means the refusal.
    """

    nodes: tuple[str, ...] = ()
    edges: tuple[DependencyEdge, ...] = ()
    #: `(source, target)` pairs the workspace refuses outright.
    forbidden: frozenset[tuple[str, str]] = frozenset()
    source: DependencySource = DependencySource.BOTH

    @property
    def known(self) -> bool:
        """Whether the workspace carries any dependency information at all."""
        return bool(self.edges)

    def targets_of(self, module: str) -> tuple[str, ...]:
        """Return the modules one module depends on directly, in edge order."""
        found: list[str] = []
        for edge in self.edges:
            if edge.source == module and edge.target not in found:
                found.append(edge.target)
        return tuple(found)

    def has_edges_from(self, module: str) -> bool:
        """Whether anything is known about what this module depends on."""
        return any(edge.source == module for edge in self.edges)

    def reachable(self, source: str, target: str) -> Reachability:
        """Answer whether `source` may use code that lives in `target`.

        A module always reaches itself (DIRECT). A forbidden pair is FORBIDDEN whatever
        the edges say. A module nothing is known about is UNKNOWN, and every other pair
        fails closed to UNREACHABLE rather than being assumed from `order`.
        """
        if (source, target) in self.forbidden:
            return Reachability.FORBIDDEN
        if source == target:
            return Reachability.DIRECT
        if not self.has_edges_from(source):
            return Reachability.UNKNOWN
        direct = self.targets_of(source)
        if target in direct:
            return Reachability.DIRECT
        seen = {source, *direct}
        queue = list(direct)
        while queue:
            current = queue.pop(0)
            for step in self.targets_of(current):
                if step == target:
                    return Reachability.TRANSITIVE
                if step not in seen:
                    seen.add(step)
                    queue.append(step)
        return Reachability.UNREACHABLE

    def nearest_common_dependency(self, modules: Sequence[str], rank: Callable[[str], int]) -> str | None:
        """Return the DEEPEST module every given module may reach, if there is one.

        Nearest means nearest to the modules that would share the mechanism, not nearest to
        the core: of the modules they all depend on, the ones that are not themselves a
        dependency of another shared module. Two `zenit-*` siblings share `protoblast`,
        `hawkeye` and `zenit`, and the answer is `zenit` - `protoblast` is dropped because
        `zenit` already reaches it, and putting widget-and-flow state in the root library
        would be the wrong advice. A tie between two maximal modules is broken by `order`,
        most core first, and the answer says the tie happened.

        :param modules: The modules that would share the mechanism.
        :param rank: A module's distance from the core, lowest first.
        :return: The module to put the shared mechanism in, or None when they share nothing.
        """
        candidates = self.common_dependencies(modules)
        return min(candidates, key=lambda node: (rank(node), node)) if candidates else None

    def common_dependencies(self, modules: Sequence[str]) -> tuple[str, ...]:
        """Return the maximal modules every given module may reach.

        The members themselves are never candidates: siblings do not reach each other, and
        a module is not its own shared home. A shared module reachable FROM another shared
        module is dropped, which is what makes the result the deepest shared layer rather
        than the most core one.
        """
        wanted = [module for module in dict.fromkeys(modules) if module]
        if not wanted or not self.known:
            return ()
        shared = [
            node
            for node in self.nodes
            if node not in wanted and all(self.reachable(module, node).usable for module in wanted)
        ]
        maximal = [
            node for node in shared if not any(other != node and self.reachable(other, node).usable for other in shared)
        ]
        # AIDEV-NOTE: a dependency CYCLE makes every shared module reachable from another
        # one and would leave nothing maximal. Falling back to the whole shared set keeps
        # the answer total instead of reporting that the modules share nothing.
        return tuple(maximal or shared)

    def to_dict(self) -> dict[str, Any]:
        """Render the graph as JSON-ready data."""
        return {
            "source": self.source.value,
            "known": self.known,
            "nodes": list(self.nodes),
            "edges": [edge.to_dict() for edge in self.edges],
            "forbidden": [{"from": rule[0], "to": rule[1]} for rule in sorted(self.forbidden)],
        }


def build_graph(
    root: str | Path,
    modules: Sequence[str],
    module_of: Callable[[str], str],
    declared: Mapping[str, Sequence[str]],
    forbidden: Iterable[tuple[str, str]] = (),
    source: DependencySource = DependencySource.BOTH,
    gradle_roots: tuple[str, ...] = (),
) -> DependencyGraph:
    """Merge what a workspace declares with what its build files say.

    A module that declares `depends_on` REPLACES discovery for itself entirely, including
    when it declares an empty list: a workspace that writes down its boundaries means them.

    :param root: The workspace root, scanned for build files unless discovery is off.
    :param modules: Every module the workspace declares; they are the graph's nodes.
    :param module_of: Maps a workspace-relative path to the module holding it.
    :param declared: Module -> the modules it declares a dependency on.
    :param forbidden: Pairs the workspace refuses.
    :param source: Which lanes may contribute edges.
    :param gradle_roots: Directories to scan, empty meaning the whole tree.
    :return: The merged graph.
    """
    known = tuple(modules)
    edges: list[DependencyEdge] = []
    if source in (DependencySource.DECLARED, DependencySource.BOTH):
        for module, targets in declared.items():
            edges.extend(
                DependencyEdge(source=module, target=target, origin=EdgeOrigin.DECLARED)
                for target in targets
                if target != module
            )
    if source in (DependencySource.GRADLE, DependencySource.BOTH):
        overridden = set(declared) if source is DependencySource.BOTH else set()
        edges.extend(
            edge for edge in discovered_edges(root, known, module_of, gradle_roots) if edge.source not in overridden
        )
    return DependencyGraph(nodes=known, edges=_deduplicate(edges), forbidden=frozenset(forbidden), source=source)


def discovered_edges(
    root: str | Path,
    modules: Sequence[str],
    module_of: Callable[[str], str],
    gradle_roots: tuple[str, ...] = (),
) -> list[DependencyEdge]:
    """Read the Gradle build files under a root as edges between the declared modules.

    A reference that resolves to no declared module - an external library, a project the
    workspace does not check out - contributes nothing: an edge is only ever between two
    modules this answer already knows about.
    """
    projects = discover(root, gradle_roots)
    by_path = {project.gradle_path: project for project in projects}
    by_name = {project.name: project for project in projects}
    names = {module.lower(): module for module in modules}
    edges: list[DependencyEdge] = []
    for project in projects:
        owner = module_of(f"{project.directory}/build.gradle" if project.directory else "build.gradle")
        if owner not in names.values():
            continue
        for ref in project.refs:
            target = _target_module(ref, by_path, by_name, names, module_of)
            if target is None or target == owner:
                continue
            edges.append(
                DependencyEdge(
                    source=owner, target=target, origin=EdgeOrigin.DISCOVERED, configuration=ref.configuration
                )
            )
    return edges


def module_of_artifact(artifact: str, names: Mapping[str, str]) -> str | None:
    """Resolve a published artifact name to a declared module.

    AIDEV-NOTE: a workspace of sibling repositories declares its internal dependencies as
    coordinates (`be.elevenways:zenit-common:...`) rather than as `project(':zenit')`, so
    the artifact name is the only edge evidence there is. Only the exact module name and
    a name with one known fold suffix removed resolve; anything else is an external
    library and contributes no edge.
    """
    lowered = artifact.lower()
    if lowered in names:
        return names[lowered]
    for suffix in FOLD_SUFFIXES:
        if lowered.endswith(suffix) and lowered[: -len(suffix)] in names:
            return names[lowered[: -len(suffix)]]
    return None


def _target_module(
    ref: Any,
    by_path: Mapping[str, GradleProject],
    by_name: Mapping[str, GradleProject],
    names: Mapping[str, str],
    module_of: Callable[[str], str],
) -> str | None:
    """Resolve one dependency reference to a declared module, or None."""
    if ref.kind is RefKind.PROJECT:
        project = by_path.get(ref.project_path) or by_name.get(ref.project_path.lstrip(":"))
        if project is None:
            return names.get(ref.project_path.lstrip(":").rsplit(":", 1)[-1].lower())
        owner = module_of(f"{project.directory}/build.gradle" if project.directory else "build.gradle")
        return owner if owner in names.values() else None
    if ref.kind is RefKind.COORDINATE:
        return module_of_artifact(ref.artifact, names)
    raise ValueError(f"unhandled dependency reference kind: {ref.kind!r}")


def _deduplicate(edges: Sequence[DependencyEdge]) -> tuple[DependencyEdge, ...]:
    """Keep one edge per (source, target), the first-seen configuration."""
    seen: dict[tuple[str, str], DependencyEdge] = {}
    for edge in edges:
        seen.setdefault((edge.source, edge.target), edge)
    return tuple(seen.values())


__all__ = [
    "FOLD_SUFFIXES",
    "DependencyEdge",
    "DependencyGraph",
    "DependencySource",
    "EdgeOrigin",
    "Reachability",
    "build_graph",
    "discovered_edges",
    "module_of_artifact",
]
