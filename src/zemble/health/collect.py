"""The raw facts `zemble health` judges, gathered from the graph in bulk and from the workspace config.

AIDEV-NOTE: every read here is one whole-graph query (`all_symbols`, `inbound_tallies`, ...), never
one per symbol: javaweb has 120k symbols and 1.2M edges. Paths and ids are the provider's, so a
sub-directory view hands over paths relative to the requested directory; `module_of` maps them
onto the root that declared `.zemble/home.toml` for `HomeConfig.module_of`.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatch
from functools import cache, cached_property
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from zemble.graph.model import (
    ANONYMOUS_MARKER,
    CALLABLE_KINDS,
    DECLARED_TYPE_KINDS,
    Edge,
    EdgeKind,
    EdgeTally,
    FileDependency,
    Resolution,
    Symbol,
    SymbolKind,
)
from zemble.graph.provider import AnyProvider
from zemble.health.copying import CopyingFacts, measure_copying
from zemble.health.git import ChangedLines
from zemble.health.history import HistoryFacts, measure_history
from zemble.health.model import Comparison, Signal
from zemble.home.config import HomeConfig
from zemble.index.files import detect_language
from zemble.languages import Visibility, spec_for
from zemble.languages.metrics import CallableMetrics, FileMetrics
from zemble.languages.read import visibility_of
from zemble.languages.visibility import VISIBILITY_WORDS

if TYPE_CHECKING:  # pragma: no cover
    from zemble.health.style import StyleMeasurement

#: The language every unmeasurable file is reported under.
UNKNOWN_LANGUAGE = "unknown"

#: Edges that say "a subtype of this".
HIERARCHY_KINDS = frozenset({EdgeKind.EXTENDS, EdgeKind.IMPLEMENTS})
#: How confidently a supertype edge must be resolved for the view to see into that supertype: the
#: strictest floor of the absence signals, which are the ones an unseen supertype could defeat.
_PROVABLE: Resolution = max(
    (signal.facts.floor for signal in Signal if signal.facts.floor and signal.facts.comparison is Comparison.AT_MOST),
    key=lambda resolution: resolution.strength,
)
#: Visibility levels whose references can only be written near the declaration.
_NEARBY = frozenset({Visibility.PRIVATE, Visibility.PACKAGE})
_WORD = re.compile(r"[A-Za-z_$][\w$]*")

#: Edges a module depends on another module through.
DEPENDENCY_KINDS = (EdgeKind.IMPORTS, EdgeKind.CALLS, EdgeKind.REFERENCES_TYPE)

#: Container kinds whose members are part of a contract: as visible as the container unless they say otherwise.
_CONTRACT_KINDS = frozenset({SymbolKind.INTERFACE, SymbolKind.ANNOTATION})

_NAME_SPLIT = re.compile(r"[.:/\\#]")

#: Containers whose declared types count as top-level.
_FILE_LEVEL = frozenset({SymbolKind.MODULE, SymbolKind.PACKAGE})


@cache
def language_of(file_path: str) -> str:
    """The language a file is measured as, by its extension."""
    return detect_language(PurePosixPath(file_path)) or UNKNOWN_LANGUAGE


def within(file_path: str, paths: Sequence[str]) -> bool:
    """Whether a path is one of `paths` or lies under one; no paths means everything."""
    return not paths or any(file_path == path or file_path.startswith(f"{path}/") for path in paths)


def written_name(text: str) -> str:
    """The simple name an unresolved edge wrote: `java.util.List<String>` reads `List`."""
    head = text.split("<", 1)[0].split("(", 1)[0].split("[", 1)[0]
    return _NAME_SPLIT.split(head)[-1].strip()


def anonymous(symbol: Symbol | None) -> bool:
    """Whether a symbol is an anonymous class, which nothing can name."""
    return symbol is not None and symbol.name.startswith(ANONYMOUS_MARKER)


def named_type(symbol: Symbol) -> bool:
    """Whether a symbol is a type declaration with a name."""
    return symbol.kind in DECLARED_TYPE_KINDS and not anonymous(symbol)


@dataclass
class HealthFacts:
    """Everything the signal producers read, already filtered to production code in the scope."""

    #: The absolute directory the report is about; every path is relative to it.
    root: str
    config: HomeConfig
    #: Every symbol of the view, test code included, by id: containers are looked up here.
    symbols: dict[str, Symbol]
    metrics: dict[str, CallableMetrics]
    #: (path, metrics) of every measured production file in the scope.
    files: list[tuple[str, FileMetrics]]
    #: Inbound tallies per symbol id, every kind.
    inbound: dict[str, list[EdgeTally]]
    #: Ids of the callables that override, or may override, a supertype's member (see `_may_override`).
    may_override: set[str]
    #: Edges with no destination, by kind and the simple name they wrote.
    unresolved_names: dict[tuple[EdgeKind, str], int]
    #: Edges leaving production files in the scope; empty without a declared `home.toml`.
    dependencies: list[FileDependency] = field(default_factory=list)
    #: Paths the report is restricted to, relative to `root`; empty means everything.
    paths: tuple[str, ...] = ()
    #: Every file of the graph, test code included, as (absolute path, is test): where a name may be written.
    sources: tuple[tuple[str, bool], ...] = ()
    #: The duplication run, when the report measures the duplication tier.
    copying: CopyingFacts | None = None
    #: (path, metrics) of every measured production file of the view, in the scope or not: the peers a
    #: file's style is compared with, so `--paths` and `--since` judge against the whole view.
    population: list[tuple[str, FileMetrics]] = field(default_factory=list)
    #: The history window's commits per production file of the view, when the report measures history.
    history: HistoryFacts | None = None
    _modules: dict[str, str] = field(default_factory=dict)

    def in_scope(self, file_path: str) -> bool:
        """Whether a path is inside one of the requested sub-paths."""
        return within(file_path, self.paths)

    @cached_property
    def everywhere(self) -> list[Symbol]:
        """Every non-test symbol of the view, in the scope or not, by file and line."""
        return [symbol for symbol in self.symbols.values() if not symbol.is_test]

    @cached_property
    def production(self) -> list[Symbol]:
        """Every non-test symbol in the scope, by file and line."""
        return [symbol for symbol in self.everywhere if self.in_scope(symbol.file_path)]

    @cached_property
    def _nested_lines(self) -> Counter[str]:
        """Per callable id, the lines of the named callables declared directly inside it."""
        nested: Counter[str] = Counter()
        for symbol in self.symbols.values():
            container = self.container(symbol)
            if symbol.kind in CALLABLE_KINDS and container is not None and container.kind in CALLABLE_KINDS:
                nested[container.id] += symbol.end_line - symbol.start_line + 1
        return nested

    def own_lines(self, symbol: Symbol, metric: CallableMetrics) -> int:
        """A callable's length in lines, less the named callables declared directly inside it.

        A nested function (a decorated tool inside its `register_*`, a closure) is measured on its own,
        so counting its lines again in the enclosing callable would count one body twice.
        """
        return max(metric.lines - self._nested_lines[symbol.id], 0)

    @cached_property
    def top_level_types(self) -> Counter[str]:
        """Named production type declarations at file level, per file of the view."""
        counts: Counter[str] = Counter()
        for symbol in self.everywhere:
            if named_type(symbol):
                container = self.container(symbol)
                if container is None or container.kind in _FILE_LEVEL:
                    counts[symbol.file_path] += 1
        return counts

    @cached_property
    def style(self) -> StyleMeasurement:
        """The consistency axis's measurement of every file of the view, computed once for both its signals."""
        from zemble.health.style import measure_style  # the style module reads these facts: a cycle at import

        return measure_style(self)

    def container(self, symbol: Symbol) -> Symbol | None:
        """The symbol's declaring container, when the view holds it."""
        return self.symbols.get(symbol.container_id) if symbol.container_id else None

    def in_contract(self, symbol: Symbol) -> bool:
        """Whether a symbol is a member of an interface, trait, protocol or annotation type."""
        container = self.container(symbol)
        return container is not None and container.kind in _CONTRACT_KINDS

    def overridden(self, symbol: Symbol) -> bool:
        """Whether anything overrides a callable, by an edge at any resolution: a weak edge still means it may."""
        return any(tally.kind is EdgeKind.OVERRIDES for tally in self.inbound.get(symbol.id, ()))

    @cached_property
    def callables_by_file(self) -> dict[str, list[Symbol]]:
        """Every production callable in the scope, by file."""
        found: dict[str, list[Symbol]] = defaultdict(list)
        for symbol in self.production:
            if symbol.kind in CALLABLE_KINDS:
                found[symbol.file_path].append(symbol)
        return dict(found)

    def visibility(self, symbol: Symbol) -> Visibility:
        """How far a symbol reaches; UNKNOWN for a language no rule places.

        A member of an interface (trait, protocol) without a word of its own is as visible as the
        interface: it is part of the contract, in every language that has one.
        """
        container = self.container(symbol)
        written = set(symbol.modifiers)
        for word, level in VISIBILITY_WORDS:
            if word in written:
                return level
        if container is not None and self.in_contract(symbol):
            return self.visibility(container)
        language = language_of(symbol.file_path)
        if language == "java":
            return Visibility.PACKAGE
        spec = spec_for(language)
        if spec is None:
            return Visibility.UNKNOWN
        in_type = container is not None and container.kind in DECLARED_TYPE_KINDS
        return visibility_of(spec, symbol.name, symbol.modifiers, in_type=in_type)

    def is_runtime_hook(self, symbol: Symbol) -> bool:
        """Whether the language runtime calls this callable by its name alone (`__repr__`)."""
        spec = spec_for(language_of(symbol.file_path))
        return spec is not None and any(fnmatch(symbol.name, pattern) for pattern in spec.runtime_hook_patterns)

    def reaching(
        self, symbol: Symbol, kinds: frozenset[EdgeKind], floor: Resolution, *, tests: bool
    ) -> tuple[int, Resolution, int]:
        """Count what reaches a symbol over some edge kinds.

        :param tests: Whether sources in test code count as reaching it.
        :return: The distinct sources at `floor` or better, the weakest resolution among them (EXACT
            for none), and the edges below `floor` that might also mean it: inbound ones resolved
            weaker plus every destination-less edge that wrote its name.
        """
        strong = 0
        weakest = Resolution.EXACT
        doubt = sum(self.unresolved_names.get((kind, symbol.name), 0) for kind in kinds)
        for tally in self.inbound.get(symbol.id, ()):
            if tally.kind not in kinds:
                continue
            if not tally.resolution.at_least(floor):
                doubt += tally.sources
            elif tests or not tally.from_test:
                strong += tally.sources
                weakest = min(weakest, tally.resolution, key=lambda resolution: resolution.strength)
        return strong, weakest, doubt

    @cached_property
    def _config_root(self) -> str:
        """The resolved root that declared the config."""
        return os.path.realpath(self.config.root)

    def module_of(self, file_path: str) -> str:
        """The declared module of a path relative to `root`, memoised: dependency rows repeat their files."""
        module = self._modules.get(file_path)
        if module is None:
            module = self._modules[file_path] = self.config.module_of(self._config_path(file_path))
        return module

    def _config_path(self, file_path: str) -> str:
        """A path relative to `root` respelled relative to the root that declared the config."""
        if self._config_root == self.root:
            return file_path
        absolute = os.path.normpath(os.path.join(self.root, file_path))
        return Path(os.path.relpath(absolute, self._config_root)).as_posix()

    def unmentioned(self, candidates: Sequence[Symbol], *, tests: bool) -> list[Symbol]:
        """Keep the candidates whose name is written nowhere it could be referenced from, its own span aside.

        AIDEV-NOTE: the graph has no edge for a callable used as a VALUE (a dispatch table, `key=f`,
        `Foo::bar`), so an absence the graph reports is checked against the source text: a private
        name against its own file, a package-private one against its directory, any other against
        every file of the graph. Comments and strings count too, which only ever withholds a finding.

        :param tests: Whether a mention in test code counts.
        """
        reach = {symbol.id: self.visibility(symbol) for symbol in candidates}
        wanted = {symbol.name for symbol in candidates if reach[symbol.id] is not Visibility.PRIVATE}
        directories = {
            os.path.dirname(self.absolute(symbol.file_path))
            for symbol in candidates
            if reach[symbol.id] is Visibility.PACKAGE
        }
        anywhere = any(level not in _NEARBY for level in reach.values())
        found: dict[str, set[str]] = defaultdict(set)
        for path, is_test in self.sources if wanted else ():
            if (tests or not is_test) and (anywhere or os.path.dirname(path) in directories):
                for name in _words(path) & wanted:
                    found[name].add(path)
        kept: list[Symbol] = []
        for symbol in candidates:
            own = self.absolute(symbol.file_path)
            level = reach[symbol.id]
            elsewhere = [path for path in found.get(symbol.name, ()) if path != own and _reaches(level, own, path)]
            if not elsewhere and not _written_outside(own, symbol):
                kept.append(symbol)
        return kept

    def absolute(self, file_path: str) -> str:
        """The absolute path of a path relative to `root`."""
        return os.path.normpath(os.path.join(self.root, file_path))


def collect(
    provider: AnyProvider,
    config: HomeConfig,
    paths: Sequence[str] = (),
    *,
    copying: bool = False,
    changed: ChangedLines | None = None,
    history_window: int | None = None,
) -> HealthFacts:
    """Read every fact the signals need from an open provider and the workspace config.

    :param provider: The graph answering for the requested directory (its own, or an ancestor's view).
    :param config: The `home.toml` of the nearest ancestor declaring one, or the generic config.
    :param paths: Sub-paths to restrict the report to, relative to the directory or absolute.
    :param copying: Whether to run the duplication channels (`zemble.health.copying`).
    :param changed: The lines a `--since` run reports on; the duplication run then compares only their files.
    :param history_window: Days of git history to read for the history axis; None reads none.
    """
    root = os.path.realpath(provider.path)
    scope = tuple(relative for path in paths if (relative := _relative(root, path)))
    unresolved: dict[tuple[EdgeKind, str], int] = defaultdict(int)
    for (kind, name), edges in provider.unresolved_name_tallies().items():
        unresolved[(kind, written_name(name))] += edges
    inbound: dict[str, list[EdgeTally]] = defaultdict(list)
    for tally in provider.inbound_tallies():
        inbound[tally.symbol_id].append(tally)
    symbols = {symbol.id: symbol for symbol in provider.all_symbols()}
    links = provider.outgoing_edges([EdgeKind.OVERRIDES, *HIERARCHY_KINDS])
    facts = HealthFacts(
        root=root,
        config=config,
        symbols=symbols,
        metrics={symbol.id: metric for symbol, metric in provider.all_metrics()},
        files=[],
        inbound=dict(inbound),
        may_override=_may_override(symbols, links),
        unresolved_names=dict(unresolved),
        paths=scope,
        sources=tuple(provider.source_files()),
    )
    facts.population = [(path, metric) for path, metric, is_test in provider.file_measurements() if not is_test]
    facts.files = [(path, metric) for path, metric in facts.population if facts.in_scope(path)]
    if not config.generic:
        production = {path for path, _ in facts.files}
        facts.dependencies = [
            dependency
            for dependency in provider.file_dependencies(DEPENDENCY_KINDS)
            if dependency.source_path in production
        ]
    if copying:
        files = None if changed is None else [path for path in changed.files if facts.in_scope(path)]
        facts.copying = measure_copying(provider, scope, files)
    if history_window is not None:
        facts.history = measure_history(root, [path for path, _ in facts.population], history_window)
    return facts


def _may_override(symbols: dict[str, Symbol], links: Sequence[Edge]) -> set[str]:
    """Ids of the callables that override a supertype's member, by an edge or by inheritance.

    Not every extractor writes OVERRIDES edges (Python has none), so a callable also overrides
    when a supertype in its type's chain declares a callable of the same name. A type with a
    supertype the view cannot see into (unresolved, ambiguous, or outside the folder) may inherit
    anything, so every callable it declares counts: an absence there is not provable. So does a
    callable declared on an owner the graph holds no symbol for, such as the methods of a Rust
    `impl Trait for ForeignType`, whose qualified name names that owner while their container is the file.
    """
    overriding = {edge.src_id for edge in links if edge.kind is EdgeKind.OVERRIDES}
    chains = _Inheritance(symbols, links)
    for symbol in symbols.values():
        if symbol.kind not in CALLABLE_KINDS or not symbol.container_id:
            continue
        container = symbols.get(symbol.container_id)
        if container is None or _owner(symbol) not in (None, container.qualified_name):
            overriding.add(symbol.id)
        elif symbol.container_id in chains.parents:
            above = chains.names_above(symbol.container_id, frozenset({symbol.container_id}))
            if above is None or symbol.name in above:
                overriding.add(symbol.id)
    return overriding


def _owner(symbol: Symbol) -> str | None:
    """The qualified name a member is declared on, as its own qualified name spells it, if it does."""
    suffix = f".{symbol.name}"
    return symbol.qualified_name[: -len(suffix)] if symbol.qualified_name.endswith(suffix) else None


class _Inheritance:
    """The callable names each type inherits through the supertype edges the view can see into."""

    def __init__(self, symbols: dict[str, Symbol], links: Sequence[Edge]) -> None:
        """Index every supertype edge (None for one the view cannot see into) and every type's callables."""
        self.parents: dict[str, list[str | None]] = defaultdict(list)
        for edge in links:
            if edge.kind in HIERARCHY_KINDS:
                known = edge.dst_id in symbols and edge.resolution.at_least(_PROVABLE)
                self.parents[edge.src_id].append(edge.dst_id if known else None)
        self.members: dict[str, set[str]] = defaultdict(set)
        for symbol in symbols.values():
            if symbol.kind in CALLABLE_KINDS and symbol.container_id:
                self.members[symbol.container_id].add(symbol.name)
        self._inherited: dict[str, set[str] | None] = {}

    def names_above(self, type_id: str, seen: frozenset[str]) -> set[str] | None:
        """The callable names a type inherits, or None when its chain leaves what the view can see."""
        if type_id in self._inherited:
            return self._inherited[type_id]
        names: set[str] | None = set()
        for parent in self.parents.get(type_id, ()):
            if parent in seen:
                continue
            above = None if parent is None else self.names_above(parent, seen | {parent})
            if parent is None or above is None:
                names = None
                break
            names |= self.members.get(parent, set()) | above
        self._inherited[type_id] = names
        return names


def _reaches(level: Visibility, own: str, path: str) -> bool:
    """Whether code in `path` may reference a declaration of this level written in `own`."""
    if level is Visibility.PRIVATE:
        return False
    if level is Visibility.PACKAGE:
        return os.path.dirname(path) == os.path.dirname(own)
    return True


def _words(path: str) -> set[str]:
    """Every identifier-like word written in a file, or none when it cannot be read."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return set(_WORD.findall(handle.read()))
    except OSError:
        return set()


def _written_outside(path: str, symbol: Symbol) -> bool:
    """Whether a symbol's name is written in its own file outside its own declaration's lines."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.read().split("\n")
    except OSError:
        return True
    pattern = re.compile(rf"(?<![\w$]){re.escape(symbol.name)}(?![\w$])")
    return any(
        pattern.search(line)
        for number, line in enumerate(lines, start=1)
        if not symbol.start_line <= number <= symbol.end_line
    )


def _relative(root: str, path: str) -> str:
    """A requested sub-path relative to the report root, in POSIX form without a trailing slash."""
    absolute = path if os.path.isabs(path) else os.path.join(root, path)
    relative = Path(os.path.relpath(os.path.normpath(absolute), root)).as_posix()
    return "" if relative == "." else relative


__all__ = [
    "DEPENDENCY_KINDS",
    "HIERARCHY_KINDS",
    "UNKNOWN_LANGUAGE",
    "HealthFacts",
    "anonymous",
    "collect",
    "named_type",
    "language_of",
    "within",
    "written_name",
]
