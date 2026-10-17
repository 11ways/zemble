"""Turning collected facts into samples: one producer per `Signal`, each yielding its whole population.

A producer yields every unit it MEASURED, past the threshold or not: the population is what
percentiles and scores are computed over. A unit a weaker edge might also reach is not measured
at all (an absence it cannot vouch for, a count it cannot finish), so it neither scores nor ranks.
"""

from __future__ import annotations

import posixpath
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from zemble.dedup.languages import SiteKind, profile_for
from zemble.dedup.model import CloneKind, Unit
from zemble.graph.model import (
    CALLABLE_KINDS,
    DECLARED_TYPE_KINDS,
    TYPE_KINDS,
    EdgeKind,
    FileDependency,
    Resolution,
    Symbol,
    SymbolKind,
)
from zemble.graph.provider import display_name
from zemble.health.collect import HIERARCHY_KINDS, HealthFacts, anonymous, language_of, named_type
from zemble.health.copying import CODE_KINDS, CopyingFacts
from zemble.health.git import merge_ranges
from zemble.health.history import HOTSPOT_MIN_COMMITS
from zemble.health.model import Signal
from zemble.health.structure import ancestor_depths, strongly_connected
from zemble.health.style import Distance
from zemble.languages import Visibility
from zemble.languages.metrics import MARKER, CallableMetrics


@dataclass(frozen=True)
class Sample:
    """One measured unit of one signal."""

    file_path: str
    start_line: int
    end_line: int
    #: The symbol's display name, or a short label for a file-level unit.
    name: str
    #: Path-free where the unit has a qualified name: what a finding key is made of.
    identity: str
    value: int
    resolution: Resolution = Resolution.EXACT
    #: What this unit adds to the signal's advice, or "".
    detail: str = ""
    #: Other places the same unit lives, as (path, start, end): a vocabulary's other declarations.
    elsewhere: tuple[tuple[str, int, int], ...] = ()
    #: The whole the unit is one member of, as `Finding.group` carries it, or "".
    group: str = ""

    @property
    def language(self) -> str:
        """The language the unit is measured in."""
        return language_of(self.file_path)


Producer = Callable[[HealthFacts], Iterator[Sample]]

_CALLS = frozenset({EdgeKind.CALLS})
_ANY_EDGE = frozenset(EdgeKind)
#: Visibility levels whose declarations only their own code may reference.
_RESTRICTED = frozenset({Visibility.PRIVATE, Visibility.PACKAGE})
#: Callables an outside caller reaches by name: the rest (lambdas, nested helpers) only their scope does.
_MEMBER_CALLABLE_KINDS = frozenset({SymbolKind.METHOD, SymbolKind.FUNCTION})
#: The name every language's process entry point goes by.
_MAIN = "main"
#: Files a cloned body's detail names before it only counts the rest; directories a cycle's detail names too.
_FILES_SHOWN = 3


def _weakest(resolutions: Iterable[Resolution]) -> Resolution:
    """The weakest of some resolutions, EXACT for none."""
    return min(resolutions, key=lambda resolution: resolution.strength, default=Resolution.EXACT)


def _floor(signal: Signal) -> Resolution:
    """A graph signal's floor."""
    floor = signal.facts.floor
    assert floor is not None, f"{signal.value} is a graph signal"
    return floor


def _symbol_sample(symbol: Symbol, value: int, resolution: Resolution = Resolution.EXACT, detail: str = "") -> Sample:
    """A sample anchored on a declaration."""
    identity = symbol.id.split("#", 1)[-1]
    return Sample(
        symbol.file_path, symbol.start_line, symbol.end_line, display_name(symbol), identity, value, resolution, detail
    )


def _callables(facts: HealthFacts) -> Iterator[Symbol]:
    """Every production callable in the scope."""
    return (symbol for symbol in facts.production if symbol.kind in CALLABLE_KINDS)


def _types(facts: HealthFacts) -> Iterator[Symbol]:
    """Every named production type declaration in the scope."""
    return (symbol for symbol in facts.production if named_type(symbol))


def _measured(facts: HealthFacts, read: Callable[[Symbol, CallableMetrics], int]) -> Iterator[Sample]:
    """One sample per measured production callable, valued by one of its measurements."""
    for symbol in _callables(facts):
        metric = facts.metrics.get(symbol.id)
        if metric is not None:
            yield _symbol_sample(symbol, read(symbol, metric))


def _long_callable(facts: HealthFacts) -> Iterator[Sample]:
    """Callable length in lines, less the named callables declared directly inside it (`HealthFacts.own_lines`)."""
    return _measured(facts, facts.own_lines)


def _deep_nesting(facts: HealthFacts) -> Iterator[Sample]:
    """Deepest control-flow nesting."""
    return _measured(facts, lambda _, metric: metric.nesting)


def _complex_callable(facts: HealthFacts) -> Iterator[Sample]:
    """Cognitive complexity."""
    return _measured(facts, lambda _, metric: metric.complexity)


def _many_parameters(facts: HealthFacts) -> Iterator[Sample]:
    """Declared parameters of every callable, bodiless ones included."""
    return (_symbol_sample(symbol, symbol.arity) for symbol in _callables(facts))


def _large_type(facts: HealthFacts) -> Iterator[Sample]:
    """Line span of every type declaration."""
    return (_symbol_sample(symbol, symbol.end_line - symbol.start_line + 1) for symbol in _types(facts))


def _large_file(facts: HealthFacts) -> Iterator[Sample]:
    """Lines of every measured production file but one holding exactly one top-level type, which `large_type` judges."""
    for path, metric in facts.files:
        if facts.top_level_types[path] != 1:
            yield Sample(path, 1, max(metric.lines, 1), "", path, metric.lines)


def _many_types_per_file(facts: HealthFacts) -> Iterator[Sample]:
    """Top-level type declarations of every measured production file."""
    for path, metric in facts.files:
        yield Sample(path, 1, max(metric.lines, 1), "", path, facts.top_level_types[path])


def _descendants(facts: HealthFacts, symbol: Symbol, floor: Resolution) -> tuple[list[Symbol], Resolution, bool]:
    """The concrete production descendants of an abstraction, the weakest edge on the way, and any doubt.

    An interface or abstract type below the abstraction passes its own implementors up: an interface whose
    one direct subtype is a sub-interface with two implementors has two, not one. Doubt is a weaker edge
    at any abstraction on the way, or a test subtype of one (a seam for test doubles, which inlining would
    remove).
    """
    concrete: list[Symbol] = []
    weakest = Resolution.EXACT
    doubt = False
    seen: set[str] = set()
    pending = [symbol]
    while pending:
        current = pending.pop()
        strong, _, weak = facts.reaching(current, HIERARCHY_KINDS, floor, tests=False)
        if weak or facts.reaching(current, HIERARCHY_KINDS, floor, tests=True)[0] > strong:
            doubt = True
        for sub, resolution in facts.subtypes.get(current.id, ()):
            if sub.id in seen or not resolution.at_least(floor):
                continue
            seen.add(sub.id)
            weakest = min(weakest, resolution, key=lambda level: level.strength)
            if sub.kind is SymbolKind.INTERFACE or "abstract" in sub.modifiers:
                pending.append(sub)
            else:
                concrete.append(sub)
    return concrete, weakest, doubt


def _subtypes(facts: HealthFacts, symbol: Symbol, signal: Signal) -> Sample | None:
    """A type valued by its concrete production descendants (`_descendants`), or None when unmeasured.

    Unmeasured: a weaker edge or a test subtype casts doubt; or the one descendant lives in another module
    (an extension point by design) or outside a sub-directory view, where its module cannot be read.
    """
    concrete, weakest, doubt = _descendants(facts, symbol, _floor(signal))
    if doubt:
        return None
    if len(concrete) == 1 and facts.module_of(concrete[0].file_path) != facts.module_of(symbol.file_path):
        return None
    return _symbol_sample(symbol, len(concrete), weakest)


def _single_implementation_interface(facts: HealthFacts) -> Iterator[Sample]:
    """Production implementors of every interface, trait or protocol."""
    for symbol in _types(facts):
        if symbol.kind is SymbolKind.INTERFACE:
            sample = _subtypes(facts, symbol, Signal.SINGLE_IMPLEMENTATION_INTERFACE)
            if sample is not None:
                yield sample


def _single_subclass_abstract(facts: HealthFacts) -> Iterator[Sample]:
    """Production subtypes of every abstract type."""
    for symbol in _types(facts):
        if symbol.kind is not SymbolKind.INTERFACE and "abstract" in symbol.modifiers:
            sample = _subtypes(facts, symbol, Signal.SINGLE_SUBCLASS_ABSTRACT)
            if sample is not None:
                yield sample


def _deep_inheritance(facts: HealthFacts) -> Iterator[Sample]:
    """Ancestors of every production class on its longest chain of EXTENDS edges the view can see into.

    A supertype the graph does not hold ends the chain uncounted; an interface is not measured.
    """
    floor = _floor(Signal.DEEP_INHERITANCE)
    parents: dict[str, list[str]] = defaultdict(list)
    strength: dict[tuple[str, str], Resolution] = {}
    for edge in facts.links:
        dst = edge.dst_id
        if edge.kind is not EdgeKind.EXTENDS or dst is None or dst not in facts.symbols or dst == edge.src_id:
            continue
        if edge.resolution.at_least(floor):
            pair = (edge.src_id, dst)
            if pair not in strength:
                parents[edge.src_id].append(dst)
            strength[pair] = max(strength.get(pair, edge.resolution), edge.resolution, key=lambda level: level.strength)
    depths = ancestor_depths(parents)
    for symbol in _types(facts):
        if symbol.kind is not SymbolKind.CLASS:
            continue
        depth, chain = depths.get(symbol.id, (0, ()))
        steps = zip((symbol.id, *chain), chain, strict=False)
        detail = f"up to {display_name(facts.symbols[chain[-1]])}" if chain else ""
        yield _symbol_sample(symbol, depth, _weakest(strength[step] for step in steps), detail)


def _entry_point(facts: HealthFacts, symbol: Symbol) -> bool:
    """Whether something other than a written call reaches this callable.

    A constructor is reached through its type, `main` by the launcher, an annotated declaration by
    the framework the annotation registers it with, an override through its supertype, a runtime
    hook by the interpreter, and a member of an anonymous class through the type it implements.
    """
    return (
        symbol.kind is SymbolKind.CONSTRUCTOR
        or symbol.name == _MAIN
        or bool(symbol.annotations)
        or symbol.id in facts.may_override
        or facts.is_runtime_hook(symbol)
        or anonymous(facts.container(symbol))
    )


def _named_callables(facts: HealthFacts) -> Iterator[Symbol]:
    """Production methods and functions declared in a type or file that no rule says are reached otherwise."""
    for symbol in _callables(facts):
        if symbol.kind not in _MEMBER_CALLABLE_KINDS or _entry_point(facts, symbol):
            continue
        container = facts.container(symbol)
        if container is None or container.kind in TYPE_KINDS:
            yield symbol


def _lone_public_callable(facts: HealthFacts) -> Iterator[Sample]:
    """Production callers outside its own outermost type of every called public method or function.

    A callable no production code calls is not measured: a caller outside the workspace and dead code
    look the same. One whose name is written in another file is not measured either: the graph has no
    edge for a reference used as a value (`Type::member`), and making that one private would break it.
    """
    floor = _floor(Signal.LONE_PUBLIC_CALLABLE)
    counted: list[tuple[Symbol, int, int, Resolution]] = []
    for symbol in _named_callables(facts):
        if facts.visibility(symbol) is not Visibility.PUBLIC or facts.in_contract(symbol):
            continue
        strong, weakest, doubt = facts.reaching(symbol, _CALLS, floor, tests=False)
        if doubt or not strong:
            continue
        home = facts.home_of(symbol.id)
        outside = sum(
            sources
            for caller, resolution, from_test, sources in facts.callers.get(symbol.id, ())
            if not from_test and resolution.at_least(floor) and caller != home
        )
        counted.append((symbol, strong, outside, weakest))
    inside = [symbol for symbol, _, outside, _ in counted if not outside]
    silent = {symbol.id for symbol in facts.unmentioned(inside, tests=False, own_file=False)}
    for symbol, strong, outside, weakest in counted:
        if outside:
            yield _symbol_sample(symbol, outside, weakest)
        elif symbol.id in silent:
            home = facts.home_of(symbol.id)
            owner = facts.symbols.get(home)
            where = display_name(owner) if owner is not None else PurePosixPath(home).name
            tested = facts.reaching(symbol, _CALLS, floor, tests=True)[0] > strong
            detail = f"{strong} caller(s), all in {where}" + ("; tests call it too" if tested else "")
            yield _symbol_sample(symbol, 0, weakest, detail)


def _unreferenced_symbol(facts: HealthFacts) -> Iterator[Sample]:
    """Inbound edges of any kind, tests included, of every non-public callable and type.

    A type is also referenced through its constructors: `new Foo()` is a call to one of them. A
    type that is annotated, or declares an annotated member (`@Test`, `@Inject`), is one a framework
    instantiates. An unreferenced one must also go unmentioned in the source it could be referenced from.
    """
    floor = Signal.UNREFERENCED_SYMBOL.facts.floor
    assert floor is not None
    constructors: dict[str, list[Symbol]] = defaultdict(list)
    annotated: set[str] = set()
    for symbol in facts.symbols.values():
        if symbol.kind is SymbolKind.CONSTRUCTOR and symbol.container_id:
            constructors[symbol.container_id].append(symbol)
        if symbol.annotations:
            annotated.update(ident for ident in (symbol.id, symbol.container_id) if ident)
    types = [symbol for symbol in _types(facts) if symbol.id not in annotated]
    counted: list[tuple[Symbol, int, Resolution]] = []
    for symbol in (*_named_callables(facts), *types):
        if facts.visibility(symbol) in _RESTRICTED:
            reached = _reached(facts, (symbol, *constructors.get(symbol.id, ())), floor)
            if reached is not None:
                counted.append((symbol, *reached))
    silent = _unmentioned_absences(facts, counted, tests=True)
    for symbol, strong, weakest in counted:
        if strong or symbol.id in silent:
            yield _symbol_sample(symbol, strong, weakest, "" if strong else "nothing references it")


def _reached(facts: HealthFacts, targets: tuple[Symbol, ...], floor: Resolution) -> tuple[int, Resolution] | None:
    """Sources reaching any of the targets by any edge and their weakest resolution, or None when unprovable."""
    strong, weakest = 0, Resolution.EXACT
    for target in targets:
        count, resolution, doubt = facts.reaching(target, _ANY_EDGE, floor, tests=True)
        if doubt:
            return None
        strong += count
        weakest = min(weakest, resolution, key=lambda level: level.strength)
    return strong, weakest


def _unmentioned_absences(
    facts: HealthFacts, counted: list[tuple[Symbol, int, Resolution]], *, tests: bool
) -> set[str]:
    """Ids of the counted symbols nothing reaches whose name the source does not write elsewhere either."""
    return {
        symbol.id for symbol in facts.unmentioned([symbol for symbol, strong, _ in counted if not strong], tests=tests)
    }


def _forbidden_crossing(facts: HealthFacts) -> Iterator[Sample]:
    """Forbidden edges from each production file into each other module it reaches.

    Measured only where the workspace declares forbidden dependencies; a (file, module) pair is
    measured when at least one of its edges is resolved at the signal's floor, and only those
    edges count.
    """
    floor = Signal.FORBIDDEN_CROSSING.facts.floor
    assert floor is not None
    config = facts.config
    if config.generic or not config.forbidden:
        return
    pairs: dict[tuple[str, str], list[FileDependency]] = defaultdict(list)
    for dependency in facts.dependencies:
        if dependency.resolution.at_least(floor):
            target = facts.module_of(dependency.target_path)
            if target != facts.module_of(dependency.source_path):
                pairs[(dependency.source_path, target)].append(dependency)
    lines = {path: metric.lines for path, metric in facts.files}
    for (path, target), dependencies in sorted(pairs.items()):
        rule = config.forbids(facts.module_of(path), target)
        edges = sum(dependency.edges for dependency in dependencies) if rule is not None else 0
        weakest = min((dependency.resolution for dependency in dependencies), key=lambda level: level.strength)
        detail = rule.describe() if rule is not None else ""
        yield Sample(path, 1, max(lines.get(path, 1), 1), f"-> {target}", f"{path}->{target}", edges, weakest, detail)


def _package_cycle(facts: HealthFacts) -> Iterator[Sample]:
    """Directories of every production directory's dependency cycle inside its own module, 1 when acyclic.

    A node is a (module, directory) pair: an edge between modules is never part of a cycle here, so a
    by-name edge resolved into the wrong module cannot close one. A directory in a cycle is anchored at the
    first file of its lightest link into the cycle (the fewest edges into one other directory of it): the
    cheapest place to cut. The weakest edge reported is the weakest of its links into the cycle.
    """
    floor = _floor(Signal.PACKAGE_CYCLE)
    lines = {path: max(metric.lines, 1) for path, metric in facts.files}
    members: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in lines:
        members[(facts.module_of(path), posixpath.dirname(path))].append(path)
    successors: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    links: dict[tuple[tuple[str, str], tuple[str, str]], tuple[Resolution, int, set[str]]] = {}
    for dependency in facts.dependencies:
        if not dependency.resolution.at_least(floor) or dependency.target_path not in lines:
            continue
        source = (facts.module_of(dependency.source_path), posixpath.dirname(dependency.source_path))
        target = (facts.module_of(dependency.target_path), posixpath.dirname(dependency.target_path))
        if source == target or source[0] != target[0]:
            continue
        successors[source].add(target)
        strongest, edges, files = links.get((source, target), (dependency.resolution, 0, set()))
        strongest = max(strongest, dependency.resolution, key=lambda level: level.strength)
        files.add(dependency.source_path)
        links[(source, target)] = (strongest, edges + dependency.edges, files)
    for component in strongly_connected(sorted(members), successors):
        cycle = set(component)
        group = f"cycle {component[0][0]}:{min(directory or '.' for _, directory in component)}"
        for node in component:
            module, directory = node
            name = directory or "."
            inward = [(links[(node, other)], other[1] or ".") for other in successors.get(node, ()) if other in cycle]
            if not inward:
                path = min(members[node])
                yield Sample(path, 1, lines[path], name, f"{module}:{name}", 1)
                continue
            (_, edges, files), into = min(inward, key=lambda link: (link[0][1], link[1]))
            anchor = min(files)
            others = sorted(other[1] or "." for other in component if other != node)
            more = f" and {len(others) - _FILES_SHOWN} more" if len(others) > _FILES_SHOWN else ""
            also = f" and {len(files) - 1} more file(s)" if len(files) > 1 else ""
            detail = (
                f"in a cycle with {', '.join(others[:_FILES_SHOWN])}{more}; its lightest link into it: "
                f"{edges} edge(s) from {PurePosixPath(anchor).name}{also} into {into}"
            )
            weakest = _weakest(resolution for (resolution, _, _), _ in inward)
            yield Sample(
                anchor, 1, lines[anchor], name, f"{module}:{name}", len(component), weakest, detail, group=group
            )


def _feature_envy(facts: HealthFacts) -> Iterator[Sample]:
    """Getter calls into the most-read other type of every production method that reads it more than its own.

    The other type's calls are the resolved data reads (`HealthFacts.reads_data`: written on a receiver that
    is no type name, into a getter); its own are every resolved call into its own outermost type or a
    supertype of it, bare ones included, an inherited member being the type's own. Not measured: a method
    without a measured body, declared in no type, or whose place a contract or framework fixes (a member of
    an interface, `_entry_point`: an override, an annotated method), which cannot move. A type the view does
    not hold is never the envied one.
    """
    floor = _floor(Signal.FEATURE_ENVY)
    above = _supertypes(facts, floor)
    for symbol in _callables(facts):
        if symbol.kind is not SymbolKind.METHOD or symbol.id not in facts.metrics or facts.in_contract(symbol):
            continue
        if _entry_point(facts, symbol):
            continue
        home = facts.home_of(symbol.id)
        if home not in facts.symbols:
            continue
        own_types = {home, *above(home)}
        own = 0
        foreign: Counter[str] = Counter()
        weakest: dict[str, Resolution] = {}
        for callee, resolution, reads, calls in facts.member_calls.get(symbol.id, ()):
            if not resolution.at_least(floor):
                continue
            if callee in own_types:
                own += calls
            elif reads and (target := facts.symbols.get(callee)) is not None and target.kind in DECLARED_TYPE_KINDS:
                foreign[callee] += calls
                weakest[callee] = _weakest((weakest.get(callee, resolution), resolution))
        envied, most = min(foreign.items(), key=lambda item: (-item[1], item[0]), default=("", 0))
        if most > own:
            detail = f"{most} getter calls into {display_name(facts.symbols[envied])}, {own} calls into its own type"
            yield _symbol_sample(symbol, most, weakest[envied], detail)
        else:
            yield _symbol_sample(symbol, 0)


def _supertypes(facts: HealthFacts, floor: Resolution) -> Callable[[str], set[str]]:
    """A memoised lookup of every supertype the view holds of a type, over hierarchy edges at `floor` or better."""
    parents: dict[str, set[str]] = defaultdict(set)
    for edge in facts.links:
        if edge.kind in HIERARCHY_KINDS and edge.dst_id in facts.symbols and edge.resolution.at_least(floor):
            parents[edge.src_id].add(facts.home_of(edge.dst_id))
    found: dict[str, set[str]] = {}

    def above(type_id: str) -> set[str]:
        if type_id not in found:
            seen: set[str] = set()
            pending = list(parents.get(type_id, ()))
            while pending:
                current = pending.pop()
                if current not in seen and current != type_id:
                    seen.add(current)
                    pending.extend(parents.get(current, ()))
            found[type_id] = seen
        return found[type_id]

    return above


def _place(unit: Unit) -> tuple[str, int, int]:
    """Where a duplication unit lives."""
    return unit.file_path, unit.start_line, unit.end_line


def _callable_of(facts: HealthFacts, unit: Unit) -> Symbol | None:
    """The production callable a duplication body is: same file and simple name, the widest line overlap."""
    simple = unit.name.rsplit(".", 1)[-1]
    best, overlap = None, 0
    for symbol in facts.callables_by_file.get(unit.file_path, ()):
        shared = min(symbol.end_line, unit.end_line) - max(symbol.start_line, unit.start_line) + 1
        if symbol.name == simple and shared > overlap:
            best, overlap = symbol, shared
    return best


def _bodies(facts: HealthFacts) -> Iterator[tuple[Unit, Symbol]]:
    """Every production shaped body the duplication run judged, once per graph callable it is; unmatched ones never."""
    copying = facts.copying
    if copying is None:
        return
    seen: set[str] = set()
    for unit in copying.shaped:
        if facts.in_scope(unit.file_path) and copying.judged(unit.file_path):
            symbol = _callable_of(facts, unit)
            if symbol is not None and symbol.id not in seen:
                seen.add(symbol.id)
                yield unit, symbol


def _clone_mass(facts: HealthFacts) -> Iterator[Sample]:
    """Thousandths of every judged production file's lines that sit inside an exact or renamed clone class."""
    copying = facts.copying
    if copying is None:
        return
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for clone in copying.classes:
        if clone.kind in CODE_KINDS:
            for member in clone.members:
                spans[member.file_path].append((member.start_line, member.end_line))
    for path, metric in facts.files:
        if profile_for(path) is None or not copying.judged(path):
            continue
        cloned = sum(end - start + 1 for start, end in merge_ranges(spans.get(path, [])))
        total = max(metric.lines, 1)
        yield Sample(path, 1, total, "", path, cloned * 1000 // total, detail=f"{cloned} of {metric.lines} lines")


def _copies_elsewhere(copying: CopyingFacts) -> dict[tuple[str, int, int], set[str]]:
    """Per body member of an exact or renamed clone class, the other files holding a copy of it."""
    others: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for clone in copying.classes:
        if clone.kind not in CODE_KINDS:
            continue
        files = {member.file_path for member in clone.members}
        for member in clone.members:
            if member.is_body:
                others[_place(member)] |= files - {member.file_path}
    return others


def _cloned_body(facts: HealthFacts) -> Iterator[Sample]:
    """Other files holding an exact or renamed copy of every judged production body of clone size."""
    copying = facts.copying
    if copying is None:
        return
    others = _copies_elsewhere(copying)
    for unit, symbol in _bodies(facts):
        if unit.token_count < copying.min_tokens:
            continue
        elsewhere = sorted(others.get(_place(unit), ()))
        more = f" and {len(elsewhere) - _FILES_SHOWN} more" if len(elsewhere) > _FILES_SHOWN else ""
        detail = f"copies in {', '.join(elsewhere[:_FILES_SHOWN])}{more}" if elsewhere else ""
        yield _symbol_sample(symbol, len(elsewhere), detail=detail)


def _pass_through(facts: HealthFacts) -> Iterator[Sample]:
    """Whether every judged public production body with parameters, outside any type contract, only forwards.

    A body taking no parameter passes nothing through: it names a value (`return Set.of();`, a hook's
    default). A body that is part of a contract (declared on an interface, annotated as implementing one,
    overriding or overridden) is a role, and a constructor handing over to another is an overload; none is
    measured. Neither is a body in a language whose profile cannot read forwarding (no shape hooks).
    """
    for unit, symbol in _bodies(facts):
        profile = profile_for(unit.file_path)
        if profile is None or profile.shapes is None or not symbol.arity:
            continue
        if unit.visibility is not Visibility.PUBLIC or unit.implements_contract or unit.delegates:
            continue
        if facts.in_contract(symbol) or symbol.id in facts.may_override or facts.overridden(symbol):
            continue
        forwards = unit.forwards_to is not None
        yield _symbol_sample(symbol, int(forwards), detail=f"forwards to {unit.forwards_to}" if forwards else "")


def _vocabulary_drift(facts: HealthFacts) -> Iterator[Sample]:
    """Places of every vocabulary class with a judged member, and 1 for every other judged declaration site.

    A class is anchored at its first judged member and carries every other place it is declared in.
    """
    copying = facts.copying
    if copying is None:
        return

    def judged(unit: Unit) -> bool:
        return facts.in_scope(unit.file_path) and copying.judged(unit.file_path)

    classed: set[tuple[str, int, int]] = set()
    for clone in copying.classes:
        if clone.kind is not CloneKind.VOCABULARY:
            continue
        classed.update(_place(member) for member in clone.members)
        anchor = next((member for member in clone.members if judged(member)), None)
        if anchor is not None:
            elsewhere = tuple(_place(member) for member in clone.members if member is not anchor)
            detail = clone.notes[0] if clone.notes else ""
            places = len(clone.members)
            yield Sample(*_place(anchor), anchor.name, clone.key, places, detail=detail, elsewhere=elsewhere)
    for site in copying.sites:
        if SiteKind(site.kind).declares and judged(site) and _place(site) not in classed:
            yield Sample(*_place(site), site.name, f"{site.file_path}:{site.name}", 1)


def _style_samples(facts: HealthFacts, distances: dict[str, Distance]) -> Iterator[Sample]:
    """One sample per file of the scope a style norm judged, valued by its distance in hundredths of the cut."""
    for path, metric in facts.files:
        distance = distances.get(path)
        if distance is not None:
            yield Sample(path, 1, max(metric.lines, 1), "", path, distance.value, detail=distance.detail)


def _style_outlier_repo(facts: HealthFacts) -> Iterator[Sample]:
    """Every measured file's style distance from its language's files in the view (`zemble.health.style`)."""
    return _style_samples(facts, facts.style.repo)


def _style_outlier_directory(facts: HealthFacts) -> Iterator[Sample]:
    """Every measured file's style distance from its directory's files, where those are not its whole language."""
    return _style_samples(facts, facts.style.directory)


def _hotspot_cause(facts: HealthFacts, path: str, lines: int) -> str:
    """What makes a file a hotspot candidate (its most complex callable, its size), or "" when nothing does."""
    complex_limit = Signal.COMPLEX_CALLABLE.facts.threshold
    causes = []
    measured = [
        (metric.complexity, symbol)
        for symbol in facts.callables_by_file.get(path, ())
        if (metric := facts.metrics.get(symbol.id)) is not None
    ]
    if measured:
        complexity, symbol = max(measured, key=lambda item: (item[0], -item[1].start_line))
        if complexity > complex_limit:
            causes.append(f"complexity {complexity} in {display_name(symbol)}")
    if lines > Signal.LARGE_FILE.facts.threshold:
        causes.append(f"{lines} lines")
    return ", ".join(causes)


def _hotspot(facts: HealthFacts) -> Iterator[Sample]:
    """Every changed production file of the scope, valued by its churn rank when it is complex or large, else 0.

    The rank is in thousandths of the changed production files of the view in the same checkout that
    changed strictly less often: each checkout of a workspace is its own history, so one busy repository
    does not set the bar for the others. Files tied at the top rank below every one of them, so an
    evenly changed tree has none. A file neither complex nor large, or with fewer than
    `HOTSPOT_MIN_COMMITS` commits in the window, is measured at 0: it changed, and it is no hotspot.
    A file the window never touched is not measured.
    """
    history = facts.history
    if history is None:
        return
    churns: dict[str, list[int]] = defaultdict(list)
    for path, ids in history.commits.items():
        churns[history.checkout_of(path)].append(len(ids))
    for values in churns.values():
        values.sort()
    for path, metric in facts.files:
        churn = history.churn(path)
        if not churn:
            continue
        cause = _hotspot_cause(facts, path, metric.lines) if churn >= HOTSPOT_MIN_COMMITS else ""
        peers = churns[history.checkout_of(path)]
        rank = bisect_left(peers, churn) * 1000 // len(peers) if cause else 0
        detail = f"{churn} commits in {history.window_days} days" + (f"; {cause}" if cause else "")
        yield Sample(path, 1, max(metric.lines, 1), "", path, rank, detail=detail)


def _boundary_co_change(facts: HealthFacts) -> Iterator[Sample]:
    """Every co-changing pair of production files in different modules, valued by their commits' Jaccard.

    A pair is anchored at its first file in the scope and carries the other as a place it also lives.
    Two files of the same name (a generated per-package `sst-env.d.ts`, one `cli.py` per package) are
    instances of one convention that a single sweep edits: they are not measured.
    """
    history = facts.history
    if history is None:
        return
    lines = {path: max(metric.lines, 1) for path, metric in facts.population}
    for pair in history.co_changes:
        if facts.module_of(pair.first) == facts.module_of(pair.second):
            continue
        if PurePosixPath(pair.first).name == PurePosixPath(pair.second).name:
            continue
        anchor, other = (pair.first, pair.second) if facts.in_scope(pair.first) else (pair.second, pair.first)
        if not facts.in_scope(anchor):
            continue
        detail = f"changed together in {pair.shared} of {pair.either} commits with {other}"
        value = pair.shared * 1000 // pair.either
        identity = f"{pair.first}<->{pair.second}"
        elsewhere = ((other, 1, lines[other]),)
        yield Sample(anchor, 1, lines[anchor], "", identity, value, detail=detail, elsewhere=elsewhere)


def _public_api(facts: HealthFacts, symbol: Symbol) -> bool:
    """Whether a symbol is public and so is every type around it, as Checkstyle's surrounding scope reads it."""
    current: Symbol | None = symbol
    while current is not None and (current is symbol or current.kind in DECLARED_TYPE_KINDS):
        if facts.visibility(current) is not Visibility.PUBLIC:
            return False
        current = facts.container(current)
    return True


def _api_declaration(facts: HealthFacts, symbol: Symbol) -> bool:
    """Whether a reader of the public API needs this declaration's doc, which `_undocumented_public` measures.

    A named type or a method or function declared in a type or file counts. Not a constructor (its
    type says what it builds), an entry point (`_entry_point`: an override's supertype documents the
    contract, an annotation's registration does, `__repr__` the language, `main` the launcher), a
    member of an anonymous type, or a one-statement body (`CallableMetrics.is_boilerplate`: an accessor
    says what its name says).
    """
    container = facts.container(symbol)
    if container is not None and container.kind in CALLABLE_KINDS:
        return False
    if named_type(symbol):
        return _public_api(facts, symbol)
    if symbol.kind not in _MEMBER_CALLABLE_KINDS or _entry_point(facts, symbol):
        return False
    metric = facts.metrics.get(symbol.id)
    if metric is not None and metric.is_boilerplate:
        return False
    return _public_api(facts, symbol)


def _undocumented_public(facts: HealthFacts) -> Iterator[Sample]:
    """Doc lines of every public API declaration of the scope (`_api_declaration`), anchored on its first line.

    The doc goes above that line, so a `--since` run lists the finding when the declaration is new or
    re-declared, not on every edit inside a body or a type. UNKNOWN visibility is never public.
    """
    for symbol in facts.production:
        if _api_declaration(facts, symbol):
            yield replace(_symbol_sample(symbol, symbol.doc_lines), end_line=symbol.start_line)


def _long_docblock(facts: HealthFacts) -> Iterator[Sample]:
    """Doc lines of every documented declaration of the scope (the extractors document types, callables, fields)."""
    for symbol in facts.production:
        if symbol.doc_lines:
            yield _symbol_sample(symbol, symbol.doc_lines)


def _commented_out_code(facts: HealthFacts) -> Iterator[Sample]:
    """Lines of comment blocks that are nothing but code (`metrics.disabled_code`), per measured production file."""
    for path, metric in facts.files:
        yield Sample(path, 1, max(metric.lines, 1), "", path, metric.commented_code_lines)


def _stale_marker(facts: HealthFacts) -> Iterator[Sample]:
    """Comment lines holding an unfinished-work marker, of every measured production file.

    A file past the bar is anchored on its first marker line, read off the file now (the graph keeps the
    count only), with the other lines in the detail; one that moved since the graph read it keeps the whole
    file as its span.
    """
    for path, metric in facts.files:
        if not metric.marker_lines:
            yield Sample(path, 1, max(metric.lines, 1), "", path, 0)
            continue
        rows = _marker_rows(facts.absolute(path))
        if not rows:
            yield Sample(path, 1, max(metric.lines, 1), "", path, metric.marker_lines)
            continue
        more = f" and {len(rows) - _FILES_SHOWN} more" if len(rows) > _FILES_SHOWN else ""
        detail = "line " + ", ".join(map(str, rows[:_FILES_SHOWN])) + more
        yield Sample(path, rows[0], rows[0], "", path, metric.marker_lines, detail=detail)


def _marker_rows(absolute: str) -> list[int]:
    """The 1-based lines of a file holding a marker (`MARKER`), in order; empty when it cannot be read."""
    try:
        with open(absolute, encoding="utf-8", errors="replace") as handle:
            return [number for number, line in enumerate(handle, start=1) if MARKER.search(line)]
    except OSError:
        return []


#: The one producer of every signal; a member without one refuses at import below.
PRODUCERS: dict[Signal, Producer] = {
    Signal.LONG_CALLABLE: _long_callable,
    Signal.MANY_PARAMETERS: _many_parameters,
    Signal.DEEP_NESTING: _deep_nesting,
    Signal.COMPLEX_CALLABLE: _complex_callable,
    Signal.LARGE_TYPE: _large_type,
    Signal.LARGE_FILE: _large_file,
    Signal.MANY_TYPES_PER_FILE: _many_types_per_file,
    Signal.SINGLE_IMPLEMENTATION_INTERFACE: _single_implementation_interface,
    Signal.SINGLE_SUBCLASS_ABSTRACT: _single_subclass_abstract,
    Signal.DEEP_INHERITANCE: _deep_inheritance,
    Signal.LONE_PUBLIC_CALLABLE: _lone_public_callable,
    Signal.UNREFERENCED_SYMBOL: _unreferenced_symbol,
    Signal.FORBIDDEN_CROSSING: _forbidden_crossing,
    Signal.PACKAGE_CYCLE: _package_cycle,
    Signal.FEATURE_ENVY: _feature_envy,
    Signal.CLONE_MASS: _clone_mass,
    Signal.CLONED_BODY: _cloned_body,
    Signal.PASS_THROUGH: _pass_through,
    Signal.VOCABULARY_DRIFT: _vocabulary_drift,
    Signal.STYLE_OUTLIER_REPO: _style_outlier_repo,
    Signal.STYLE_OUTLIER_DIRECTORY: _style_outlier_directory,
    Signal.HOTSPOT: _hotspot,
    Signal.BOUNDARY_CO_CHANGE: _boundary_co_change,
    Signal.UNDOCUMENTED_PUBLIC: _undocumented_public,
    Signal.LONG_DOCBLOCK: _long_docblock,
    Signal.COMMENTED_OUT_CODE: _commented_out_code,
    Signal.STALE_MARKER: _stale_marker,
}

_missing = set(Signal) - set(PRODUCERS)
if _missing:  # pragma: no cover - a drift the import itself refuses
    raise RuntimeError(f"health signals without a producer: {sorted(signal.value for signal in _missing)}")


def measure(facts: HealthFacts, signals: tuple[Signal, ...]) -> dict[Signal, list[Sample]]:
    """Run the producers of the selected signals, in declaration order."""
    return {signal: list(PRODUCERS[signal](facts)) for signal in Signal if signal in signals}


__all__ = ["PRODUCERS", "Sample", "measure"]
