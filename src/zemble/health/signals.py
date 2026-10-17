"""Turning collected facts into samples: one producer per `Signal`, each yielding its whole population.

A producer yields every unit it MEASURED, past the threshold or not: the population is what
percentiles and scores are computed over. A unit a weaker edge might also reach is not measured
at all (an absence it cannot vouch for, a count it cannot finish), so it neither scores nor ranks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from zemble.graph.model import (
    ANONYMOUS_MARKER,
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
from zemble.health.collect import HIERARCHY_KINDS, HealthFacts, language_of
from zemble.health.model import Signal
from zemble.languages import Visibility
from zemble.languages.metrics import CallableMetrics


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
#: Containers whose declared types count as top-level.
_FILE_LEVEL = frozenset({SymbolKind.MODULE, SymbolKind.PACKAGE})
#: The name every language's process entry point goes by.
_MAIN = "main"


def _symbol_sample(symbol: Symbol, value: int, resolution: Resolution = Resolution.EXACT, detail: str = "") -> Sample:
    """A sample anchored on a declaration."""
    identity = symbol.id.split("#", 1)[-1]
    return Sample(
        symbol.file_path, symbol.start_line, symbol.end_line, display_name(symbol), identity, value, resolution, detail
    )


def _anonymous(symbol: Symbol | None) -> bool:
    """Whether a symbol is an anonymous class, which nothing can name."""
    return symbol is not None and symbol.name.startswith(ANONYMOUS_MARKER)


def _callables(facts: HealthFacts) -> Iterator[Symbol]:
    """Every production callable in the scope."""
    return (symbol for symbol in facts.production if symbol.kind in CALLABLE_KINDS)


def _types(facts: HealthFacts) -> Iterator[Symbol]:
    """Every named production type declaration in the scope."""
    return (symbol for symbol in facts.production if symbol.kind in DECLARED_TYPE_KINDS and not _anonymous(symbol))


def _measured(facts: HealthFacts, read: Callable[[Symbol, CallableMetrics], int]) -> Iterator[Sample]:
    """One sample per measured production callable, valued by one of its measurements."""
    for symbol in _callables(facts):
        metric = facts.metrics.get(symbol.id)
        if metric is not None:
            yield _symbol_sample(symbol, read(symbol, metric))


def _long_callable(facts: HealthFacts) -> Iterator[Sample]:
    """Callable length in lines, less the named callables declared directly inside it.

    A nested function (a decorated tool inside its `register_*`, a closure) is measured on its own,
    so counting its lines again in the enclosing callable would flag one body twice.
    """
    nested: Counter[str] = Counter()
    for symbol in facts.symbols.values():
        container = facts.container(symbol)
        if symbol.kind in CALLABLE_KINDS and container is not None and container.kind in CALLABLE_KINDS:
            nested[container.id] += symbol.end_line - symbol.start_line + 1
    return _measured(facts, lambda symbol, metric: max(metric.lines - nested[symbol.id], 0))


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
    """Lines of every measured production file."""
    for path, metric in facts.files:
        yield Sample(path, 1, max(metric.lines, 1), "", path, metric.lines)


def _many_types_per_file(facts: HealthFacts) -> Iterator[Sample]:
    """Top-level type declarations of every measured production file."""
    counts: Counter[str] = Counter()
    for symbol in _types(facts):
        container = facts.container(symbol)
        if container is None or container.kind in _FILE_LEVEL:
            counts[symbol.file_path] += 1
    for path, metric in facts.files:
        yield Sample(path, 1, max(metric.lines, 1), "", path, counts[path])


def _subtypes(facts: HealthFacts, symbol: Symbol, signal: Signal) -> Sample | None:
    """A type valued by its production subtypes, or None when a weaker edge might add one."""
    floor = signal.facts.floor
    assert floor is not None, f"{signal.value} is a graph signal"
    strong, weakest, doubt = facts.reaching(symbol, HIERARCHY_KINDS, floor, tests=False)
    return None if doubt else _symbol_sample(symbol, strong, weakest)


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
        or _anonymous(facts.container(symbol))
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
    """Production callers of every public method or function; an uncalled one must also go unmentioned."""
    floor = Signal.LONE_PUBLIC_CALLABLE.facts.floor
    assert floor is not None
    counted: list[tuple[Symbol, int, Resolution]] = []
    for symbol in _named_callables(facts):
        if facts.visibility(symbol) is not Visibility.PUBLIC:
            continue
        strong, weakest, doubt = facts.reaching(symbol, _CALLS, floor, tests=False)
        if not doubt:
            counted.append((symbol, strong, weakest))
    silent = _unmentioned_absences(facts, counted, tests=False)
    for symbol, strong, weakest in counted:
        if strong:
            yield _symbol_sample(symbol, strong, weakest, "one production caller")
        elif symbol.id in silent:
            tested = facts.reaching(symbol, _CALLS, floor, tests=True)[0]
            yield _symbol_sample(symbol, 0, weakest, "called only from tests" if tested else "never called")


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
    Signal.LONE_PUBLIC_CALLABLE: _lone_public_callable,
    Signal.UNREFERENCED_SYMBOL: _unreferenced_symbol,
    Signal.FORBIDDEN_CROSSING: _forbidden_crossing,
}

_missing = set(Signal) - set(PRODUCERS)
if _missing:  # pragma: no cover - a drift the import itself refuses
    raise RuntimeError(f"health signals without a producer: {sorted(signal.value for signal in _missing)}")


def measure(facts: HealthFacts, signals: tuple[Signal, ...]) -> dict[Signal, list[Sample]]:
    """Run the producers of the selected signals, in declaration order."""
    return {signal: list(PRODUCERS[signal](facts)) for signal in Signal if signal in signals}


__all__ = ["PRODUCERS", "Sample", "measure"]
