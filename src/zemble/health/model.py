"""The vocabularies `zemble health` speaks in: axes, tiers, units, comparisons, signals and findings.

AIDEV-NOTE: every per-signal decision (axis, tier, unit, threshold, comparison, resolution floor)
reads `Signal.facts`, whose table names every member; a member missing from it raises on first use,
and `zemble.health.signals` refuses at import to run without a producer for every member.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from zemble.graph.model import Resolution

if TYPE_CHECKING:  # pragma: no cover
    from zemble.health.git import ChangedLines


class Axis(str, Enum):
    """What a group of signals says about the code; one weight each in `zemble.health.score`.

    CONSISTENCY and HISTORY join when a signal measures them; an axis without a signal would only
    ever read "not measured".
    """

    SIZE = "size"
    MACHINERY = "machinery"
    LAYERING = "layering"
    COPYING = "copying"
    VOCABULARY = "vocabulary"


class Tier(str, Enum):
    """What a signal rests on."""

    #: The parse alone: true for every measured language.
    SYNTACTIC = "syntactic"
    #: Graph edges: only as good as their resolution, so each such signal names a floor.
    GRAPH = "graph"
    #: The duplication channels' units and clone classes (`zemble.dedup`): a parse of every file beside the
    #: graph, so a run over a whole workspace measures it only when asked (`zemble.health.copying`).
    DUPLICATION = "duplication"


class Unit(str, Enum):
    """What a signal's value counts."""

    LINES = "lines"
    COUNT = "count"
    LEVEL = "level"
    #: Thousandths of a whole: a share kept integral, as every value is.
    PERMILLE = "permille"

    @property
    def weighs_by_value(self) -> bool:
        """Whether a measured unit weighs its own value in the score (lines) rather than 1."""
        return self is Unit.LINES


class Comparison(str, Enum):
    """How a value is judged against its signal's threshold."""

    #: Past the threshold when strictly above it; a larger value is worse.
    ABOVE = "above"
    #: Past the threshold when at most it; a smaller value is worse.
    AT_MOST = "at_most"
    #: Past the threshold when equal to it; the further from it, the less suspect.
    EQUALS = "equals"

    def past(self, value: float, threshold: float) -> bool:
        """Whether a value is a finding."""
        return _PAST[self](value, threshold)

    def badness(self, value: float, threshold: float) -> float:
        """A key that grows with how suspect a value is, for percentiles."""
        return _BADNESS[self](value, threshold)

    @property
    def symbol(self) -> str:
        """How a report writes the comparison between a value and its threshold."""
        return _SYMBOLS[self]


_PAST = {
    Comparison.ABOVE: lambda value, threshold: value > threshold,
    Comparison.AT_MOST: lambda value, threshold: value <= threshold,
    Comparison.EQUALS: lambda value, threshold: value == threshold,
}
_BADNESS = {
    Comparison.ABOVE: lambda value, threshold: value,
    Comparison.AT_MOST: lambda value, threshold: -value,
    Comparison.EQUALS: lambda value, threshold: -abs(value - threshold),
}
_SYMBOLS = {Comparison.ABOVE: ">", Comparison.AT_MOST: "<=", Comparison.EQUALS: "=="}


@dataclass(frozen=True, slots=True)
class SignalFacts:
    """The per-signal facts every consumer reads instead of switching over the signal."""

    axis: Axis
    tier: Tier
    description: str
    unit: Unit
    #: Absolute and declared; never derived from the repo being measured.
    threshold: int
    comparison: Comparison
    #: GRAPH only: the weakest resolution an edge may have to count; weaker edges never support a finding.
    floor: Resolution | None
    #: Where the threshold comes from.
    source: str
    #: What a reader does about a finding, as the start of its reason line.
    advice: str


class Signal(str, Enum):
    """One measured code shape; its facts live in `_SIGNAL_FACTS`."""

    LONG_CALLABLE = "long_callable"
    MANY_PARAMETERS = "many_parameters"
    DEEP_NESTING = "deep_nesting"
    COMPLEX_CALLABLE = "complex_callable"
    LARGE_TYPE = "large_type"
    LARGE_FILE = "large_file"
    MANY_TYPES_PER_FILE = "many_types_per_file"
    SINGLE_IMPLEMENTATION_INTERFACE = "single_implementation_interface"
    SINGLE_SUBCLASS_ABSTRACT = "single_subclass_abstract"
    LONE_PUBLIC_CALLABLE = "lone_public_callable"
    UNREFERENCED_SYMBOL = "unreferenced_symbol"
    FORBIDDEN_CROSSING = "forbidden_crossing"
    CLONE_MASS = "clone_mass"
    CLONED_BODY = "cloned_body"
    PASS_THROUGH = "pass_through"
    VOCABULARY_DRIFT = "vocabulary_drift"

    @property
    def facts(self) -> SignalFacts:
        """What this signal measures, against what, and on which evidence."""
        return _SIGNAL_FACTS[self]


def _syntactic(axis: Axis, description: str, unit: Unit, threshold: int, source: str, advice: str) -> SignalFacts:
    """Facts of a signal read off the parse alone, past its threshold when above it."""
    return SignalFacts(axis, Tier.SYNTACTIC, description, unit, threshold, Comparison.ABOVE, None, source, advice)


def _graph(
    axis: Axis,
    description: str,
    threshold: int,
    comparison: Comparison,
    floor: Resolution,
    source: str,
    advice: str,
) -> SignalFacts:
    """Facts of a signal that rests on graph edges, each counted only at `floor` or better."""
    return SignalFacts(axis, Tier.GRAPH, description, Unit.COUNT, threshold, comparison, floor, source, advice)


def _duplication(axis: Axis, description: str, unit: Unit, threshold: int, source: str, advice: str) -> SignalFacts:
    """Facts of a signal read off the duplication channels, past its threshold when above it."""
    return SignalFacts(axis, Tier.DUPLICATION, description, unit, threshold, Comparison.ABOVE, None, source, advice)


_SIGNAL_FACTS: dict[Signal, SignalFacts] = {
    Signal.LONG_CALLABLE: _syntactic(
        Axis.SIZE,
        "callable body longer than the threshold, in lines",
        Unit.LINES,
        60,
        "PMD ExcessiveMethodLength reports at 100 lines by default; 60 flags the long tail before it gets there",
        "split it into named steps",
    ),
    Signal.MANY_PARAMETERS: _syntactic(
        Axis.SIZE,
        "callable declaring more parameters than the threshold",
        Unit.COUNT,
        5,
        "PMD ExcessiveParameterList reports at 10 by default; 5 is the common review convention",
        "group the parameters that travel together into one type",
    ),
    Signal.DEEP_NESTING: _syntactic(
        Axis.SIZE,
        "control flow nested deeper than the threshold",
        Unit.LEVEL,
        3,
        "Sonar rule S134 (nested control flow) default maximum 3",
        "return early or extract the inner block",
    ),
    Signal.COMPLEX_CALLABLE: _syntactic(
        Axis.SIZE,
        "cognitive complexity above the threshold",
        Unit.COUNT,
        15,
        "Sonar rule S3776 (cognitive complexity) default threshold 15",
        "extract the branches into named helpers",
    ),
    Signal.LARGE_TYPE: _syntactic(
        Axis.SIZE,
        "type declaration spanning more lines than the threshold",
        Unit.LINES,
        500,
        "PMD ExcessiveClassLength reports at 1000 lines by default; 500 flags the long tail before it gets there",
        "split its responsibilities into separate types",
    ),
    Signal.LARGE_FILE: _syntactic(
        Axis.SIZE,
        "file longer than the threshold, in lines",
        Unit.LINES,
        1000,
        "Sonar rule S104 (too many lines in a file) default maximum 1000",
        "split the file by responsibility",
    ),
    Signal.MANY_TYPES_PER_FILE: _syntactic(
        Axis.SIZE,
        "file declaring more top-level types than the threshold",
        Unit.COUNT,
        5,
        "declared here: no common tool default; Java allows one public top-level type per file",
        "give the types files of their own",
    ),
    Signal.SINGLE_IMPLEMENTATION_INTERFACE: _graph(
        Axis.MACHINERY,
        "interface, trait or protocol with exactly one production implementor",
        1,
        Comparison.EQUALS,
        Resolution.UNIQUE_NAME,
        "speculative generality (Fowler, Refactoring): an abstraction with one implementation",
        "inline the interface into its implementor, or name the second implementation it waits for",
    ),
    Signal.SINGLE_SUBCLASS_ABSTRACT: _graph(
        Axis.MACHINERY,
        "abstract type with exactly one production subtype",
        1,
        Comparison.EQUALS,
        Resolution.UNIQUE_NAME,
        "speculative generality (Fowler, Refactoring): a hierarchy with one branch",
        "merge the abstract type into its subtype, or name the second subtype it waits for",
    ),
    Signal.LONE_PUBLIC_CALLABLE: _graph(
        Axis.MACHINERY,
        "public callable with at most one production caller",
        1,
        Comparison.AT_MOST,
        Resolution.UNIQUE_NAME,
        "declared here: a public seam nobody else uses is a private helper or dead code",
        "make it private beside its caller, or delete it",
    ),
    Signal.UNREFERENCED_SYMBOL: _graph(
        Axis.MACHINERY,
        "non-public callable or type that nothing references",
        0,
        Comparison.AT_MOST,
        Resolution.UNIQUE_NAME,
        "declared here: dead code",
        "delete it, or wire the caller it was written for",
    ),
    Signal.FORBIDDEN_CROSSING: _graph(
        Axis.LAYERING,
        "edges from a file into a module its own module may not depend on",
        0,
        Comparison.ABOVE,
        Resolution.EXACT,
        "the workspace's own [[forbidden]] rules in .zemble/home.toml",
        "move the code or the dependency so the module stops crossing the rule",
    ),
    Signal.CLONE_MASS: _duplication(
        Axis.COPYING,
        "file whose lines inside exact or renamed clone classes exceed the threshold, in thousandths",
        Unit.PERMILLE,
        50,
        "declared here: PMD CPD reports every copy past its token minimum but has no per-file share default",
        "extract the copied code into one shared home",
    ),
    Signal.CLONED_BODY: _duplication(
        Axis.COPYING,
        "callable body that is an exact or renamed copy of a body in another file, counted in other files",
        Unit.COUNT,
        0,
        "membership: any copy in another file (Fowler, Refactoring: duplicated code)",
        "keep one copy and call it from the other places",
    ),
    Signal.PASS_THROUGH: _duplication(
        Axis.MACHINERY,
        "public callable whose whole body hands its parameters, in order, to one other callable",
        Unit.COUNT,
        0,
        "existence: middle man (Fowler, Refactoring); contract implementations and constructor overloads excepted",
        "call the target directly and delete the pass-through",
    ),
    Signal.VOCABULARY_DRIFT: _duplication(
        Axis.VOCABULARY,
        "value or value set declared in more than one place, counted in places",
        Unit.COUNT,
        1,
        "existence: a vocabulary class of `zemble dupes --kind vocabulary`",
        "declare the vocabulary once and derive the other places from it",
    ),
}


@dataclass(frozen=True)
class Finding:
    """One unit past its signal's threshold."""

    signal: Signal
    language: str
    file_path: str
    start_line: int
    end_line: int
    #: The symbol's display name, or "" for a file.
    name: str
    value: int
    #: Share of every measured value of the same signal and language in the scope that is no more suspect, 0-100,
    #: to one decimal so a large population still ranks its extremes apart.
    percentile: float
    #: The weakest edge supporting the finding; EXACT for syntactic signals and for an absence.
    resolution: Resolution
    reason: str
    #: Path-free where the unit has a qualified name, never a line number, so a baseline survives moves.
    key: str
    #: Other places the same finding lives (a vocabulary's other declarations), as (path, start, end).
    elsewhere: tuple[tuple[str, int, int], ...] = ()

    def touched(self, changes: ChangedLines) -> bool:
        """Whether the finding's span, or any place it also lives, intersects a changed line."""
        return any(
            changes.touches(path, start, end)
            for path, start, end in ((self.file_path, self.start_line, self.end_line), *self.elsewhere)
        )

    @property
    def threshold(self) -> int:
        """The signal's threshold, carried for the reader."""
        return self.signal.facts.threshold

    def to_dict(self) -> dict[str, Any]:
        """Render the finding as JSON-ready data."""
        facts = self.signal.facts
        return {
            "signal": self.signal.value,
            "axis": facts.axis.value,
            "tier": facts.tier.value,
            "language": self.language,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "name": self.name,
            "value": self.value,
            "threshold": facts.threshold,
            "comparison": facts.comparison.value,
            "percentile": self.percentile,
            "resolution": self.resolution.value,
            "reason": self.reason,
            "key": self.key,
            "elsewhere": [f"{path}:{start}-{end}" for path, start, end in self.elsewhere],
        }


__all__ = ["Axis", "Comparison", "Finding", "Signal", "SignalFacts", "Tier", "Unit"]
