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
from zemble.health.history import CO_CHANGE_MIN, MAX_CHANGESET_FILES

if TYPE_CHECKING:  # pragma: no cover
    from zemble.health.git import ChangedLines


class Axis(str, Enum):
    """What a group of signals says about the code; one weight each in `zemble.health.score`."""

    SIZE = "size"
    MACHINERY = "machinery"
    LAYERING = "layering"
    COUPLING = "coupling"
    COPYING = "copying"
    VOCABULARY = "vocabulary"
    CONSISTENCY = "consistency"
    HISTORY = "history"
    DOCUMENTATION = "documentation"


class Tier(str, Enum):
    """What a signal rests on."""

    #: The parse alone: true for every measured language.
    SYNTACTIC = "syntactic"
    #: Graph edges: only as good as their resolution, so each such signal names a floor.
    GRAPH = "graph"
    #: The duplication channels' units and clone classes (`zemble.dedup`): a parse of every file beside the
    #: graph, so a run over a whole workspace measures it only when asked (`zemble.health.copying`).
    DUPLICATION = "duplication"
    #: The parse judged against its own peers (`zemble.health.style`): a finding can come or go when
    #: OTHER files change, since the norm it is measured from is theirs.
    POPULATION = "population"
    #: The commits of a window of git history (`zemble.health.history`): a finding can come or go as the
    #: window slides, with no change to the code.
    HISTORY = "history"


class Unit(str, Enum):
    """What a signal's value counts."""

    LINES = "lines"
    COUNT = "count"
    LEVEL = "level"
    #: Thousandths of a whole: a share kept integral, as every value is.
    PERMILLE = "permille"
    #: Hundredths of a reference the signal declares (a distance as a share of its cut): past 100 is beyond it.
    PERCENT = "percent"

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


#: The chi-square quantile a style distance is cut at: under a normal assumption this share of files stays inside.
STYLE_CUT_QUANTILE = 0.99


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
    DEEP_INHERITANCE = "deep_inheritance"
    LONE_PUBLIC_CALLABLE = "lone_public_callable"
    UNREFERENCED_SYMBOL = "unreferenced_symbol"
    FORBIDDEN_CROSSING = "forbidden_crossing"
    PACKAGE_CYCLE = "package_cycle"
    FEATURE_ENVY = "feature_envy"
    CLONE_MASS = "clone_mass"
    CLONED_BODY = "cloned_body"
    PASS_THROUGH = "pass_through"
    VOCABULARY_DRIFT = "vocabulary_drift"
    STYLE_OUTLIER_REPO = "style_outlier_repo"
    STYLE_OUTLIER_DIRECTORY = "style_outlier_directory"
    HOTSPOT = "hotspot"
    BOUNDARY_CO_CHANGE = "boundary_co_change"
    UNDOCUMENTED_PUBLIC = "undocumented_public"
    LONG_DOCBLOCK = "long_docblock"
    COMMENTED_OUT_CODE = "commented_out_code"
    STALE_MARKER = "stale_marker"

    @property
    def facts(self) -> SignalFacts:
        """What this signal measures, against what, and on which evidence."""
        return _SIGNAL_FACTS[self]


def _syntactic(
    axis: Axis,
    description: str,
    unit: Unit,
    threshold: int,
    source: str,
    advice: str,
    comparison: Comparison = Comparison.ABOVE,
) -> SignalFacts:
    """Facts of a signal read off the parse alone, past its threshold when above it unless it says otherwise."""
    return SignalFacts(axis, Tier.SYNTACTIC, description, unit, threshold, comparison, None, source, advice)


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


def _population(description: str, advice: str) -> SignalFacts:
    """Facts of a style signal: a robust distance in hundredths of its chi-square cut, past it above 100."""
    source = (
        "declared here: the squared robust distance (median centre, MAD scale, each dimension on its own) "
        f"past the chi-square {STYLE_CUT_QUANTILE} quantile of the dimensions kept, so under a normal "
        f"assumption {1 - STYLE_CUT_QUANTILE:.0%} of files would pass"
    )
    return SignalFacts(
        Axis.CONSISTENCY, Tier.POPULATION, description, Unit.PERCENT, 100, Comparison.ABOVE, None, source, advice
    )


def _history(description: str, threshold: int, source: str, advice: str) -> SignalFacts:
    """Facts of a signal read off the history window, in thousandths, past its threshold when above it."""
    return SignalFacts(
        Axis.HISTORY, Tier.HISTORY, description, Unit.PERMILLE, threshold, Comparison.ABOVE, None, source, advice
    )


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
        "file longer than the threshold, in lines, unless it holds exactly one top-level type (large_type judges that)",
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
        "interface, trait or protocol with exactly one production implementor, unless that one lives in another module",
        1,
        Comparison.EQUALS,
        Resolution.UNIQUE_NAME,
        "speculative generality (Fowler, Refactoring): an abstraction with one implementation",
        "inline the interface into its implementor, or name the second implementation it waits for",
    ),
    Signal.SINGLE_SUBCLASS_ABSTRACT: _graph(
        Axis.MACHINERY,
        "abstract type with exactly one production subtype, unless that one lives in another module",
        1,
        Comparison.EQUALS,
        Resolution.UNIQUE_NAME,
        "speculative generality (Fowler, Refactoring): a hierarchy with one branch",
        "merge the abstract type into its subtype, or name the second subtype it waits for",
    ),
    Signal.DEEP_INHERITANCE: _graph(
        Axis.MACHINERY,
        "class with more ancestors over extends edges than the threshold, counted from the topmost one the graph holds",
        5,
        Comparison.ABOVE,
        Resolution.UNIQUE_NAME,
        "Sonar rule S110 (inheritance tree depth) default maximum 5",
        "prefer composition: flatten the hierarchy or delegate",
    ),
    Signal.LONE_PUBLIC_CALLABLE: _graph(
        Axis.MACHINERY,
        "called public callable whose production callers are all declared in its own outermost type, valued "
        "by the callers outside it",
        0,
        Comparison.AT_MOST,
        Resolution.UNIQUE_NAME,
        "declared here: a public seam only its own type uses is a private helper; one nothing calls is not "
        "measured, since a caller outside the workspace and dead code look the same without a declared "
        "library/application split",
        "make it private beside its callers",
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
    Signal.PACKAGE_CYCLE: _graph(
        Axis.COUPLING,
        "directory of production files whose exact dependencies inside its module lead back to it, valued by "
        "the directories of its cycle",
        1,
        Comparison.ABOVE,
        Resolution.EXACT,
        "Acyclic Dependencies Principle (Martin, Agile Software Development, 2002); cyclic physical dependencies "
        "(Lakos, Large-Scale C++ Software Design, 1996); JDepend reports package cycles",
        "break the cycle: move the shared part below both packages, or invert one dependency with an interface",
    ),
    Signal.FEATURE_ENVY: _graph(
        Axis.COUPLING,
        "method reading the data of one other type (getter calls on a receiver) more often than it calls its "
        "own type, valued by those reads (0 when its own type is called as often)",
        2,
        Comparison.ABOVE,
        Resolution.UNIQUE_NAME,
        "Feature Envy (Lanza and Marinescu, Object-Oriented Metrics in Practice, 2006: ATFD > 2 and LAA < 1/3), "
        "read off resolved calls into one-statement getters instead of attribute accesses",
        "move it onto the type it uses, or hand it the data it keeps asking for",
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
    Signal.STYLE_OUTLIER_REPO: _population(
        "file whose style vector lies past the cut from its language's files, in hundredths of the cut",
        "bring it into the style its language's files share, or say why it reads different",
    ),
    Signal.STYLE_OUTLIER_DIRECTORY: _population(
        "file whose style vector lies past the cut from its directory's files, in hundredths of the cut",
        "bring it into the style its neighbours share, or say why it reads different",
    ),
    Signal.HOTSPOT: _history(
        "file past the complex_callable or large_file threshold whose churn (commits in the window) is in the "
        "view's top decile, valued in thousandths of the view's changed files that changed less often",
        900,
        "hotspots, change frequency x complexity (Tornhill, Your Code as a Crime Scene, 2015; CodeScene): "
        "the top decile of churn",
        "simplify or split it first: it is complex and it keeps changing",
    ),
    Signal.BOUNDARY_CO_CHANGE: _history(
        "two files in different modules whose commit sets in the window overlap (Jaccard) above the "
        f"threshold, in thousandths, over at least {CO_CHANGE_MIN} shared commits",
        300,
        "temporal coupling (Tornhill, Your Code as a Crime Scene, 2015); code-maat's min-coupling default 30%, "
        "read as a Jaccard (code-maat divides by the mean of the two files' commits, which is never stricter); "
        f"changesets over {MAX_CHANGESET_FILES} files left out as code-maat does",
        "move what changes together into one module, or extract the concept both keep changing",
    ),
    Signal.UNDOCUMENTED_PUBLIC: _syntactic(
        Axis.DOCUMENTATION,
        "public named type, method or function with no doc comment or docstring, in doc lines; constructors, "
        "overrides, runtime hooks, `main`, one-statement bodies and members of anonymous or non-public types "
        "are not measured",
        Unit.COUNT,
        0,
        "Sonar rule S1176 (public API should be documented); Checkstyle MissingJavadocType and "
        "MissingJavadocMethod at their default public scope",
        "write a one-sentence docblock: what it is for, then only the gotchas",
        Comparison.AT_MOST,
    ),
    Signal.LONG_DOCBLOCK: _syntactic(
        Axis.DOCUMENTATION,
        "doc comment and docstring of a type, callable or field longer than the threshold, in lines",
        Unit.LINES,
        20,
        "declared here: no common tool default; the workspace convention is a one-sentence summary plus "
        "the gotchas, and twenty lines leave room for a long list of those",
        "cut it to the one-sentence summary and the gotchas; examples and standard patterns go to docs",
    ),
    Signal.COMMENTED_OUT_CODE: _syntactic(
        Axis.DOCUMENTATION,
        "file whose comment blocks that are nothing but code hold more lines than the threshold",
        Unit.COUNT,
        2,
        "Sonar rule S125 (sections of code should not be commented out), read by a line heuristic instead "
        "of a re-parse; two lines tolerate one explanatory snippet. Counted per file (each weighs 1): weighed "
        "by its lines, a file under the bar would weigh 0 and the score would say how concentrated the "
        "disabled code is, not how widespread",
        "delete it; git remembers",
    ),
    Signal.STALE_MARKER: _syntactic(
        Axis.DOCUMENTATION,
        "file whose comments hold upper-case TODO, FIXME, XXX or HACK markers, in lines",
        Unit.COUNT,
        0,
        "Sonar rules S1135 (TODO tags) and S1134 (FIXME tags); counted per file, each weighing 1",
        "resolve it, or turn it into a tracked issue and delete the comment",
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
    #: The whole this finding is one member of (`cycle <module>:<directory>`), or "": a report lists one
    #: member per group and folds the rest under it, since a 59-directory cycle is one thing to act on.
    group: str = ""

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
            "group": self.group,
        }


__all__ = ["Axis", "Comparison", "STYLE_CUT_QUANTILE", "Finding", "Signal", "SignalFacts", "Tier", "Unit"]
