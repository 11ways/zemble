"""The consistency axis: one style vector per file and its robust distance from its language's and directory's files.

No house style is declared up front: a file is compared with the other files of its own language in
the view. Each `StyleDimension` is centred on the language's median and scaled by its MAD times
1.4826 (the normal-consistency constant); where the MAD is 0, by the mean absolute deviation from the
median times 1.2533, the fallback of the modified z-score (Iglewicz and Hoaglin); a dimension with
neither is constant for that language and dropped there. A difference from a centre is clipped at
`DIFFERENCE_CAP` scales (Huber's clipping), and a distance is the root of the summed squared clipped
differences over the kept dimensions (the diagonal scale only: no covariance is inverted), in
hundredths of the root of the chi-square `STYLE_CUT_QUANTILE` quantile of their count
(Wilson-Hilferty), so 100 is the cut whatever the number of dimensions kept.

AIDEV-NOTE: the vector reads the graph's measurements plus one plain read of the file's text (blank
lines, line length); nothing is parsed again. One-statement callables (`is_boilerplate`) stay out
of the callable-shape dimensions and the public share, or every data class reads different from
every logic class. The directory norm uses the LANGUAGE's scale, so both distances are in the same
units: what reads different here, against how much files vary anywhere.
"""

from __future__ import annotations

import math
import posixpath
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from statistics import NormalDist
from typing import TYPE_CHECKING

from zemble.graph.model import CALLABLE_KINDS, Symbol, SymbolKind
from zemble.health.collect import language_of, named_type
from zemble.health.model import STYLE_CUT_QUANTILE
from zemble.languages import Visibility
from zemble.languages.metrics import CallableMetrics

if TYPE_CHECKING:  # pragma: no cover
    from zemble.health.collect import HealthFacts

#: Measured callables a file needs before its medians and percentiles mean anything.
MIN_CALLABLES = 3
#: Measured files a language needs for a norm: with fewer, the median and the MAD rest on a handful.
MIN_LANGUAGE_FILES = 6
#: Measured files a directory needs to be a norm of its own; a smaller one borrows its parent's.
MIN_DIRECTORY_FILES = 3
#: How many of the language's scales one dimension's difference counts for at most. Past 3 a value is
#: plainly unusual on that dimension and more says nothing new; since 3 squared is under the cut of 2
#: or more dimensions (9.2 for 2, 23.2 for 10), no single measurement makes a file an outlier: the size
#: signals judge those, and reading different takes several dimensions at once.
DIFFERENCE_CAP = 3.0
#: Kept dimensions a norm needs, so that `DIFFERENCE_CAP` holds: one dimension alone is a size question.
MIN_KEPT = 2
#: Dimensions a reason names at most.
_REASONS_SHOWN = 3
#: How far (in the language's scale) a dimension must differ to be named, unless it is the largest.
_REASON_FLOOR = 1.0
#: The MAD's factor for a normal distribution's standard deviation.
_MAD_FACTOR = 1.4826
#: The mean absolute deviation's factor for a normal distribution's standard deviation (sqrt(pi / 2)).
_MEAN_AD_FACTOR = 1.2533
#: Callables whose names their author chose; a constructor repeats its type's.
_NAMED_CALLABLE_KINDS = frozenset({SymbolKind.METHOD, SymbolKind.FUNCTION})


class Scale(str, Enum):
    """How a dimension's raw value is transformed before it is standardised."""

    #: As measured: a share in [0, 1].
    LINEAR = "linear"
    #: `log1p` of the value: lengths and counts are right-skewed, and 400 lines is not ten times as foreign as 40.
    LOG = "log"

    def apply(self, value: float) -> float:
        """The transformed value."""
        return _TRANSFORMS[self](value)


_TRANSFORMS: dict[Scale, Callable[[float], float]] = {Scale.LINEAR: lambda value: value, Scale.LOG: math.log1p}


class NameShape(str, Enum):
    """The spelling of a declared name, leading and trailing underscores aside."""

    SNAKE = "snake_case"
    CAMEL = "camelCase"
    PASCAL = "PascalCase"

    @classmethod
    def of(cls, name: str) -> NameShape | None:
        """The name's shape; None for one word in lower case, UPPER_CASE or a mix no shape names."""
        core = name.strip("_$")
        if not core or not core[0].isalpha():
            return None
        upper = any(char.isupper() for char in core)
        if "_" in core:
            return None if upper else cls.SNAKE
        if core[0].isupper():
            return cls.PASCAL if any(char.islower() for char in core) else None
        return cls.CAMEL if upper else None


@dataclass(frozen=True)
class Body:
    """One measured callable that is more than boilerplate."""

    #: Own lines (`HealthFacts.own_lines`).
    lines: int
    complexity: int
    nesting: int
    parameters: int


@dataclass(frozen=True)
class StyleInput:
    """What one file's style vector is computed from."""

    lines: int
    comment_lines: int
    #: Every measured callable, boilerplate included.
    callables: int
    #: The measured callables that are more than one straight-line statement (see `is_boilerplate`).
    bodies: tuple[Body, ...]
    top_level_types: int
    #: The shape of every declared method and function name, runtime hooks aside. Type names are left out:
    #: every language this reads spells them one way, so counting them would only count types again.
    shapes: tuple[NameShape | None, ...]
    #: Declared named types and methods and functions whose body is not boilerplate (an accessor is public
    #: by its role), and how many of them are public.
    declarations: int
    public: int
    #: From the file's text: blank lines, and the count and total length of the others.
    blank_lines: int
    text_lines: int
    text_chars: int


def is_boilerplate(metric: CallableMetrics) -> bool:
    """Whether a callable is one statement without control flow: an accessor, a delegation, a constant.

    Such bodies are shaped by the language and the file's role (a record's getters), not by how its
    author writes logic, so they would make every data class read different from every logic class.
    Nesting rather than complexity decides: a delegation to a same-named method elsewhere
    (`size() { return items.size(); }`) reads as recursion and scores a complexity of 1.
    """
    return metric.statements <= 1 and metric.nesting == 0


def _over_bodies(read: Callable[[Sequence[Body]], float]) -> Callable[[StyleInput], float | None]:
    """A dimension read off the file's non-boilerplate bodies; None with fewer than `MIN_CALLABLES` of them."""
    return lambda style: read(style.bodies) if len(style.bodies) >= MIN_CALLABLES else None


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _p90(values: Sequence[float]) -> float:
    """The nearest-rank 90th percentile."""
    ordered = sorted(values)
    return float(ordered[math.ceil(0.9 * len(ordered)) - 1])


def _share(part: int, whole: int) -> float | None:
    """`part / whole`, or None (not measured) when there is no whole to share."""
    return part / whole if whole else None


def _shape_share(style: StyleInput, shape: NameShape) -> float | None:
    """The share of the shaped callable names spelled `shape`; None when no name has a shape.

    A one-word lower-case name (`run`, `facts`) fits snake_case and camelCase alike, so it is
    evidence for neither: a file of such names is not measured on naming rather than read as neither.
    """
    shaped = [found for found in style.shapes if found is not None]
    return _share(sum(found is shape for found in shaped), len(shaped))


def _lines(value: float) -> str:
    return f"{value:.0f} lines"


def _percent(value: float) -> str:
    return f"{value:.0%}"


def _amount(value: float) -> str:
    return f"{value:.1f}"


def _chars(value: float) -> str:
    return f"{value:.0f} chars"


@dataclass(frozen=True, slots=True)
class DimensionFacts:
    """How one style dimension is computed, transformed and worded."""

    description: str
    #: The raw value, or None when the file gives the dimension nothing to measure.
    compute: Callable[[StyleInput], float | None]
    scale: Scale
    #: How a reason words a file above, and below, its norm.
    more: str
    less: str
    shown: Callable[[float], str]
    #: The name shape a naming dimension counts.
    shape: NameShape | None = None


def _naming(shape: NameShape) -> DimensionFacts:
    """Facts of the dimension counting one name shape."""
    return DimensionFacts(
        f"share of the file's declared method and function names spelled {shape.value}",
        lambda style: _shape_share(style, shape),
        Scale.LINEAR,
        f"more {shape.value} names",
        f"fewer {shape.value} names",
        _percent,
        shape,
    )


class StyleDimension(str, Enum):
    """One coordinate of a file's style vector; its facts live in `_DIMENSION_FACTS`."""

    CALLABLE_LINES_MEDIAN = "callable_lines_median"
    CALLABLE_LINES_P90 = "callable_lines_p90"
    COMPLEXITY_MEDIAN = "complexity_median"
    NESTING_MAX = "nesting_max"
    CALLABLE_DENSITY = "callable_density"
    TOP_LEVEL_TYPES = "top_level_types"
    COMMENT_DENSITY = "comment_density"
    PARAMETERS_MEAN = "parameters_mean"
    BLANK_LINE_RATIO = "blank_line_ratio"
    LINE_LENGTH_MEAN = "line_length_mean"
    SNAKE_CASE_NAMES = "snake_case_names"
    CAMEL_CASE_NAMES = "camel_case_names"
    PASCAL_CASE_NAMES = "pascal_case_names"
    PUBLIC_SHARE = "public_share"

    @property
    def facts(self) -> DimensionFacts:
        """How this dimension is computed and worded."""
        return _DIMENSION_FACTS[self]


_DIMENSION_FACTS: dict[StyleDimension, DimensionFacts] = {
    StyleDimension.CALLABLE_LINES_MEDIAN: DimensionFacts(
        "median own lines of the non-boilerplate callables",
        _over_bodies(lambda bodies: _median([body.lines for body in bodies])),
        Scale.LOG,
        "longer callables (median)",
        "shorter callables (median)",
        _lines,
    ),
    StyleDimension.CALLABLE_LINES_P90: DimensionFacts(
        "nearest-rank 90th percentile of the non-boilerplate callables' own lines",
        _over_bodies(lambda bodies: _p90([body.lines for body in bodies])),
        Scale.LOG,
        "longer callables (p90)",
        "shorter callables (p90)",
        _lines,
    ),
    StyleDimension.COMPLEXITY_MEDIAN: DimensionFacts(
        "median cognitive complexity of the non-boilerplate callables",
        _over_bodies(lambda bodies: _median([body.complexity for body in bodies])),
        Scale.LOG,
        "more complex callables (median)",
        "simpler callables (median)",
        _amount,
    ),
    StyleDimension.NESTING_MAX: DimensionFacts(
        "deepest control-flow nesting of any non-boilerplate callable",
        _over_bodies(lambda bodies: float(max(body.nesting for body in bodies))),
        Scale.LOG,
        "deeper nesting",
        "shallower nesting",
        _amount,
    ),
    StyleDimension.CALLABLE_DENSITY: DimensionFacts(
        "measured callables, boilerplate included, per 100 lines of the file",
        lambda style: 100 * style.callables / style.lines,
        Scale.LOG,
        "more callables per 100 lines",
        "fewer callables per 100 lines",
        _amount,
    ),
    StyleDimension.TOP_LEVEL_TYPES: DimensionFacts(
        "named type declarations at file level",
        lambda style: float(style.top_level_types),
        Scale.LOG,
        "more top-level types",
        "fewer top-level types",
        _amount,
    ),
    StyleDimension.COMMENT_DENSITY: DimensionFacts(
        "share of the file's lines a comment touches",
        lambda style: _share(style.comment_lines, style.lines),
        Scale.LINEAR,
        "more commented",
        "less commented",
        _percent,
    ),
    StyleDimension.PARAMETERS_MEAN: DimensionFacts(
        "mean declared parameters of the non-boilerplate callables",
        _over_bodies(lambda bodies: statistics.fmean(body.parameters for body in bodies)),
        Scale.LOG,
        "more parameters per callable (mean)",
        "fewer parameters per callable (mean)",
        _amount,
    ),
    StyleDimension.BLANK_LINE_RATIO: DimensionFacts(
        "share of the file's text lines that are blank",
        lambda style: _share(style.blank_lines, style.blank_lines + style.text_lines),
        Scale.LINEAR,
        "more blank lines",
        "fewer blank lines",
        _percent,
    ),
    StyleDimension.LINE_LENGTH_MEAN: DimensionFacts(
        "mean length of the file's non-blank lines, trailing whitespace stripped",
        lambda style: _share(style.text_chars, style.text_lines),
        Scale.LOG,
        "longer lines (mean)",
        "shorter lines (mean)",
        _chars,
    ),
    StyleDimension.SNAKE_CASE_NAMES: _naming(NameShape.SNAKE),
    StyleDimension.CAMEL_CASE_NAMES: _naming(NameShape.CAMEL),
    StyleDimension.PASCAL_CASE_NAMES: _naming(NameShape.PASCAL),
    StyleDimension.PUBLIC_SHARE: DimensionFacts(
        "share of the file's named types and non-boilerplate methods and functions that are public",
        lambda style: _share(style.public, style.declarations),
        Scale.LINEAR,
        "more public declarations",
        "fewer public declarations",
        _percent,
    ),
}

_missing = set(StyleDimension) - set(_DIMENSION_FACTS)
if _missing:  # pragma: no cover - a drift the import itself refuses
    raise RuntimeError(f"style dimensions without facts: {sorted(dimension.value for dimension in _missing)}")
_counted = sorted(shape.value for facts in _DIMENSION_FACTS.values() if (shape := facts.shape) is not None)
if _counted != sorted(shape.value for shape in NameShape):  # pragma: no cover - a drift the import itself refuses
    raise RuntimeError(f"every name shape needs exactly one style dimension, got {_counted}")


@dataclass(frozen=True)
class FileStyle:
    """One measured file's raw style vector; None where the file gives a dimension nothing to measure."""

    path: str
    language: str
    values: dict[StyleDimension, float | None]


@dataclass(frozen=True)
class Norm:
    """One language's centre and scale per kept dimension, in transformed values."""

    language: str
    files: int
    kept: tuple[StyleDimension, ...]
    centre: tuple[float, ...]
    spread: tuple[float, ...]
    #: The root of the chi-square cut for `len(kept)` dimensions.
    cut: float

    def standardised(self, style: FileStyle) -> tuple[float | None, ...]:
        """A file's robust z-score on every kept dimension, None where it is not measured."""
        scores: list[float | None] = []
        for dimension, centre, spread in zip(self.kept, self.centre, self.spread, strict=True):
            value = style.values[dimension]
            scores.append(None if value is None else (dimension.facts.scale.apply(value) - centre) / spread)
        return tuple(scores)


@dataclass(frozen=True)
class Distance:
    """How far one file reads from a norm, in hundredths of the cut, and what differs most."""

    value: int
    #: Where the norm comes from: "<n> <language> files" or "<directory>/ (<n> files)".
    against: str
    reasons: tuple[str, ...]

    @property
    def detail(self) -> str:
        """The finding's detail: the norm, then the dimensions that differ most."""
        return f"against {self.against}: {'; '.join(self.reasons)}"


@dataclass(frozen=True)
class StyleMeasurement:
    """Every measured file's vector, each language's norm and both distances of every file a norm judges."""

    styles: dict[str, FileStyle]
    norms: dict[str, Norm]
    repo: dict[str, Distance]
    #: Only files whose directory norm is not their whole language's.
    directory: dict[str, Distance]
    #: Per file, the z-score vector over its language's kept dimensions.
    standardised: dict[str, tuple[float | None, ...]]


def chi_square_quantile(degrees: int, quantile: float) -> float:
    """The chi-square quantile by the Wilson-Hilferty cube approximation (within 1% from 3 degrees up)."""
    ratio = 2 / (9 * degrees)
    return degrees * (1 - ratio + NormalDist().inv_cdf(quantile) * math.sqrt(ratio)) ** 3


def measure_style(facts: HealthFacts) -> StyleMeasurement:
    """Measure every production file of the view and judge each against its language and its directory."""
    styles = _styles(facts)
    by_language: dict[str, list[FileStyle]] = defaultdict(list)
    for style in styles.values():
        by_language[style.language].append(style)
    norms: dict[str, Norm] = {}
    repo: dict[str, Distance] = {}
    directory: dict[str, Distance] = {}
    standardised: dict[str, tuple[float | None, ...]] = {}
    for language, members in sorted(by_language.items()):
        norm = _norm(language, members)
        if norm is None:
            continue
        norms[language] = norm
        scores = {style.path: norm.standardised(style) for style in members}
        standardised.update(scores)
        centre = tuple(0.0 for _ in norm.kept)
        everywhere = _raw_centre(norm, members)
        against = f"{len(members)} {language} files"
        for style in members:
            repo[style.path] = _distance(norm, scores[style.path], centre, style, everywhere, against)
        directory.update(_directory_distances(norm, members, scores))
    return StyleMeasurement(styles, norms, repo, directory, standardised)


def _styles(facts: HealthFacts) -> dict[str, FileStyle]:
    """The style vector of every production file of the view with enough measured callables, by path."""
    measured: dict[str, list[tuple[Symbol, CallableMetrics]]] = defaultdict(list)
    callables: dict[str, list[Symbol]] = defaultdict(list)
    types: dict[str, list[Symbol]] = defaultdict(list)
    boilerplate: set[str] = set()
    for symbol in facts.everywhere:
        metric = facts.metrics.get(symbol.id) if symbol.kind in CALLABLE_KINDS else None
        if metric is not None:
            measured[symbol.file_path].append((symbol, metric))
            if is_boilerplate(metric):
                boilerplate.add(symbol.id)
        if symbol.kind in _NAMED_CALLABLE_KINDS and not facts.is_runtime_hook(symbol):
            callables[symbol.file_path].append(symbol)
        elif named_type(symbol):
            types[symbol.file_path].append(symbol)
    styles: dict[str, FileStyle] = {}
    for path, file_metric in sorted(facts.population):
        found = measured.get(path, [])
        if len(found) < MIN_CALLABLES:
            continue
        text = _text_measure(facts.absolute(path))
        if text is None:
            continue
        names = callables.get(path, [])
        declared = [*(symbol for symbol in names if symbol.id not in boilerplate), *types.get(path, [])]
        style_input = StyleInput(
            lines=max(file_metric.lines, 1),
            comment_lines=file_metric.comment_lines,
            callables=len(found),
            bodies=tuple(
                Body(facts.own_lines(symbol, metric), metric.complexity, metric.nesting, symbol.arity)
                for symbol, metric in found
                if symbol.id not in boilerplate
            ),
            top_level_types=facts.top_level_types[path],
            shapes=tuple(NameShape.of(symbol.name) for symbol in names),
            declarations=len(declared),
            public=sum(facts.visibility(symbol) is Visibility.PUBLIC for symbol in declared),
            blank_lines=text[0],
            text_lines=text[1],
            text_chars=text[2],
        )
        values = {dimension: dimension.facts.compute(style_input) for dimension in StyleDimension}
        styles[path] = FileStyle(path, language_of(path), values)
    return styles


def _text_measure(path: str) -> tuple[int, int, int] | None:
    """Blank lines, other lines and their total length (trailing whitespace stripped), or None when unreadable."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    stripped = [line.rstrip() for line in lines]
    written = [line for line in stripped if line.strip()]
    return len(stripped) - len(written), len(written), sum(len(line) for line in written)


def _spread(values: Sequence[float], centre: float) -> float:
    """The robust scale: MAD times 1.4826, else the mean absolute deviation times 1.2533; 0 for a constant."""
    deviations = [abs(value - centre) for value in values]
    mad = statistics.median(deviations) * _MAD_FACTOR
    return mad if mad > 0 else statistics.fmean(deviations) * _MEAN_AD_FACTOR


def _norm(language: str, members: Sequence[FileStyle]) -> Norm | None:
    """A language's norm over its kept dimensions, or None with too few files or no dimension that varies.

    A dimension is kept when at least `MIN_LANGUAGE_FILES` files measure it and it varies among them.
    """
    if len(members) < MIN_LANGUAGE_FILES:
        return None
    kept: list[StyleDimension] = []
    centres: list[float] = []
    spreads: list[float] = []
    for dimension in StyleDimension:
        scale = dimension.facts.scale
        values = [scale.apply(value) for style in members if (value := style.values[dimension]) is not None]
        if len(values) < MIN_LANGUAGE_FILES:
            continue
        centre = statistics.median(values)
        spread = _spread(values, centre)
        if spread > 0:
            kept.append(dimension)
            centres.append(centre)
            spreads.append(spread)
    if len(kept) < MIN_KEPT:
        return None
    cut = math.sqrt(chi_square_quantile(len(kept), STYLE_CUT_QUANTILE))
    return Norm(language, len(members), tuple(kept), tuple(centres), tuple(spreads), cut)


def _middle(values: Sequence[float | None]) -> float:
    """The median of the measured values; 0.0 when none is, which only a difference of 0 ever reads."""
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else 0.0


def _raw_centre(norm: Norm, members: Sequence[FileStyle]) -> tuple[float, ...]:
    """The median raw value of each kept dimension, for the reader."""
    return tuple(_middle([style.values[dimension] for style in members]) for dimension in norm.kept)


def _distance(
    norm: Norm,
    scores: tuple[float | None, ...],
    centre: tuple[float, ...],
    style: FileStyle,
    raw_centre: tuple[float, ...],
    against: str,
) -> Distance:
    """A file's distance from a centre, in hundredths of the cut, with the dimensions that differ most.

    A dimension the file does not measure differs by nothing: no evidence is not a difference. The
    reasons are ranked by the unclipped difference, so the most extreme of several capped ones leads.
    """
    differences = [0.0 if score is None else score - middle for score, middle in zip(scores, centre, strict=True)]
    clipped = [max(-DIFFERENCE_CAP, min(DIFFERENCE_CAP, difference)) for difference in differences]
    value = round(100 * math.sqrt(sum(difference * difference for difference in clipped)) / norm.cut)
    order = sorted(range(len(differences)), key=lambda index: (-abs(differences[index]), index))
    reasons: list[str] = []
    for rank, index in enumerate(order[:_REASONS_SHOWN]):
        difference = differences[index]
        if not difference or (rank and abs(difference) < _REASON_FLOOR):
            break
        dimension = norm.kept[index]
        facts = dimension.facts
        word = facts.more if difference > 0 else facts.less
        written = style.values[dimension]
        assert written is not None, "an unmeasured dimension differs by nothing"
        reasons.append(f"{word} ({facts.shown(written)} vs {facts.shown(raw_centre[index])})")
    return Distance(value, against, tuple(reasons))


def _directory_distances(
    norm: Norm, members: Sequence[FileStyle], scores: dict[str, tuple[float | None, ...]]
) -> dict[str, Distance]:
    """Each file's distance from its directory's files, where that directory is not the whole language.

    A directory with fewer than `MIN_DIRECTORY_FILES` measured files of the language (its sub-directories
    included) borrows its parent's, up to the root; a file whose norm would be its whole language is
    left to the repo signal.
    """
    under: dict[str, list[FileStyle]] = defaultdict(list)
    for style in members:
        folder = posixpath.dirname(style.path)
        while True:
            under[folder].append(style)
            if not folder:
                break
            folder = posixpath.dirname(folder)
    centres: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    found: dict[str, Distance] = {}
    for style in members:
        folder = posixpath.dirname(style.path)
        while folder and len(under[folder]) < MIN_DIRECTORY_FILES:
            folder = posixpath.dirname(folder)
        group = under[folder]
        if len(group) == len(members):
            continue
        if folder not in centres:
            centres[folder] = (
                tuple(_middle([scores[peer.path][index] for peer in group]) for index in range(len(norm.kept))),
                _raw_centre(norm, group),
            )
        centre, raw = centres[folder]
        found[style.path] = _distance(
            norm, scores[style.path], centre, style, raw, f"{folder}/ ({len(group)} {norm.language} files)"
        )
    return found


__all__ = [
    "MIN_CALLABLES",
    "MIN_DIRECTORY_FILES",
    "MIN_LANGUAGE_FILES",
    "DimensionFacts",
    "Distance",
    "FileStyle",
    "NameShape",
    "Norm",
    "Scale",
    "StyleDimension",
    "StyleInput",
    "StyleMeasurement",
    "chi_square_quantile",
    "measure_style",
]
