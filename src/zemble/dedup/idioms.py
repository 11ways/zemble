"""The holed and idiom channels: bodies equal up to literal values, and call shapes repeated at many sites.

Both read the holed stream (`zemble.dedup.units.holed_tokens`): literals become typed holes and qualified
constant references one `<const>` hole, declared names positional placeholders. A holed class is whole
bodies with one holed stream; an idiom class is call-chain or paired-call sites with one erased shape,
where every hole and every name the site uses once is the same `_` and only the data flow stays.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from zemble.dedup.languages import SiteKind, profile_for
from zemble.dedup.model import SPREAD_PER_FILE, CloneClass, CloneKind, Unit

#: A holed class needs tokens x copies of at least this: two copies of 20 tokens, three of 14, five of 8.
HOLED_MIN_MASS = 40
#: An idiom needs this many sites, across at least this many files.
IDIOM_MIN_SITES = 3
IDIOM_MIN_FILES = 2
#: Share of an idiom's sites one literal value must appear at to be a repeated hidden constant.
_FIXED_SHARE = 0.9
#: Share of the larger site set a shape and its prefix-extension must share to be reported as one family.
SAME_SITES_SHARE = 0.75
#: Distinct call names that make a chain more than one API's ordinary call.
_MIN_CHAIN_NAMES = 4
#: Shortest shared camel-case noun that makes two calls a read/write pair.
_MIN_NOUN = 4
#: Operator tokens that make a short body compute rather than declare.
_COMPUTING = frozenset({"?", "==", "!=", "<=", ">=", "&&", "||", "!", "+", "-", "*", "/", "%"})
#: Control keywords every body may end in without deciding anything.
_NOT_A_DECISION = frozenset({"return", "throw"})
#: One token of a rendered shape: a hole, a placeholder, a word, a two-character operator or one character.
_SHAPE_TOKEN = re.compile(r"<[a-z]+>|\$\d+|\w+|==|!=|<=|>=|&&|\|\||[^\s\w]")
#: How much of a shape a class note quotes, and how much a class root line names it by.
_SHAPE_CHARS = 160
_ROOT_CHARS = 60
#: How many names a note lists before summing up the rest.
_NAMES_SHOWN = 4
_IDIOM_KINDS = frozenset({SiteKind.CHAIN.value, SiteKind.WRAPPER.value, SiteKind.PAIR.value})


def _ordered(members: Iterable[Unit]) -> tuple[Unit, ...]:
    """Members by location, the order every class lists them in."""
    return tuple(sorted(members, key=lambda unit: (unit.file_path, unit.start_line, unit.shape)))


def quoted(shape: str, limit: int = _SHAPE_CHARS) -> str:
    """A shape cut down to what a note line can carry."""
    return shape if len(shape) <= limit else shape[: limit - 3] + "..."


def _names(units: Sequence[Unit]) -> str:
    """The first few member names, then a count of the rest."""
    names = list(dict.fromkeys(unit.name for unit in units))
    shown = ", ".join(names[:_NAMES_SHOWN])
    return shown + (f" and {len(names) - _NAMES_SHOWN} more" if len(names) > _NAMES_SHOWN else "")


def _declares_data(unit: Unit) -> bool:
    """Whether a small body declares rather than computes: it makes at most one call and no decision.

    An override that returns its own constant (`return Icon.of("x");`), a setter or constructor that stores
    its parameter, or an overload binding one argument (`return this.writer(CREATE, row);`) is how a
    language declares a value or a convenience, not copied code. A body with an operator or a branch, or
    with two different calls, computes something even when it is short.
    """
    if len(set(unit.calls)) >= 2:
        return False
    tokens = set(_SHAPE_TOKEN.findall(unit.shape))
    return not tokens & _COMPUTING and not set(unit.skeleton) - _NOT_A_DECISION


def _declares_value(unit: Unit) -> bool:
    """Whether a body takes no input and decides nothing: whatever it calls, it declares the value it returns.

    `return schedulesWhen(List.of(fallback("0 3 * * *")), Role.DATABASES);` and `return find().orderBy(NAME).all();`
    are each body's own declaration; only a profile that reads signatures can tell, so a body without one never is.
    """
    return unit.signature is not None and not unit.signature.parameters and not set(unit.skeleton) - _NOT_A_DECISION


def _language_only(unit: Unit) -> bool:
    """Whether every call a body makes is its language's standard library (`substring`, `lastIndexOf`), or none."""
    profile = profile_for(unit.file_path)
    if profile is None or profile.shapes is None:
        return False
    return set(unit.calls) <= profile.shapes.standard.members


def _values_only(members: Sequence[Unit]) -> bool:
    """Whether a small holed group is no copy: its bodies differ in their values and share nothing beyond them.

    What the copies share is then a constructor overload binding its defaults (`this(new Service())`), a body
    declaring its own value through one API, or the language applied to different data (`"prefix" + value`,
    `path.substring(path.lastIndexOf('/') + 1)` beside `.lastIndexOf('.')`). Copies holding the same values are
    one helper written twice, whatever they call.
    """
    if all(unit.delegates for unit in members):
        return True
    if len({(unit.literals, unit.constants) for unit in members}) < 2:
        return False
    return all(_declares_value(unit) for unit in members) or all(_language_only(unit) for unit in members)


def holed_classes(shaped: Sequence[Unit], min_tokens: int) -> list[CloneClass]:
    """Group whole bodies by holed stream: the same code, its literal values and constants free to differ.

    A group whose members all share one alpha-renamed stream and reach `min_tokens` is exact or renamed
    duplication and is reported there alone. Below `min_tokens` a class must span two files and its
    bodies must compute (`_declares_data`): `return Icon.of("x");` is a language requirement.

    :param shaped: Every shaped body of the run.
    :param min_tokens: The clone channels' smallest unit.
    :return: The holed classes, unranked.
    """
    buckets: dict[str, list[Unit]] = defaultdict(list)
    for unit in shaped:
        buckets[unit.shape_hash].append(unit)
    classes = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        tokens = min(unit.token_count for unit in members)
        if len({unit.renamed_hash for unit in members}) == 1 and tokens >= min_tokens:
            continue
        small = tokens < min_tokens
        if small and (all(_declares_data(unit) for unit in members) or len({unit.file_path for unit in members}) < 2):
            continue  # a small twin inside one file is a family of conveniences (`yes()`/`no()`), not a copy
        if small and _values_only(members):
            continue
        if tokens * len(members) < HOLED_MIN_MASS:
            continue
        variants = len({unit.literals for unit in members})
        notes = (
            f"shape: {quoted(members[0].shape)}",
            f"{len(members)} copies, {variants} literal variant(s): {_names(members)}",
        )
        classes.append(CloneClass(CloneKind.HOLED, _ordered(members), tokens, notes=notes))
    return classes


def _same_noun(left: str, right: str) -> bool:
    """Whether two call names are two verbs on one camel-case noun (`getAttribute`, `setAttribute`)."""
    suffix = os.path.commonprefix([left[::-1], right[::-1]])[::-1]
    hump = next((index for index, char in enumerate(suffix) if char.isupper()), len(suffix))
    return left != right and len(suffix) - hump >= _MIN_NOUN


@dataclass
class _Idiom:
    """One candidate idiom: its sites, its representative shape and what makes it more than an API call."""

    members: list[Unit]
    shape: str
    wrappers: list[Unit]
    #: Literal values nearly every site repeats.
    fixed: list[str]
    #: Evidence other than the repeated literals.
    other: list[str]
    #: Prefix-extensions of this shape cut at the same sites, folded into it (`_.tabs(...).build()`).
    variants: list[str] = field(default_factory=list)

    @property
    def spread(self) -> int:
        """The spread ranking of the class this would become."""
        files = len({unit.file_path for unit in self.members})
        return min(len(self.members), SPREAD_PER_FILE * files) * files


def _idiom(members: list[Unit]) -> _Idiom:
    """Read why a repeated shape is duplication rather than an API's ordinary use.

    A fluent API is called the same way everywhere (`.icon(Icon.of("x"))`) and that is its design. A shape
    only counts when a site wraps it already while others inline it, when nearly every site repeats one
    literal, when it wires one value into two calls, when it chains `_MIN_CHAIN_NAMES` different calls, or
    when it is a read and a write of one noun under one key.
    """
    shape, _count = Counter(unit.shape for unit in members).most_common(1)[0]
    wrappers = [unit for unit in members if unit.kind == SiteKind.WRAPPER.value]
    values = Counter(value for unit in members for value in set(unit.literals))
    fixed = sorted(value for value, count in values.items() if count >= _FIXED_SHARE * len(members))
    other = []
    if wrappers and len(wrappers) < len(members):
        other.append(f"{len(wrappers)} site(s) already wrap it, {len(members) - len(wrappers)} inline it")
    first = members[0]
    placeholders = Counter(re.findall(r"\$\d+", shape))
    if any(count > 1 for count in placeholders.values()) and first.kind != SiteKind.PAIR.value:
        other.append("one value is wired into several calls at every site")
    if len(first.calls) >= _MIN_CHAIN_NAMES:
        other.append(f"{len(first.calls)} different calls chained")
    if first.kind == SiteKind.PAIR.value and _same_noun(*first.calls):
        other.append("one receiver read and written under the same first argument; one helper owns the pair")
    return _Idiom(members, shape, wrappers, fixed, other)


def _sites(idiom: _Idiom) -> frozenset[tuple[str, int]]:
    """Where an idiom's sites are, whatever shape each was cut at."""
    return frozenset((unit.file_path, unit.start_line) for unit in idiom.members)


def _extends(longer: str, shorter: str) -> bool:
    """Whether one shape is another with more calls chained on its result."""
    return len(longer) > len(shorter) and longer.startswith(shorter) and longer[len(shorter) :].startswith(".")


def _merge_extensions(candidates: Sequence[_Idiom]) -> list[_Idiom]:
    """One idiom per family: a shape and its prefix-extension cut at the same sites are reported once.

    Nearly every site of `_.tabs(...)` goes on as `_.tabs(...).build()`, so the two are one finding. The
    variant with more sites stays (the longer shape on a tie, being what every site does) and names the
    other; an extension found at only some of the prefix's sites is a narrower finding and stays apart.
    """
    kept: list[tuple[_Idiom, frozenset[tuple[str, int]]]] = []
    for idiom in sorted(candidates, key=lambda candidate: (-len(candidate.members), -len(candidate.shape))):
        sites = _sites(idiom)
        family = next(
            (
                (other, where)
                for other, where in kept
                if (_extends(other.shape, idiom.shape) or _extends(idiom.shape, other.shape))
                and len(sites & where) >= SAME_SITES_SHARE * max(len(sites), len(where))
            ),
            None,
        )
        if family is None:
            kept.append((idiom, sites))
        else:
            family[0].variants.append(f"{len(sites & family[1])} of these sites as {idiom.shape}")
    return [idiom for idiom, _ in kept]


def _explained(idiom: _Idiom, kept: Sequence[_Idiom]) -> bool:
    """Whether an idiom's only evidence is literals a better-ranked idiom inside it already repeats.

    `_.label(Microcopy.of("x").withFilter("scope", "y"))` repeats "scope" because it holds the Microcopy
    idiom; reporting it again says nothing new.
    """
    if idiom.other:
        return False
    return any(other.shape in idiom.shape and set(idiom.fixed) <= set(other.fixed) for other in kept)


def idiom_classes(sites: Sequence[Unit]) -> list[CloneClass]:
    """Group call-chain and paired-call sites by erased shape, keeping shapes repeated at enough sites and files.

    :param sites: Every site of the run; vocabulary sites are ignored.
    :return: The idiom classes, unranked.
    """
    buckets: dict[str, list[Unit]] = defaultdict(list)
    for unit in sites:
        if unit.kind in _IDIOM_KINDS:
            buckets[unit.shape_hash].append(unit)
    candidates = []
    for members in buckets.values():
        files = {unit.file_path for unit in members}
        if len(members) >= IDIOM_MIN_SITES and len(files) >= IDIOM_MIN_FILES:
            idiom = _idiom(members)
            if idiom.fixed or idiom.other:
                candidates.append(idiom)
    kept: list[_Idiom] = []
    classes = []
    for idiom in sorted(_merge_extensions(candidates), key=lambda candidate: (-candidate.spread, candidate.shape)):
        if _explained(idiom, kept):
            continue
        kept.append(idiom)
        tokens = min(unit.token_count for unit in idiom.members)
        kind = Counter(unit.kind for unit in idiom.members).most_common(1)[0][0]
        root = f"{kind} {quoted(idiom.shape, _ROOT_CHARS)}"
        classes.append(CloneClass(CloneKind.IDIOM, _ordered(idiom.members), tokens, notes=_notes(idiom), root=root))
    return classes


def _notes(idiom: _Idiom) -> tuple[str, ...]:
    """The class-level findings of one reported idiom: its shape, its spread and its evidence."""
    files = len({unit.file_path for unit in idiom.members})
    notes = [f"idiom: {quoted(idiom.shape)}", f"{len(idiom.members)} sites in {files} files"]
    if idiom.fixed:
        notes.append(f"nearly every site repeats {', '.join(idiom.fixed[:_NAMES_SHOWN])}")
    notes.extend(idiom.other)
    if idiom.variants:
        notes.extend(f"one family: {quoted(variant)}" for variant in idiom.variants)
    if idiom.wrappers:
        notes.append(f"wrapper method(s) whose whole body is this idiom: {_names(idiom.wrappers)}")
    return tuple(notes)
