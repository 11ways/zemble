"""The vocabulary channel: one value declared in several places, parallel value sets, literals beside a constant.

It reads the vocabulary sites a language's shape hooks report (`zemble.dedup.languages.VocabularyFact`):
constants with a literal value, literal uses, value sets (an enum, a run of prefixed constants, the
string labels of one switch) and regex literals. Every class names each place the value lives and a
suggested home, and is keyed by the values themselves, so a moved declaration keeps its key.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from zemble.dedup.languages import SiteKind, Visibility
from zemble.dedup.model import CloneClass, CloneKind, Unit

#: A value is shared vocabulary once this many constants in at least this many files declare it.
VALUE_MIN_CONSTANTS = 3
VALUE_MIN_FILES = 2
#: Two value sets are parallel when they share this many values, and that much of the smaller one.
SET_MIN_SHARED = 3
SET_MIN_OVERLAP = 0.6
#: A regex is duplicated once it is written this many times; shorter ones are separators (`\\s+`, `\\.`).
REGEX_MIN_SITES = 2
REGEX_MIN_LENGTH = 8
#: Switches dispatching on one call (`column.name()`) in this many files each restate their own vocabulary.
DISPATCH_MIN_FILES = 3
#: A status word, slug or key: what a constant of a vocabulary holds, never prose.
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_.:/][A-Za-z0-9]+)*")
#: A compound token (`instance-devices`, `host_admission`, `editTarget`): specific enough that a literal
#: equal to a constant's value is that constant written out, not a common word that happens to match.
_COMPOUND = re.compile(r".*(?:[-_.:/]|[a-z][A-Z]).*")
#: What makes a string a regular expression rather than a separator: an anchor, a class or an escape.
_REGEX = re.compile(r"^\^|\$$|\[[^\]]+\]|\\[dswbDSWB.]|\(\?|[+*]\)?$|\{\d")
_VALUES_SHOWN = 6
_PAIRS_SHOWN = 3
#: Characters two values must share at the start to read as drift of one word (`success`, `succeeded`).
_DRIFT_PREFIX = 5


def _of_kind(sites: Iterable[Unit], kind: SiteKind) -> list[Unit]:
    """The sites of one kind."""
    return [unit for unit in sites if unit.kind == kind.value]


def _ordered(members: Iterable[Unit]) -> tuple[Unit, ...]:
    """Members by location."""
    return tuple(sorted(members, key=lambda unit: (unit.file_path, unit.start_line, unit.name)))


def _place(unit: Unit) -> str:
    """A site as a note names it."""
    return f"{unit.name} ({unit.file_path}:{unit.start_line})"


def _common_directory(members: Sequence[Unit]) -> str:
    """The deepest directory every member lives under, or `.`."""
    directories = [os.path.dirname(unit.file_path) or "." for unit in members]
    try:
        return os.path.commonpath(directories) or "."
    except ValueError:
        return "."


def _home_rank(unit: Unit) -> tuple[bool, bool, int, str]:
    """Order declarations by how good a home they are: public, shared source set, shallow, then by path."""
    reusable = unit.visibility is Visibility.PUBLIC and unit.container_visibility is Visibility.PUBLIC
    shared = "/common/" in f"/{unit.file_path}"
    return (not reusable, not shared, unit.file_path.count("/"), unit.file_path)


def _shown(values: Sequence[str]) -> str:
    """A value list cut to what a note carries."""
    head = ", ".join(f'"{value}"' for value in values[:_VALUES_SHOWN])
    return head + (f" (+{len(values) - _VALUES_SHOWN} more)" if len(values) > _VALUES_SHOWN else "")


def _reachable(constant: Unit, use: Unit) -> bool:
    """Whether a literal use could have named the constant: it is not private, or it is in the same file."""
    return constant.visibility is not Visibility.PRIVATE or constant.file_path == use.file_path


def _meaning(unit: Unit) -> str:
    """What a constant's name says it holds: "" when the name spells its value (`STATUS_FAILED` = "failed")."""
    simple = unit.name.rsplit(".", 1)[-1]
    spelled = re.sub(r"[^A-Za-z0-9]+", "_", unit.literals[0]).upper()
    return "" if simple == spelled or simple.endswith(f"_{spelled}") else simple


def _value_class(value: str, decls: list[Unit], uses: list[Unit]) -> CloneClass | None:
    """The class of one declared value, or None when it is neither shared vocabulary nor written out by hand.

    A value is shared when enough constants declare it under one meaning: names spelling the value
    (`FAILED`, `STATUS_FAILED`) or one repeated name (`STATE_COLUMN`); six constants that merely happen to
    hold "hohenheim" for six purposes are not one vocabulary. Literal uses are listed only for a compound
    value: a common word ("failed") is written for other reasons too, so its uses are counted, not listed.
    """
    meanings = Counter(_meaning(unit) for unit in decls)
    meaning, count = meanings.most_common(1)[0]
    agreeing = [unit for unit in decls if _meaning(unit) == meaning]
    shared = count >= VALUE_MIN_CONSTANTS and len({unit.file_path for unit in agreeing}) >= VALUE_MIN_FILES
    compound = _COMPOUND.fullmatch(value) is not None
    stray = [use for use in uses if any(_reachable(decl, use) for decl in decls)]
    listed = stray if compound else []
    if not shared and not listed:
        return None
    declarations = agreeing if shared else decls
    home = min(declarations, key=_home_rank)
    files = len({unit.file_path for unit in declarations})
    notes = [f'value "{value}": {len(declarations)} constant(s) in {files} file(s), {len(stray)} literal use(s)']
    if shared:
        names = sorted({unit.name.rsplit(".", 1)[-1] for unit in declarations})
        more = " ..." if len(names) > _VALUES_SHOWN else ""
        notes.append(f"declared as {', '.join(names[:_VALUES_SHOWN])}{more}")
    if stray and not compound:
        notes.append("its literal uses are not listed: a common word is written for other reasons too")
    notes.append(f"suggested home: {_place(home)}; every other place reads it")
    return CloneClass(CloneKind.VOCABULARY, _ordered([*declarations, *listed]), 1, notes=tuple(notes))


def value_classes(constants: Sequence[Unit], literals: Sequence[Unit]) -> list[CloneClass]:
    """One class per value declared by several constants, or written out as a literal where a constant has it.

    :param constants: Every constant site.
    :param literals: Every literal-use site.
    :return: The classes, unranked.
    """
    declared: dict[str, list[Unit]] = defaultdict(list)
    for unit in constants:
        if unit.literals[0]:
            declared[unit.literals[0]].append(unit)
    used: dict[str, list[Unit]] = defaultdict(list)
    for unit in literals:
        # Only a token (a status word, slug or key) written out is that constant; prose that matches is chance.
        if unit.literals[0] in declared and _TOKEN.fullmatch(unit.literals[0]):
            used[unit.literals[0]].append(unit)
    found = (_value_class(value, decls, used.get(value, [])) for value, decls in declared.items())
    return [clone for clone in found if clone is not None]


class _Families:
    """Disjoint sets of value-set sites."""

    def __init__(self, size: int) -> None:
        """Every set in its own family."""
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        """The family of one set."""
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        """Merge two families."""
        self.parent[self.find(right)] = self.find(left)

    def groups(self) -> list[list[int]]:
        """Every family of two or more sets."""
        grouped: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.parent)):
            grouped[self.find(index)].append(index)
        return [indices for indices in grouped.values() if len(indices) >= 2]


def _drift(values: list[set[str]]) -> list[str]:
    """Pairs of values no single set holds both of that read as one word spelled two ways (`success`, `succeeded`)."""
    union = sorted(set().union(*values))
    pairs = []
    for index, left in enumerate(union):
        for right in union[index + 1 :]:
            if len(left) < _DRIFT_PREFIX or left[:_DRIFT_PREFIX] != right[:_DRIFT_PREFIX]:
                continue
            if not any(left in members and right in members for members in values):
                pairs.append(f"{left} ~ {right}")
    return pairs[:_PAIRS_SHOWN]


def _parallel(values: Sequence[set[str]]) -> _Families:
    """Union every two sets sharing `SET_MIN_SHARED` values and `SET_MIN_OVERLAP` of the smaller one."""
    holders: dict[str, list[int]] = defaultdict(list)
    for index, members in enumerate(values):
        for value in members:
            holders[value].append(index)
    families = _Families(len(values))
    for index, members in enumerate(values):
        counts: Counter[int] = Counter(other for value in members for other in holders[value] if other > index)
        for other, exact in counts.items():
            # One value short, a near twin (`success` beside `succeeded`) counts: that drift is the finding.
            shared = exact + (_near(members, values[other]) if exact == SET_MIN_SHARED - 1 else 0)
            if shared >= SET_MIN_SHARED and shared / min(len(members), len(values[other])) >= SET_MIN_OVERLAP:
                families.union(index, other)
    return families


def _near(left: set[str], right: set[str]) -> int:
    """How many values of one set have a near twin, the same word spelled another way, in the other."""
    loose_left, loose_right = left - right, right - left
    return sum(
        1
        for value in loose_left
        if len(value) >= _DRIFT_PREFIX and any(other[:_DRIFT_PREFIX] == value[:_DRIFT_PREFIX] for other in loose_right)
    )


def set_classes(sets: Sequence[Unit]) -> list[CloneClass]:
    """Families of value sets (enums, prefixed constant runs, switch labels) that declare the same values.

    :param sets: Every value-set and switch site.
    :return: One class per family of two or more parallel sets, unranked.
    """
    values = [set(unit.literals) for unit in sets]
    classes = []
    for indices in _parallel(values).groups():
        members = [sets[index] for index in indices]
        family = [values[index] for index in indices]
        union = set().union(*family)
        common = set.intersection(*family)
        home = min(members, key=lambda unit: (*_home_rank(unit)[:2], -len(unit.literals), unit.file_path))
        notes = [
            f"{len(members)} value sets declare the same vocabulary; shared by all: {_shown(sorted(common)) or 'none'}",
            f"union: {_shown(sorted(union))}",
        ]
        drift = _drift(family)
        if drift:
            notes.append(f"drift: {'; '.join(drift)}")
        notes.append(f"suggested home: {_place(home)}; the other sets derive from it")
        classes.append(CloneClass(CloneKind.VOCABULARY, _ordered(members), len(common), notes=tuple(notes)))
    return classes


def dispatch_classes(switches: Sequence[Unit]) -> list[CloneClass]:
    """Switches in several files dispatching on one call (`switch (column.name())`), each restating its labels.

    :param switches: Every switch site; its shape is what it dispatches on.
    :return: One class per call dispatched on in `DISPATCH_MIN_FILES` files, unranked.
    """
    by_selector: dict[str, list[Unit]] = defaultdict(list)
    for unit in switches:
        if "(" in unit.shape:
            by_selector[unit.shape].append(unit)
    classes = []
    for selector, members in by_selector.items():
        files = {unit.file_path for unit in members}
        if len(files) < DISPATCH_MIN_FILES:
            continue
        labels = sum(len(unit.literals) for unit in members)
        notes = (
            f"{len(members)} switches on {selector} in {len(files)} files restate {labels} string labels by hand",
            "suggested home: derive the dispatch from the declarations the labels repeat, "
            f"in {_common_directory(members)}",
        )
        classes.append(CloneClass(CloneKind.VOCABULARY, _ordered(members), 1, notes=notes))
    return classes


def regex_classes(regexes: Sequence[Unit], constants: Sequence[Unit]) -> list[CloneClass]:
    """One class per regular expression written in two or more places, as a call argument or a constant.

    :param regexes: Every regex-literal site.
    :param constants: Every constant site; one holding a value used as a regex is a place it lives too.
    :return: The classes, unranked.
    """
    written: dict[str, list[Unit]] = defaultdict(list)
    for unit in regexes:
        value = unit.literals[0]
        if len(value) >= REGEX_MIN_LENGTH and _REGEX.search(value):
            written[value].append(unit)
    # A constant is a place a regex lives only when that value is used as a regex somewhere: a CSS selector
    # constant has brackets too.
    for unit in constants:
        if unit.literals[0] in written:
            written[unit.literals[0]].append(unit)
    classes = []
    for value, members in written.items():
        if len(members) < REGEX_MIN_SITES:
            continue
        notes = (
            f"regex {value!r} written {len(members)} times",
            f"suggested home: one Pattern constant in {_common_directory(members)}",
        )
        classes.append(CloneClass(CloneKind.VOCABULARY, _ordered(members), 1, notes=notes))
    return classes


def vocabulary_classes(sites: Sequence[Unit]) -> list[list[CloneClass]]:
    """Every vocabulary class of a run, one list per flavour so each is ranked on its own.

    :param sites: Every site of the run; idiom sites are ignored.
    :return: The value, value-set, switch-dispatch and regex classes.
    """
    constants = _of_kind(sites, SiteKind.CONSTANT)
    switches = _of_kind(sites, SiteKind.SWITCH)
    return [
        value_classes(constants, _of_kind(sites, SiteKind.LITERAL)),
        set_classes([*_of_kind(sites, SiteKind.SET), *switches]),
        dispatch_classes(switches),
        regex_classes(_of_kind(sites, SiteKind.REGEX), constants),
    ]
