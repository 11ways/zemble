"""The re-implementation channel: code that redoes what an existing method already offers.

Two findings, both pointing from a copy at the API it should call. A forwarding facade is a type whose
methods only hand their parameters to one other type, read off the syntax. A re-implementation is a
body whose embedding sits next to a public method elsewhere AND repeats that method's calls and
control flow without calling it; embedding similarity alone never reports anything.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from zemble.dedup.languages import Visibility
from zemble.dedup.model import CloneClass, CloneKind, Unit
from zemble.dedup.structure import edit_distance
from zemble.graph.model import is_test_path

#: Cosine an API must reach among a copy's nearest bodies.
REIMPLEMENT_THRESHOLD = 0.85
REIMPLEMENT_TOP_K = 10
#: Smallest body either side may be: below it a body is a delegate, not an implementation.
REIMPLEMENT_MIN_TOKENS = 12
#: Calls the API must make, and the share of them the copy must repeat. An API making more different calls
#: than the ceiling is a composition root (a panel declaration, a builder), not a mechanism to call instead.
MIN_API_CALLS = 2
MAX_API_CALLS = 20
MIN_CONTAINMENT = 0.75
#: The API's calls the copy repeats must include this many that are not generic in the scan.
MIN_SPECIFIC_CALLS = 2
#: A call made by more than this share of the scanned bodies (and more than the floor) is generic.
GENERIC_SHARE = 0.005
GENERIC_FLOOR = 20
#: Share of the API's literal values the copy must repeat, when the API has two or more.
MIN_LITERAL_SHARE = 0.5
#: Share of an API's calls chained onto another call's result at which it is a builder declaration.
MAX_CHAINED_SHARE = 0.4
#: A call in a rendered shape, and a call made on another call's result.
_CALL = re.compile(r"\w\s*\(")
_CHAINED = re.compile(r"\)\s*\.\s*\w+\s*\(")
#: Control-flow edits the copy may differ by.
MAX_FLOW_EDITS = 3
#: How much bigger than the API a copy may be before it is doing more than re-implementing it.
MAX_SIZE_RATIO = 4.0
#: A facade forwards at least this many methods, and this share of its bodies, to one type.
FACADE_MIN_METHODS = 2
FACADE_MIN_SHARE = 0.75
#: Visibilities a same-named member may override or be called through; only a private or package copy is a copy.
_OVERRIDABLE = frozenset({Visibility.PUBLIC, Visibility.PROTECTED})
#: Unit kinds nothing calls by name.
_UNCALLABLE = frozenset({"constructor", "initializer"})
_CALLS_SHOWN = 5
#: Bodies whose similarity row is computed at once.
_BLOCK = 1024


@lru_cache(maxsize=None)
def _test_file(file_path: str) -> bool:
    """`is_test_path`, once per file: a workspace asks it a few hundred thousand times."""
    return is_test_path(file_path)


def _in_tests(unit: Unit) -> bool:
    """Whether a body lives in a test source set."""
    return _test_file(unit.file_path)


def _owner(unit: Unit) -> str:
    """The qualified type a member belongs to, or "" for a top-level function."""
    return unit.name.rsplit(".", 1)[0] if "." in unit.name else ""


def _simple(unit: Unit) -> str:
    """A member's own name."""
    return unit.name.rsplit(".", 1)[-1]


def _is_api(unit: Unit) -> bool:
    """Whether a body is a utility other code may call: public in a public type, outside the tests, no override."""
    return (
        unit.visibility is Visibility.PUBLIC
        and unit.container_visibility is Visibility.PUBLIC
        and not _in_tests(unit)
        and unit.kind not in _UNCALLABLE
        and not unit.implements_contract
        and not _simple(unit).startswith("<")
    )


def _core_rank(unit: Unit) -> tuple[bool, int, str]:
    """Which of two public bodies is the more likely home: a shared source set, then the shallower path."""
    return ("/common/" not in f"/{unit.file_path}", unit.file_path.count("/"), unit.file_path)


def reimplementation_candidates(shaped: Sequence[Unit]) -> list[Unit]:
    """The bodies the embedding comparison reads: big enough to implement something, with their text."""
    return [unit for unit in shaped if unit.text and unit.token_count >= REIMPLEMENT_MIN_TOKENS]


def forwarding_classes(shaped: Sequence[Unit]) -> list[CloneClass]:
    """Types whose methods only forward their parameters to one other type: call that type instead.

    :param shaped: Every shaped body of the run.
    :return: One class per facade: its forwarding methods, then the targets the scan holds.
    """
    by_owner: dict[tuple[str, str], list[Unit]] = defaultdict(list)
    for unit in shaped:
        if _owner(unit):
            by_owner[(unit.file_path, _owner(unit))].append(unit)
    targets: dict[str, list[Unit]] = defaultdict(list)
    for unit in shaped:
        if "." in unit.name:
            targets[".".join(unit.name.rsplit(".", 2)[-2:])].append(unit)
    classes = []
    for (_path, owner), bodies in by_owner.items():
        forwards = [unit for unit in bodies if unit.forwards_to]
        if not forwards:
            continue
        target, count = Counter(str(unit.forwards_to).rsplit(".", 1)[0] for unit in forwards).most_common(1)[0]
        if count < FACADE_MIN_METHODS or count / len(bodies) < FACADE_MIN_SHARE:
            continue
        facade = [unit for unit in forwards if str(unit.forwards_to).startswith(f"{target}.")]
        reached = [api for unit in facade for api in targets.get(str(unit.forwards_to), ()) if _owner(api) != owner]
        if not reached:
            continue  # the target is outside the scan (a JDK type): nothing here to call instead
        notes = (
            f"{owner} is a forwarding facade over {target}: {count} of its {len(bodies)} bodies only pass "
            "their parameters on",
            f"call {target} directly and retire {owner}",
        )
        members = tuple(dict.fromkeys([*facade, *reached]))
        classes.append(CloneClass(CloneKind.REIMPLEMENTS, members, min(u.token_count for u in members), notes=notes))
    return classes


def _chained_share(unit: Unit) -> float:
    """How much of a body's calling is chained onto another call's result (`a().b().c()`), read off its shape."""
    calls = len(_CALL.findall(unit.shape))
    return len(_CHAINED.findall(unit.shape)) / calls if calls else 0.0


def _generic_calls(candidates: Sequence[Unit]) -> frozenset[str]:
    """Call names so common in this scan that sharing them says nothing (`get`, `append`, `charAt`)."""
    frequency = Counter(name for unit in candidates for name in set(unit.calls))
    ceiling = max(GENERIC_FLOOR, GENERIC_SHARE * len(candidates))
    return frozenset(name for name, count in frequency.items() if count > ceiling)


def _literal_agreement(copy: Unit, api: Unit) -> bool:
    """Whether the copy repeats most of the API's literal values; parallel code differs exactly there."""
    wanted = set(api.literals)
    if len(wanted) < 2:
        return True
    return len(wanted & set(copy.literals)) / len(wanted) >= MIN_LITERAL_SHARE


@dataclass(frozen=True, slots=True)
class _Tier:
    """How a candidate pair was found, and the bar it must then clear."""

    found_by: str
    min_similarity: float
    min_containment: float
    max_flow_edits: int
    min_specific: int
    min_calls: int = MIN_API_CALLS


#: A close embedding neighbour must repeat most of the API's calls and its control flow.
_NEIGHBOURS = _Tier("embedding neighbour", REIMPLEMENT_THRESHOLD, MIN_CONTAINMENT, MAX_FLOW_EDITS, MIN_SPECIFIC_CALLS)
#: A private copy of a public API's very code (locals renamed) needs no further evidence.
_TWINS = _Tier("the same code", 0.0, 0.0, 0, 0, min_calls=1)
#: A pair sharing many uncommon calls may sit further apart in the embedding and drift a little more.
_SHARED_CALLS = _Tier("shared uncommon calls", 0.75, 0.6, 4, 4)


def _verdict(copy: Unit, api: Unit, generic: frozenset[str], tier: _Tier) -> tuple[float, str] | None:
    """How closely a copy repeats an API under one tier, or None when it does not: its call containment and a reason."""
    if _in_tests(copy) or (_is_api(copy) and _core_rank(copy) < _core_rank(api)):
        return None  # test code is never the copy; of two public bodies the more core one is the home
    if api.file_path == copy.file_path or _owner(api) == _owner(copy) or not _is_api(api):
        return None
    if _simple(api) == _simple(copy) and copy.visibility in _OVERRIDABLE:
        return None  # a same-named member others can see is a parallel implementation or an override, not a copy
    if _simple(api) in copy.calls or _simple(copy) in api.calls or copy.forwards_to or api.forwards_to:
        return None
    api_calls, copy_calls = set(api.calls), set(copy.calls)
    repeated = api_calls & copy_calls
    specific = sorted(repeated - generic)
    if not tier.min_calls <= len(api_calls) <= MAX_API_CALLS or len(specific) < tier.min_specific:
        return None
    if _chained_share(api) >= MAX_CHAINED_SHARE:
        return None  # an API that mostly chains calls declares configuration (a builder), it implements nothing
    containment = len(repeated) / len(api_calls)
    if containment < tier.min_containment or not _literal_agreement(copy, api):
        return None
    if not 1 / MAX_SIZE_RATIO <= copy.token_count / api.token_count <= MAX_SIZE_RATIO:
        return None
    edits = edit_distance(copy.skeleton, api.skeleton, tier.max_flow_edits)
    if edits > tier.max_flow_edits:
        return None
    shown = ", ".join(specific[:_CALLS_SHOWN]) + (", ..." if len(specific) > _CALLS_SHOWN else "")
    flow = "control flow identical" if edits == 0 else f"control flow within {edits} edit(s)"
    return (
        containment,
        f"repeats {len(repeated)} of {len(api_calls)} calls of {api.name}, uncommon ones: {shown}; {flow}",
    )


def _twin_pairs(candidates: Sequence[Unit]) -> Iterator[tuple[int, int]]:
    """Every private or package body with each public API it equals once locals are renamed."""
    twins: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(candidates):
        twins[unit.renamed_hash].append(index)
    for indices in twins.values():
        apis = [index for index in indices if _is_api(candidates[index])]
        if not apis:
            continue
        for copy_index in indices:
            if candidates[copy_index].visibility not in _OVERRIDABLE:
                yield from ((copy_index, api) for api in apis)


def _neighbour_pairs(
    candidates: Sequence[Unit], unit_vectors: np.ndarray, production: Sequence[bool]
) -> Iterator[tuple[int, int, float]]:
    """Every production body with each of its nearest APIs in the embedding, at the neighbour tier's similarity.

    Only APIs are columns and only production bodies rows: the matrix a whole workspace multiplies is the
    bodies that may be a copy against the methods that may be called instead. Only pairs over the threshold
    are ranked, so a row costs a comparison, not a partition of every API.
    """
    apis = np.array([index for index, unit in enumerate(candidates) if _is_api(unit)], dtype=np.int64)
    copies = [index for index in range(len(candidates)) if production[index]]
    if not len(apis) or not copies:
        return
    columns = unit_vectors[apis].T
    for start in range(0, len(copies), _BLOCK):
        rows = copies[start : start + _BLOCK]
        similarities = unit_vectors[rows].dot(columns)
        hits: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for row, column in zip(*np.nonzero(similarities >= _NEIGHBOURS.min_similarity)):
            hits[int(row)].append((float(similarities[row, column]), int(apis[column])))
        for row, found in hits.items():
            copy_index = rows[row]
            for similarity, api_index in sorted(found, reverse=True)[:REIMPLEMENT_TOP_K]:
                if api_index != copy_index:
                    yield copy_index, api_index, similarity


def _call_pairs(candidates: Sequence[Unit], generic: frozenset[str]) -> Iterator[tuple[int, int]]:
    """Every body with each public API sharing `_SHARED_CALLS.min_specific` uncommon calls with it."""
    holders: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(candidates):
        if _is_api(unit):
            for name in set(unit.calls) - generic:
                holders[name].append(index)
    for copy_index, copy in enumerate(candidates):
        uncommon = set(copy.calls) - generic
        if _in_tests(copy) or len(uncommon) < _SHARED_CALLS.min_specific:
            continue
        shared = Counter(api for name in uncommon for api in holders.get(name, ()) if api != copy_index)
        yield from ((copy_index, api) for api, count in shared.items() if count >= _SHARED_CALLS.min_specific)


def reimplementation_classes(candidates: Sequence[Unit], vectors: np.ndarray) -> list[CloneClass]:
    """One class per public method that bodies re-implement: the copies first, the method to call last.

    Candidates come from two lanes: close embedding neighbours, and pairs sharing many uncommon calls;
    each lane then has its own bar (`_Tier`). Embedding similarity alone never reports anything.

    :param candidates: The bodies to compare (:func:`reimplementation_candidates`).
    :param vectors: Their embeddings, row for row.
    :return: The classes, each copy under its best-evidenced API; unranked.
    """
    if len(candidates) < 2:
        return []
    from vicinity.utils import normalize_or_copy

    generic = _generic_calls(candidates)
    unit_vectors = normalize_or_copy(vectors)
    best: dict[int, tuple[float, int, str]] = {}

    def offer(copy_index: int, api_index: int, similarity: float, tier: _Tier) -> None:
        if similarity < tier.min_similarity:
            return
        found = _verdict(candidates[copy_index], candidates[api_index], generic, tier)
        if found is not None and (copy_index not in best or similarity * found[0] > best[copy_index][0]):
            reason = f"{tier.found_by}, similarity {similarity:.2f}; {found[1]}"
            best[copy_index] = (similarity * found[0], api_index, reason)

    for copy_index, api_index in _twin_pairs(candidates):
        offer(copy_index, api_index, float(unit_vectors[copy_index].dot(unit_vectors[api_index])), _TWINS)
    production = [not _in_tests(unit) for unit in candidates]
    for copy_index, api_index, similarity in _neighbour_pairs(candidates, unit_vectors, production):
        offer(copy_index, api_index, similarity, _NEIGHBOURS)
    for copy_index, api_index in _call_pairs(candidates, generic):
        offer(copy_index, api_index, float(unit_vectors[copy_index].dot(unit_vectors[api_index])), _SHARED_CALLS)
    classes = []
    by_api: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for copy_index, (_score, api_index, reason) in best.items():
        by_api[api_index].append((copy_index, reason))
    for api_index, found in by_api.items():
        api = candidates[api_index]
        copies = sorted((candidates[index] for index, _ in found), key=lambda unit: (unit.file_path, unit.start_line))
        names = ", ".join(unit.name for unit in copies[:_CALLS_SHOWN]) + (" ..." if len(copies) > _CALLS_SHOWN else "")
        verb = "re-implements" if len(copies) == 1 else f"({len(copies)} bodies) re-implement"
        notes = [f"{names} {verb} {api.name}; call {api.name} ({api.location})"]
        notes.extend(reason for _, reason in sorted(found)[:_CALLS_SHOWN])
        tokens = min(unit.token_count for unit in (*copies, api))
        classes.append(CloneClass(CloneKind.REIMPLEMENTS, (*copies, api), tokens, notes=tuple(notes)))
    return classes
