"""The re-implementation channel: code that redoes what an existing method already offers.

Two findings, both pointing from a copy at the API it should call. A forwarding facade is a type whose
methods only hand their parameters to one other type, read off the syntax. A re-implementation is a
body whose embedding sits next to a public method elsewhere AND repeats that method's calls and
control flow without calling it, or a static helper that states the same intent as a public method
(name, signature, documentation, the calls it makes) and could be replaced by a call to it. Embedding
similarity alone never reports anything.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import numpy as np

from zemble.dedup.languages import Signature, Visibility
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


class Signal(Enum):
    """The evidence the intent lane weighs, each with its share of the score; the shares sum to 1.

    AIDEV-NOTE: measured on Hohenheim b9b8f221 plus zenit, zenit-cms and protoblast (docs/dedup.md): the body
    embedding is the weakest of the four for a re-implementation, which by definition uses different calls.
    """

    #: Cosine of the two members' intent texts (:func:`intent_text`): name, signature, documentation, calls.
    INTENT = 0.45
    #: Cosine of the two raw bodies.
    BODY = 0.25
    #: How much of each member's name the other's name spells, rare words counting most.
    NAME = 0.20
    #: Whether the copy could hand its own inputs to the API and use what it returns.
    SIGNATURE = 0.10

    @property
    def weight(self) -> float:
        """The share of the score this signal carries."""
        return float(self.value)


#: Intent cosine an API must reach among a copy's nearest intents to be weighed at all.
INTENT_FLOOR = 0.6
INTENT_TOP_K = 10
#: Weighted score (:class:`Signal`) an intent pair must reach.
INTENT_MIN_SCORE = 0.80
#: Largest body either side of an intent pair may be: the lane judges helpers, not whole mechanisms.
HELPER_MAX_TOKENS = 160
#: Share of the API's inputs the copy must be able to supply.
MIN_SUBSTITUTION = 0.5
#: A body this close to an accepted intent copy (cosine, control-flow edits, shared calls) is that copy again.
TWIN_SIMILARITY = 0.9
TWIN_FLOW_EDITS = 2
TWIN_SHARED_CALLS = 0.6
#: Words a member name spells that say nothing about what it does.
_NAME_STOPWORDS = frozenset({"a", "an", "and", "as", "by", "for", "from", "get", "in", "is", "of", "on", "or", "to"})
#: Suffixes folded off a name word (`parsed`, `parses` and `parse` are one word), longest first.
_NAME_SUFFIXES = ("ing", "ed", "es", "s")
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
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


@dataclass(frozen=True, slots=True)
class _Copy:
    """One accepted copy: how strongly, under which API, why, and whether only intent evidences it."""

    score: float
    api: int
    reason: str
    inferred: bool


#: A close embedding neighbour must repeat most of the API's calls and its control flow.
_NEIGHBOURS = _Tier("embedding neighbour", REIMPLEMENT_THRESHOLD, MIN_CONTAINMENT, MAX_FLOW_EDITS, MIN_SPECIFIC_CALLS)
#: A private copy of a public API's very code (locals renamed) needs no further evidence.
_TWINS = _Tier("the same code", 0.0, 0.0, 0, 0, min_calls=1)
#: A pair sharing many uncommon calls may sit further apart in the embedding and drift a little more.
_SHARED_CALLS = _Tier("shared uncommon calls", 0.75, 0.6, 4, 4)


def _may_copy(copy: Unit, api: Unit) -> bool:
    """Whether one body may be reported as a copy of a public API at all, whatever the evidence."""
    if _in_tests(copy) or (_is_api(copy) and _core_rank(copy) < _core_rank(api)):
        return False  # test code is never the copy; of two public bodies the more core one is the home
    if api.file_path == copy.file_path or _owner(api) == _owner(copy) or not _is_api(api):
        return False
    if _simple(api) == _simple(copy) and copy.visibility in _OVERRIDABLE:
        return False  # a same-named member others can see is a parallel implementation or an override, not a copy
    if _simple(api) in copy.calls or _simple(copy) in api.calls or copy.forwards_to or api.forwards_to:
        return False
    return 1 / MAX_SIZE_RATIO <= copy.token_count / api.token_count <= MAX_SIZE_RATIO


def _verdict(copy: Unit, api: Unit, generic: frozenset[str], tier: _Tier) -> tuple[float, str] | None:
    """How closely a copy repeats an API under one tier, or None when it does not: its call containment and a reason."""
    if not _may_copy(copy, api):
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


def _words(name: str) -> list[str]:
    """The words a member name spells, lower-cased and folded (`parsedInt` is `pars`, `int`)."""
    words = []
    for word in _CAMEL.sub(r"\1 \2", name).replace("_", " ").lower().split():
        if word in _NAME_STOPWORDS:
            continue
        suffix = next((end for end in _NAME_SUFFIXES if word.endswith(end) and len(word) - len(end) >= 3), "")
        words.append(word[: len(word) - len(suffix)] if suffix else word)
    return words


def intent_text(unit: Unit) -> str | None:
    """What a member says it does, for the intent embedding: its name, signature, documentation and calls.

    :return: The text, or None for a body whose profile reads no signature (it then takes no part in the lane).
    """
    signature = unit.signature
    if signature is None:
        return None
    name = " ".join(_CAMEL.sub(r"\1 \2", _simple(unit)).lower().split())
    calls = " ".join(_CAMEL.sub(r"\1 \2", call).lower() for call in unit.calls)
    returns = signature.returns or "nothing"
    return f"{name} ({', '.join(signature.parameters)}) -> {returns}. {signature.summary} calls: {calls}".strip()


class _NameWords:
    """Every candidate's name words with their rarity in the scan, so a shared `duration` outweighs a shared `of`."""

    def __init__(self, candidates: Sequence[Unit]) -> None:
        """Read every candidate's name once."""
        self.words = [frozenset(_words(_simple(unit))) for unit in candidates]
        frequency = Counter(word for words in self.words for word in words)
        self.rarity = {word: math.log(len(candidates) / count) for word, count in frequency.items()}

    def _spelled(self, words: frozenset[str], other: frozenset[str]) -> float:
        """The rarity-weighted share of `words` the other name spells; a word matches its own prefix (`int`)."""
        total = sum(self.rarity[word] for word in words)
        shared = sum(
            self.rarity[word]
            for word in words
            if any(
                word == mate
                or (
                    min(len(word), len(mate)) >= 3
                    and word[:3] == mate[:3]
                    and (word.startswith(mate) or mate.startswith(word))
                )
                for mate in other
            )
        )
        return shared / total if total else 0.0

    def similarity(self, left: int, right: int) -> float:
        """How much each of two names spells of the other, the weaker direction counting."""
        one, other = self.words[left], self.words[right]
        if not one or not other:
            return 0.0
        return min(self._spelled(one, other), self._spelled(other, one))


def _inputs(unit: Unit, signature: Signature) -> list[str]:
    """What a member works on: its parameters, and its own type when it reads an instance."""
    owner = _owner(unit).rsplit(".", 1)[-1]
    return [*signature.parameters, *([owner] if owner and not signature.static else [])]


def _substitution(copy: Unit, api: Unit) -> float:
    """How far the copy could be replaced by a call: 0 when the results disagree, else the API inputs it can supply."""
    mine, theirs = copy.signature, api.signature
    if mine is None or theirs is None:
        return 0.0
    if mine.returns != theirs.returns and not (
        mine.returns and theirs.returns and (theirs.returns in theirs.open_types or mine.returns in mine.open_types)
    ):
        return 0.0
    needed = _inputs(api, theirs)
    if not needed:
        return 1.0
    have = set(_inputs(copy, mine))
    untyped = bool(have & mine.open_types)  # an untyped input (`Object raw`) is converted, whatever the API takes
    return sum(1 for kind in needed if kind in have or kind in theirs.open_types or untyped) / len(needed)


def _helper(unit: Unit) -> bool:
    """Whether a body is helper-sized and carries the signature the intent lane reads."""
    return unit.signature is not None and unit.token_count <= HELPER_MAX_TOKENS


def _may_replace(copy: Unit, api: Unit) -> bool:
    """Whether a helper may be reported as replaceable by an API: a copy at all, holding every value the API holds.

    A helper with its own literal (`copy(key)` with its own `"scope"` value) is a parallel helper, not a copy:
    a call to the API would change what it does.
    """
    return _may_copy(copy, api) and set(api.literals) <= set(copy.literals)


def _intent_verdict(copy: Unit, api: Unit, signals: Mapping[Signal, float]) -> tuple[float, str] | None:
    """The weighted score of an intent pair, or None when the copy is no static helper the API could replace."""
    if not (_helper(copy) and _helper(api)) or copy.signature is None or not copy.signature.static:
        return None
    if signals[Signal.SIGNATURE] < MIN_SUBSTITUTION or not _may_replace(copy, api):
        return None
    score = sum(signal.weight * signals[signal] for signal in Signal)
    if score < INTENT_MIN_SCORE:
        return None
    shown = ", ".join(f"{signal.name.lower()} {signals[signal]:.2f}" for signal in Signal)
    return score, f"same intent, score {score:.2f} ({shown})"


def _intent_pairs(candidates: Sequence[Unit], intents: Mapping[int, np.ndarray]) -> Iterator[tuple[int, int, float]]:
    """Every static helper outside the tests with each of its nearest helper APIs in the intent embedding."""
    from vicinity.utils import normalize_or_copy

    apis = [index for index in intents if _is_api(candidates[index]) and _helper(candidates[index])]
    copies = [
        index
        for index in intents
        if not _in_tests(candidates[index]) and _helper(candidates[index]) and candidates[index].signature.static
    ]
    if not apis or not copies:
        return
    vectors = normalize_or_copy(np.stack([intents[index] for index in [*apis, *copies]]))
    columns, rows = vectors[: len(apis)].T, vectors[len(apis) :]
    for start in range(0, len(copies), _BLOCK):
        similarities = rows[start : start + _BLOCK].dot(columns)
        for row, line in enumerate(similarities):
            nearest = np.argpartition(-line, min(INTENT_TOP_K, len(apis) - 1))[:INTENT_TOP_K]
            for column in nearest:
                if line[column] >= INTENT_FLOOR and apis[column] != copies[start + row]:
                    yield copies[start + row], apis[column], float(line[column])


def _twins_of_copies(
    candidates: Sequence[Unit], unit_vectors: np.ndarray, accepted: Sequence[int], taken: Mapping[int, object]
) -> Iterator[tuple[int, int, float]]:
    """Every static helper outside the tests that is an accepted intent copy again (`TWIN_SIMILARITY`, flow)."""
    if not accepted:
        return
    rows = [
        index
        for index, unit in enumerate(candidates)
        if index not in taken and not _in_tests(unit) and _helper(unit) and unit.signature.static
    ]
    columns = unit_vectors[list(accepted)].T
    for start in range(0, len(rows), _BLOCK):
        similarities = unit_vectors[rows[start : start + _BLOCK]].dot(columns)
        for row, column in zip(*np.nonzero(similarities >= TWIN_SIMILARITY)):
            twin, copy = candidates[rows[start + row]], candidates[accepted[column]]
            calls, mates = set(twin.calls), set(copy.calls)
            if len(calls & mates) < TWIN_SHARED_CALLS * len(calls | mates):
                continue  # `value.trim()` and `String.valueOf(value)` embed alike and do different things
            if edit_distance(twin.skeleton, copy.skeleton, TWIN_FLOW_EDITS) <= TWIN_FLOW_EDITS:
                yield rows[start + row], accepted[column], float(similarities[row, column])


def _intent_lane(
    candidates: Sequence[Unit],
    unit_vectors: np.ndarray,
    intents: Mapping[int, np.ndarray],
    best: dict[int, _Copy],
) -> None:
    """Add the intent lane's copies to `best`, never replacing what a code lane found, then their twins."""
    names = _NameWords(candidates)
    found: dict[int, _Copy] = {}
    for copy_index, api_index, intent in _intent_pairs(candidates, intents):
        if copy_index in best:
            continue
        copy, api = candidates[copy_index], candidates[api_index]
        signals = {
            Signal.INTENT: intent,
            Signal.BODY: float(unit_vectors[copy_index].dot(unit_vectors[api_index])),
            Signal.NAME: names.similarity(copy_index, api_index),
            Signal.SIGNATURE: _substitution(copy, api),
        }
        verdict = _intent_verdict(copy, api, signals)
        if verdict is not None and (copy_index not in found or verdict[0] > found[copy_index].score):
            found[copy_index] = _Copy(verdict[0], api_index, verdict[1], inferred=True)
    best.update(found)
    for twin, copy_index, similarity in _twins_of_copies(candidates, unit_vectors, sorted(found), best):
        source = found[copy_index]
        twin_unit, api = candidates[twin], candidates[source.api]
        if (
            _may_copy(twin_unit, api)
            and _literal_agreement(twin_unit, api)
            and _substitution(twin_unit, api) >= MIN_SUBSTITUTION
        ):
            if twin not in best or source.score * similarity > best[twin].score:
                reason = f"the same code as {candidates[copy_index].name}, similarity {similarity:.2f}"
                best[twin] = _Copy(source.score * similarity, source.api, reason, inferred=True)


def reimplementation_classes(
    candidates: Sequence[Unit], vectors: np.ndarray, intents: Mapping[int, np.ndarray] | None = None
) -> list[CloneClass]:
    """One class per public method that bodies re-implement: the copies first, the method to call last.

    Candidates come from three code lanes: the same code renamed, close embedding neighbours, and pairs sharing
    many uncommon calls, each with its own bar (`_Tier`). A body none of them claims may still be a static
    helper stating an API's intent (:class:`Signal`), and a body that is such a helper again joins it.
    Embedding similarity alone never reports anything.

    :param candidates: The bodies to compare (:func:`reimplementation_candidates`).
    :param vectors: Their body embeddings, row for row.
    :param intents: The embeddings of their intent texts (:func:`intent_text`), by candidate index; None or
        empty runs the code lanes alone.
    :return: The classes, each copy under its best-evidenced API; unranked.
    """
    if len(candidates) < 2:
        return []
    from vicinity.utils import normalize_or_copy

    generic = _generic_calls(candidates)
    unit_vectors = normalize_or_copy(vectors)
    best: dict[int, _Copy] = {}

    def offer(copy_index: int, api_index: int, similarity: float, tier: _Tier) -> None:
        if similarity < tier.min_similarity:
            return
        found = _verdict(candidates[copy_index], candidates[api_index], generic, tier)
        if found is not None and (copy_index not in best or similarity * found[0] > best[copy_index].score):
            reason = f"{tier.found_by}, similarity {similarity:.2f}; {found[1]}"
            best[copy_index] = _Copy(similarity * found[0], api_index, reason, inferred=False)

    for copy_index, api_index in _twin_pairs(candidates):
        offer(copy_index, api_index, float(unit_vectors[copy_index].dot(unit_vectors[api_index])), _TWINS)
    production = [not _in_tests(unit) for unit in candidates]
    for copy_index, api_index, similarity in _neighbour_pairs(candidates, unit_vectors, production):
        offer(copy_index, api_index, similarity, _NEIGHBOURS)
    for copy_index, api_index in _call_pairs(candidates, generic):
        offer(copy_index, api_index, float(unit_vectors[copy_index].dot(unit_vectors[api_index])), _SHARED_CALLS)
    if intents:
        _intent_lane(candidates, unit_vectors, intents, best)
    return _classes(candidates, best)


def _classes(candidates: Sequence[Unit], best: Mapping[int, _Copy]) -> list[CloneClass]:
    """One class per API and kind of evidence: the copies by location, then the API.

    Copies only intent evidences form their own class beside the code-evidenced copies of the same API, so a
    class the code lanes report is the same class whatever the intent lane adds.
    """
    classes = []
    by_api: dict[tuple[int, bool], list[tuple[int, str]]] = defaultdict(list)
    for copy_index, verdict in best.items():
        by_api[(verdict.api, verdict.inferred)].append((copy_index, verdict.reason))
    for (api_index, inferred), found in by_api.items():
        api = candidates[api_index]
        copies = sorted((candidates[index] for index, _ in found), key=lambda unit: (unit.file_path, unit.start_line))
        names = ", ".join(unit.name for unit in copies[:_CALLS_SHOWN]) + (" ..." if len(copies) > _CALLS_SHOWN else "")
        verb = "re-implements" if len(copies) == 1 else f"({len(copies)} bodies) re-implement"
        notes = [f"{names} {verb} {api.name}; call {api.name} ({api.location})"]
        notes.extend(reason for _, reason in sorted(found)[:_CALLS_SHOWN])
        tokens = min(unit.token_count for unit in (*copies, api))
        members = (*copies, api)
        classes.append(CloneClass(CloneKind.REIMPLEMENTS, members, tokens, notes=tuple(notes), inferred=inferred))
    return classes
