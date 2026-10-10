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
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache

import numpy as np

from zemble.dedup.languages import LanguageProfile, Signature, Visibility, profile_for
from zemble.dedup.model import CloneClass, CloneKind, Unit
from zemble.dedup.settings import ShapeSettings
from zemble.dedup.structure import edit_distance
from zemble.graph.model import is_test_path
from zemble.home.config import HomeConfig
from zemble.home.deps import Reachability

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
#: Largest body either side of an intent pair may be on intent alone: a helper can say one thing with other calls.
HELPER_MAX_TOKENS = 160
#: Largest body the intent lane judges at all; a pair with a side past `HELPER_MAX_TOKENS` must also share an
#: uncommon call, because a mechanism that redoes another with none of its calls is a parallel one.
#: AIDEV-NOTE: measured on the scratch root (docs/dedup.md): unbounded, the lane multiplies 8 749 APIs by 22 552
#: bodies in 2.7 s against 1.0 s at 160 tokens; past 400 tokens no pair reached the bar, below it every pair that did
#: without a shared uncommon call was a parallel registration or collector.
INTENT_MAX_TOKENS = 400
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
    """Which of two public bodies in one module is the likelier home: a shared source set, then the shallower path."""
    return ("/common/" not in f"/{unit.file_path}", unit.file_path.count("/"), unit.file_path)


class Architecture:
    """Where each body lives among the workspace's declared modules (`home.toml`) and which module may call which.

    The original of a copy is the API in the most core module the copy's module may depend on. Without a
    declaration every body ranks alike, every reach is unknown, and the path alone orders two public bodies.
    """

    def __init__(self, config: HomeConfig | None = None, settings: ShapeSettings | None = None) -> None:
        """Read modules, order and dependencies from a loaded config; None or a generic one declares nothing.

        The source sets come from `settings` (the built-in folds without one): a server-fold API is unreachable
        from common code whatever the modules say.
        """
        self.config = config if config is not None and not config.generic else None
        self.settings = settings or ShapeSettings()
        self._modules: dict[str, str] = {}
        self._reach: dict[tuple[str, str], Reachability] = {}

    def module(self, unit: Unit) -> str:
        """The declared module a body lives in, "" when nothing is declared."""
        if self.config is None:
            return ""
        module = self._modules.get(unit.file_path)
        if module is None:
            module = self._modules[unit.file_path] = self.config.module_of(unit.file_path)
        return module

    def core(self, unit: Unit) -> tuple[int, bool, int, str]:
        """How close to the core a body lives: its module's rank, then its place in the module (lowest first)."""
        return (self.config.rank(self.module(unit)) if self.config is not None else 0, *_core_rank(unit))

    def reach(self, copy: Unit, api: Unit) -> Reachability:
        """Whether the copy's module may depend on the API's, and its source set read the API's."""
        if not self.settings.reaches(copy.file_path, api.file_path):
            return Reachability.UNREACHABLE
        if self.config is None:
            return Reachability.UNKNOWN
        key = (self.module(copy), self.module(api))
        if key not in self._reach:
            self._reach[key] = self.config.reachable(*key)
        return self._reach[key]

    def blocked(self, copy: Unit, api: Unit) -> bool:
        """Whether the copy's module is known not to reach the API's: forbidden, unreachable, or any other refusal."""
        reach = self.reach(copy, api)
        return not reach.usable and reach is not Reachability.UNKNOWN

    def preference(self, copy: Unit, api: Unit, score: float) -> tuple[int, int, float]:
        """The order an API is preferred in as a copy's original: callable, then most core, then best evidenced."""
        reach = self.reach(copy, api)
        callable_order = 0 if reach.usable else 2 if self.blocked(copy, api) else 1
        return callable_order, self.core(api)[0], -score


def reimplementation_candidates(shaped: Sequence[Unit]) -> list[Unit]:
    """The bodies the embedding comparison reads: big enough to implement something, with their text."""
    return [unit for unit in shaped if unit.text and unit.token_count >= REIMPLEMENT_MIN_TOKENS]


@lru_cache(maxsize=None)
def _standard_shapes(profile: LanguageProfile) -> dict[str, str]:
    """Holed shape hash -> the standard call one canonical body of a language spells (`StandardLibrary.source`)."""
    from zemble.dedup.units import extract_file

    if profile.shapes is None or not profile.shapes.standard.source or not profile.extensions:
        return {}
    standard = profile.shapes.standard
    extracted = extract_file(standard.source.encode(), f"StandardHomes{profile.extensions[0]}", shaped=True)
    shapes = {}
    for unit in extracted.shaped:
        key = unit.name.rsplit(".", 1)[-1].rsplit("_", 1)[0]
        home = standard.homes.get(key)
        if home is None:  # pragma: no cover - a canonical body without a declared home is a profile bug
            raise ValueError(f"{profile.name} standard body {unit.name} names no declared home")
        shapes[unit.shape_hash] = home
    return shapes


def standard_bodies(shaped: Sequence[Unit]) -> list[tuple[Unit, str]]:
    """Every production body that is, literal values aside, one call of its language's standard library.

    :param shaped: Every shaped body of the run.
    :return: Each such body with the call that replaces it (`Objects.toString(value, fallback)`).
    """
    found = []
    for unit in shaped:
        profile = profile_for(unit.file_path)
        home = _standard_shapes(profile).get(unit.shape_hash) if profile is not None else None
        if home is not None and not _in_tests(unit) and not unit.forwards_to:
            found.append((unit, home))
    return found


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


def _may_copy(copy: Unit, api: Unit, architecture: Architecture) -> bool:
    """Whether one body may be reported as a copy of a public API at all, whatever the evidence."""
    if _in_tests(copy) or (_is_api(copy) and architecture.core(copy) < architecture.core(api)):
        return False  # test code is never the copy; of two public bodies the more core one is the home
    if api.file_path == copy.file_path or _owner(api) == _owner(copy) or not _is_api(api):
        return False
    if _simple(api) == _simple(copy) and copy.visibility in _OVERRIDABLE:
        return False  # a same-named member others can see is a parallel implementation or an override, not a copy
    if _simple(api) in copy.calls or _simple(copy) in api.calls or copy.forwards_to or api.forwards_to:
        return False
    return 1 / MAX_SIZE_RATIO <= copy.token_count / api.token_count <= MAX_SIZE_RATIO


def _rebinding(copy: Unit, api: Unit) -> list[str] | None:
    """The constants the API names and the copy does not (`CONFIG` where the copy binds `MANAGE`), or None.

    Such a pair is one mechanism bound to two capabilities, tiers or registries: calling the API would change what
    the copy does, so it is a parameter to extract rather than a call to make, and its class is demoted.
    """
    mine = {constant.rsplit(".", 1)[-1] for constant in copy.constants}
    missing = sorted({constant.rsplit(".", 1)[-1] for constant in api.constants} - mine)
    return missing or None


def _verdict(
    copy: Unit, api: Unit, generic: frozenset[str], tier: _Tier, architecture: Architecture
) -> tuple[float, str] | None:
    """How closely a copy repeats an API under one tier, or None when it does not: its call containment and a reason."""
    if not _may_copy(copy, api, architecture):
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


def reimplementation_texts(candidates: Sequence[Unit]) -> tuple[list[int], list[str]]:
    """Every text the channel embeds in one purchase: each body, then each intent (:func:`intent_text`).

    :return: The candidates that state an intent, and the texts: the bodies row for row, then those intents.
    """
    intents = {index: text for index, unit in enumerate(candidates) if (text := intent_text(unit))}
    return list(intents), [unit.text or "" for unit in candidates] + list(intents.values())


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


def _inputs(unit: Unit, signature: Signature, receiver: bool) -> list[str]:
    """What a member works on: its parameters, and its own type when `receiver` says it needs an instance."""
    owner = _owner(unit).rsplit(".", 1)[-1]
    return [*signature.parameters, *([owner] if owner and receiver else [])]


def _substitution(copy: Unit, api: Unit) -> float:
    """How far the copy could be replaced by a call: 0 when the results disagree, else the API inputs it can supply."""
    mine, theirs = copy.signature, api.signature
    if mine is None or theirs is None:
        return 0.0
    if mine.returns != theirs.returns and not (
        mine.returns and theirs.returns and (theirs.returns in theirs.open_types or mine.returns in mine.open_types)
    ):
        return 0.0
    needed = _inputs(api, theirs, receiver=False)
    have = set(_inputs(copy, mine, receiver=mine.reads_instance))
    if not theirs.static:
        # Calling an instance method takes an instance of its type, and no conversion of an untyped input makes one.
        owner = _owner(api).rsplit(".", 1)[-1]
        if owner and owner not in have:
            return 0.0
    if not needed:
        return 1.0
    untyped = bool(have & mine.open_types)  # an untyped input (`Object raw`) is converted, whatever the API takes
    return sum(1 for kind in needed if kind in have or kind in theirs.open_types or untyped) / len(needed)


def _judged(unit: Unit) -> bool:
    """Whether the intent lane judges a body at all: it carries a signature and is at most `INTENT_MAX_TOKENS`."""
    return unit.signature is not None and unit.token_count <= INTENT_MAX_TOKENS


def _helper_copy(unit: Unit) -> bool:
    """Whether a body may be an intent copy: judged, outside the tests, and reading no instance state of its own."""
    return _judged(unit) and not _in_tests(unit) and not unit.signature.reads_instance


def _may_replace(copy: Unit, api: Unit, architecture: Architecture) -> bool:
    """Whether a helper may be reported as replaceable by an API: a copy at all, holding every value the API holds.

    A helper with its own literal (`copy(key)` with its own `"scope"` value) is a parallel helper, not a copy:
    a call to the API would change what it does.
    """
    return _may_copy(copy, api, architecture) and set(api.literals) <= set(copy.literals)


def _shares_mechanism(copy: Unit, api: Unit, generic: frozenset[str]) -> bool:
    """Whether a pair is helper-sized on both sides, or else shares at least one uncommon call (`INTENT_MAX_TOKENS`)."""
    if max(copy.token_count, api.token_count) <= HELPER_MAX_TOKENS:
        return True
    return bool((set(copy.calls) & set(api.calls)) - generic)


def _intent_gate(copy: Unit, api: Unit, generic: frozenset[str], architecture: Architecture) -> float | None:
    """The copy's substitution signal when it is a helper the API could replace at all, else None."""
    if not (_helper_copy(copy) and _judged(api)) or not _shares_mechanism(copy, api, generic):
        return None
    substitution = _substitution(copy, api)
    if substitution < MIN_SUBSTITUTION or not _may_replace(copy, api, architecture):
        return None
    return substitution


def _intent_verdict(signals: Mapping[Signal, float]) -> tuple[float, str] | None:
    """The weighted score of a gated intent pair (:func:`_intent_gate`), or None below `INTENT_MIN_SCORE`."""
    score = sum(signal.weight * signals[signal] for signal in Signal)
    if score < INTENT_MIN_SCORE:
        return None
    shown = ", ".join(f"{signal.name.lower()} {signals[signal]:.2f}" for signal in Signal)
    return score, f"same intent, score {score:.2f} ({shown})"


def _intent_pairs(candidates: Sequence[Unit], intents: Mapping[int, np.ndarray]) -> Iterator[tuple[int, int, float]]:
    """Every helper outside the tests reading no instance with each of its nearest judged APIs in the intent space."""
    from vicinity.utils import normalize_or_copy

    apis = [index for index in intents if _is_api(candidates[index]) and _judged(candidates[index])]
    copies = [index for index in intents if _helper_copy(candidates[index])]
    if not apis or not copies:
        return
    vectors = normalize_or_copy(np.stack([intents[index] for index in [*apis, *copies]]))
    columns, rows = vectors[: len(apis)].T, vectors[len(apis) :]
    keep = min(INTENT_TOP_K, len(apis))
    for start in range(0, len(copies), _BLOCK):
        similarities = rows[start : start + _BLOCK].dot(columns)
        nearest = np.argpartition(-similarities, keep - 1, axis=1)[:, :keep]
        scores = np.take_along_axis(similarities, nearest, axis=1)
        for row, slot in zip(*np.nonzero(scores >= INTENT_FLOOR)):
            column = int(nearest[row, slot])
            if apis[column] != copies[start + row]:
                yield copies[start + row], apis[column], float(scores[row, slot])


def _twins_of_copies(
    candidates: Sequence[Unit], unit_vectors: np.ndarray, accepted: Sequence[int], taken: Mapping[int, object]
) -> Iterator[tuple[int, int, float]]:
    """Every intent-copy-shaped body that is an accepted intent copy again (`TWIN_SIMILARITY`, flow)."""
    if not accepted:
        return
    rows = [index for index, unit in enumerate(candidates) if index not in taken and _helper_copy(unit)]
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


class _Originals:
    """Every accepted verdict per copy; the original is chosen among them by architecture, not by first sight."""

    def __init__(
        self, candidates: Sequence[Unit], architecture: Architecture, roots: frozenset[tuple[str, int, str]]
    ) -> None:
        """Start empty; `roots` are the standard bodies, preferred over any other original a copy matches."""
        self.candidates = candidates
        self.architecture = architecture
        self.roots = roots
        self.verdicts: dict[int, list[_Copy]] = defaultdict(list)

    def add(self, copy_index: int, verdict: _Copy) -> None:
        """Record one accepted verdict."""
        self.verdicts[copy_index].append(verdict)

    def chosen(self) -> dict[int, _Copy]:
        """Each copy's original: callable, a standard call, bound to the same constants, most core, best evidenced."""
        copies = self.candidates

        def order(index: int, verdict: _Copy) -> tuple[int, bool, bool, int, float]:
            copy, api = copies[index], copies[verdict.api]
            callable_order, core, score = self.architecture.preference(copy, api, verdict.score)
            standard = _identity(api) in self.roots
            return callable_order, not standard, _rebinding(copy, api) is not None, core, score

        return {index: min(found, key=lambda v: order(index, v)) for index, found in self.verdicts.items()}


def _intent_lane(
    candidates: Sequence[Unit],
    unit_vectors: np.ndarray,
    intents: Mapping[int, np.ndarray],
    best: dict[int, _Copy],
    generic: frozenset[str],
    architecture: Architecture,
    roots: frozenset[tuple[str, int, str]] = frozenset(),
) -> None:
    """Add the intent lane's copies to `best`, never replacing what a code lane found, then their twins."""
    names = _NameWords(candidates)
    originals = _Originals(candidates, architecture, roots)
    for copy_index, api_index, intent in _intent_pairs(candidates, intents):
        if copy_index in best:
            continue
        substitution = _intent_gate(candidates[copy_index], candidates[api_index], generic, architecture)
        if substitution is None:
            continue
        signals = {
            Signal.INTENT: intent,
            Signal.BODY: float(unit_vectors[copy_index].dot(unit_vectors[api_index])),
            Signal.NAME: names.similarity(copy_index, api_index),
            Signal.SIGNATURE: substitution,
        }
        verdict = _intent_verdict(signals)
        if verdict is not None:
            originals.add(copy_index, _Copy(verdict[0], api_index, verdict[1], inferred=True))
    found = originals.chosen()
    best.update(found)
    for twin, copy_index, similarity in _twins_of_copies(candidates, unit_vectors, sorted(found), best):
        source = found[copy_index]
        twin_unit, api = candidates[twin], candidates[source.api]
        if (
            _may_copy(twin_unit, api, architecture)
            and _literal_agreement(twin_unit, api)
            and _substitution(twin_unit, api) >= MIN_SUBSTITUTION
        ):
            if twin not in best or source.score * similarity > best[twin].score:
                reason = f"the same code as {candidates[copy_index].name}, similarity {similarity:.2f}"
                best[twin] = _Copy(source.score * similarity, source.api, reason, inferred=True)


def _chase(candidates: Sequence[Unit], best: Mapping[int, _Copy], architecture: Architecture) -> dict[int, _Copy]:
    """Point every copy whose API is itself a copy at that API's own original, so a class names one root to call.

    A code-evidenced copy only follows code-evidenced links (its class must not depend on the intent lane), and
    no copy follows a link to an original its module reaches worse than the API it already points at.
    """
    chased: dict[int, _Copy] = {}
    for copy_index, verdict in best.items():
        copy, api, via, seen = candidates[copy_index], verdict.api, None, {copy_index, verdict.api}
        while api in best:
            step = best[api]
            if step.api in seen or (step.inferred and not verdict.inferred):
                break  # a loop, or a code-evidenced copy reaching an intent-only link
            here = architecture.preference(copy, candidates[api], 0.0)[0]
            if architecture.preference(copy, candidates[step.api], 0.0)[0] > here:
                break
            seen.add(step.api)
            via, api = via if via is not None else api, step.api
        if via is None:
            chased[copy_index] = verdict
        else:
            reason = f"{verdict.reason}; {candidates[via].name} is itself a copy of {candidates[api].name}"
            chased[copy_index] = replace(verdict, api=api, reason=reason)
    return chased


def reimplementation_classes(
    candidates: Sequence[Unit],
    vectors: np.ndarray,
    intents: Mapping[int, np.ndarray] | None = None,
    architecture: Architecture | None = None,
    standard: Sequence[tuple[Unit, str]] = (),
) -> list[CloneClass]:
    """One class per public method that bodies re-implement: the copies first, the method to call last.

    Candidates come from three code lanes: the same code renamed, close embedding neighbours, and pairs sharing
    many uncommon calls, each with its own bar (`_Tier`). A body none of them claims may still be a helper
    stating an API's intent (:class:`Signal`), and a body that is such a helper again joins it. Of the APIs a copy
    matches, the original is the one in the most core module its module may depend on (:class:`Architecture`),
    and an original that is itself a copy hands its copies on to its own. Embedding similarity alone never
    reports anything.

    :param candidates: The bodies to compare (:func:`reimplementation_candidates`).
    :param vectors: Their body embeddings, row for row.
    :param intents: The embeddings of their intent texts (:func:`intent_text`), by candidate index; None or
        empty runs the code lanes alone.
    :param architecture: The workspace's declared modules; None ranks bodies by path alone.
    :param standard: Bodies that are one standard-library call (:func:`standard_bodies`), with that call: the
        most core home there is, so they and every copy whose original is one of them point at it instead.
    :return: The classes, each copy under its original; unranked.
    """
    if len(candidates) < 2:
        return []
    from vicinity.utils import normalize_or_copy

    architecture = architecture or Architecture()
    generic = _generic_calls(candidates)
    unit_vectors = normalize_or_copy(vectors)
    replaced = frozenset(_identity(unit) for unit, _ in standard)
    originals = _Originals(candidates, architecture, replaced)

    def offer(copy_index: int, api_index: int, similarity: float, tier: _Tier) -> None:
        if similarity < tier.min_similarity:
            return
        found = _verdict(candidates[copy_index], candidates[api_index], generic, tier, architecture)
        if found is not None:
            reason = f"{tier.found_by}, similarity {similarity:.2f}; {found[1]}"
            originals.add(copy_index, _Copy(similarity * found[0], api_index, reason, inferred=False))

    for copy_index, api_index in _twin_pairs(candidates):
        offer(copy_index, api_index, float(unit_vectors[copy_index].dot(unit_vectors[api_index])), _TWINS)
    production = [not _in_tests(unit) for unit in candidates]
    for copy_index, api_index, similarity in _neighbour_pairs(candidates, unit_vectors, production):
        offer(copy_index, api_index, similarity, _NEIGHBOURS)
    for copy_index, api_index in _call_pairs(candidates, generic):
        offer(copy_index, api_index, float(unit_vectors[copy_index].dot(unit_vectors[api_index])), _SHARED_CALLS)
    best = originals.chosen()
    if intents:
        _intent_lane(candidates, unit_vectors, intents, best, generic, architecture, replaced)
    # A standard body is a root: its own copy verdicts are dropped, so a chase stops at it and hands on to the call.
    best = {index: verdict for index, verdict in best.items() if _identity(candidates[index]) not in replaced}
    chased = _chase(candidates, best, architecture)
    return [*_classes(candidates, chased, architecture, standard), *_standard_classes(candidates, chased, standard)]


def _identity(unit: Unit) -> tuple[str, int, str]:
    """Where a body is, which tells it apart from every other body of a run."""
    return unit.file_path, unit.start_line, unit.name


def _classes(
    candidates: Sequence[Unit],
    best: Mapping[int, _Copy],
    architecture: Architecture,
    standard: Sequence[tuple[Unit, str]],
) -> list[CloneClass]:
    """One class per API, kind of evidence: the copies by location, then the API.

    Copies only intent evidences form their own class beside the code-evidenced copies of the same API, so a
    class the code lanes report is the same class whatever the intent lane adds. A copy whose module or source set
    may not reach the API has no original to call and is not reported; neither is a copy that is itself, or whose
    API is, a standard-library call (:func:`_standard_classes` reports those).
    """
    replaced = {_identity(unit) for unit, _ in standard}
    classes = []
    by_api: dict[tuple[int, bool, bool], list[tuple[int, str]]] = defaultdict(list)
    for copy_index, verdict in best.items():
        copy, api = candidates[copy_index], candidates[verdict.api]
        if architecture.blocked(copy, api) or {_identity(copy), _identity(api)} & replaced:
            continue
        rebinding = _rebinding(copy, api)
        reason = verdict.reason
        if rebinding is not None:
            reason = f"{reason}; {api.name} binds {', '.join(rebinding)}, which {copy.name} does not"
        by_api[(verdict.api, verdict.inferred, rebinding is not None)].append((copy_index, reason))
    for (api_index, inferred, rebound), found in by_api.items():
        api = candidates[api_index]
        copies = sorted((candidates[index] for index, _ in found), key=lambda unit: (unit.file_path, unit.start_line))
        if rebound:
            head = (
                f"{_listed(copies)} {api.name} ({api.location}) over other constants: one mechanism bound to two "
                "capabilities or registries; extract the constant as a parameter rather than call it"
            )
        else:
            head = f"{_listed(copies)} {api.name}; call {api.name} ({api.location})"
        notes = [head, *(reason for _, reason in sorted(found)[:_CALLS_SHOWN])]
        tokens = min(unit.token_count for unit in (*copies, api))
        members = (*copies, api)
        classes.append(
            CloneClass(CloneKind.REIMPLEMENTS, members, tokens, notes=tuple(notes), inferred=inferred, demoted=rebound)
        )
    return classes


def _listed(copies: Sequence[Unit]) -> str:
    """The copies a class names, then the verb: `A, B (2 bodies) re-implement`."""
    names = ", ".join(unit.name for unit in copies[:_CALLS_SHOWN]) + (" ..." if len(copies) > _CALLS_SHOWN else "")
    return f"{names} {'re-implements' if len(copies) == 1 else f'({len(copies)} bodies) re-implement'}"


def _standard_classes(
    candidates: Sequence[Unit], best: Mapping[int, _Copy], standard: Sequence[tuple[Unit, str]]
) -> list[CloneClass]:
    """One class per standard-library call and kind of evidence: the bodies that are it, and the copies of those.

    A body equal to the call is code evidence; a copy reaches the call through the API it copies, with that
    copy's own evidence (`B.helper is itself Objects.toString(value, fallback)`).
    """
    homes = {_identity(unit): home for unit, home in standard}
    grouped: dict[tuple[str, bool], dict[tuple[str, int, str], tuple[Unit, str]]] = defaultdict(dict)
    for unit, home in standard:
        grouped[(home, False)][_identity(unit)] = (unit, f"{unit.name} is {home}, its literal values aside")
    for copy_index, verdict in best.items():
        copy, api = candidates[copy_index], candidates[verdict.api]
        home = homes.get(_identity(api))
        if home is not None and _identity(copy) not in homes:
            reason = f"{verdict.reason}; {api.name} is itself {home}"
            grouped[(home, verdict.inferred)][_identity(copy)] = (copy, reason)
    classes = []
    for (home, inferred), found in grouped.items():
        bodies = sorted(found.values(), key=lambda entry: (entry[0].file_path, entry[0].start_line))
        copies = [unit for unit, _ in bodies]
        notes = (
            f"{_listed(copies)} {home}; call it (the language's standard library)",
            *(reason for _, reason in bodies[:_CALLS_SHOWN]),
        )
        tokens = min(unit.token_count for unit in copies)
        classes.append(CloneClass(CloneKind.REIMPLEMENTS, tuple(copies), tokens, notes=notes, inferred=inferred))
    return classes
