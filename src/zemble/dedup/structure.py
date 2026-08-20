"""The structural check that stands between an embedding neighbour and a reported logic clone."""

from __future__ import annotations

from dataclasses import dataclass

from zemble.dedup.model import Unit

#: Largest control-flow edit distance two units may differ by and still be called a logic clone.
MAX_SKELETON_DISTANCE = 2
#: Smallest Jaccard overlap of called names two units must share.
MIN_CALL_JACCARD = 0.6


@dataclass(frozen=True, slots=True)
class StructuralVerdict:
    """Whether a candidate pair survived the structural check, and why."""

    accepted: bool
    reason: str


def edit_distance(left: tuple[str, ...], right: tuple[str, ...], limit: int) -> int:
    """Levenshtein distance between two keyword sequences, giving up once it exceeds a limit.

    :param left: The first sequence.
    :param right: The second sequence.
    :param limit: The largest distance the caller cares about.
    :return: The distance, or ``limit + 1`` once it is certainly larger.
    """
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard overlap of two name sets; two empty sets overlap completely."""
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _literal_note(left: Unit, right: Unit) -> str:
    """Describe how the two units' literals differ."""
    differing = sorted(set(left.literals) ^ set(right.literals))
    if not differing:
        return "literals identical"
    shown = ", ".join(differing[:4])
    more = "" if len(differing) <= 4 else f", +{len(differing) - 4} more"
    return f"{len(differing)} literals differ ({shown}{more})"


def _call_note(left: frozenset[str], right: frozenset[str]) -> str:
    """Describe the shared and the differing calls of a pair."""
    if not left and not right:
        return "neither body calls anything"
    shared = sorted(left & right)
    differing = sorted(left ^ right)
    shared_text = ", ".join(shared[:5]) + ("" if len(shared) <= 5 else ", ...")
    if not differing:
        return f"calls {{{shared_text}}} shared, none differ"
    differing_text = ", ".join(differing[:5]) + ("" if len(differing) <= 5 else ", ...")
    return f"calls {{{shared_text}}} shared, {{{differing_text}}} differs"


def check_pair(left: Unit, right: Unit) -> StructuralVerdict:
    """Decide whether two embedding neighbours are structurally close enough to report.

    Control flow must be identical or within :data:`MAX_SKELETON_DISTANCE` edits, and the
    called names must overlap by at least :data:`MIN_CALL_JACCARD`. Nothing is reported on
    embedding similarity alone.

    :param left: The first unit.
    :param right: The second unit.
    :return: The verdict, carrying the reason the report prints.
    """
    distance = edit_distance(left.skeleton, right.skeleton, MAX_SKELETON_DISTANCE)
    if distance > MAX_SKELETON_DISTANCE:
        return StructuralVerdict(False, "control flow differs")
    left_calls, right_calls = frozenset(left.calls), frozenset(right.calls)
    overlap = jaccard(left_calls, right_calls)
    if overlap < MIN_CALL_JACCARD:
        return StructuralVerdict(False, f"call overlap {overlap:.2f} below {MIN_CALL_JACCARD}")
    flow = "control flow identical" if distance == 0 else f"control flow within {distance} edit(s)"
    return StructuralVerdict(True, f"{flow}; {_call_note(left_calls, right_calls)}; {_literal_note(left, right)}")
