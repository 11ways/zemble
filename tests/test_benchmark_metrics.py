"""The hit-rate half of the benchmark metrics, and how a run aggregates it."""

from benchmarks.metrics import HIT_CUTOFFS, hit_at_k
from benchmarks.run_benchmark import QueryResult, _hit_rates


def test_hit_at_k_walks_a_result_down_the_ranking() -> None:
    """One relevant result sliding down the list crosses each cutoff in turn."""
    # 1. At rank 1 it is a hit at every cutoff.
    assert [hit_at_k([1], k) for k in HIT_CUTOFFS] == [True, True, True], "rank 1 hits every cutoff"

    # 2. At rank 5 it is inside hit@5 and hit@10 but no longer hit@1.
    assert [hit_at_k([5], k) for k in HIT_CUTOFFS] == [False, True, True], "rank 5 is the hit@5 boundary"

    # 3. At rank 11 it is outside every reported cutoff.
    assert [hit_at_k([11], k) for k in HIT_CUTOFFS] == [False, False, False], "rank 11 is past hit@10"

    # 4. Nothing found at all is never a hit.
    assert [hit_at_k([], k) for k in HIT_CUTOFFS] == [False, False, False], "no ranks is no hit"

    # 5. Only the best rank matters: a deep hit does not cancel a shallow one.
    assert hit_at_k([9, 2], 5) is True, "any rank inside the cutoff counts"


def test_hit_at_k_ignores_how_many_relevant_targets_a_query_has() -> None:
    """Unlike NDCG, the rate answers "did anything relevant show up", not "how much of it"."""
    assert hit_at_k([3], 5) == hit_at_k([3, 4, 5], 5), "one hit and three hits score the same"


def _query(ranks: list[int], kind: str = "symbol") -> QueryResult:
    """Build a QueryResult with its hit flags derived from the ranks."""
    return QueryResult(
        query="q",
        kind=kind,
        category="symbol",
        relevant=["a.java"],
        relevant_ranks=ranks,
        ndcg10=0.0,
        hit1=hit_at_k(ranks, 1),
        hit5=hit_at_k(ranks, 5),
        hit10=hit_at_k(ranks, 10),
    )


def test_hit_rates_are_the_fraction_of_queries_that_hit() -> None:
    """Four queries, one at rank 1, one at rank 4, one at rank 8 and one missed."""
    rates = _hit_rates([_query([1]), _query([4]), _query([8]), _query([])])
    assert rates == {"hit1": 0.25, "hit5": 0.5, "hit10": 0.75}, "each cutoff counts the queries at or above it"


def test_hit_rates_of_no_queries_are_zero_not_a_division() -> None:
    """An empty group reports zeros rather than raising."""
    assert _hit_rates([]) == {"hit1": 0.0, "hit5": 0.0, "hit10": 0.0}, "empty group is all zeros"
