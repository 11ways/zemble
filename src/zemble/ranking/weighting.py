from zemble.embedding.base import semantic_weight_bonus
from zemble.ranking.boosting import is_symbol_query

_ALPHA_SYMBOL = 0.3  # lean BM25 for exact keyword matching
_ALPHA_NL = 0.5  # balanced semantic + BM25


def resolve_alpha(query: str, alpha: float | None, embedder: object = None) -> float:
    """Return the blending weight for semantic scores, auto-detecting from query type.

    The shipped constants were tuned around the static default embedder; an embedder that
    declares a fusion bonus shifts both of them up by that much. An explicit alpha is the
    caller's number and is never adjusted.

    :param query: The search query, used only to tell a symbol lookup from natural language.
    :param alpha: An explicit weight, or None to auto-detect.
    :param embedder: The embedder whose dense lane is being weighted, or None for the shipped weights.
    :return: A weight clamped to [0, 1].
    """
    if alpha is not None:
        return alpha
    base = _ALPHA_SYMBOL if is_symbol_query(query) else _ALPHA_NL
    return min(1.0, max(0.0, base + semantic_weight_bonus(embedder)))
