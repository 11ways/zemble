"""Pairwise reranking of the top fused candidates, off unless a spec asks for it."""

from zemble.rerank.apply import apply_reranker, passage_text
from zemble.rerank.base import Reranker
from zemble.rerank.registry import (
    NONE_SPEC,
    PassageMode,
    RerankerSpecError,
    RerankSettings,
    load_reranker,
    parse_reranker_spec,
    resolve_reranker_spec,
)

__all__ = [
    "NONE_SPEC",
    "PassageMode",
    "RerankSettings",
    "Reranker",
    "RerankerSpecError",
    "apply_reranker",
    "load_reranker",
    "parse_reranker_spec",
    "passage_text",
    "resolve_reranker_spec",
]
