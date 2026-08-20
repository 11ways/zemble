"""Pluggable embedding backends."""

from zemble.embedding.base import Embedder, EmbeddingMatrix, declared_dimensions, is_remote
from zemble.embedding.cache import CachingEmbedder, EmbeddingCache
from zemble.embedding.pricing import (
    EmbeddingBudgetExceeded,
    estimate_tokens,
    price_per_million,
)
from zemble.embedding.registry import (
    DEFAULT_EMBEDDER_SPEC,
    EmbedderSpecError,
    ResolvedEmbedder,
    build_embedder,
    load_embedder,
    parse_embedder_spec,
    resolve_embedder_spec,
)

__all__ = [
    "DEFAULT_EMBEDDER_SPEC",
    "CachingEmbedder",
    "Embedder",
    "EmbedderSpecError",
    "EmbeddingBudgetExceeded",
    "EmbeddingCache",
    "EmbeddingMatrix",
    "ResolvedEmbedder",
    "declared_dimensions",
    "estimate_tokens",
    "is_remote",
    "load_embedder",
    "parse_embedder_spec",
    "price_per_million",
    "build_embedder",
    "resolve_embedder_spec",
]
