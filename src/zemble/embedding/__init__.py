"""Pluggable embedding backends."""

from zemble.embedding.base import Embedder, EmbeddingMatrix
from zemble.embedding.cache import CachingEmbedder, EmbeddingCache
from zemble.embedding.registry import (
    DEFAULT_EMBEDDER_SPEC,
    EmbedderSpecError,
    load_embedder,
    parse_embedder_spec,
    resolve_embedder_spec,
)

__all__ = [
    "DEFAULT_EMBEDDER_SPEC",
    "CachingEmbedder",
    "Embedder",
    "EmbedderSpecError",
    "EmbeddingCache",
    "EmbeddingMatrix",
    "load_embedder",
    "parse_embedder_spec",
    "resolve_embedder_spec",
]
