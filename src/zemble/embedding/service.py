"""The embedding server's work, without HTTP: one shared cache, one buyer per text.

Every machine asks this service instead of a provider, so a chunk is paid for once across all of
them. Two machines asking for the same uncached text at the same moment do not both buy it: the
first request claims the text, and every other request waits for that claim and then reads the
stored vector.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zemble.embedding.base import Embedder, EmbeddingMatrix, semantic_weight_bonus
from zemble.embedding.cache import CachingEmbedder, EmbeddingCache, text_hash
from zemble.embedding.gc import file_size
from zemble.embedding.registry import EmbedderSpecError, build_provider, cached_family
from zemble.rerank.base import Reranker
from zemble.rerank.registry import HOSTED_SCHEMES, RerankerSpecError, build_hosted_reranker, split_reranker_spec

logger = logging.getLogger(__name__)

#: A stored vector's key: a sha256 hex digest.
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class SpecNotServed(ValueError):
    """A client asked for an embedder or reranker this server was not configured to pay for."""


def _hosted_reranker_spec(spec: str) -> str:
    """Normalize a reranker spec to ``<scheme>:<model>``, refusing one no provider is paid for.

    :param spec: The reranker spec.
    :return: The normalized spec.
    :raises RerankerSpecError: If the spec is malformed, ``none``, or names a local reranker.
    """
    parts = split_reranker_spec(spec)
    if parts is None or parts[0] not in HOSTED_SCHEMES:
        raise RerankerSpecError(f"{spec!r} is not a hosted reranker; only {sorted(HOSTED_SCHEMES)} are served")
    return f"{parts[0]}:{parts[1]}"


def build_reranker(spec: str) -> Reranker:
    """Build the bare hosted reranker for a normalized spec, never one that asks an embedding server."""
    scheme, _, model = _hosted_reranker_spec(spec).partition(":")
    return build_hosted_reranker(scheme, model)


@dataclass
class _Counters:
    """What the service has done since it started, for the status report."""

    documents_requested: int = 0
    #: Distinct texts no cache held when they were asked for; the provider's own token count is
    #: the bill, because a concurrent `store` can fill a miss between the lookup and the purchase.
    documents_missed: int = 0
    queries: int = 0
    rerank_passages: int = 0
    stored: int = 0

    def add(self, name: str, amount: int) -> None:
        """Add to one counter; called under the service's lock."""
        setattr(self, name, getattr(self, name) + amount)


class EmbeddingService:
    """Serves vectors for the families it is configured for, buying each distinct text once."""

    def __init__(
        self,
        directory: Path,
        embedder_specs: Sequence[str],
        reranker_specs: Sequence[str] = (),
        provider: Callable[[str], Embedder] | None = None,
        reranker: Callable[[str], Reranker] | None = None,
    ) -> None:
        """Configure what may be served; nothing is built until a client asks for it.

        :param directory: Where the per-family sqlite files live.
        :param embedder_specs: Specs whose FAMILY may be served; the width a client asks for is its own.
        :param reranker_specs: Reranker specs that may be served, matched exactly.
        :param provider: Builds the bare provider for a spec; the registry's by default.
        :param reranker: Builds the bare reranker for a spec; Voyage's by default.
        :raises EmbedderSpecError: If an embedder spec names no paid, cacheable family.
        """
        self.directory = directory
        self.started_at = time.time()
        self._families: dict[str, str] = {}
        for spec in embedder_specs:
            family = cached_family(spec)
            if family is None:
                raise EmbedderSpecError(f"{spec!r} names no paid embedder family; nothing to serve")
            self._families[family] = spec
        self._reranker_specs = frozenset(_hosted_reranker_spec(spec) for spec in reranker_specs)
        self._provider = provider or (lambda spec: build_provider(spec)[0])
        self._build_reranker = reranker or build_reranker
        self._embedders: dict[str, CachingEmbedder] = {}
        self._rerankers: dict[str, Reranker] = {}
        self._build_lock = threading.Lock()
        self._claims_lock = threading.Lock()
        self._claims: dict[tuple[str, str, int], threading.Event] = {}
        self.counters = _Counters()

    @property
    def families(self) -> list[str]:
        """The embedder families this service pays for."""
        return sorted(self._families)

    @property
    def rerankers(self) -> list[str]:
        """The reranker specs this service pays for."""
        return sorted(self._reranker_specs)

    def embedder(self, spec: str) -> CachingEmbedder:
        """Return the caching embedder for a client's spec, building it on first use.

        :param spec: The client's embedder spec.
        :return: The embedder that serves it.
        :raises SpecNotServed: If the spec's family is not one this service serves.
        """
        spec = spec.strip()
        family = cached_family(spec)
        if family is None or family not in self._families:
            served = ", ".join(self.families) or "nothing"
            raise SpecNotServed(f"this server does not serve {spec!r}; it serves {served}")
        with self._build_lock:
            built = self._embedders.get(spec)
            if built is None:
                built = CachingEmbedder(self._provider(spec), family, self.directory)
                self._embedders[spec] = built
        return built

    def describe(self, spec: str) -> dict[str, Any]:
        """Return what a client needs to stand in for this embedder without asking anyone else."""
        embedder = self.embedder(spec)
        return {
            "model_id": embedder.model_id,
            "dimensions": embedder.dimensions,
            "family": embedder.family,
            "semantic_weight_bonus": semantic_weight_bonus(embedder.inner),
        }

    def documents(self, spec: str, texts: list[str]) -> EmbeddingMatrix:
        """Return document vectors, buying only what no cache holds and no other request is buying.

        :param spec: The client's embedder spec.
        :param texts: The texts to embed.
        :return: A float32 matrix with L2-normalized rows, in input order.
        """
        embedder = self.embedder(spec)
        dims = embedder.dimensions
        self._count("documents_requested", len(texts))
        result = np.zeros((len(texts), dims), dtype=np.float32)
        positions: dict[str, list[int]] = {}
        for position, digest in enumerate(text_hash(text) for text in texts):
            positions.setdefault(digest, []).append(position)
        unresolved = self._fill_cached(embedder.cache, dims, positions, result)
        while unresolved:
            mine, waits = self._claim(embedder.family, dims, unresolved)
            if mine:
                try:
                    bought = embedder.embed_documents([texts[positions[digest][0]] for digest in mine])
                finally:
                    self._release(embedder.family, dims, mine)
                self._count("documents_missed", len(mine))
                for row, digest in enumerate(mine):
                    result[positions[digest]] = bought[row]
            for event in waits.values():
                event.wait()
            # A claim another request held is now stored, unless that request failed: then the
            # text is unclaimed again and the next round buys it here.
            unresolved = self._fill_cached(
                embedder.cache, dims, {digest: positions[digest] for digest in waits}, result
            )
        return result

    def _fill_cached(
        self, cache: EmbeddingCache, dims: int, positions: dict[str, list[int]], result: EmbeddingMatrix
    ) -> list[str]:
        """Copy every stored vector into its rows and return the digests still missing."""
        missing: list[str] = []
        served: list[str] = []
        for digest, rows in positions.items():
            vector = cache.get(digest, dims)
            if vector is None:
                missing.append(digest)
                continue
            result[rows] = vector
            served.append(digest)
        cache.touch(served)
        return missing

    def _count(self, name: str, amount: int) -> None:
        """Add to one status counter; request threads share them."""
        with self._claims_lock:
            self.counters.add(name, amount)

    def _claim(self, family: str, dims: int, digests: list[str]) -> tuple[list[str], dict[str, threading.Event]]:
        """Claim the digests nobody is buying; return them and the claims to wait for instead."""
        mine: list[str] = []
        waits: dict[str, threading.Event] = {}
        with self._claims_lock:
            for digest in digests:
                key = (family, digest, dims)
                held = self._claims.get(key)
                if held is not None:
                    waits[digest] = held
                else:
                    self._claims[key] = threading.Event()
                    mine.append(digest)
        return mine, waits

    def _release(self, family: str, dims: int, digests: list[str]) -> None:
        """Release claims, waking every request waiting on them, whether the purchase stored or failed."""
        with self._claims_lock:
            for digest in digests:
                event = self._claims.pop((family, digest, dims), None)
                if event is not None:
                    event.set()

    def queries(self, spec: str, texts: list[str]) -> EmbeddingMatrix:
        """Return query vectors; a query is never stored, so an asymmetric model stays honest."""
        embedder = self.embedder(spec)
        self._count("queries", len(texts))
        return embedder.embed_queries(texts)

    def covered(self, spec: str, digests: list[str], probe: bool) -> tuple[set[str], int | None]:
        """Return which text hashes are already paid for, and the width that was judged.

        :param spec: The client's embedder spec.
        :param digests: The text hashes to look up.
        :param probe: Whether the provider may be asked for a width no spec or file declares.
        :return: The covered hashes and the width, None when no width could be read.
        """
        embedder = self.embedder(spec)
        width = embedder.dimensions if probe else embedder.known_dimensions
        if width is None:
            return set(), None
        return embedder.cache.covered(digests, width), width

    def store(self, spec: str, rows: list[tuple[str, int, bytes]]) -> int:
        """Store vectors a client's own cache already bought, keeping every vector already here.

        :param spec: An embedder spec naming the family the rows belong to.
        :param rows: ``(text hash, dims, raw float32 bytes)`` triples.
        :return: How many were new.
        :raises ValueError: If a row is malformed.
        """
        for digest, dims, vector in rows:
            if not _DIGEST.match(digest) or dims <= 0 or len(vector) != dims * 4:
                raise ValueError(f"malformed vector row for {digest[:16]!r} at {dims} dimensions")
        added = self.embedder(spec).cache.put_missing(rows)
        self._count("stored", added)
        return added

    def rerank(self, spec: str, query: str, passages: list[str]) -> list[float]:
        """Score passages with a served reranker.

        :param spec: The client's reranker spec.
        :param query: The search query.
        :param passages: The candidate passages.
        :return: One score per passage.
        :raises SpecNotServed: If the spec is not one this service serves.
        """
        try:
            spec = _hosted_reranker_spec(spec)
        except RerankerSpecError as exc:
            raise SpecNotServed(str(exc)) from None
        if spec not in self._reranker_specs:
            served = ", ".join(self.rerankers) or "no reranker"
            raise SpecNotServed(f"this server does not serve reranker {spec!r}; it serves {served}")
        with self._build_lock:
            reranker = self._rerankers.get(spec)
            if reranker is None:
                reranker = self._build_reranker(spec)
                self._rerankers[spec] = reranker
        self._count("rerank_passages", len(passages))
        return reranker.score(query, passages)

    def status(self) -> dict[str, Any]:
        """Report what the service holds and what it has done since it started."""
        families = []
        for family in self.families:
            cache = EmbeddingCache(family, self.directory)
            try:
                rows = cache.count()
            finally:
                cache.close()
            families.append(
                {"family": family, "vectors": rows, "bytes": file_size(cache.path), "path": str(cache.path)}
            )
        tokens = sum(int(getattr(embedder.inner, "total_tokens", 0)) for embedder in self._embedders.values())
        return {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "families": families,
            "rerankers": self.rerankers,
            "documents_requested": self.counters.documents_requested,
            "documents_missed": self.counters.documents_missed,
            "provider_tokens": tokens,
            "queries": self.counters.queries,
            "rerank_passages": self.counters.rerank_passages,
            "vectors_pushed": self.counters.stored,
        }


__all__ = ["EmbeddingService", "SpecNotServed", "build_reranker"]
