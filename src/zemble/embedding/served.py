"""Embedders and rerankers that ask an embedding server instead of a provider.

A client configured with ``ZEMBLE_EMBED_SERVER`` builds these in place of the provider and its
local sqlite cache: the server holds the only cache and the only provider key. Everything an
index records about its embedder - model id, width, fusion bonus - is the provider's own, so an
index built before the switch stays valid after it.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from zemble.embedding.base import Embedder, EmbeddingMatrix, declared_dimensions, semantic_weight_bonus
from zemble.embedding.cache import text_hash, uncovered_texts
from zemble.embedding.http import EmbeddingRequestError, batched, post_json
from zemble.embedding.wire import (
    AUTH_HEADER,
    COVERED_ROUTE,
    DESCRIBE_ROUTE,
    DIGESTS_PER_REQUEST,
    DOCUMENTS_PER_REQUEST,
    DOCUMENTS_ROUTE,
    HEALTH_ROUTE,
    MAX_BODY_BYTES,
    QUERIES_ROUTE,
    RERANK_ROUTE,
    ROWS_PER_REQUEST,
    STATUS_ROUTE,
    STORE_ROUTE,
    ServerSettings,
    decode_vectors,
)
from zemble.version import __version__

#: Characters of text one documents request may carry, well under the server's body limit
#: once JSON escaping has had its way with them.
_MAX_REQUEST_CHARS = MAX_BODY_BYTES // 4
#: Seconds a GET to the server may take.
_GET_TIMEOUT_SECONDS = 30.0


class ServerClient:
    """The HTTP calls one configured embedding server answers."""

    def __init__(self, settings: ServerSettings) -> None:
        """Remember where the server is and the key to present."""
        self.settings = settings

    def _headers(self) -> dict[str, str]:
        """Return the auth header, plus the client version for the server's log."""
        return {AUTH_HEADER: f"Bearer {self.settings.key}", "User-Agent": f"zemble/{__version__}"}

    def post(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to a route, retrying an unreachable or overloaded server like any provider; refusals raise."""
        return post_json(f"{self.settings.url}{route}", payload, self._headers())

    def get(self, route: str) -> dict[str, Any]:
        """GET a route once; a status probe should fail fast rather than back off.

        :param route: The route to ask.
        :return: The decoded answer.
        :raises EmbeddingRequestError: If the server is unreachable or refuses.
        """
        url = f"{self.settings.url}{route}"
        request = urllib.request.Request(url, headers=self._headers())  # noqa: S310 - a configured https/http URL
        try:
            with urllib.request.urlopen(request, timeout=_GET_TIMEOUT_SECONDS) as response:  # noqa: S310
                decoded: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return decoded
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(body).get("error", body)
            except (ValueError, AttributeError):
                message = body
            raise EmbeddingRequestError(f"{url} returned {exc.code}: {message}") from None
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise EmbeddingRequestError(f"{url} failed: {exc}") from None

    def health(self) -> dict[str, Any]:
        """Return the server's liveness answer."""
        return self.get(HEALTH_ROUTE)

    def status(self) -> dict[str, Any]:
        """Return what the server holds and has done."""
        return self.get(STATUS_ROUTE)

    def store(self, spec: str, rows: list[tuple[str, int, bytes]]) -> int:
        """Upload vectors a local cache bought; the server keeps what it already has.

        :param spec: An embedder spec naming the family the rows belong to.
        :param rows: ``(text hash, dims, raw float32 bytes)`` triples.
        :return: How many were new to the server.
        """
        added = 0
        for start in range(0, len(rows), ROWS_PER_REQUEST):
            batch = rows[start : start + ROWS_PER_REQUEST]
            payload = {
                "spec": spec,
                "rows": [[digest, dims, base64.b64encode(vector).decode("ascii")] for digest, dims, vector in batch],
            }
            added += int(self.post(STORE_ROUTE, payload)["added"])
        return added


def _vectors(response: dict[str, Any], expected_rows: int) -> EmbeddingMatrix:
    """Decode a vectors answer, refusing one that does not hold a row per text."""
    rows = int(response.get("rows", -1))
    if rows != expected_rows:
        raise EmbeddingRequestError(f"embedding server returned {rows} vectors for {expected_rows} texts")
    return decode_vectors(str(response["vectors"]), rows, int(response["dimensions"]))


class ServerEmbedder:
    """An embedder whose vectors come from the embedding server, which caches and pays for them.

    :attr:`is_remote` stays True: the server still buys every text it has not seen, so the
    budget guard asks the server what a build would buy, exactly as it asks a local cache.
    """

    is_remote = True

    def __init__(self, spec: str, family: str, settings: ServerSettings, provider: Embedder) -> None:
        """Stand in for a provider.

        :param spec: The spec the server is asked to serve.
        :param family: Its cache family, which the price table is keyed by.
        :param settings: Where the server is and the key to present.
        :param provider: The bare provider for the same spec, read for its identity only; it is
            never asked to embed, so this process needs no provider key.
        """
        self.spec = spec
        self._family = family
        self.client = ServerClient(settings)
        self._provider = provider
        self._described: dict[str, Any] | None = None

    def _describe(self) -> dict[str, Any]:
        """Ask the server once for what the spec alone does not say, such as an undocumented width."""
        if self._described is None:
            self._described = self.client.post(DESCRIBE_ROUTE, {"spec": self.spec})
        return self._described

    @property
    def family(self) -> str:
        """The cache family key (scheme plus model, no dimensions) the price table is keyed by."""
        return self._family

    @property
    def store_location(self) -> str:
        """Where the vectors this embedder reuses are kept, for a report."""
        return self.client.settings.url

    @property
    def declared_dimensions(self) -> int | None:
        """The width the spec or the provider's model table declares, without a request."""
        return declared_dimensions(self._provider)

    @property
    def known_dimensions(self) -> int | None:
        """The width a report may judge stored vectors by; the server reads its own file for the rest."""
        return self.declared_dimensions

    @property
    def dimensions(self) -> int:
        """The vector width: the declared one, else the one the server reports."""
        declared = self.declared_dimensions
        return declared if declared is not None else int(self._describe()["dimensions"])

    @property
    def model_id(self) -> str:
        """The provider's own spec string, so an index built before the server was configured stays valid."""
        if self.declared_dimensions is not None:
            return self._provider.model_id
        return str(self._describe()["model_id"])

    @property
    def semantic_weight_bonus(self) -> float:
        """The provider's fusion bonus; who serves a vector does not change how good it is."""
        return semantic_weight_bonus(self._provider)

    def embed_documents(self, texts: list[str]) -> EmbeddingMatrix:
        """Embed documents through the server, one flush-sized request at a time.

        :param texts: The texts to embed.
        :return: A float32 matrix with L2-normalized rows.
        """
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        parts = [
            _vectors(self.client.post(DOCUMENTS_ROUTE, {"spec": self.spec, "texts": batch}), len(batch))
            for _, batch in batched(texts, DOCUMENTS_PER_REQUEST, _MAX_REQUEST_CHARS)
        ]
        return np.concatenate(parts).astype(np.float32, copy=False)

    def embed_queries(self, texts: list[str]) -> EmbeddingMatrix:
        """Embed queries through the server, which never stores them.

        :param texts: The texts to embed.
        :return: A float32 matrix with L2-normalized rows.
        """
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return _vectors(self.client.post(QUERIES_ROUTE, {"spec": self.spec, "texts": texts}), len(texts))

    def covered(self, digests: list[str], *, probe: bool) -> set[str]:
        """Return which text hashes the server already holds a usable vector for.

        :param digests: The text hashes to look up.
        :param probe: Whether the server may ask its provider for a width nothing declares.
        :return: The covered subset.
        """
        found: set[str] = set()
        for start in range(0, len(digests), DIGESTS_PER_REQUEST):
            batch = digests[start : start + DIGESTS_PER_REQUEST]
            answer = self.client.post(COVERED_ROUTE, {"spec": self.spec, "digests": batch, "probe": probe})
            found.update(answer["covered"])
        return found

    def stored_digests(self, digests: list[str]) -> set[str]:
        """Return which text hashes are already paid for, without anybody asking a provider."""
        return self.covered(digests, probe=False)

    def pending_documents(self, texts: list[str]) -> list[str]:
        """Return the distinct texts the server would still have to buy.

        :param texts: The texts a build is about to embed.
        :return: Those with no usable vector stored, first occurrence first.
        """
        digests = [text_hash(text) for text in texts]
        return uncovered_texts(texts, digests, self.covered(digests, probe=True))

    def pending_documents_unprobed(self, texts: list[str]) -> list[str]:
        """Return the same answer for a caller that may not cost a provider request."""
        digests = [text_hash(text) for text in texts]
        return uncovered_texts(texts, digests, self.covered(digests, probe=False))


class ServerReranker:
    """A reranker whose scores come from the embedding server, which holds the provider key."""

    def __init__(self, spec: str, settings: ServerSettings) -> None:
        """Stand in for a hosted reranker.

        :param spec: The normalized reranker spec, e.g. ``voyage:rerank-2.5-lite``.
        :param settings: Where the server is and the key to present.
        """
        self._spec = spec
        self.client = ServerClient(settings)

    @property
    def model_id(self) -> str:
        """The normalized spec string."""
        return self._spec

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score every passage against the query through the server.

        :param query: The search query.
        :param passages: The candidate passages, in candidate order.
        :return: One score per passage, in the same order.
        :raises EmbeddingRequestError: If the server answers with the wrong number of scores.
        """
        if not passages:
            return []
        scores = self.client.post(RERANK_ROUTE, {"spec": self._spec, "query": query, "passages": passages})["scores"]
        if not isinstance(scores, list) or len(scores) != len(passages):
            raise EmbeddingRequestError(f"embedding server returned {len(scores or [])} scores for {len(passages)}")
        return [float(score) for score in scores]


__all__ = ["ServerClient", "ServerEmbedder", "ServerReranker"]
