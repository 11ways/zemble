"""Base class for embedders that call an HTTP endpoint returning OpenAI-shaped embedding data."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from zemble.embedding.base import EmbeddingMatrix, normalize_rows
from zemble.embedding.http import CHARS_PER_TOKEN, EmbeddingRequestError, batched, post_json


class HttpEmbedder:
    """Embeds by POSTing batches to an ``/embeddings`` endpoint and reading ``data[].embedding``.

    Subclasses supply the URL, the auth header and any provider-specific payload fields.
    Token usage reported by the provider is accumulated in :attr:`total_tokens`, and every
    HTTP round trip is counted in :attr:`request_count` (the cache tests assert on it).
    """

    def __init__(self, model: str, dimensions: int | None, max_texts: int, max_tokens: int, api_key_env: str) -> None:
        """Initialise the embedder.

        :param model: The provider's model name.
        :param dimensions: The requested output width, or None to accept the provider's default.
        :param max_texts: Maximum texts the provider accepts per request.
        :param max_tokens: Maximum tokens the provider accepts per request.
        :param api_key_env: Environment variable holding the API key.
        """
        self._model = model
        self._dimensions = dimensions
        self._max_texts = max_texts
        self._max_chars = int(max_tokens * CHARS_PER_TOKEN)
        self._api_key_env = api_key_env
        self.total_tokens = 0
        self.request_count = 0

    #: Every HTTP provider is a paid round trip, so the budget guard applies to all of them.
    is_remote = True

    @property
    def model_id(self) -> str:
        """The normalized spec string."""
        raise NotImplementedError

    @property
    def declared_dimensions(self) -> int | None:
        """The width already known from the spec or the model table; None means only a probe knows."""
        return self._dimensions

    @property
    def dimensions(self) -> int:
        """The vector width, probed with a one-token request when the provider default is unknown."""
        if self._dimensions is None:
            self._dimensions = int(self._embed(["dimension probe"], role="document").shape[1])
        return self._dimensions

    def _endpoint(self) -> str:
        """Return the full URL to POST to."""
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        """Return the request headers, including auth."""
        raise NotImplementedError

    def _payload(self, texts: list[str], role: str | None) -> dict[str, Any]:
        """Return the JSON body for one batch.

        :param texts: The batch of texts.
        :param role: ``"document"``, ``"query"`` or None when the provider has no such notion.
        :return: The request body.
        :raises NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def _api_key(self) -> str:
        """Return the API key, refusing loudly when the environment variable is unset.

        :return: The API key.
        :raises EmbeddingRequestError: If the variable is unset or empty.
        """
        key = os.environ.get(self._api_key_env, "").strip()
        if not key:
            raise EmbeddingRequestError(f"{self._api_key_env} is not set; it is required for {self.model_id}")
        return key

    def _embed(self, texts: list[str], role: str | None) -> EmbeddingMatrix:
        """Embed texts in provider-sized batches, preserving input order.

        :param texts: The texts to embed.
        :param role: ``"document"``, ``"query"`` or None.
        :return: A float32 matrix with L2-normalized rows.
        :raises EmbeddingRequestError: If the provider returns a batch of the wrong size.
        """
        rows: list[list[float]] = []
        for _, batch in batched(texts, self._max_texts, self._max_chars):
            response = post_json(self._endpoint(), self._payload(batch, role), self._headers())
            self.request_count += 1
            usage = response.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
                self.total_tokens += int(usage["total_tokens"])
            data = response.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise EmbeddingRequestError(
                    f"{self._endpoint()} returned {len(data or [])} embeddings for {len(batch)} inputs"
                )
            for item in sorted(data, key=lambda entry: int(entry.get("index", 0))):
                rows.append(item["embedding"])
        return normalize_rows(np.asarray(rows, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> EmbeddingMatrix:
        """Embed documents.

        :param texts: The texts to embed.
        :return: A float32 matrix with L2-normalized rows.
        """
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return self._embed(texts, role="document")

    def embed_queries(self, texts: list[str]) -> EmbeddingMatrix:
        """Embed queries.

        :param texts: The texts to embed.
        :return: A float32 matrix with L2-normalized rows.
        """
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return self._embed(texts, role="query")
