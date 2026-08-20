"""The Voyage AI embeddings provider."""

from __future__ import annotations

from typing import Any

from zemble.embedding.remote import HttpEmbedder

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
API_KEY_ENV = "VOYAGE_API_KEY"

#: Documented ceilings: 1,000 texts per request for every model, 320K tokens for
#: voyage-4/voyage-code-4 and 1M for voyage-4-lite. We stay well under both so a
#: character-based token estimate cannot push a request over.
MAX_TEXTS_PER_REQUEST = 128
DEFAULT_MAX_TOKENS_PER_REQUEST = 100_000

#: Models whose default output width is documented, so no probe request is needed.
KNOWN_DIMENSIONS = {
    "voyage-code-4": 1024,
    "voyage-4": 1024,
    "voyage-4-lite": 1024,
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-code-3": 1024,
}

_LARGE_TOKEN_BUDGET_MODELS = frozenset({"voyage-4-lite", "voyage-3.5-lite"})


class VoyageEmbedder(HttpEmbedder):
    """Embeds through https://api.voyageai.com/v1/embeddings, using ``input_type`` for asymmetry."""

    def __init__(self, model: str, dimensions: int | None = None) -> None:
        """Initialise the embedder.

        :param model: A Voyage model name, e.g. ``voyage-code-4``.
        :param dimensions: Requested Matryoshka output width, or None for the model default.
        """
        max_tokens = 300_000 if model in _LARGE_TOKEN_BUDGET_MODELS else DEFAULT_MAX_TOKENS_PER_REQUEST
        super().__init__(
            model=model,
            dimensions=dimensions if dimensions is not None else KNOWN_DIMENSIONS.get(model),
            max_texts=MAX_TEXTS_PER_REQUEST,
            max_tokens=max_tokens,
            api_key_env=API_KEY_ENV,
        )

    @property
    def model_id(self) -> str:
        """The normalized spec string, with the resolved width made explicit."""
        return f"voyage:{self._model}@{self.dimensions}"

    def _endpoint(self) -> str:
        """Return the Voyage embeddings URL."""
        return VOYAGE_URL

    def _headers(self) -> dict[str, str]:
        """Return the bearer auth header."""
        return {"Authorization": f"Bearer {self._api_key()}"}

    def _payload(self, texts: list[str], role: str | None) -> dict[str, Any]:
        """Return the Voyage request body.

        :param texts: The batch of texts.
        :param role: ``"document"`` or ``"query"``; passed through as ``input_type``.
        :return: The request body.
        """
        payload: dict[str, Any] = {"input": texts, "model": self._model, "truncation": True}
        if role is not None:
            payload["input_type"] = role
        if self._dimensions is not None:
            payload["output_dimension"] = self._dimensions
        return payload
