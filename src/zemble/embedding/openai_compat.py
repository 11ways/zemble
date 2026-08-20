"""The OpenAI-compatible ``/v1/embeddings`` provider: OpenAI, Ollama, LM Studio, vLLM."""

from __future__ import annotations

import os
from typing import Any

from zemble.embedding.remote import HttpEmbedder

DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

#: Conservative for a local server, still far above one HTTP round trip per chunk.
MAX_TEXTS_PER_REQUEST = 64
MAX_TOKENS_PER_REQUEST = 100_000


class OpenAICompatibleEmbedder(HttpEmbedder):
    """Embeds through an OpenAI-shaped ``/embeddings`` endpoint.

    The OpenAI schema has no ``input_type``, so documents and queries are embedded
    identically; an asymmetric model behind this endpoint cannot be told apart here.
    An unset API key is tolerated because local servers (Ollama, LM Studio) need none.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        dimensions: int | None = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
    ) -> None:
        """Initialise the embedder.

        :param base_url: Base URL up to and including the API version, e.g. ``http://localhost:11434/v1``.
        :param model: The model name to request.
        :param dimensions: Requested output width, passed through when the server supports it.
        :param api_key_env: Environment variable holding the API key, if any.
        """
        super().__init__(
            model=model,
            dimensions=dimensions,
            max_texts=MAX_TEXTS_PER_REQUEST,
            max_tokens=MAX_TOKENS_PER_REQUEST,
            api_key_env=api_key_env,
        )
        self._base_url = base_url.rstrip("/")

    @property
    def model_id(self) -> str:
        """The normalized spec string, with the resolved width made explicit."""
        return f"openai:{self._base_url}#{self._model}@{self.dimensions}"

    def _endpoint(self) -> str:
        """Return the embeddings URL under the configured base."""
        return f"{self._base_url}/embeddings"

    def _headers(self) -> dict[str, str]:
        """Return the auth header, omitted entirely when no key is configured."""
        key = os.environ.get(self._api_key_env, "").strip()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _payload(self, texts: list[str], role: str | None) -> dict[str, Any]:
        """Return the OpenAI request body; ``role`` is ignored because the schema has no field for it.

        :param texts: The batch of texts.
        :param role: Ignored.
        :return: The request body.
        """
        payload: dict[str, Any] = {"input": texts, "model": self._model}
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions
        return payload
