"""The local, default embedder: a Model2Vec static model."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from zemble.embedding.base import EmbeddingMatrix, normalize_rows

if TYPE_CHECKING:
    from model2vec import StaticModel


@cache
def load_static_model(model_path: str) -> "StaticModel":
    """Load a Model2Vec model, retrying once with a forced download when the local cache is unusable."""
    # model2vec drags in huggingface_hub, which is a fifth of a search's import budget and is
    # needed only once a model is actually loaded.
    from huggingface_hub.utils.tqdm import disable_progress_bars
    from model2vec import StaticModel

    # Disable HF progress bars since the model is loaded silently in the background during indexing.
    disable_progress_bars()
    try:
        try:
            return StaticModel.from_pretrained(model_path, force_download=False)
        except ValueError:
            return StaticModel.from_pretrained(model_path, force_download=True)
    finally:
        disable_progress_bars()


class Model2VecEmbedder:
    """Embeds through a local static model; documents and queries are treated identically."""

    #: Runs on this machine: free, offline, never budget-gated.
    is_remote = False

    #: A static model is exactly what the shipped fusion weights were tuned on, and every
    #: increase measured worse (docs/voyage.md): it earns no extra share of the fusion.
    semantic_weight_bonus = 0.0

    def __init__(self, model_path: str) -> None:
        """Initialise the embedder.

        :param model_path: A Hugging Face model id or a local directory.
        """
        self._model_path = model_path
        self._model: "StaticModel | None" = None

    @property
    def model(self) -> "StaticModel":
        """The underlying static model, loaded on first use."""
        if self._model is None:
            self._model = load_static_model(self._model_path)
        return self._model

    @property
    def model_id(self) -> str:
        """The normalized spec string."""
        return f"model2vec:{self._model_path}"

    @property
    def dimensions(self) -> int:
        """The model's vector width."""
        return int(self.model.dim)

    @property
    def declared_dimensions(self) -> int | None:
        """The width, if the model is already loaded; loading one just to report a width is not worth it."""
        return None if self._model is None else int(self._model.dim)

    def embed_documents(self, texts: list[str]) -> EmbeddingMatrix:
        """Embed documents.

        :param texts: The texts to embed.
        :return: A float32 matrix with L2-normalized rows.
        """
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        encoded = np.asarray(self.model.encode(texts, use_multiprocessing=False), dtype=np.float32)
        return normalize_rows(encoded)

    def embed_queries(self, texts: list[str]) -> EmbeddingMatrix:
        """Embed queries; identical to :meth:`embed_documents` for a symmetric static model.

        :param texts: The texts to embed.
        :return: A float32 matrix with L2-normalized rows.
        """
        return self.embed_documents(texts)
