"""A local cross-encoder reranker, behind the optional ``zemble[rerank]`` extra."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Pairs scored per forward pass. Padding is per batch, so a small batch wastes less
#: compute on short passages than one big padded tensor would.
BATCH_SIZE = 16
#: Token ceiling for one pair. Only the passage is truncated; the query is never cut.
MAX_LENGTH = 512

_MISSING_EXTRA = (
    "A cross-encoder reranker needs torch and transformers, which are not installed. "
    "Install them with: pip install 'zemble[rerank]'"
)


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a sequence-classification cross-encoder.

    The model is loaded on the first :meth:`score` call, never at import: importing zemble
    must not drag in torch, and a configured-but-unused reranker must not cost a model load.
    """

    def __init__(self, model: str, *, batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH) -> None:
        """Initialise the reranker.

        :param model: A Hugging Face model id, e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
        :param batch_size: Pairs scored per forward pass.
        :param max_length: Token ceiling for one pair.
        """
        self._model_name = model
        self._batch_size = batch_size
        self._max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

    @property
    def model_id(self) -> str:
        """The normalized spec string."""
        return f"cross:{self._model_name}"

    def _load(self) -> None:
        """Load the tokenizer and model once.

        :raises ImportError: If the optional extra is not installed.
        """
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(_MISSING_EXTRA) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        model.eval()
        self._model = model

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score every passage against the query.

        A model with a multi-class head is read on its last logit, which is the
        relevance class for every reranker head shipped in this shape.

        :param query: The search query.
        :param passages: The candidate passages, in candidate order.
        :return: One score per passage, in the same order; higher = more relevant.
        """
        if not passages:
            return []
        self._load()
        torch = self._torch
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(passages), self._batch_size):
                batch = passages[start : start + self._batch_size]
                encoded = self._tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation="only_second",
                    max_length=self._max_length,
                    return_tensors="pt",
                )
                logits = self._model(**encoded).logits
                column = logits[:, -1] if logits.shape[-1] > 1 else logits.reshape(-1)
                scores.extend(float(value) for value in column.tolist())
        return scores
