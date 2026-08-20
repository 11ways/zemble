from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from zemble.embedding.cache import CachingEmbedder
from zemble.embedding.pricing import (
    BUDGET_ENV,
    CONFIRM_ENV,
    DEFAULT_BUDGET_TOKENS,
    FREE_SCHEMES,
    PRICES_USD_PER_MILLION_TOKENS,
    EmbeddingBudgetExceeded,
    budget_tokens,
    check_budget,
    estimate_cost,
    estimate_tokens,
    format_cost,
    price_per_million,
)
from zemble.embedding.registry import SCHEMES


class PricedEmbedder:
    """A remote-looking embedder that records what it was asked to embed."""

    is_remote = True

    def __init__(self, dimensions: int = 4, remote: bool = True) -> None:
        """Initialise the fake, optionally as a local (never billed) embedder."""
        self._dimensions = dimensions
        self.is_remote = remote
        self.document_batches: list[list[str]] = []

    @property
    def model_id(self) -> str:
        """The normalized spec string."""
        return f"voyage:voyage-4-lite@{self._dimensions}"

    @property
    def dimensions(self) -> int:
        """The vector width."""
        return self._dimensions

    @property
    def declared_dimensions(self) -> int:
        """The width, known without a request."""
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed documents, recording the batch."""
        self.document_batches.append(list(texts))
        rows = np.ones((len(texts), self._dimensions), dtype=np.float32)
        return rows / np.sqrt(self._dimensions)

    def embed_queries(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed queries."""
        return self.embed_documents(texts)


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("voyage:voyage-code-4", 0.12),
        ("voyage:voyage-4", 0.06),
        ("voyage:voyage-4-lite", 0.02),
        ("openai:https://api.openai.com/v1#text-embedding-3-small", 0.02),
        ("openai:https://api.openai.com/v1#text-embedding-3-large", 0.13),
        ("model2vec:minishlab/potion-code-16M-v2", 0.0),
        ("voyage:voyage-9-imaginary", None),
        ("openai:http://localhost:11434/v1#nomic-embed-text", None),
        ("nonsense", None),
    ],
)
def test_price_lookup(family: str, expected: float | None) -> None:
    """A known model is priced, a local one is free, and anything else is honestly unknown."""
    assert price_per_million(family) == expected, f"{family} must price at {expected}"


def test_every_scheme_is_classified() -> None:
    """Adding an embedder scheme without pricing it is a build-breaking omission, not a silent None."""
    for scheme in SCHEMES:
        assert scheme in FREE_SCHEMES or scheme in PRICES_USD_PER_MILLION_TOKENS, (
            f"scheme {scheme!r} is neither free nor priced; add it to one of the two tables"
        )


def test_estimate_arithmetic() -> None:
    """Tokens come from characters at the documented density, and cost from the price table."""
    texts = ["a" * 360, "b" * 360]
    assert estimate_tokens(texts) == 200, "720 chars at 3.6 chars per token is 200 tokens"
    assert estimate_tokens([]) == 0, "nothing to embed is nothing to pay"
    assert estimate_cost(1_000_000, 0.02) == pytest.approx(0.02), "a million tokens costs the list price"
    assert estimate_cost(1_000_000, None) is None, "an unknown price cannot become a number"
    assert format_cost(1_000_000, None) == "unknown price", "an unknown price says so"
    assert format_cost(15_500_000, 0.02) == "$0.31", "the measured javaweb index at voyage-4-lite"


def test_budget_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    """The budget comes from the environment, and nonsense falls back to the default."""
    assert budget_tokens() == DEFAULT_BUDGET_TOKENS, "unset means the default"
    monkeypatch.setenv(BUDGET_ENV, "500")
    assert budget_tokens() == 500, "an integer is honoured"
    monkeypatch.setenv(BUDGET_ENV, "many")
    assert budget_tokens() == DEFAULT_BUDGET_TOKENS, "nonsense falls back rather than crashing a build"


def test_check_budget_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal names the estimate, the price, the budget and both ways out."""
    monkeypatch.setenv(BUDGET_ENV, "100")
    check_budget("voyage:voyage-4-lite@1024", "voyage:voyage-4-lite", 1, 100)
    with pytest.raises(EmbeddingBudgetExceeded) as raised:
        check_budget("voyage:voyage-4-lite@1024", "voyage:voyage-4-lite", 3, 101)
    message = str(raised.value)
    for fragment in ("101", "100", "voyage:voyage-4-lite@1024", "--yes", CONFIRM_ENV, BUDGET_ENV):
        assert fragment in message, f"the refusal must name {fragment}"


def test_budget_guard_journey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A paid build walks under budget -> over budget -> confirmed -> local, and only refuses once."""
    inner = PricedEmbedder()
    embedder = CachingEmbedder(inner, "voyage:voyage-4-lite", tmp_path)

    # 1. Under budget: the provider is called and the vectors are cached.
    monkeypatch.setenv(BUDGET_ENV, "1000")
    embedder.embed_documents(["a" * 360])
    assert inner.document_batches == [["a" * 360]], "step 1: an affordable build embeds normally"

    # 2. Over budget: nothing reaches the provider at all.
    with pytest.raises(EmbeddingBudgetExceeded) as raised:
        embedder.embed_documents(["b" * 36_000])
    assert len(inner.document_batches) == 1, "step 2: a refused build must send nothing"
    assert "10,000 estimated tokens" in str(raised.value), "step 2: the estimate is named"

    # 3. Already-cached text is not pending, so it is not counted against the budget.
    embedder.embed_documents(["a" * 360])
    assert len(inner.document_batches) == 1, "step 3: a cache hit costs nothing and is never gated"

    # 4. Confirmed: the same call goes through.
    monkeypatch.setenv(CONFIRM_ENV, "1")
    embedder.embed_documents(["b" * 36_000])
    assert len(inner.document_batches) == 2, "step 4: an explicit confirmation embeds anyway"

    # 5. A local embedder is never gated, confirmation or not.
    monkeypatch.delenv(CONFIRM_ENV)
    local = PricedEmbedder(remote=False)
    local_embedder = CachingEmbedder(local, "model2vec:test", tmp_path)
    local_embedder.embed_documents(["c" * 36_000])
    assert len(local.document_batches) == 1, "step 5: a local embedder costs nothing and is never refused"


def test_paid_embed_logs_one_line(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Every paid embed announces its size and cost before the first request."""
    embedder = CachingEmbedder(PricedEmbedder(), "voyage:voyage-4-lite", tmp_path)
    with caplog.at_level("INFO", logger="zemble.embedding.cache"):
        embedder.embed_documents(["a" * 3600])
    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 1, f"exactly one announcement, got {lines}"
    assert "embedding 1 uncached chunk(s), ~1000 tokens, ~$0.0000 with voyage:voyage-4-lite@4" == lines[0], lines[0]


def test_local_embed_announces_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A free embedder does not narrate a bill it never sends."""
    embedder = CachingEmbedder(PricedEmbedder(remote=False), "model2vec:test", tmp_path)
    with caplog.at_level("INFO", logger="zemble.embedding.cache"):
        embedder.embed_documents(["a" * 3600])
    assert caplog.records == [], "a local embed is not a paid embed"
