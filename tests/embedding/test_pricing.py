from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from zemble.embedding.cache import CachingEmbedder
from zemble.embedding.pricing import (
    BUDGET_ENV,
    BUDGET_USD_ENV,
    CONFIRM_ENV,
    DEFAULT_BUDGET_USD,
    DEFAULT_UNPRICED_BUDGET_TOKENS,
    FREE_SCHEMES,
    PRICES_USD_PER_MILLION_TOKENS,
    EmbeddingBudgetExceeded,
    bill_refusal,
    budget_tokens,
    budget_usd,
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
    """Both ceilings come from the environment, and nonsense falls back to the default."""
    assert budget_usd() == DEFAULT_BUDGET_USD, "unset means the default spending ceiling"
    monkeypatch.setenv(BUDGET_USD_ENV, "0.50")
    assert budget_usd() == 0.50, "a number is honoured"
    monkeypatch.setenv(BUDGET_USD_ENV, "plenty")
    assert budget_usd() == DEFAULT_BUDGET_USD, "nonsense falls back rather than crashing a build"

    assert budget_tokens(price=None) == DEFAULT_UNPRICED_BUDGET_TOKENS, "an unpriced model is capped by volume"
    assert budget_tokens(price=0.02) is None, "a priced model is capped by money, not by volume"
    assert budget_tokens(price=0.0) is None, "a free model is capped by nothing at all"
    monkeypatch.setenv(BUDGET_ENV, "500")
    assert budget_tokens(price=0.02) == 500, "an explicitly named token ceiling applies to a priced model too"
    assert budget_tokens(price=0.0) is None, "but a free model still costs nothing to cap"
    monkeypatch.setenv(BUDGET_ENV, "many")
    assert budget_tokens(price=None) == DEFAULT_UNPRICED_BUDGET_TOKENS, "nonsense falls back to the default"


def test_check_budget_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal names the estimate, the cost, the ceiling that refused and every way out."""
    monkeypatch.setenv(BUDGET_USD_ENV, "0.05")
    check_budget("voyage:voyage-4-lite@1024", "voyage:voyage-4-lite", 1, 2_500_000)
    with pytest.raises(EmbeddingBudgetExceeded) as raised:
        check_budget("voyage:voyage-4-lite@1024", "voyage:voyage-4-lite", 3, 5_000_000)
    message = str(raised.value)
    remedy_fragments = (".zembleignore", "sub-path", "--yes", CONFIRM_ENV, BUDGET_USD_ENV)
    for fragment in ("5,000,000", "$0.10", "$0.05", "voyage:voyage-4-lite@1024", *remedy_fragments):
        assert fragment in message, f"the refusal must name {fragment}"
    assert BUDGET_ENV not in message, "the money ceiling is the one that refused, so it is the one named"


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


def test_the_two_harms_have_two_ceilings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bill is refused in money, an unpriced model in volume, and a free one never."""
    # 1. The measured full javaweb index is affordable at the configured model, and at the dearest.
    javaweb = 15_500_000
    assert bill_refusal(javaweb, "voyage:voyage-4-lite") is None, "step 1: $0.31 is not a runaway bill"
    assert bill_refusal(javaweb, "openai:https://api.openai.com/v1#text-embedding-3-large") is None, (
        "step 1: the same index at $0.13 per million ($2.02) is still an ordinary workspace index"
    )

    # 2. A build that would really spend money is refused, in money, naming the money knob.
    runaway = bill_refusal(2_500_000_000, "voyage:voyage-4-lite")
    assert runaway is not None and "$50.00" in runaway, f"step 2: a $50 build is refused, got {runaway}"
    assert BUDGET_USD_ENV in runaway, "step 2: the refusal names the ceiling a caller would raise"

    # 3. A model with no documented price cannot be billed, so its VOLUME is what is capped.
    unpriced = bill_refusal(DEFAULT_UNPRICED_BUDGET_TOKENS + 1, "voyage:voyage-9-imaginary")
    assert unpriced is not None, "step 3: an unknown price is never treated as unlimited"
    assert "no documented price" in unpriced, f"step 3: and the reason says why it is capped, got {unpriced}"
    assert BUDGET_ENV in unpriced, "step 3: naming the volume knob, not the money one"
    assert bill_refusal(DEFAULT_UNPRICED_BUDGET_TOKENS, "voyage:voyage-9-imaginary") is None, "step 3: at the ceiling"

    # 4. A local family is never billed, however much of it there is.
    assert bill_refusal(2_500_000_000, "model2vec:minishlab/potion-code-16M-v2") is None, "step 4: free stays free"

    # 5. An explicitly named token ceiling is an additional cap on the priced lane.
    monkeypatch.setenv(BUDGET_ENV, "500")
    explicit = bill_refusal(501, "voyage:voyage-4-lite")
    assert explicit is not None and BUDGET_ENV in explicit, "step 5: a caller who names a ceiling means it"
    assert bill_refusal(501, "model2vec:minishlab/potion-code-16M-v2") is None, "step 5: except where nothing is paid"
