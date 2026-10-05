from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from zemble.embedding.base import Embedder
from zemble.embedding.cache import CachingEmbedder
from zemble.embedding.pricing import (
    BUDGET_ENV,
    BUDGET_USD_ENV,
    CONFIRM_ENV,
    DEFAULT_BUDGET_USD,
    DEFAULT_UNPRICED_BUDGET_TOKENS,
    FREE_SCHEMES,
    MAX_BUDGET_TOKENS,
    PRICE_REVIEW_INTERVAL_DAYS,
    PRICES_CHECKED_ON,
    PRICES_USD_PER_MILLION_TOKENS,
    UNNAMED_EMBEDDER,
    EmbeddingBudgetExceeded,
    bill_refusal,
    budget_tokens,
    budget_usd,
    check_budget,
    estimate_cost,
    estimate_tokens,
    format_cost,
    format_usd,
    price_per_million,
    require_affordable_bill,
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


def buy(embedder: Embedder, texts: list[str]) -> None:
    """Buy vectors the way a build seam does: the bill guard first, then the embedder.

    This is exactly what `zemble.index.create.write_index` and the dupes logic lane do, spelled
    out here so a journey can name its own token counts instead of a capsule's.
    """
    require_affordable_bill(embedder, texts)
    embedder.embed_documents(texts)


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


@pytest.mark.parametrize("raw", ["plenty", "0,50", "nan", "NaN", "inf", "-inf", "infinity", "1e400"])
def test_a_ceiling_that_is_not_a_number_is_not_a_ceiling(
    raw: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A budget that cannot be compared against falls back to the default, and says so.

    `float()` accepts `nan`, `inf`, `infinity` and an overflowing literal, and every one of
    them DISABLES the money ceiling silently: nan compares false against everything and inf is
    never exceeded. A European "fifty cents" (`0,50`) is the same defect from the other side.
    """
    monkeypatch.setenv(BUDGET_USD_ENV, raw)
    with caplog.at_level("WARNING", logger="zemble.envknob"):
        limit = budget_usd()
    assert limit == DEFAULT_BUDGET_USD, f"{raw!r} must fall back, not become the ceiling"
    assert bill_refusal(2_500_000_000, "voyage:voyage-4-lite") is not None, f"{raw!r} must not disarm the guard"
    assert any(raw in record.getMessage() for record in caplog.records), f"{raw!r} was swallowed instead of reported"


def test_budget_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both ceilings come from the environment, and a caller who names one means it."""
    assert budget_usd() == DEFAULT_BUDGET_USD, "unset means the default spending ceiling"
    monkeypatch.setenv(BUDGET_USD_ENV, "0.50")
    assert budget_usd() == 0.50, "a number is honoured"
    monkeypatch.setenv(BUDGET_USD_ENV, "0")
    assert budget_usd() == 0.0, "and a deliberate zero still disables the money half"

    assert budget_tokens(price=None) == DEFAULT_UNPRICED_BUDGET_TOKENS, "an unpriced model is capped by volume"
    assert budget_tokens(price=0.02) == MAX_BUDGET_TOKENS, "a priced model is capped by money AND by the backstop"
    assert budget_tokens(price=0.0) is None, "a free model is capped by nothing at all"
    monkeypatch.setenv(BUDGET_ENV, "500")
    assert budget_tokens(price=0.02) == 500, "an explicitly named token ceiling applies to a priced model too"
    assert budget_tokens(price=0.0) is None, "but a free model still costs nothing to cap"
    monkeypatch.setenv(BUDGET_ENV, "many")
    assert budget_tokens(price=None) == DEFAULT_UNPRICED_BUDGET_TOKENS, "nonsense falls back to the default"
    assert budget_tokens(price=0.02) == MAX_BUDGET_TOKENS, "and a priced model falls back to the backstop"


def test_the_price_table_is_dated_and_sane() -> None:
    """The prices divide the money ceiling, so nobody may inherit them unread or mistyped.

    A rate that is 10x too low admits 10x the tokens while the guard keeps printing "$5.00":
    the table stopped being display data the day the ceiling became money.
    """
    # 1. Somebody read the providers' price lists recently enough to answer for them.
    age = date.today() - PRICES_CHECKED_ON
    assert age <= timedelta(days=PRICE_REVIEW_INTERVAL_DAYS), (
        f"the price table was last checked {age.days} days ago; re-read each provider's price "
        f"list, correct PRICES_USD_PER_MILLION_TOKENS and move PRICES_CHECKED_ON forward"
    )
    assert PRICES_CHECKED_ON <= date.today(), "a table checked in the future has not been checked"

    # 2. Every rate is in the documented unit: USD per MILLION tokens, not per token or per thousand.
    for scheme, models in PRICES_USD_PER_MILLION_TOKENS.items():
        for model, price in models.items():
            assert 0.001 <= price <= 10.0, f"{scheme}:{model} is priced {price}, which is not USD per million tokens"


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

    # An embedder declaring neither a model nor a family is still named: a refusal a reader has
    # to act on may not read "Refusing to embed 1 uncached chunk(s) with : ...".
    with pytest.raises(EmbeddingBudgetExceeded) as unnamed:
        check_budget("", "", 1, 5_000_000)
    assert f"with {UNNAMED_EMBEDDER}:" in str(unnamed.value), f"got {unnamed.value}"


def test_budget_guard_journey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A paid build walks under budget -> over budget -> cached -> confirmed -> local, refusing once."""
    inner = PricedEmbedder()
    embedder = CachingEmbedder(inner, "voyage:voyage-4-lite", tmp_path)

    # 1. Under budget: the provider is called and the vectors are cached.
    monkeypatch.setenv(BUDGET_ENV, "1000")
    buy(embedder, ["a" * 360])
    assert inner.document_batches == [["a" * 360]], "step 1: an affordable build embeds normally"

    # 2. Over budget: nothing reaches the provider at all.
    with pytest.raises(EmbeddingBudgetExceeded) as raised:
        buy(embedder, ["b" * 36_000])
    assert len(inner.document_batches) == 1, "step 2: a refused build must send nothing"
    assert "10,000 estimated tokens" in str(raised.value), "step 2: the estimate is named"

    # 3. Already-cached text is not pending, so it is not counted against the budget.
    buy(embedder, ["a" * 360])
    assert len(inner.document_batches) == 1, "step 3: a cache hit costs nothing and is never gated"

    # 4. Confirmed: the same call goes through.
    monkeypatch.setenv(CONFIRM_ENV, "1")
    buy(embedder, ["b" * 36_000])
    assert len(inner.document_batches) == 2, "step 4: an explicit confirmation embeds anyway"

    # 5. A local embedder is never gated, confirmation or not.
    monkeypatch.delenv(CONFIRM_ENV)
    local = PricedEmbedder(remote=False)
    local_embedder = CachingEmbedder(local, "model2vec:test", tmp_path)
    buy(local_embedder, ["c" * 36_000])
    assert len(local.document_batches) == 1, "step 5: a local embedder costs nothing and is never refused"


def test_paid_embed_logs_one_line(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Every paid embed announces its size and cost before the first request."""
    embedder = CachingEmbedder(PricedEmbedder(), "voyage:voyage-4-lite", tmp_path)
    with caplog.at_level("INFO", logger="zemble.embedding.pricing"):
        buy(embedder, ["a" * 3600])
    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 1, f"exactly one announcement, got {lines}"
    assert "embedding 1 uncached chunk(s), ~1000 tokens, ~$0.00002 with voyage:voyage-4-lite@4" == lines[0], lines[0]


def test_a_money_figure_reads_as_zero_only_when_it_is_zero() -> None:
    """The one money format keeps a small figure readable however small it gets.

    Rounding to `$0.00` is what two formatters did to a refusal ("~$0.0002 exceeds the budget of
    $0.00"); fixing that at four decimals only moved the same defect one order of magnitude
    down, where a ceiling of `$0.0000001` read "exceeds the budget of $0.0000".
    """
    # 1. Ordinary money is ordinary: two decimals, and zero is allowed to look like zero.
    assert format_usd(0.0) == "$0.00", "step 1: nothing costs nothing"
    assert (format_usd(5.0), format_usd(0.01)) == ("$5.00", "$0.01"), "step 1: a cent is still two decimals"

    # 2. Under a cent the precision follows the figure, so nothing positive renders as zero.
    for amount in (0.009, 0.0002, 0.00002, 1e-7, 1e-12, 1e-13, 1e-300):
        rendered = format_usd(amount)
        assert float(rendered.removeprefix("$")) > 0, f"step 2: {amount} rendered as {rendered}, which reads as zero"
    assert format_usd(1e-7) == "$0.0000001", "step 2: and it is still a decimal figure, not an exponent"
    assert format_usd(1e-13) == "$1.00e-13", "step 2: until digits stop being readable at all"


def test_local_embed_announces_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A free embedder does not narrate a bill it never sends."""
    embedder = CachingEmbedder(PricedEmbedder(remote=False), "model2vec:test", tmp_path)
    with caplog.at_level("INFO", logger="zemble.embedding.pricing"):
        buy(embedder, ["a" * 3600])
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
    monkeypatch.delenv(BUDGET_ENV)

    # 6. A price the table gets WRONG narrows the ceiling; it can never delete it. At a tenth of
    #    the real rate $5.00 would buy 2.5 billion tokens, and the volume backstop still refuses.
    monkeypatch.setitem(PRICES_USD_PER_MILLION_TOKENS["voyage"], "voyage-4-lite", 0.002)
    mistyped = bill_refusal(MAX_BUDGET_TOKENS + 1, "voyage:voyage-4-lite")
    assert mistyped is not None, "step 6: a rate 10x too low must not lift the ceiling off the build"
    assert f"{MAX_BUDGET_TOKENS:,} tokens" in mistyped, f"step 6: refused on volume, got {mistyped}"
    assert "no price may lift" in mistyped, f"step 6: and it says the volume is what refused, got {mistyped}"
    assert bill_refusal(MAX_BUDGET_TOKENS, "voyage:voyage-4-lite") is None, "step 6: at the backstop, not over it"
