"""What a paid embedder costs, and the budget that refuses a surprise bill.

The price table is data with one declaring home: a model that is not in it has an
unknown price, never a guessed one, and an unknown price never silently becomes free.
"""

from __future__ import annotations

import math
import os

#: Characters per token used to estimate a bill before anything is sent. Measured on the
#: javaweb workspace (docs/voyage.md: 15,526,808 provider-reported tokens for 73,957 chunks
#: of ~600 chars). Deliberately different from the batching constant in ``http.py``, which
#: is pessimistic on purpose so a batch cannot overshoot a provider ceiling.
ESTIMATE_CHARS_PER_TOKEN = 3.6

#: Names the token budget a single build may spend before it is refused.
BUDGET_ENV = "ZEMBLE_EMBED_BUDGET_TOKENS"
#: Set to 1 to embed whatever the budget would have refused.
CONFIRM_ENV = "ZEMBLE_EMBED_CONFIRM"
#: Roughly a full javaweb index; a bill of a few tens of cents passes, a runaway one does not.
DEFAULT_BUDGET_TOKENS = 2_000_000

#: Schemes that run on this machine: no round trip, no bill, never gated.
FREE_SCHEMES = frozenset({"model2vec"})

#: USD per million tokens, by scheme and model name, from each provider's own price list.
#: A remote model missing here is priced None ("unknown price"), which is reported, not assumed.
PRICES_USD_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "voyage": {
        "voyage-code-4": 0.12,
        "voyage-4": 0.06,
        "voyage-4-lite": 0.02,
        "voyage-3.5": 0.06,
        "voyage-3.5-lite": 0.02,
    },
    "openai": {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
    },
}


class EmbeddingBudgetExceeded(RuntimeError):
    """A build would have embedded more tokens than the budget allows, so nothing was sent."""


def model_of_family(family: str) -> tuple[str, str]:
    """Split a cache family key into its scheme and the model name the price table is keyed by.

    :param family: ``voyage:<model>``, ``model2vec:<model>`` or ``openai:<base_url>#<model>``.
    :return: The scheme and the model name.
    """
    scheme, _, rest = family.partition(":")
    return scheme, rest.rpartition("#")[2]


def price_per_million(family: str) -> float | None:
    """Return the USD-per-million-tokens price for an embedder family.

    :param family: The cache family key, e.g. ``voyage:voyage-4-lite``.
    :return: The price, 0.0 for a local family, or None when the model has no documented price.
    """
    scheme, model = model_of_family(family)
    if scheme in FREE_SCHEMES:
        return 0.0
    return PRICES_USD_PER_MILLION_TOKENS.get(scheme, {}).get(model)


def estimate_tokens(texts: list[str]) -> int:
    """Estimate how many tokens a set of texts costs, from their character count."""
    return math.ceil(sum(len(text) for text in texts) / ESTIMATE_CHARS_PER_TOKEN)


def estimate_cost(tokens: int, price: float | None) -> float | None:
    """Return the estimated USD for a token count at a price, or None when the price is unknown."""
    return None if price is None else tokens * price / 1_000_000


def format_cost(tokens: int, price: float | None) -> str:
    """Render an estimated cost for a human, naming an unknown price instead of hiding it."""
    cost = estimate_cost(tokens, price)
    if cost is None:
        return "unknown price"
    return f"${cost:.2f}" if cost >= 0.01 or cost == 0 else f"${cost:.4f}"


def budget_tokens() -> int:
    """Return the per-build token budget; 0 or less disables the guard entirely."""
    raw = os.environ.get(BUDGET_ENV, "").strip()
    if not raw:
        return DEFAULT_BUDGET_TOKENS
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_BUDGET_TOKENS


def confirmed() -> bool:
    """Return whether the caller has already agreed to pay whatever this build costs."""
    return os.environ.get(CONFIRM_ENV, "").strip().lower() in {"1", "true", "yes"}


def exceeds_budget(tokens: int) -> bool:
    """Return whether this many tokens would be refused right now."""
    budget = budget_tokens()
    return budget > 0 and not confirmed() and tokens > budget


def check_budget(model_id: str, family: str, count: int, tokens: int) -> None:
    """Refuse a paid embed that would blow the budget, before a single text is sent.

    :param model_id: The embedder's normalized spec string, for the message.
    :param family: The cache family key, used to price the estimate.
    :param count: How many uncached texts are pending.
    :param tokens: The estimated token count for them.
    :raises EmbeddingBudgetExceeded: If the estimate is over budget and nothing confirmed it.
    """
    if not exceeds_budget(tokens):
        return
    price = price_per_million(family)
    raise EmbeddingBudgetExceeded(
        f"Refusing to embed {count} uncached chunk(s) with {model_id}: "
        f"~{tokens:,} estimated tokens (~{format_cost(tokens, price)}) exceeds the budget of "
        f"{budget_tokens():,} tokens. Pass --yes (or set {CONFIRM_ENV}=1) to embed anyway, "
        f"or raise {BUDGET_ENV}."
    )
