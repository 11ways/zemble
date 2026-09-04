"""What a paid embedder costs, and the budget that refuses a surprise bill.

The price table is data with one declaring home: a model that is not in it has an
unknown price, never a guessed one, and an unknown price never silently becomes free.
Money is refused in money: the ceiling is USD, and the token ceiling beside it exists for
the one case a bill cannot be computed. Runaway WORK is a different harm with its own
guard, in bytes, in :mod:`zemble.index.scope`.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

#: Characters per token used to estimate a bill before anything is sent. Measured on the
#: javaweb workspace (docs/voyage.md: 15,526,808 provider-reported tokens for 73,957 chunks
#: of ~600 chars). Deliberately different from the batching constant in ``http.py``, which
#: is pessimistic on purpose so a batch cannot overshoot a provider ceiling.
ESTIMATE_CHARS_PER_TOKEN = 3.6

#: Names the ceiling on what one build may SPEND.
BUDGET_USD_ENV = "ZEMBLE_EMBED_BUDGET_USD"
#: 16x the measured full javaweb index at the configured model ($0.31, docs/voyage.md) and
#: 2.7x the same index at the dearest code model ($1.86), so a legitimate workspace index
#: never prompts and a runaway still does.
DEFAULT_BUDGET_USD = 5.00

#: Names the volume ceiling that applies when a model has NO documented price, and an
#: additional token cap on every lane when a caller sets it deliberately.
BUDGET_ENV = "ZEMBLE_EMBED_BUDGET_TOKENS"
#: An unpriced model's bill cannot be bounded, so its VOLUME is. 2M tokens is $0.26 even at
#: the dearest documented rate, so an unknown model can never surprise by more than that.
DEFAULT_UNPRICED_BUDGET_TOKENS = 2_000_000

#: Set to 1 to embed whatever the budget would have refused.
CONFIRM_ENV = "ZEMBLE_EMBED_CONFIRM"

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
    """A build would have cost more than the budget allows, so nothing was sent."""

    def __init__(self, message: str, knob: str = BUDGET_USD_ENV) -> None:
        """Refuse a bill, carrying the ceiling's own environment variable rather than a guess.

        :param message: The refusal text, which already names the ceiling in its own unit.
        :param knob: The environment variable that raises the ceiling this refusal hit.
        """
        super().__init__(message)
        self.knob = knob


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


def budget_usd() -> float:
    """Return the per-build spending ceiling in USD; 0 or less disables the money half.

    :return: The USD ceiling for this build.
    """
    raw = os.environ.get(BUDGET_USD_ENV, "").strip()
    if not raw:
        return DEFAULT_BUDGET_USD
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_BUDGET_USD


def budget_tokens(price: float | None = None) -> int | None:
    """Return the token ceiling this build is capped by, or None when tokens do not cap it.

    A caller who names a token ceiling means it, so an explicit value applies to any model
    that costs anything. Without one, only a model with NO documented price is capped by
    volume: a priced model is capped by :func:`budget_usd`, and a free one by nothing.

    :param price: The model's USD per million tokens; 0.0 when free, None when undocumented.
    :return: The ceiling in tokens, 0 or less when disabled, or None when no token cap applies.
    """
    if price == 0.0:
        return None
    raw = os.environ.get(BUDGET_ENV, "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return None if price is not None else DEFAULT_UNPRICED_BUDGET_TOKENS


def confirmed() -> bool:
    """Return whether the caller has already agreed to pay whatever this build costs."""
    return os.environ.get(CONFIRM_ENV, "").strip().lower() in {"1", "true", "yes"}


def _bill_refusal(tokens: int, family: str) -> tuple[str, str] | None:
    """Decide a bill refusal once, returning both its wording and the knob that named the ceiling."""
    if confirmed():
        return None
    price = price_per_million(family)
    ceiling = budget_tokens(price)
    if ceiling is not None and 0 < ceiling < tokens:
        unpriced = "" if price is not None else " for a model with no documented price"
        return (
            f"~{tokens:,} estimated tokens exceeds the ceiling of {ceiling:,} tokens{unpriced} ({BUDGET_ENV})",
            BUDGET_ENV,
        )
    if price is None:
        # FAIL CLOSED: an unknown price is never treated as free, only as un-billable, which is
        # why the volume ceiling above is the one that governs it.
        return None
    limit = budget_usd()
    cost = estimate_cost(tokens, price)
    if cost is not None and 0 < limit < cost:
        return (
            f"~{tokens:,} estimated tokens (~{format_cost(tokens, price)}) exceeds the budget of "
            f"${limit:.2f} ({BUDGET_USD_ENV})",
            BUDGET_USD_ENV,
        )
    return None


def bill_refusal(tokens: int, family: str) -> str | None:
    """Return why buying this many uncached tokens is refused, or None when it is affordable.

    :param tokens: The estimated token count of the texts that would actually be bought.
    :param family: The cache family key, used to price them.
    :return: The reason, naming the ceiling and the knob that sets it, or None.
    """
    refusal = _bill_refusal(tokens, family)
    return None if refusal is None else refusal[0]


def check_budget(model_id: str, family: str, count: int, tokens: int) -> None:
    """Refuse a paid embed that would blow the budget, before a single text is sent.

    :param model_id: The embedder's normalized spec string, for the message.
    :param family: The cache family key, used to price the estimate.
    :param count: How many uncached texts are pending.
    :param tokens: The estimated token count for them.
    :raises EmbeddingBudgetExceeded: If the estimate is over budget and nothing confirmed it.
    """
    refusal = _bill_refusal(tokens, family)
    if refusal is None:
        return
    reason, knob = refusal
    raise EmbeddingBudgetExceeded(
        f"Refusing to embed {count} uncached chunk(s) with {model_id}: {reason}. {remedies(None, knob)}", knob
    )


def embedder_family(embedder: object) -> str:
    """Return the price/cache family key of an embedder, without asking a provider anything.

    A wrapped embedder carries the family the cache file and the price table are keyed by; a
    bare one only has its spec string, whose ``@<dims>`` suffix the price table is not keyed by.
    Reading that spec can cost a provider round trip on a model with no documented width, so a
    failure to name the family is reported as an unknown price rather than raised.
    """
    family = getattr(getattr(embedder, "cache", None), "family", None)
    if isinstance(family, str):
        return family
    try:
        model_id = str(getattr(embedder, "model_id", ""))
    except Exception:  # pragma: no cover - only a provider probe can fail here
        return ""
    scheme, separator, rest = model_id.partition(":")
    body, at, _dimensions = rest.rpartition("@")
    return f"{scheme}{separator}{body if at else rest}"


def remedies(root: str | Path | None = None, knob: str = BUDGET_USD_ENV) -> str:
    """Name the ways past a refusal, cheapest first, in the one wording every guard uses.

    :param root: The tree being indexed, named in the paths; None renders a placeholder.
    :param knob: The environment variable naming the ceiling that refused, which a caller raises.
    :return: One sentence listing the exclusion file, the sub-path narrowing and the env escapes.
    """
    where = str(root) if root is not None else "<root>"
    return (
        f"Exclude paths with {where}/.zembleignore (gitignore syntax), "
        f"or point repo at a sub-path such as {where}/src, "
        f"or raise {knob} / set {CONFIRM_ENV}=1 (--yes on the CLI) in the environment of the process "
        f"that builds "
        "(a running daemon does not see a client's environment; restart it)."
    )
