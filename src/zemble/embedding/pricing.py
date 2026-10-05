"""What a paid embedder costs, and the budget that refuses a surprise bill.

The price table is data with one declaring home: a model that is not in it has an
unknown price, never a guessed one, and an unknown price never silently becomes free.
Money is refused in money: the ceiling is USD, and the token ceilings beside it exist for
the two cases a bill cannot be trusted - a model with no documented price, and a price that
has moved since anybody read it. Runaway WORK is a different harm with its own guard, in
bytes, in :mod:`zemble.index.scope`.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import date
from pathlib import Path

from zemble.embedding.base import is_remote
from zemble.envknob import env_float, env_int
from zemble.refusal import Refused

logger = logging.getLogger(__name__)

#: Characters per token used to estimate a bill before anything is sent. Measured on the
#: javaweb workspace (docs/voyage.md: 15,526,808 provider-reported tokens for 73,957 chunks
#: of ~600 chars). Deliberately different from the batching constant in ``http.py``, which
#: is pessimistic on purpose so a batch cannot overshoot a provider ceiling.
ESTIMATE_CHARS_PER_TOKEN = 3.6

#: How much bigger the text a build embeds is than the source bytes it came from: a context
#: capsule prefixes every chunk with a header whose size is roughly constant, so the overhead
#: scales with FILE SIZE rather than with the tree. Measured over real builds: 1.05x on 20 KB
#: files, 1.10x at 400 B, 1.50x at 80 B, 2.60x at 20 B. THE declaring home of that range - the
#: argument that the volume backstop binds needs the LOW end, and the worst-case bill a build
#: at the work ceiling can carry needs the HIGH one, so both are quoted, never re-typed.
CAPSULE_OVERHEAD_LOW = 1.05
CAPSULE_OVERHEAD_HIGH = 2.60

#: Names the ceiling on what one build may SPEND.
BUDGET_USD_ENV = "ZEMBLE_EMBED_BUDGET_USD"
#: 16x the measured full javaweb index at the configured model ($0.31, docs/voyage.md) and
#: 2.7x the same index at the dearest code model ($1.86), so a legitimate workspace index
#: never prompts and a runaway still does.
DEFAULT_BUDGET_USD = 5.00

#: Names the volume ceiling that applies when a model has NO documented price, the absolute
#: backstop below, and an additional token cap on every lane when a caller sets it deliberately.
BUDGET_ENV = "ZEMBLE_EMBED_BUDGET_TOKENS"
#: An unpriced model's bill cannot be bounded, so its VOLUME is. This caps how much text an
#: unknown model may be handed; it says nothing about what that model charges for it.
DEFAULT_UNPRICED_BUDGET_TOKENS = 2_000_000

#: The day every price below was last read off its provider's own price list.
#: ``test_the_price_table_is_dated_and_sane`` fails once it is older than the review interval:
#: the table is not display data any more, so nobody may inherit it unread.
PRICES_CHECKED_ON = date(2026, 9, 4)
#: How long the table may go unread before that test asks for a re-read.
PRICE_REVIEW_INTERVAL_DAYS = 180

#: Set to 1 to embed whatever the budget would have refused.
CONFIRM_ENV = "ZEMBLE_EMBED_CONFIRM"

#: Schemes that run on this machine: no round trip, no bill, never gated.
FREE_SCHEMES = frozenset({"model2vec"})

#: What a refusal calls an embedder that declares neither a model id nor a family, because
#: "Refusing to embed 1 uncached chunk(s) with : ..." names nothing a reader can act on.
UNNAMED_EMBEDDER = "an unnamed embedder"

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

#: The dearest rate the table above documents. Derived, never typed twice: adding a dearer
#: model moves the volume backstop below it with no second edit.
DEAREST_DOCUMENTED_RATE = max(price for prices in PRICES_USD_PER_MILLION_TOKENS.values() for price in prices.values())

#: The volume no build passes however cheap the price table claims a model is. Dividing a money
#: ceiling by a price makes that price load-bearing: a rate 10x too low admits 10x the tokens.
#: Denominating the backstop in the DEAREST documented rate is what bounds that - at this many
#: tokens even the dearest model in the table bills exactly the default budget - so a rate
#: mistyped low for any other model NARROWS the ceiling instead of deleting it. A mistyped
#: dearest entry does lift it, which is what ``PRICES_CHECKED_ON`` and the unit-sanity test are
#: for. It has to sit BELOW what the 180 MB work ceiling can produce, which is that volume over
#: :data:`ESTIMATE_CHARS_PER_TOKEN` times the measured capsule overhead: ~52M estimated tokens at
#: :data:`CAPSULE_OVERHEAD_LOW` and ~130M at :data:`CAPSULE_OVERHEAD_HIGH`.
#: 38.5M sits under even the low end, so it binds whatever a tree is made of; a
#: legitimate build that big names ``ZEMBLE_EMBED_BUDGET_TOKENS`` deliberately. Raising the
#: money knob does NOT raise it.
MAX_BUDGET_TOKENS = int(DEFAULT_BUDGET_USD / DEAREST_DOCUMENTED_RATE * 1_000_000)


class EmbeddingBudgetExceeded(Refused):
    """A build would have cost more than the budget allows, so nothing was sent."""

    DEFAULT_KNOB = BUDGET_USD_ENV


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


def format_usd(amount: float) -> str:
    """Render a USD amount, keeping a sub-cent figure readable rather than rounding it to $0.00.

    THE one money format: an estimate and the ceiling it is compared against are printed side
    by side, and two formatters made a refusal read ``~$0.0002 exceeds the budget of $0.00``.
    Four decimals was the same defect one order of magnitude down - a ceiling of $0.0000001 read
    ``exceeds the budget of $0.0000`` - so the precision follows the figure: no amount that is
    not zero ever renders as one. Below a picodollar the digits stop being readable at all and
    the exponent is what a reader can act on.
    """
    # A money formatter that raises takes the refusal message down with it, and this one exists
    # to make refusals readable, so a non-finite amount renders rather than reaching math.floor.
    if amount == 0 or not math.isfinite(amount) or abs(amount) >= 0.01:
        return f"${amount:.2f}"
    digits = -math.floor(math.log10(abs(amount)))
    return f"${amount:.{digits}f}" if digits <= 12 else f"${amount:.2e}"


def format_cost(tokens: int, price: float | None) -> str:
    """Render an estimated cost for a human, naming an unknown price instead of hiding it."""
    cost = estimate_cost(tokens, price)
    return "unknown price" if cost is None else format_usd(cost)


def budget_usd() -> float:
    """Return the per-build spending ceiling in USD; 0 or less disables the money half.

    :return: The USD ceiling for this build.
    """
    return env_float(BUDGET_USD_ENV, DEFAULT_BUDGET_USD)


def explicit_budget_tokens() -> int | None:
    """Return the token ceiling a caller named deliberately, or None when nobody named one.

    :return: The value of the token knob, or None when it is unset or unreadable.
    """
    return env_int(BUDGET_ENV, None)


def budget_tokens(price: float | None = None) -> int | None:
    """Return the token ceiling this build is capped by, or None when tokens do not cap it.

    A caller who names a token ceiling means it, so an explicit value applies to any model
    that costs anything - including a value of 0 or less, which disables the volume half.
    Without one, a model with NO documented price is capped at the small unpriced volume, a
    priced one at :data:`MAX_BUDGET_TOKENS` (the money ceiling is only as good as the price
    that divides it), and a free one at nothing.

    :param price: The model's USD per million tokens; 0.0 when free, None when undocumented.
    :return: The ceiling in tokens, 0 or less when disabled, or None when no token cap applies.
    """
    if price == 0.0:
        return None
    explicit = explicit_budget_tokens()
    if explicit is not None:
        return explicit
    return MAX_BUDGET_TOKENS if price is not None else DEFAULT_UNPRICED_BUDGET_TOKENS


def applicable_budget_usd(price: float | None) -> float | None:
    """Return the USD ceiling governing a build at this price, or None when money does not cap it.

    FAIL CLOSED: an unknown price yields None here because it cannot be turned into a bill, never
    because the build is free - :func:`budget_tokens` is what caps that build, by volume.

    :param price: The model's USD per million tokens; 0.0 when free, None when undocumented.
    :return: The ceiling in USD, or None when this build is not capped by money.
    """
    if price is None or price == 0.0:
        return None
    limit = budget_usd()
    return limit if limit > 0 else None


def confirmed() -> bool:
    """Return whether the caller has already agreed to pay whatever this build costs."""
    return os.environ.get(CONFIRM_ENV, "").strip().lower() in {"1", "true", "yes"}


def _volume_reason(price: float | None) -> str:
    """Say why a VOLUME ceiling applies, unless the reason is simply that a caller named one."""
    if explicit_budget_tokens() is not None:
        return ""
    if price is None:
        return " for a model with no documented price"
    return ", the volume ceiling no price may lift"


def _bill_refusal(tokens: int, family: str) -> tuple[str, str] | None:
    """Decide a bill refusal once, returning both its wording and the knob that named the ceiling.

    Money is judged FIRST: where a bill can be computed it is the harm, and the USD knob is
    the one worth naming. The volume ceiling below it covers the two cases money cannot -
    a model with no documented price, and a documented price nobody can prove is current.
    """
    if confirmed():
        return None
    price = price_per_million(family)
    limit = applicable_budget_usd(price)
    cost = estimate_cost(tokens, price)
    if limit is not None and cost is not None and cost > limit:
        return (
            f"~{tokens:,} estimated tokens (~{format_cost(tokens, price)}) exceeds the budget of "
            f"{format_usd(limit)} ({BUDGET_USD_ENV})",
            BUDGET_USD_ENV,
        )
    ceiling = budget_tokens(price)
    if ceiling is not None and 0 < ceiling < tokens:
        return (
            f"~{tokens:,} estimated tokens exceeds the ceiling of {ceiling:,} tokens"
            f"{_volume_reason(price)} ({BUDGET_ENV})",
            BUDGET_ENV,
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
        f"Refusing to embed {count} uncached chunk(s) with {model_id or UNNAMED_EMBEDDER}: "
        f"{reason}. {remedies(None, knob)}",
        knob,
    )


def embedder_family(embedder: object) -> str:
    """Return the price/cache family key of an embedder, without asking a provider anything.

    A wrapped embedder carries the family the cache file and the price table are keyed by; a
    bare one only has its spec string, whose ``@<dims>`` suffix the price table is not keyed by.
    Reading that spec can cost a provider round trip on a model with no documented width, so a
    failure to name the family is reported as an unknown price rather than raised.
    """
    family = getattr(embedder, "family", None)
    if isinstance(family, str):
        return family
    try:
        model_id = str(getattr(embedder, "model_id", ""))
    except Exception:  # pragma: no cover - only a provider probe can fail here
        return ""
    scheme, separator, rest = model_id.partition(":")
    body, at, _dimensions = rest.rpartition("@")
    return f"{scheme}{separator}{body if at else rest}"


def pending_purchase(embedder: object, texts: list[str], may_probe: bool = True) -> list[str]:
    """Return the texts an embedder would really have to buy out of these.

    An embedder that stores what it bought answers for itself, because only it knows which of
    these texts are already paid for and which copies of a repeated text share one provider
    slot. Anything else buys every one of them: FAIL CLOSED, never an assumption that some
    invisible cache will pick up the bill.

    :param embedder: The embedder a build resolved.
    :param texts: Every text the build is about to embed.
    :param may_probe: Whether the caller may cost a provider round trip to learn a vector width.
        A pre-flight report may not, and asks the unprobed question instead of skipping the
        buyer - skipping it billed every copy of a repeated chunk for a model with no declared
        width, 20x what the caching build then bought.
    :return: The subset that would actually reach a provider.
    """
    resolve = getattr(embedder, "pending_documents" if may_probe else "pending_documents_unprobed", None)
    if resolve is None:
        return texts
    pending: list[str] = resolve(texts)
    return pending


def require_affordable_bill(embedder: object, texts: list[str]) -> None:
    """Refuse, or announce, the paid part of one embed before a single text is sent.

    THE money verdict on a real build. It lives at the seam that buys vectors rather than
    inside the caching wrapper, because that wrapper is optional - ``ZEMBLE_EMBED_CACHE=0``
    turns it off, and a library caller may hand in a bare remote embedder - and a ceiling one
    environment variable can delete is not a ceiling. Cache awareness comes from asking the
    embedder what it would buy, which is the only cache-aware number that exists here.

    Over budget, :func:`check_budget` raises ``EmbeddingBudgetExceeded`` here, before the caller
    reaches its provider.

    :param embedder: The embedder a build resolved.
    :param texts: Every text this embed would cover, already-bought ones included.
    """
    if not texts or not is_remote(embedder):
        return
    buying = pending_purchase(embedder, texts)
    if not buying:
        return
    family = embedder_family(embedder)
    tokens = estimate_tokens(buying)
    # The announcement below names the embedder too, so the fallback is resolved here rather
    # than only where the refusal is worded.
    model_id = str(getattr(embedder, "model_id", "") or family or UNNAMED_EMBEDDER)
    check_budget(model_id, family, len(buying), tokens)
    logger.info(
        "embedding %d uncached chunk(s), ~%d tokens, ~%s with %s",
        len(buying),
        tokens,
        format_cost(tokens, price_per_million(family)),
        model_id,
    )


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
