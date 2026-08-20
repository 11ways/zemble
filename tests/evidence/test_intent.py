"""Query intent: which rule fires, what it decides, and what it refuses."""

from __future__ import annotations

import pytest

from zemble.evidence.intent import Intent, IntentMatch, classify, parse_intent

# Query, expected intent, expected rule. One row per rule in the table.
_CASES = [
    ("PageWindow", Intent.SYMBOL, "identifier"),
    ("SessionIds.forToken", Intent.SYMBOL, "identifier"),
    ("zc-inbox", Intent.SYMBOL, "identifier"),
    ("who calls the page window helper", Intent.CONSUMER, "who-uses"),
    ("callers of PageWindow.of", Intent.CONSUMER, "callers-of"),
    ("where is the entity tag helper used", Intent.CONSUMER, "where-used"),
    ("templates that hand a picker a data provider", Intent.CONSUMER, "that-uses"),
    ("tests covering the outbound retry schedule", Intent.CONSUMER, "tests-for"),
    ("how does a stored chain re-check authority at every step", Intent.ARCHITECTURE, "how-does"),
    ("the pieces the settings editor is wired through", Intent.ARCHITECTURE, "wiring"),
    ("asking for a page beyond the last one gives an empty list instead of the final page", Intent.BUG, "symptom"),
    ("clamp a requested page number against the row total", Intent.BEHAVIOUR, "default"),
]


@pytest.mark.parametrize(("query", "intent", "rule"), _CASES)
def test_rule_table(query: str, intent: Intent, rule: str) -> None:
    """Every rule in the table fires on the phrasing it was written for."""
    match = classify(query)

    assert match.intent is intent, f"{query!r} is a {intent.value} question"
    assert match.rule == rule, f"{query!r} is decided by the {rule} rule"


def test_classification_journey() -> None:
    """Walk one query through the shapes the classifier has to separate."""
    # 1. A lone code name is a symbol lookup, whatever it is spelled like.
    assert classify("MediaIngestion").intent is Intent.SYMBOL, "step 1: a CamelCase token is a symbol"
    assert classify("ingestion").intent is Intent.BEHAVIOUR, "step 1: an English word is not"

    # 2. The same subject asked about differently lands on different intents.
    assert classify("deduplicate an upload by its digest").intent is Intent.BEHAVIOUR, "step 2: a behaviour"
    assert classify("who uses the ingestion helper").intent is Intent.CONSUMER, "step 2: a consumer question"
    assert classify("how does an upload reach storage").intent is Intent.ARCHITECTURE, "step 2: an architecture one"

    # 3. A mixed query resolves by rule ORDER, and says which rule won.
    mixed = classify("how does MediaIngestion decide who calls it")
    assert mixed.intent is Intent.CONSUMER, "step 3: the consumer rule is tested before the how rule"
    assert mixed.rule == "who-uses", "step 3: the answer names the rule that decided"

    # 4. Nothing to classify is UNKNOWN rather than a guess.
    assert classify("   ") == IntentMatch(Intent.UNKNOWN, "empty"), "step 4: an empty query is unknown"

    # 5. The decision renders the way a bundle header states it.
    assert classify("callers of PageWindow").describe() == "consumer (rule: callers-of)", "step 5: header form"


def test_parse_intent_refuses_an_unknown_name() -> None:
    """A written intent outside the vocabulary is refused, not defaulted."""
    assert parse_intent("Consumer") is Intent.CONSUMER, "a written name is case-insensitive"
    with pytest.raises(ValueError, match="unknown intent"):
        parse_intent("callers")
