"""What a query is asking for, decided from its text alone.

The intent picks the tier order an evidence bundle packs with: a question about
who calls something wants call sites, a question about how parts fit together
wants outlines, and a bug report wants the code plus the tests that name the
behaviour it lost. The rules are deliberately cheap, ordered and inspectable -
every answer says which rule fired, so a wrong order can be argued with.

AIDEV-NOTE: the eval set carries a `kind` label per query; nothing here may read
it. It is ground truth for measuring the classifier, never an input to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    """What kind of answer a query is asking for."""

    SYMBOL = "symbol"
    BEHAVIOUR = "behaviour"
    ARCHITECTURE = "architecture"
    BUG = "bug"
    CONSUMER = "consumer"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentMatch:
    """The decided intent and the name of the rule that decided it."""

    intent: Intent
    rule: str

    def describe(self) -> str:
        """Render the decision the way a bundle header states it."""
        return f"{self.intent.value} (rule: {self.rule})"


# A token that could be a code name: `PageWindow`, `SessionIds.forToken`, `zc-inbox`.
_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$-]*(\.[A-Za-z0-9_$-]+)*(\(\))?$")
# ... but only when it is spelled like one, so a plain English word is not a symbol.
_IDENTIFIER_SHAPE = re.compile(r"(?<=[a-z0-9])[A-Z]|[._$-]|^[A-Z]{2,}")

_CONSUMER_WHO_USES = re.compile(
    r"\bwho\s+(uses|calls|consumes|implements|extends|depends|invokes)\b"
    r"|\b(what|which)\b[^.?]{0,40}?\b(uses|calls|consumes|implements|invokes|depends on)\b",
    re.IGNORECASE,
)
_CONSUMER_CALLERS_OF = re.compile(
    r"\b(callers?|call sites?|consumers?|users?|usages?|implementors?|implementers?|implementations?|subclasses)\s+of\b"
    r"|\bused by\b",
    re.IGNORECASE,
)
_CONSUMER_WHERE_USED = re.compile(
    r"\bwhere\s+(is|are)\b[^.?]{0,40}?\b(used|called|consumed|referenced|implemented)\b",
    re.IGNORECASE,
)
# Code described by what it uses ("templates that hand a picker a data provider").
_CONSUMER_THAT_USES = re.compile(
    r"\b(that|which)\s+(use|uses|using|call|calls|consume|consumes|invoke|invokes|implement|implements|pass|passes"
    r"|hand|hands)\b",
    re.IGNORECASE,
)
# A query whose subject is a test is asking for a consumer too: a test uses what it covers.
_CONSUMER_TESTS = re.compile(
    r"^\s*(unit\s+|integration\s+|browser\s+)?tests?\b"
    r"|\btests?\s+(for|of|covering|proving|asserting|pinning|rejecting|verifying|exercising|that)\b",
    re.IGNORECASE,
)

_ARCHITECTURE_HOW = re.compile(
    r"^\s*how\s+(does|do|is|are|can|could|did|would|should|a|an|the)\b",
    re.IGNORECASE,
)
_ARCHITECTURE_WIRING = re.compile(
    r"\b(wired|wiring|work together|fit together|hang together|plug into|flows? through|end to end)\b",
    re.IGNORECASE,
)

_BUG_SYMPTOM = re.compile(
    r"\binstead of\b"
    r"|\bshould\s+(be|have|show|return|stay|fire|get)\b"
    r"|\b(fails?|failing|throws?|crashes?|breaks?|hangs?|locks?\s+\w+\s+out)\b"
    r"|\b(vanish|vanishes|disappears?|leaks?)\b"
    r"|\bkeeps?\s+\w+ing\b"
    r"|\b(does not|doesn't|do not|don't|won't)\s+\w+\b"
    r"|\b(shows?|showing|returns?|gives?|reports?|renders?|logs?|creates?|produces?)\b[^.?]{0,30}?"
    r"\b(wrong|empty|nothing|duplicate|stale|blank|null|the same)\b",
    re.IGNORECASE,
)


def _looks_like_identifier(query: str) -> bool:
    """Return True when the whole query is one code name rather than a sentence."""
    token = query.strip()
    if not token or " " in token:
        return False
    return bool(_IDENTIFIER.match(token)) and bool(_IDENTIFIER_SHAPE.search(token))


#: Ordered rules; the first that fires decides. Name, intent, and the test itself.
_RULES: tuple[tuple[str, Intent, object], ...] = (
    ("identifier", Intent.SYMBOL, _looks_like_identifier),
    ("who-uses", Intent.CONSUMER, _CONSUMER_WHO_USES),
    ("callers-of", Intent.CONSUMER, _CONSUMER_CALLERS_OF),
    ("where-used", Intent.CONSUMER, _CONSUMER_WHERE_USED),
    ("that-uses", Intent.CONSUMER, _CONSUMER_THAT_USES),
    ("tests-for", Intent.CONSUMER, _CONSUMER_TESTS),
    ("how-does", Intent.ARCHITECTURE, _ARCHITECTURE_HOW),
    ("wiring", Intent.ARCHITECTURE, _ARCHITECTURE_WIRING),
    ("symptom", Intent.BUG, _BUG_SYMPTOM),
)

#: What a query that fired no rule is treated as.
DEFAULT_RULE = "default"


def classify(query: str) -> IntentMatch:
    """Decide what a query is asking for, and say which rule decided it.

    :param query: The query text, exactly as the caller wrote it.
    :return: The intent and the name of the rule that fired.
    """
    if not query.strip():
        return IntentMatch(Intent.UNKNOWN, "empty")
    for name, intent, test in _RULES:
        matched = test(query) if callable(test) else bool(test.search(query))  # type: ignore[union-attr]
        if matched:
            return IntentMatch(intent, name)
    return IntentMatch(Intent.BEHAVIOUR, DEFAULT_RULE)


def parse_intent(value: str) -> Intent:
    """Resolve a written intent name, refusing anything outside the vocabulary."""
    try:
        return Intent(value.strip().lower())
    except ValueError:
        known = ", ".join(item.value for item in Intent)
        raise ValueError(f"unknown intent {value!r}; expected one of {known}") from None


INTENT_NAMES = tuple(item.value for item in Intent)

__all__ = ["DEFAULT_RULE", "INTENT_NAMES", "Intent", "IntentMatch", "classify", "parse_intent"]
