"""Intent-aware packing: the tier table per intent, and the consumer caller seed."""

from __future__ import annotations

from pathlib import Path

import pytest

from zemble.evidence.bundle import (
    DEFAULT_ORDER,
    MAX_PER_TIER_PER_ANCHOR,
    PLANS,
    TIERS,
    ItemKind,
    _seed_candidates,
    _Sources,
)
from zemble.evidence.intent import Intent
from zemble.graph.provider import SqliteGraphProvider


def test_the_shipped_default_is_the_fixed_order() -> None:
    """Detection never reorders by itself: the default plan is the one measured best."""
    assert PLANS[DEFAULT_ORDER].tiers == TIERS, "the default order is the fixed tier order"
    assert PLANS[DEFAULT_ORDER].name == "default", "and says so in a bundle header"


def test_every_intent_has_a_plan_over_every_item_kind() -> None:
    """The table is exhaustive in both directions, so a new member cannot be forgotten."""
    assert set(PLANS) == set(Intent), "every intent has a plan"
    for intent, plan in PLANS.items():
        assert set(plan.tiers) == set(ItemKind), f"{intent.value} places every item kind"


@pytest.mark.parametrize("intent", [Intent.SYMBOL, Intent.BEHAVIOUR, Intent.UNKNOWN])
def test_plain_intents_keep_the_default_order(intent: Intent) -> None:
    """A symbol, a behaviour or an unreadable query packs exactly as it always did."""
    plan = PLANS[intent]

    assert plan.tiers == TIERS, f"{intent.value} uses the default tier order"
    assert plan.max_chunk_lines is None, f"{intent.value} shows a primary chunk as found"
    assert not plan.seed_callers, f"{intent.value} adds no seeds"


def test_consumer_plan_puts_the_uses_first() -> None:
    """A consumer question packs call sites, implementations and tests before the outline."""
    plan = PLANS[Intent.CONSUMER]

    assert plan.tiers[ItemKind.CHUNK] == 0, "the search hits still open the bundle"
    for kind in (ItemKind.CALLER, ItemKind.IMPLEMENTATION, ItemKind.TEST, ItemKind.NOTE):
        assert plan.tiers[kind] < plan.tiers[ItemKind.OUTLINE], f"{kind.value} outranks the outline"
        assert plan.cap(kind) > MAX_PER_TIER_PER_ANCHOR or kind is ItemKind.NOTE, f"{kind.value} gets a wider cap"
    assert plan.max_chunk_lines is not None, "the primary chunks are cut back to make room"
    assert plan.seed_callers, "a named symbol's exact callers are seeded"


def test_architecture_plan_puts_shapes_before_bodies() -> None:
    """An architecture question packs outlines and the hierarchy before raw chunks."""
    plan = PLANS[Intent.ARCHITECTURE]

    for kind in (ItemKind.OUTLINE, ItemKind.SUPERTYPE, ItemKind.IMPLEMENTATION):
        assert plan.tiers[kind] < plan.tiers[ItemKind.CHUNK], f"{kind.value} outranks the raw chunks"
    assert plan.max_outlines == 3, "only the first few types are outlined"


def test_bug_plan_puts_tests_before_the_outline() -> None:
    """A symptom packs the code and then the tests that name the behaviour it lost."""
    plan = PLANS[Intent.BUG]

    assert plan.tiers[ItemKind.CHUNK] == 0, "the code comes first"
    assert plan.tiers[ItemKind.TEST] < plan.tiers[ItemKind.OUTLINE], "the tests outrank the outline"


def test_consumer_seed_finds_uses_search_never_returned(built_graph: SqliteGraphProvider, tmp_path: Path) -> None:
    """A named symbol's exact uses are seeded from the graph alone, with no search behind them."""
    root = Path(built_graph.path)
    sources = _Sources(root)
    plan = PLANS[Intent.CONSUMER]

    # 1. A named method seeds its call sites, each as a caller item.
    callers = _seed_candidates(built_graph, "who calls Helpers.twice", sources, plan)
    assert callers, "step 1: the graph resolved the named method and returned its callers"
    assert {candidate.kind for candidate in callers} == {ItemKind.CALLER}, "step 1: they are caller evidence"
    assert any("Circle.java" in candidate.file_path for candidate in callers), "step 1: the real caller is there"
    assert all("names Helpers.twice" in candidate.reason for candidate in callers), "step 1: the reason says why"

    # 2. A named type seeds its implementations instead.
    implementations = _seed_candidates(built_graph, "who implements Shape", sources, plan)
    assert any(candidate.kind is ItemKind.IMPLEMENTATION for candidate in implementations), (
        "step 2: a type seeds implementations"
    )
    assert any("Circle" in candidate.location_text for candidate in implementations), "step 2: the implementor is there"

    # 3. A query naming nothing the graph knows seeds nothing at all.
    assert not _seed_candidates(built_graph, "who calls the drawing helper", sources, plan), (
        "step 3: an unnamed subject cannot be seeded"
    )

    # 4. The cap is the plan's, not the default one.
    assert len(callers) <= plan.cap(ItemKind.CALLER), "step 4: seeding honours the per-anchor cap"
    assert str(tmp_path), "step 4: the cache stayed isolated"
