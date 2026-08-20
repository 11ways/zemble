"""Packing rules: budget, tier order, degrade before drop, truncation, dedupe, omissions."""

from __future__ import annotations

from zemble.evidence.bundle import (
    ItemKind,
    Presentation,
    _Candidate,
    pack,
    truncate_lines,
)


def _candidate(
    kind: ItemKind,
    *,
    lines: int = 4,
    score: float = 1.0,
    path: str = "src/Main.java",
    start: int = 1,
    body: str = "int value = 1;",
) -> _Candidate:
    """Build a candidate with a body of the requested line count."""
    text = "\n".join(f"{body} // {index}" for index in range(lines))
    return _Candidate(
        kind=kind,
        file_path=path,
        start_line=start,
        end_line=start + lines,
        reason=f"{kind.value} evidence",
        text=text,
        location_text=f"method {kind.value}Method()",
        score=score,
        key=(path, start, start + lines),
    )


def test_truncate_lines_marks_what_it_dropped() -> None:
    """A truncated text says how many lines are missing."""
    text = "\n".join(str(index) for index in range(10))

    assert truncate_lines(text, 20) == text, "nothing is dropped when the keep count exceeds the text"
    truncated = truncate_lines(text, 4)
    assert truncated.splitlines()[-1] == "... (truncated, 6 more lines)", "the marker names the dropped line count"
    assert truncated.startswith("0\n1\n2\n3"), "the kept lines are the first ones"


def test_packing_journey() -> None:
    """Walk one candidate set through a generous, a tight and an impossible budget."""
    candidates = [
        _candidate(ItemKind.CALLER, lines=6, score=0.5, path="src/Caller.java", start=40),
        _candidate(ItemKind.CHUNK, lines=8, score=0.9, path="src/Main.java", start=1),
        _candidate(ItemKind.OUTLINE, lines=5, score=0.9, path="src/Main.java", start=90),
        _candidate(ItemKind.TEST, lines=5, score=0.4, path="src/MainTest.java", start=10),
    ]

    # 1. A generous budget keeps everything, as content, in tier order.
    bundle = pack("q", candidates, 4000)
    assert [item.tier for item in bundle.items] == [0, 1, 2, 3], "step 1: items are ordered by tier"
    assert all(item.presentation is Presentation.CONTENT for item in bundle.items), "step 1: everything fits as content"
    assert not bundle.omitted, "step 1: nothing is omitted"
    assert bundle.total_tokens == sum(item.tokens for item in bundle.items), "step 1: the total is the item sum"

    # 2. Every budget is respected exactly, and nothing that fits is silently dropped.
    for budget in range(10, 400, 7):
        tight = pack("q", candidates, budget)
        assert tight.rendered_tokens <= budget, f"step 2: the whole rendered answer fits budget {budget}"
        assert len(tight.items) + len(tight.omitted) + tight.unlisted_omissions == len(candidates), (
            f"step 2: at budget {budget} every candidate is packed, listed as omitted, or counted"
        )

    # 3. A budget that cannot hold the content still degrades rather than dropping.
    full = pack("q", candidates, 4000).items[3]
    degraded = pack("q", candidates, 150)
    caller = next(item for item in degraded.items if item.kind is ItemKind.CALLER)
    assert caller.presentation is Presentation.LOCATION, "step 3: a caller becomes a location line"
    assert caller.text == "method callerMethod()", "step 3: the location line carries the signature"
    assert caller.tokens < full.tokens, "step 3: degrading is cheaper than the content it replaced"

    # 4. A primary chunk is truncated instead of degraded, and says so.
    chunk_budget = pack("q", candidates, 130)
    chunk = next(item for item in chunk_budget.items if item.kind is ItemKind.CHUNK)
    assert chunk.presentation is Presentation.TRUNCATED, "step 4: the chunk is truncated"
    assert "... (truncated," in chunk.text, "step 4: the truncation is marked in the text"

    # 5. A zero budget packs nothing and reports everything as omitted.
    empty = pack("q", candidates, 0)
    assert not empty.items, "step 5: nothing fits in no budget"
    assert len(empty.omitted) + empty.unlisted_omissions == len(candidates), (
        "step 5: every candidate is at least counted"
    )
    assert empty.rendered_tokens == 0, "step 5: and the answer itself costs nothing"

    # 6. The footer is trimmed before any evidence is, and the whole answer stays inside the budget.
    footer = pack("q", candidates, 200)
    assert footer.items, "step 6: evidence survives a budget the footer cannot fit in"
    assert footer.rendered_tokens <= 200, "step 6: headings and footer are inside the budget too"
    if footer.unlisted_omissions:
        assert "not listed for budget" in footer.render(), "step 6: a trimmed footer says it was trimmed"


def test_duplicate_regions_are_packed_once() -> None:
    """Two candidates over the same lines are one item, whichever reason found them."""
    first = _candidate(ItemKind.CHUNK, path="src/Main.java", start=1)
    second = _candidate(ItemKind.CALLER, path="src/Main.java", start=1)
    assert first.key == second.key, "the dedupe key is the source region, not the reason"

    bundle = pack("q", [first, second], 4000)
    assert len(bundle.items) == 1, "the region is shown once"
    assert bundle.items[0].kind is ItemKind.CHUNK, "the earlier tier wins the region"
    assert not bundle.omitted, "a deduplicated candidate is not an omission"


def test_reserve_keeps_room_for_later_tiers() -> None:
    """A long primary chunk cannot spend the whole budget on itself."""
    candidates = [
        _candidate(ItemKind.CHUNK, lines=200, score=0.9),
        _candidate(ItemKind.OUTLINE, lines=4, score=0.9, path="src/Main.java", start=500),
        _candidate(ItemKind.TEST, lines=4, score=0.4, path="src/MainTest.java", start=10),
        _candidate(ItemKind.CALLER, lines=4, score=0.4, path="src/Caller.java", start=10),
    ]
    budget = 600
    bundle = pack("q", candidates, budget)

    chunk = next(item for item in bundle.items if item.kind is ItemKind.CHUNK)
    later = [item for item in bundle.items if item.tier > 0]
    assert chunk.presentation is Presentation.TRUNCATED, "the chunk is trimmed rather than allowed to fill the budget"
    assert chunk.tokens + sum(item.tokens for item in later) <= budget, "the reserve was spent, not merely held back"
    assert {item.kind for item in bundle.items} == {
        ItemKind.CHUNK,
        ItemKind.OUTLINE,
        ItemKind.TEST,
        ItemKind.CALLER,
    }, "every later tier still reaches the bundle"
    assert bundle.total_tokens <= budget, "the reserve does not break the budget"
