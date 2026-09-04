"""The acceptance tests for two harms, two guards, two units.

Runaway WORK is refused in bytes before a parse; a runaway BILL is refused in money once
the uncached set is known. The bug these pin is a work measurement priced as a bill: the
pre-flight said a build was affordable while the pre-parse guard refused the same tree.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tests.embedding.test_pricing import PricedEmbedder
from zemble.cache import save_index_to_cache
from zemble.daemon.server import REFUSAL_TYPES
from zemble.embedding.cache import CachingEmbedder
from zemble.embedding.preflight import embed_status
from zemble.embedding.pricing import (
    BUDGET_ENV,
    BUDGET_USD_ENV,
    DEFAULT_UNPRICED_BUDGET_TOKENS,
    ESTIMATE_CHARS_PER_TOKEN,
    EmbeddingBudgetExceeded,
    check_budget,
)
from zemble.embedding.registry import CACHE_ENV, ResolvedEmbedder, build_embedder, caching_enabled
from zemble.index import ScopeRefused, ZembleIndex
from zemble.index.scope import WORK_LIMIT_ENV, estimate_tree
from zemble.refusal import Refused
from zemble.types import ContentType

FAMILY = "voyage:voyage-4-lite"

#: The byte volume the pre-parse guard used to refuse: it converted bytes to tokens at the
#: pricing density and compared them against the old 2M-token ceiling, so any tree fatter
#: than this was refused however little of it still had to be bought.
OLD_CEILING_BYTES = math.ceil(DEFAULT_UNPRICED_BUDGET_TOKENS * ESTIMATE_CHARS_PER_TOKEN)


def _tree(root: Path, files: int, per_file: int) -> Path:
    """Write a source tree of a given byte size, with no two functions alike."""
    (root / "src").mkdir(parents=True)
    for index in range(files):
        lines: list[str] = []
        written = 0
        line = 0
        while written < per_file:
            body = f"def f_{index}_{line}():\n    return {line}\n\n"
            lines.append(body)
            written += len(body)
            line += 1
        (root / "src" / f"module_{index}.py").write_text("".join(lines), encoding="utf-8")
    return root


def _touch_every_file(root: Path) -> None:
    """Rewrite every source file with its own content, so only its modification time moves."""
    for path in sorted(root.rglob("*.py")):
        path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def test_a_tree_whose_chunks_are_already_bought_is_not_refused(tmp_path: Path) -> None:
    """A fat tree nobody has to pay for twice rebuilds: bytes are work, and work is not a bill."""
    root = _tree(tmp_path / "workspace", files=10, per_file=800_000)
    inner = PricedEmbedder(dimensions=8)
    embedder = CachingEmbedder(inner, FAMILY, tmp_path / "vectors")

    # 1. The tree is fatter than the byte volume the old ceiling refused: this is the regression.
    assert estimate_tree(root.resolve(), (ContentType.CODE,)).bytes > OLD_CEILING_BYTES, (
        "step 1: the fixture must be a tree the old ceiling would have refused"
    )

    # 2. The first build is allowed and pays for every chunk once.
    first = ZembleIndex.from_path(root, embedder=embedder)
    save_index_to_cache(first, str(root.resolve()))
    bought = len(inner.document_batches)
    assert bought > 0, "step 2: a cold build really does buy the tree"

    # 3. Every file moves, so the manifest covers none of them and the walk prices all of it.
    _touch_every_file(root.resolve())
    estimate = estimate_tree(root.resolve(), (ContentType.CODE,), (), first._manifest)
    assert estimate.bytes > OLD_CEILING_BYTES, "step 3: the walk still sees the whole tree as work to redo"

    # 4. Neither guard refuses: not the pre-parse walk, and not the budget once the set is known.
    rebuilt = ZembleIndex.from_path(root, embedder=embedder)
    assert rebuilt.stats.indexed_files == first.stats.indexed_files, "step 4: the same tree was indexed again"

    # 5. And the provider was never contacted again, because there was nothing left to buy.
    assert len(inner.document_batches) == bought, (
        f"step 5: a fully cached rebuild sends nothing, got {len(inner.document_batches)} batches"
    )


def test_the_two_lanes_agree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever `embed-status` reports for a tree, a real build of that tree does: one answer, two surfaces.

    This is the invariant whose absence produced the bug: the report said a build was allowed
    while the guard refused the very same tree, so nobody could tell which one was lying. Both
    lanes are walked twice - with the vector cache on, and with it turned off, which is a
    documented knob and must move neither verdict.
    """
    root = _tree(tmp_path / "workspace", files=12, per_file=96_000)
    inner = PricedEmbedder(dimensions=8)
    monkeypatch.setattr(
        "zemble.embedding.preflight.build_embedder",
        lambda spec: ResolvedEmbedder(spec=spec, embedder=inner, scheme="voyage", family=FAMILY),
    )

    def _verdicts(cached: bool = True) -> tuple[bool, bool]:
        """Ask the report and a real build the same question, in that order."""
        reported = embed_status(root).would_refuse
        embedder = CachingEmbedder(inner, FAMILY) if cached else inner
        try:
            ZembleIndex.from_path(root, embedder=embedder)
        except (ScopeRefused, EmbeddingBudgetExceeded):
            return reported, True
        return reported, False

    # 1. Refused as work: the tree is over the byte ceiling, and both lanes say so.
    monkeypatch.setenv(WORK_LIMIT_ENV, "1")
    reported, built = _verdicts()
    assert (reported, built) == (True, True), f"step 1: report {reported}, build {built}"

    # 2. Refused as money: affordable work, unaffordable bill, and both lanes say so.
    monkeypatch.delenv(WORK_LIMIT_ENV)
    monkeypatch.setenv(BUDGET_USD_ENV, "0.001")
    reported, built = _verdicts()
    assert (reported, built) == (True, True), f"step 2: report {reported}, build {built}"

    # 3. Allowed: the report says a build would go through, and the build goes through.
    monkeypatch.delenv(BUDGET_USD_ENV)
    reported, built = _verdicts()
    assert (reported, built) == (False, False), f"step 3: report {reported}, build {built}"
    assert inner.document_batches, "step 3: and the allowed build is the only one that bought anything"

    # 4. With the vector cache turned off, the very same three verdicts hold. The bill guard
    #    used to live INSIDE the cache wrapper, so this one variable deleted every money
    #    ceiling while the report went on saying REFUSED.
    monkeypatch.setenv(CACHE_ENV, "0")
    assert not caching_enabled(), "step 4: the documented knob really is off"
    monkeypatch.setenv(WORK_LIMIT_ENV, "1")
    assert _verdicts(cached=False) == (True, True), "step 4: work is refused with the cache off too"
    monkeypatch.delenv(WORK_LIMIT_ENV)
    monkeypatch.setenv(BUDGET_USD_ENV, "0.001")
    assert _verdicts(cached=False) == (True, True), "step 4: and so is a bill"
    monkeypatch.delenv(BUDGET_USD_ENV)
    assert _verdicts(cached=False) == (False, False), "step 4: and an affordable build still runs"


def test_the_bill_guard_is_not_deleted_by_turning_the_cache_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ZEMBLE_EMBED_CACHE=0` is a documented knob, and it must not be a way past the budget.

    The money check lived in `CachingEmbedder`, whose wrapper `build_embedder` only installs
    while caching is on - so one documented variable removed every spending ceiling, and a
    library caller handing in a bare remote embedder had none either. Measured before the fix:
    cache on refused and sent 0 texts, cache off sent 2,032.
    """
    root = _tree(tmp_path / "workspace", files=4, per_file=48_000)
    monkeypatch.setenv(BUDGET_USD_ENV, "0.0000001")

    # 1. With the knob off, `build_embedder` really does hand back an unwrapped embedder.
    monkeypatch.setenv(CACHE_ENV, "0")
    monkeypatch.setattr(
        "zemble.embedding.registry._build", lambda spec: (PricedEmbedder(dimensions=8), "voyage", FAMILY)
    )
    resolved = build_embedder("voyage:voyage-4-lite")
    assert not isinstance(resolved.embedder, CachingEmbedder), "step 1: the wrapper is what the knob removes"

    # 2. That bare embedder is refused all the same, and buys nothing.
    bare = resolved.embedder
    with pytest.raises(EmbeddingBudgetExceeded) as raised:
        ZembleIndex.from_path(root, embedder=bare)
    assert bare.document_batches == [], "step 2: a refused build sends nothing, cache or no cache"
    assert BUDGET_USD_ENV in str(raised.value), "step 2: refused in money, naming the money knob"

    # 3. And the report agrees, which is the invariant that was inverted: it said REFUSED while
    #    the build paid.
    monkeypatch.setattr(
        "zemble.embedding.preflight.build_embedder",
        lambda spec: ResolvedEmbedder(spec=spec, embedder=bare, scheme="voyage", family=FAMILY),
    )
    assert embed_status(root).would_refuse, "step 3: one answer, two surfaces, with the cache off too"


def test_a_refusal_carries_the_knob_that_refused_it_down_every_lane(tmp_path: Path) -> None:
    """A refusal names the ceiling a caller has to raise, and it stays named as it travels.

    `knob` is declared once, on the shared `Refused` base, because the daemon reads it off a
    caught refusal it never type-checks: a refusal type without one is an AttributeError inside
    an except block. Re-raising to add a hint used to reset it to the money knob, so a VOLUME
    refusal told the reader to raise a budget that was never the one that refused.
    """
    # 1. Both shipped refusal types are the same kind of answer, and both carry a knob.
    for refusal in (ScopeRefused("refused"), EmbeddingBudgetExceeded("refused")):
        assert isinstance(refusal, Refused), f"step 1: {type(refusal).__name__} must be a Refused"
        assert refusal.knob, f"step 1: {type(refusal).__name__} carries no knob"
    assert REFUSAL_TYPES == (Refused,), "step 1: the daemon catches the base, so a new type needs no edit"

    # 2. The volume lane names the volume knob, not the money one.
    with pytest.raises(EmbeddingBudgetExceeded) as raised:
        check_budget("voyage:voyage-9-imaginary@8", "voyage:voyage-9-imaginary", 1, DEFAULT_UNPRICED_BUDGET_TOKENS + 1)
    assert raised.value.knob == BUDGET_ENV, f"step 2: refused by volume, got {raised.value.knob}"

    # 3. And the ancestor hint the index cache adds keeps it, text and knob alike.
    inner = raised.value
    outer = EmbeddingBudgetExceeded(f"{inner} plus a hint", inner.knob)
    assert outer.knob == BUDGET_ENV, "step 3: a re-raise for a hint does not re-decide the ceiling"
    assert str(inner) in str(outer), "step 3: and the original reason is still readable"


#: Every place outside `zemble/embedding/` that hands document texts to a provider. Each one is
#: a paid seam and each one calls `require_affordable_bill` first; the guard lives at the seams
#: rather than inside the optional cache wrapper, so this list is what keeps them in step.
PAID_DOCUMENT_SEAMS = {"index/dense.py", "dedup/detect.py"}


def test_every_paid_document_seam_passes_the_bill_guard() -> None:
    """A new place that buys document vectors must meet the bill guard, not discover it later.

    The money check no longer sits behind `CachingEmbedder`, which one environment variable can
    remove; it sits at the seams that buy. That only holds while every seam is one of these.
    """
    source_root = Path(__file__).resolve().parents[2] / "src" / "zemble"
    found: dict[str, str] = {}
    for module in sorted(source_root.rglob("*.py")):
        relative = module.relative_to(source_root).as_posix()
        if relative.startswith("embedding/"):
            continue  # The providers and the cache are the implementations, not the buyers.
        text = module.read_text(encoding="utf-8")
        if ".embed_documents(" in text:
            found[relative] = text

    # 1. Nobody bought vectors from a seam this suite has never heard of.
    assert set(found) == PAID_DOCUMENT_SEAMS, (
        f"step 1: the seams that buy document vectors changed, got {sorted(found)}; "
        "wire require_affordable_bill into the new one and name it here"
    )

    # 2. And every one of them asks the guard before it buys.
    for relative, text in found.items():
        assert "require_affordable_bill" in text, f"step 2: {relative} buys vectors without passing the bill guard"


def test_the_report_bills_what_the_build_announces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A repeated chunk is bought once, so the report may not bill it twice.

    The caching embedder gives duplicate texts inside one call a single provider slot. A report
    that counted every copy refused trees of near-identical vendored source over a bill nobody
    was ever going to be sent.
    """
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "repeated.py").write_text("x = 1\n" * 16_000, encoding="utf-8")
    inner = PricedEmbedder(dimensions=8)
    monkeypatch.setattr(
        "zemble.embedding.preflight.build_embedder",
        lambda spec: ResolvedEmbedder(spec=spec, embedder=inner, scheme="voyage", family=FAMILY),
    )

    # 1. The file chunks into many pieces, and they are not all different pieces.
    status = embed_status(root)
    assert status.uncached > 1, f"step 1: the fixture must chunk into several, got {status.uncached}"

    # 2. The build announces exactly the tokens the report estimated: one answer, two surfaces.
    with caplog.at_level("INFO", logger="zemble.embedding.pricing"):
        ZembleIndex.from_path(root, embedder=CachingEmbedder(inner, FAMILY))
    announced = [record.getMessage() for record in caplog.records if "uncached chunk(s)" in record.getMessage()]
    assert len(announced) == 1, f"step 2: exactly one announcement, got {caplog.records}"
    bought = int(announced[0].split()[1])
    assert bought < status.uncached, f"step 2: the fixture must repeat chunks, {bought} of {status.uncached} distinct"
    assert f"~{status.estimated_tokens} tokens" in announced[0], (
        f"step 2: the report estimated {status.estimated_tokens}, the build announced {announced[0]}"
    )
