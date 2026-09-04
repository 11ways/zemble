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
from zemble.embedding.cache import CachingEmbedder
from zemble.embedding.preflight import embed_status
from zemble.embedding.pricing import (
    BUDGET_USD_ENV,
    DEFAULT_UNPRICED_BUDGET_TOKENS,
    ESTIMATE_CHARS_PER_TOKEN,
    EmbeddingBudgetExceeded,
)
from zemble.embedding.registry import ResolvedEmbedder
from zemble.index import ScopeRefused, ZembleIndex
from zemble.index.scope import WORK_LIMIT_ENV, estimate_tree
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
    while the guard refused the very same tree, so nobody could tell which one was lying.
    """
    root = _tree(tmp_path / "workspace", files=12, per_file=96_000)
    inner = PricedEmbedder(dimensions=8)
    monkeypatch.setattr(
        "zemble.embedding.preflight.build_embedder",
        lambda spec: ResolvedEmbedder(spec=spec, embedder=inner, scheme="voyage", family=FAMILY),
    )

    def _verdicts() -> tuple[bool, bool]:
        """Ask the report and a real build the same question, in that order."""
        reported = embed_status(root).would_refuse
        try:
            ZembleIndex.from_path(root, embedder=CachingEmbedder(inner, FAMILY))
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
    with caplog.at_level("INFO", logger="zemble.embedding.cache"):
        ZembleIndex.from_path(root, embedder=CachingEmbedder(inner, FAMILY))
    announced = [record.getMessage() for record in caplog.records if "uncached chunk(s)" in record.getMessage()]
    assert len(announced) == 1, f"step 2: exactly one announcement, got {caplog.records}"
    bought = int(announced[0].split()[1])
    assert bought < status.uncached, f"step 2: the fixture must repeat chunks, {bought} of {status.uncached} distinct"
    assert f"~{status.estimated_tokens} tokens" in announced[0], (
        f"step 2: the report estimated {status.estimated_tokens}, the build announced {announced[0]}"
    )
