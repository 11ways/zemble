"""The acceptance tests for two harms, two guards, two units.

Runaway WORK is refused in bytes before a parse; a runaway BILL is refused in money once
the uncached set is known. The bug these pin is a work measurement priced as a bill: the
pre-flight said a build was affordable while the pre-parse guard refused the same tree.
"""

from __future__ import annotations

import math
from pathlib import Path

from tests.embedding.test_pricing import PricedEmbedder
from zemble.cache import save_index_to_cache
from zemble.embedding.cache import CachingEmbedder
from zemble.embedding.pricing import DEFAULT_UNPRICED_BUDGET_TOKENS, ESTIMATE_CHARS_PER_TOKEN
from zemble.index import ZembleIndex
from zemble.index.scope import estimate_tree
from zemble.types import ContentType

FAMILY = "voyage:voyage-4-lite"

#: The byte volume the pre-parse guard used to refuse: it converted bytes to tokens at the
#: pricing density and compared them against the old 2M-token ceiling, so any tree fatter
#: than this was refused however little of it still had to be bought.
OLD_CEILING_BYTES = math.ceil(DEFAULT_UNPRICED_BUDGET_TOKENS * ESTIMATE_CHARS_PER_TOKEN)


def _big_tree(root: Path, files: int = 10, per_file: int = 800_000) -> Path:
    """Write a source tree comfortably past the byte volume the old token ceiling refused."""
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
    root = _big_tree(tmp_path / "workspace")
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
