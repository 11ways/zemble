"""The pre-parse size guard: a build too big for the budget is refused before anything is read."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from zemble.cache import save_index_to_cache
from zemble.chunking.capsule import CapsuleOptions, embedding_text
from zemble.embedding.pricing import BUDGET_ENV, CONFIRM_ENV, ESTIMATE_CHARS_PER_TOKEN, estimate_tokens
from zemble.index import OversizedRootRefused, ScopeRefused, ZembleIndex
from zemble.index.create import plan_files
from zemble.index.scope import BREAKDOWN_LIMIT, estimate_tree, require_affordable_scope
from zemble.types import ContentType


class LocalEmbedder:
    """A never-billed embedder, so the guard cannot be mistaken for a bill guard."""

    is_remote = False

    def __init__(self, inner) -> None:
        """Wrap a deterministic embedder and declare it local."""
        self._inner = inner

    def __getattr__(self, name: str):
        """Delegate everything the index asks for to the wrapped embedder."""
        return getattr(self._inner, name)


def _tree(root: Path, fat_files: int = 40, fat_bytes: int = 4000) -> None:
    """Plant a small `src/` beside a fat directory that dominates the tree."""
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
    fat = root / "vendored"
    fat.mkdir()
    for index in range(fat_files):
        (fat / f"copy_{index}.py").write_text(f"# {index}\n" + ("x = 1\n" * (fat_bytes // 6)), encoding="utf-8")


def _no_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any chunking attempt a test failure: the guard must run before the parse."""

    def _explode(*args: object, **kwargs: object):
        raise AssertionError("plan_files ran: the size guard did not refuse before parsing")

    monkeypatch.setattr("zemble.index.create.plan_files", _explode)


def test_oversized_root_refusal_journey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_embedder) -> None:
    """A tree over budget is refused before a parse, names its fattest child, and every way out works."""
    root = tmp_path / "workspace"
    _tree(root)
    monkeypatch.setenv(BUDGET_ENV, "1000")

    # 1. Refused before anything is chunked, and before the embedder is touched.
    _no_parsing(monkeypatch)
    before = len(mock_embedder.document_calls)
    with pytest.raises(OversizedRootRefused) as raised:
        ZembleIndex.from_path(root, embedder=mock_embedder)
    message = str(raised.value)
    assert len(mock_embedder.document_calls) == before, "step 1: nothing was embedded"

    # 2. The message carries the estimate, the fattest child first, and the three remedies in order.
    for fragment in (str(root), "estimated tokens", "1,000 tokens", "MB"):
        assert fragment in message, f"step 2: the refusal names {fragment}"
    breakdown = [line for line in message.splitlines() if line.startswith("  ")]
    assert breakdown[0].strip().startswith("vendored/"), f"step 2: the fat directory leads, got {breakdown}"
    assert "src/" in breakdown[1], f"step 2: the small directory follows, got {breakdown}"
    assert len(breakdown) <= BREAKDOWN_LIMIT, "step 2: the breakdown is bounded"
    positions = [message.index(fragment) for fragment in (".zembleignore", "sub-path", BUDGET_ENV)]
    assert positions == sorted(positions), "step 2: the remedies are ordered cheapest first"
    assert CONFIRM_ENV in message, "step 2: the confirmation escape is named"

    # 3. A local embedder is refused just the same: the budget guards work, not only bills.
    with pytest.raises(OversizedRootRefused):
        ZembleIndex.from_path(root, embedder=LocalEmbedder(mock_embedder))

    # 4. A .zembleignore for the fat directory is enough to make the same root affordable.
    (root / ".zembleignore").write_text("vendored/\n", encoding="utf-8")
    monkeypatch.undo()
    monkeypatch.setenv(BUDGET_ENV, "1000")
    index = ZembleIndex.from_path(root, embedder=mock_embedder)
    assert index.stats.indexed_files == 1, "step 4: only the small tree was indexed"

    # 5. An explicit confirmation still indexes the whole thing.
    (root / ".zembleignore").unlink()
    monkeypatch.setenv(CONFIRM_ENV, "1")
    confirmed = ZembleIndex.from_path(root, embedder=mock_embedder)
    assert confirmed.stats.indexed_files > 1, "step 5: a confirmed build indexes the fat directory too"


def test_exclude_recovers_a_refused_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_embedder) -> None:
    """The same call, with `exclude`, builds: an agent recovers from the refusal in band."""
    root = tmp_path / "workspace"
    _tree(root)
    monkeypatch.setenv(BUDGET_ENV, "1000")
    with pytest.raises(ScopeRefused):
        ZembleIndex.from_path(root, embedder=mock_embedder)
    index = ZembleIndex.from_path(root, embedder=mock_embedder, exclude=["vendored/"])
    assert index.stats.indexed_files == 1, "the pruned build holds only the small tree"
    assert index.exclude == ("vendored/",), "the index remembers what it was built without"


def test_a_reusable_index_is_not_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_embedder) -> None:
    """Files a previous build already covers cost nothing, so an incremental build is not refused."""
    root = tmp_path / "workspace"
    _tree(root, fat_files=4, fat_bytes=400)
    monkeypatch.setenv("ZEMBLE_CACHE_LOCATION", str(tmp_path / "cache"))
    monkeypatch.setenv(BUDGET_ENV, "1000000")
    first = ZembleIndex.from_path(root, embedder=mock_embedder)
    save_index_to_cache(first, str(root.resolve()))
    manifest = first._manifest
    monkeypatch.setenv(BUDGET_ENV, "10")
    estimate = estimate_tree(root.resolve(), (ContentType.CODE,), (), manifest)
    assert estimate.files == 0, "every indexed file is unchanged, so a rebuild would chunk nothing"
    require_affordable_scope(root, mock_embedder, (ContentType.CODE,))


def test_pre_walk_estimate_tracks_the_post_chunk_estimate(tmp_path: Path, mock_embedder) -> None:
    """The cheap walk-only estimate stays within 2x of the exact one a full parse would give."""
    root = tmp_path / "workspace"
    _tree(root, fat_files=10, fat_bytes=2000)
    resolved = root.resolve()
    walked = estimate_tree(resolved, (ContentType.CODE,)).tokens
    capsules = CapsuleOptions.resolve(None)
    texts = [
        embedding_text(chunk)
        for planned in plan_files(resolved, (ContentType.CODE,), display_root=resolved, capsules=capsules)
        for chunk in planned.chunks
    ]
    parsed = estimate_tokens(texts)
    assert parsed > 0, "the fixture tree has something to embed"
    ratio = walked / parsed
    assert 0.5 <= ratio <= 2.0, f"the pre-walk estimate is {walked} against {parsed} chunked tokens"


def test_estimate_is_the_walk_the_build_would_do(tmp_path: Path) -> None:
    """The estimate counts exactly the bytes the walker yields, at the one documented density."""
    root = tmp_path / "workspace"
    (root / "keep").mkdir(parents=True)
    (root / "keep" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "b.py").write_text("y" * 5000, encoding="utf-8")
    (root / "notes.md").write_text("# hello\n", encoding="utf-8")
    estimate = estimate_tree(root, (ContentType.CODE,))
    assert estimate.files == 1, "a default-ignored directory and a docs file are not code"
    assert estimate.bytes == len("x = 1\n"), "only the walked file's bytes count"
    assert estimate.tokens == math.ceil(estimate.bytes / ESTIMATE_CHARS_PER_TOKEN), "one density, one home"
    assert [child.name for child in estimate.children] == ["keep"], "the breakdown names the child directory"
