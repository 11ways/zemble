"""The pre-parse work guard: a build with more source than one build may chunk is refused."""

from __future__ import annotations

from pathlib import Path

import pytest

from zemble.chunking.capsule import CapsuleOptions, embedding_text
from zemble.embedding.pricing import CONFIRM_ENV
from zemble.index import OversizedRootRefused, ScopeRefused, ZembleIndex
from zemble.index.create import plan_files
from zemble.index.scope import (
    BREAKDOWN_LIMIT,
    DEFAULT_WORK_LIMIT_BYTES,
    WORK_LIMIT_ENV,
    estimate_tree,
    work_limit_bytes,
)
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


def _tree(root: Path, fat_files: int = 12, fat_bytes: int = 96_000) -> None:
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
        raise AssertionError("plan_files ran: the work guard did not refuse before parsing")

    monkeypatch.setattr("zemble.index.create.plan_files", _explode)


def test_work_limit_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ceiling is read in megabytes, and nonsense falls back to the default."""
    assert work_limit_bytes() == DEFAULT_WORK_LIMIT_BYTES, "unset means the default"
    monkeypatch.setenv(WORK_LIMIT_ENV, "64")
    assert work_limit_bytes() == 64_000_000, "a human names megabytes, the guard counts bytes"
    monkeypatch.setenv(WORK_LIMIT_ENV, "plenty")
    assert work_limit_bytes() == DEFAULT_WORK_LIMIT_BYTES, "nonsense falls back rather than crashing a build"


def test_the_default_admits_a_real_workspace() -> None:
    """The ceiling is where runaway work begins, not where a normal multi-repo workspace sits."""
    # Measured 2026-09-04: the javaweb workspace walks to 64.0 MB of code across 9,039 files.
    assert DEFAULT_WORK_LIMIT_BYTES > 64_000_000, "a real multi-repo workspace is ordinary work"


def test_oversized_root_refusal_journey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_embedder) -> None:
    """A tree over the work limit is refused before a parse, names its fattest child, and every way out works."""
    root = tmp_path / "workspace"
    _tree(root)
    monkeypatch.setenv(WORK_LIMIT_ENV, "1")

    # 1. Refused before anything is chunked, and before the embedder is touched.
    with monkeypatch.context() as unparsed:
        _no_parsing(unparsed)
        before = len(mock_embedder.document_calls)
        with pytest.raises(OversizedRootRefused) as raised:
            ZembleIndex.from_path(root, embedder=mock_embedder)
    message = str(raised.value)
    assert len(mock_embedder.document_calls) == before, "step 1: nothing was embedded"

    # 2. The message is denominated in work: files, MB and the MB ceiling, and never a price.
    for fragment in (str(root), "of source", "1.2 MB", "1.0 MB", "may chunk"):
        assert fragment in message, f"step 2: the refusal names {fragment}"
    for absent in ("$", "token"):
        assert absent not in message, f"step 2: a work refusal never claims to know a bill ({absent})"

    # 3. The fattest child leads the breakdown, and the remedies are ordered cheapest first.
    breakdown = [line for line in message.splitlines() if line.startswith("  ")]
    assert breakdown[0].strip().startswith("vendored/"), f"step 3: the fat directory leads, got {breakdown}"
    assert "src/" in breakdown[1], f"step 3: the small directory follows, got {breakdown}"
    assert len(breakdown) <= BREAKDOWN_LIMIT, "step 3: the breakdown is bounded"
    positions = [message.index(fragment) for fragment in (".zembleignore", "sub-path", WORK_LIMIT_ENV)]
    assert positions == sorted(positions), "step 3: the remedies are ordered cheapest first"
    assert CONFIRM_ENV in message, "step 3: the confirmation escape is named"

    # 4. A local embedder is refused just the same: this guard is about work, not about bills.
    with pytest.raises(OversizedRootRefused):
        ZembleIndex.from_path(root, embedder=LocalEmbedder(mock_embedder))

    # 5. A .zembleignore for the fat directory is enough to make the same root affordable.
    (root / ".zembleignore").write_text("vendored/\n", encoding="utf-8")
    index = ZembleIndex.from_path(root, embedder=mock_embedder)
    assert index.stats.indexed_files == 1, "step 5: only the small tree was indexed"

    # 6. An explicit confirmation still indexes the whole thing.
    (root / ".zembleignore").unlink()
    monkeypatch.setenv(CONFIRM_ENV, "1")
    confirmed = ZembleIndex.from_path(root, embedder=mock_embedder)
    assert confirmed.stats.indexed_files > 1, "step 6: a confirmed build indexes the fat directory too"


def test_exclude_recovers_a_refused_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_embedder) -> None:
    """The same call, with `exclude`, builds: an agent recovers from the refusal in band."""
    root = tmp_path / "workspace"
    _tree(root)
    monkeypatch.setenv(WORK_LIMIT_ENV, "1")
    with pytest.raises(ScopeRefused):
        ZembleIndex.from_path(root, embedder=mock_embedder)
    index = ZembleIndex.from_path(root, embedder=mock_embedder, exclude=["vendored/"])
    assert index.stats.indexed_files == 1, "the pruned build holds only the small tree"
    assert index.exclude == ("vendored/",), "the index remembers what it was built without"


def test_a_reusable_index_is_not_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_embedder) -> None:
    """Files a previous build already covers cost nothing, so an incremental build is not refused."""
    root = tmp_path / "workspace"
    _tree(root)
    monkeypatch.setenv("ZEMBLE_CACHE_LOCATION", str(tmp_path / "cache"))
    monkeypatch.setenv(CONFIRM_ENV, "1")
    first = ZembleIndex.from_path(root, embedder=mock_embedder)
    monkeypatch.delenv(CONFIRM_ENV)

    # The whole tree is over the ceiling, but only the one file that moved is work.
    monkeypatch.setenv(WORK_LIMIT_ENV, "1")
    application = root / "src" / "app.py"
    application.write_text("def app():\n    return 2\n", encoding="utf-8")
    estimate = estimate_tree(root.resolve(), (ContentType.CODE,), (), first._manifest)
    assert estimate.files == 1, "only the file that moved would be chunked again"
    rebuilt = ZembleIndex.from_path(root, embedder=mock_embedder)
    assert rebuilt.stats.indexed_files == first.stats.indexed_files, "the incremental build is not refused"


def test_pre_walk_estimate_tracks_the_post_chunk_estimate(tmp_path: Path, mock_embedder) -> None:
    """The cheap walk-only estimate stays within 2x of the bytes a full parse would chunk."""
    root = tmp_path / "workspace"
    _tree(root, fat_files=10, fat_bytes=2000)
    resolved = root.resolve()
    walked = estimate_tree(resolved, (ContentType.CODE,)).bytes
    capsules = CapsuleOptions.resolve(None)
    parsed = sum(
        len(embedding_text(chunk))
        for planned in plan_files(resolved, (ContentType.CODE,), display_root=resolved, capsules=capsules)
        for chunk in planned.chunks
    )
    assert parsed > 0, "the fixture tree has something to embed"
    ratio = walked / parsed
    assert 0.5 <= ratio <= 2.0, f"the pre-walk estimate is {walked} against {parsed} chunked bytes"


def test_estimate_is_the_walk_the_build_would_do(tmp_path: Path) -> None:
    """The estimate counts exactly the bytes the walker yields, and nothing else."""
    root = tmp_path / "workspace"
    (root / "keep").mkdir(parents=True)
    (root / "keep" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "b.py").write_text("y" * 5000, encoding="utf-8")
    (root / "notes.md").write_text("# hello\n", encoding="utf-8")
    estimate = estimate_tree(root, (ContentType.CODE,))
    assert estimate.files == 1, "a default-ignored directory and a docs file are not code"
    assert estimate.bytes == len("x = 1\n"), "only the walked file's bytes count"
    assert [child.name for child in estimate.children] == ["keep"], "the breakdown names the child directory"
