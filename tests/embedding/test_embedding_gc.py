"""Behaviour journeys over the embedding cache's use stamps and its garbage collection."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_chunk
from tests.embedding.test_embedding_cache import CountingEmbedder
from zemble.cache import resolve_cache_folder
from zemble.chunking.capsule import embedding_text
from zemble.cli import _cli_main
from zemble.embedding.cache import (
    JOURNAL_SIZE_LIMIT_BYTES,
    STAMPS_SINCE_KEY,
    CachingEmbedder,
    EmbeddingCache,
    text_hash,
    today,
)
from zemble.embedding.gc import collect_embeddings
from zemble.index.chunk_store import save_chunks
from zemble.types import Chunk

_FAMILY = "voyage:voyage-4-lite"
_DIMS = 256


def _index(cache_folder: Path, key: str, chunks: list[Chunk], embedder: str = f"{_FAMILY}@{_DIMS}") -> Path:
    """Save a minimal index folder: chunk columns plus the metadata naming its embedder."""
    folder = cache_folder / key / "index"
    save_chunks(folder / "chunks", chunks)
    (folder / "metadata.json").write_text(json.dumps({"root_path": "/nowhere", "embedder": embedder}))
    return folder


def _vector() -> np.ndarray:
    """A vector wide enough that sweeping rows visibly shrinks the file."""
    return np.ones(_DIMS, dtype=np.float32)


def _digests(path: Path) -> set[str]:
    """Every text hash a cache file holds."""
    connection = sqlite3.connect(path)
    try:
        return {row[0] for row in connection.execute("SELECT text_sha256 FROM embeddings")}
    finally:
        connection.close()


def _set(path: Path, sql: str, *params: object) -> None:
    """Run one write against a cache file, standing in for days that passed."""
    connection = sqlite3.connect(path)
    try:
        connection.execute(sql, params)
        connection.commit()
    finally:
        connection.close()


@pytest.fixture
def holder() -> Iterator[callable]:
    """Start processes that hold a file open, and stop them afterwards."""
    started: list[subprocess.Popen] = []

    def hold(path: Path) -> int:
        script = "import sys, time\nf = open(sys.argv[1], 'rb')\nprint('ready', flush=True)\ntime.sleep(120)\n"
        process = subprocess.Popen([sys.executable, "-c", script, str(path)], stdout=subprocess.PIPE, text=True)
        assert process.stdout is not None and process.stdout.readline().strip() == "ready"
        started.append(process)
        return process.pid

    yield hold
    for process in started:
        process.kill()
        process.wait()


def test_vectors_are_stamped_when_stored_and_when_served(tmp_path: Path) -> None:
    """A stored or served vector carries today's stamp, and the WAL has a size limit."""
    inner = CountingEmbedder(dimensions=8)
    embedder = CachingEmbedder(inner, "fake:counting", tmp_path)
    path = embedder.cache.path

    # 1. Storing a vector stamps it today, and the stamps record when they began.
    embedder.embed_documents(["alpha"])
    connection = sqlite3.connect(path)
    assert connection.execute("SELECT day FROM used").fetchall() == [(today(),)], "step 1: stamped on store"
    assert connection.execute("SELECT value FROM cache_meta WHERE key = ?", (STAMPS_SINCE_KEY,)).fetchone() == (
        str(today()),
    ), "step 1: the first day of stamps is recorded"
    connection.close()

    # 2. An old stamp moves to today when the vector is served from the cache again.
    _set(path, "UPDATE used SET day = ?", today() - 40)
    embedder.embed_documents(["alpha"])
    assert inner.document_batches == [["alpha"]], "step 2: served from the cache"
    connection = sqlite3.connect(path)
    assert connection.execute("SELECT day FROM used").fetchall() == [(today(),)], "step 2: re-stamped on use"
    connection.close()

    # 3. The cache's own connection truncates its WAL back to a bounded size.
    limit = embedder.cache._connection.execute("PRAGMA journal_size_limit").fetchone()[0]
    assert limit == JOURNAL_SIZE_LIMIT_BYTES, "step 3: journal_size_limit is set"


def test_a_collection_keeps_references_and_recent_use_and_sweeps_the_rest(tmp_path: Path, holder) -> None:
    """Referenced and recently used vectors survive; the rest is swept and the file shrinks."""
    cache_folder = tmp_path / "cache"
    kept_chunks = [make_chunk(f"def kept_{i}(): pass") for i in range(3)]
    _index(cache_folder, "a" * 64, kept_chunks)
    cache = EmbeddingCache(_FAMILY, cache_folder / "embeddings")
    path = cache.path
    referenced = {text_hash(embedding_text(chunk)) for chunk in kept_chunks}
    recent, stale, unstamped = text_hash("recent"), text_hash("stale"), text_hash("unstamped")
    padding = {text_hash(f"padding {i}") for i in range(400)}
    cache.put_many([(digest, _DIMS, _vector()) for digest in [*referenced, recent, stale, unstamped, *padding]])
    cache.close()
    # Days pass: stamps began long ago, `recent` was served yesterday, everything else last ages ago.
    _set(path, "UPDATE cache_meta SET value = ? WHERE key = ?", str(today() - 100), STAMPS_SINCE_KEY)
    _set(path, "UPDATE used SET day = ?", today() - 40)
    _set(path, "UPDATE used SET day = ? WHERE text_sha256 = ?", today() - 1, recent)
    _set(path, "DELETE FROM used WHERE text_sha256 = ?", unstamped)
    before = path.stat().st_size

    # 1. A dry run counts what would go and changes nothing.
    [report] = collect_embeddings(cache_folder, grace_days=14, dry_run=True)
    assert (report.rows, report.referenced, report.swept) == (406, 3, 402), "step 1: stale, unstamped and padding"
    assert report.swept_bytes > 402 * _DIMS * 4, "step 1: at least the vector bytes are counted"
    assert len(_digests(path)) == 406 and path.stat().st_size == before, "step 1: nothing was written"

    # 2. While another process has the file open a real run refuses.
    pid = holder(path)
    [refused] = collect_embeddings(cache_folder, grace_days=14, dry_run=False)
    assert refused.refused is not None and str(pid) in refused.refused, "step 2: refused, naming the holder"
    assert len(_digests(path)) == 406, "step 2: and nothing was swept"


def test_a_real_collection_sweeps_and_vacuums(tmp_path: Path) -> None:
    """The real run deletes exactly what the dry run counted, and VACUUM hands the space back."""
    cache_folder = tmp_path / "cache"
    kept_chunks = [make_chunk(f"def kept_{i}(): pass") for i in range(3)]
    _index(cache_folder, "a" * 64, kept_chunks)
    cache = EmbeddingCache(_FAMILY, cache_folder / "embeddings")
    path = cache.path
    referenced = {text_hash(embedding_text(chunk)) for chunk in kept_chunks}
    recent = text_hash("recent")
    padding = {text_hash(f"padding {i}") for i in range(400)}
    cache.put_many([(digest, _DIMS, _vector()) for digest in [*referenced, recent, *padding]])
    cache.close()
    _set(path, "UPDATE cache_meta SET value = ? WHERE key = ?", str(today() - 100), STAMPS_SINCE_KEY)
    _set(path, "UPDATE used SET day = ? WHERE text_sha256 != ?", today() - 40, recent)

    # 1. The sweep keeps the referenced and the recently used vectors only.
    [report] = collect_embeddings(cache_folder, grace_days=14, dry_run=False)
    assert report.refused is None and report.swept == 400, "step 1: the padding went"
    assert _digests(path) == referenced | {recent}, "step 1: exactly the referenced and recent ones stayed"
    connection = sqlite3.connect(path)
    assert connection.execute("SELECT COUNT(*) FROM used").fetchone()[0] == 4, "step 1: their stamps went too"
    connection.close()

    # 2. VACUUM returned the space, and the WAL was truncated.
    assert report.size_after is not None and report.size_after < report.size_before // 4, "step 2: the file shrank"
    wal = path.with_name(path.name + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0, "step 2: no WAL left over"


def test_an_unrecorded_age_fails_closed(tmp_path: Path) -> None:
    """Vectors stored before stamps existed count as used the day stamps began, so a fresh upgrade sweeps nothing."""
    cache_folder = tmp_path / "cache"
    _index(cache_folder, "a" * 64, [make_chunk("def kept(): pass")])
    path = cache_folder / "embeddings" / "voyage-voyage-4-lite.sqlite"
    path.parent.mkdir(parents=True)
    # 1. A file as an older zemble left it: vectors, no stamp tables at all.
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE embeddings (text_sha256 TEXT, dims INTEGER, vec BLOB, PRIMARY KEY (text_sha256, dims))")
    old.executemany(
        "INSERT INTO embeddings VALUES (?, ?, ?)",
        [(text_hash(f"old {i}"), _DIMS, _vector().tobytes()) for i in range(5)],
    )
    old.commit()
    old.close()

    # 2. A dry run over it counts every unstamped vector as used today.
    [dry] = collect_embeddings(cache_folder, grace_days=14, dry_run=True)
    assert (dry.swept, dry.stamps_since) == (0, None), "step 2: nothing old enough to prove unused"

    # 3. A real run opens it with stamps from today: still nothing goes, until the grace period is waived.
    [real] = collect_embeddings(cache_folder, grace_days=14, dry_run=False)
    assert (real.swept, real.stamps_since) == (0, today()), "step 3: stamps began today"
    [waived] = collect_embeddings(cache_folder, grace_days=0, dry_run=False)
    assert waived.swept == 5 and _digests(path) == set(), "step 3: with no grace the unreferenced ones go"


def test_collection_leaves_unused_files_and_refuses_an_unreadable_index(tmp_path: Path) -> None:
    """A file no index uses is left to `clear orphans`, and an index it cannot read stops its family's sweep."""
    cache_folder = tmp_path / "cache"
    EmbeddingCache("voyage:voyage-code-4", cache_folder / "embeddings").close()
    folder = _index(cache_folder, "b" * 64, [make_chunk("def kept(): pass")])
    EmbeddingCache(_FAMILY, cache_folder / "embeddings").close()
    (folder / "chunks" / "chunks.json").write_text("not json")

    reports = {report.path.name: report for report in collect_embeddings(cache_folder, grace_days=0, dry_run=False)}
    assert "clear orphans" in (reports["voyage-voyage-code-4.sqlite"].refused or ""), "an unused file is not swept"
    assert "unreadable" in (reports["voyage-voyage-4-lite.sqlite"].refused or ""), "an unreadable index refuses"


def test_clear_embeddings_on_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`zemble clear embeddings` reports a dry run in sizes and dates, then sweeps for real."""
    cache_folder = resolve_cache_folder()
    _index(cache_folder, "a" * 64, [make_chunk("def kept(): pass")])
    cache = EmbeddingCache(_FAMILY, cache_folder / "embeddings")
    cache.put_many([(text_hash(f"padding {i}"), _DIMS, _vector()) for i in range(50)])
    cache.close()

    # 1. The dry run names the counts, the grace period and what it would free, and sweeps nothing.
    monkeypatch.setattr(sys, "argv", ["zemble", "clear", "embeddings", "--dry-run", "--grace-days", "0"])
    _cli_main()
    out = capsys.readouterr().out
    assert "50 unreferenced; 50 of those last used on" in out and "would free about" in out, out
    assert len(_digests(cache.path)) == 50, "step 1: nothing swept"

    # 2. The real run sweeps them and reports the size it went from and to.
    monkeypatch.setattr(sys, "argv", ["zemble", "clear", "embeddings", "--grace-days", "0"])
    _cli_main()
    out = capsys.readouterr().out
    assert " -> " in out and _digests(cache.path) == set(), out

    # 3. --dry-run is refused where it would mean nothing.
    monkeypatch.setattr(sys, "argv", ["zemble", "clear", "index", "--dry-run"])
    with pytest.raises(SystemExit):
        _cli_main()
