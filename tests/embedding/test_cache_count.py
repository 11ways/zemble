"""The vector count a cache file keeps of itself: exact after every writer, never a scan on the status path."""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import zemble
from zemble.embedding import cache as cache_module
from zemble.embedding.cache import _SCHEMA, EmbeddingCache, text_hash
from zemble.embedding.gc import collect_unused
from zemble.embedding.service import EmbeddingService

FAMILY = "voyage:voyage-4-lite"
SPEC = "voyage:voyage-4-lite@4"


def _vector(seed: int) -> np.ndarray:
    return np.full(4, float(seed), dtype=np.float32)


def _counted(path: Path) -> int:
    """The truth the kept count is held to: a full scan."""
    connection = sqlite3.connect(path)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
    finally:
        connection.close()


def _raw(path: Path, sql: str, rows: list[tuple] | None = None) -> None:
    """Write as another program would, with no zemble code in between: an older zemble, a hand fix."""
    connection = sqlite3.connect(path, timeout=30)
    try:
        if rows is None:
            connection.execute(sql)
        else:
            connection.executemany(sql, rows)
        connection.commit()
    finally:
        connection.close()


def _old_file(directory: Path, rows: int) -> Path:
    """A cache file as today's released zemble leaves it: same schema, no kept count."""
    directory.mkdir(parents=True, exist_ok=True)
    path = cache_module.cache_file(FAMILY, directory)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(_SCHEMA)
    connection.executemany(
        "INSERT INTO embeddings (text_sha256, dims, vec) VALUES (?, 4, ?)",
        [(text_hash(f"old {index}"), _vector(index).tobytes()) for index in range(rows)],
    )
    connection.commit()
    connection.close()
    return path


def _settled(path: Path) -> None:
    """Wait for the background first count of a file, if one runs."""
    with cache_module._settling_lock:
        running = cache_module._settling.get(path)
    if running is not None:
        running.join(timeout=30)
        assert not running.is_alive()


def test_every_writer_keeps_the_count_equal_to_a_scan(tmp_path: Path) -> None:
    """Each statement that adds or removes a vector, zemble's own and a foreign program's, moves the count exactly."""
    cache = EmbeddingCache(FAMILY, tmp_path)
    path = cache.path
    try:
        # 1. A brand-new file knows its count at once: nothing to scan.
        assert cache.stored_count() == 0 == _counted(path), "step 1: a new file starts exact"

        def check(step: str) -> None:
            assert cache.stored_count() == _counted(path), f"{step}: kept {cache.stored_count()}, scan {_counted(path)}"

        # 2. put_many: new rows, a duplicate inside one batch, a replaced row.
        cache.put_many([(text_hash(f"t{i}"), 4, _vector(i)) for i in range(5)])
        check("step 2 new")
        cache.put_many([(text_hash("dup"), 4, _vector(1)), (text_hash("dup"), 4, _vector(2))])
        check("step 2 duplicate in a batch")
        cache.put_many([(text_hash("t0"), 4, _vector(9)), (text_hash("t0"), 8, np.ones(8, dtype=np.float32))])
        check("step 2 replace plus a second width")

        # 3. put_missing (push, server store): new and already held.
        added = cache.put_missing([(text_hash(f"t{i}"), 4, _vector(i).tobytes()) for i in range(3, 9)])
        assert added == 4
        check("step 3")

        # 4. A rolled-back batch leaves the count where it was.
        with pytest.raises(sqlite3.IntegrityError):
            _raw(path, "INSERT INTO embeddings (text_sha256, dims, vec) VALUES (?, 4, ?)", [(text_hash("t1"), b"x")])
        check("step 4 failed plain insert")

        # 5. Foreign writers: an older zemble's statements, an upsert, a NULL that OR IGNORE skips, deletes.
        _raw(
            path,
            "INSERT OR REPLACE INTO embeddings VALUES (?, 4, ?)",
            [(text_hash("t2"), b"y"), (text_hash("z"), b"z")],
        )
        check("step 5 replace")
        _raw(
            path, "INSERT OR IGNORE INTO embeddings VALUES (?, 4, ?)", [(text_hash("z"), b"z"), (text_hash("y"), None)]
        )
        check("step 5 ignore")
        _raw(
            path,
            "INSERT INTO embeddings VALUES (?, 4, ?) ON CONFLICT DO UPDATE SET vec = excluded.vec",
            [(text_hash("z"), b"w"), (text_hash("w"), b"w")],
        )
        check("step 5 upsert")
        _raw(path, f"DELETE FROM embeddings WHERE text_sha256 IN ('{text_hash('z')}', '{text_hash('nope')}')")
        check("step 5 delete some")

        # 6. Garbage collection deletes and VACUUMs on its own connection.
        with cache._lock:
            cache._connection.execute("UPDATE used SET day = day - 400")
        cache.close()
        [report] = collect_unused(tmp_path, grace_days=8, dry_run=False)
        assert report.swept > 0
        cache = EmbeddingCache(FAMILY, tmp_path)
        check("step 6 gc")

        # 7. Deleting everything takes no shortcut past the count.
        _raw(path, "DELETE FROM embeddings")
        check("step 7 delete all")
        assert cache.stored_count() == 0
    finally:
        cache.close()


def test_every_writer_in_the_source_is_one_the_writer_test_runs() -> None:
    """A new statement that adds or removes vectors has to be exercised above before it ships."""
    found = set()
    for path in Path(zemble.__file__).parent.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"(INSERT[A-Z ]*INTO|DELETE FROM|REPLACE INTO|UPDATE) embeddings\b[^\"]*", text):
            found.add((path.name, " ".join(match.group(0).split())))
    assert found == {
        ("cache.py", "INSERT OR IGNORE INTO embeddings (text_sha256, dims, vec) VALUES (?, ?, ?)"),
        ("cache.py", "INSERT OR REPLACE INTO embeddings (text_sha256, dims, vec) VALUES (?, ?, ?)"),
        ("gc.py", "DELETE FROM embeddings WHERE rowid IN (SELECT rid FROM doomed)"),
    }


def test_a_file_from_before_answers_at_once_and_counts_in_the_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first status over an existing file never waits for the scan; it says so, then answers exact."""
    directory = tmp_path / "data"
    path = _old_file(directory, rows=40)
    release = threading.Event()
    real_settle = cache_module.settle_count

    def slow_settle(target: Path) -> None:
        release.wait(timeout=30)
        real_settle(target)

    monkeypatch.setattr(cache_module, "settle_count", slow_settle)
    service = EmbeddingService(directory, [SPEC], provider=lambda spec: None)

    # 1. The scan is held back, yet status answers, marking the count as being computed.
    [family] = service.status()["families"]
    assert (family["vectors"], family["counting"]) == (None, True), "step 1: no number nobody counted"

    # 2. Writers keep writing while the count is pending; none of it is lost.
    cache = EmbeddingCache(FAMILY, directory)
    try:
        cache.put_many([(text_hash(f"new {i}"), 4, _vector(i)) for i in range(3)])
        _raw(path, f"DELETE FROM embeddings WHERE text_sha256 = '{text_hash('old 0')}'")
        assert cache.stored_count() is None
        release.set()
        _settled(path)

        # 3. Settled: exact, and from then on no scan.
        assert cache.stored_count() == 42 == _counted(path), "step 3: 40 old + 3 new - 1 deleted"
        [family] = service.status()["families"]
        assert (family["vectors"], family["counting"]) == (42, False)
    finally:
        cache.close()


def test_count_settles_in_line_where_a_caller_needs_the_number(tmp_path: Path) -> None:
    """`count` is the exact, blocking answer: a file never counted is scanned there, once."""
    _old_file(tmp_path, rows=7)
    cache = EmbeddingCache(FAMILY, tmp_path)
    try:
        assert cache.count() == 7
        cache.put_many([(text_hash("one more"), 4, _vector(1))])
        assert cache.count() == 8 == cache.stored_count()
    finally:
        cache.close()


_WRITER = """
import random, sqlite3, sys
from pathlib import Path
import numpy as np
from zemble.embedding.cache import EmbeddingCache, text_hash
seed, directory = int(sys.argv[1]), Path(sys.argv[2])
rng = random.Random(seed)
cache = EmbeddingCache('voyage:voyage-4-lite', directory)
vector = np.ones(4, dtype=np.float32)
for _ in range(40):
    texts = [text_hash(f"shared {rng.randrange(600)}") for _ in range(25)]
    if rng.random() < 0.5:
        cache.put_many([(digest, 4, vector) for digest in texts])
    else:
        cache.put_missing([(digest, 4, vector.tobytes()) for digest in texts])
    if seed == 0:
        foreign = sqlite3.connect(cache.path, timeout=30)
        foreign.executemany("DELETE FROM embeddings WHERE text_sha256 = ?", [(digest,) for digest in texts[:10]])
        foreign.commit()
        foreign.close()
cache.close()
"""


def test_processes_sharing_a_file_keep_one_exact_count(tmp_path: Path) -> None:
    """Writers in several processes, deletes among them, and a first count running meanwhile agree with a scan."""
    path = _old_file(tmp_path, rows=300)
    env = {**os.environ, "PYTHONPATH": str(Path(zemble.__file__).resolve().parent.parent), "OPENBLAS_NUM_THREADS": "1"}
    writers = [
        subprocess.Popen([sys.executable, "-c", _WRITER, str(seed), str(tmp_path)], env=env) for seed in range(3)
    ]
    cache = EmbeddingCache(FAMILY, tmp_path)
    try:
        assert cache.stored_count() is None
        _settled(path)
        for writer in writers:
            assert writer.wait(timeout=60) == 0
        assert cache.stored_count() == _counted(path)
    finally:
        for writer in writers:
            if writer.poll() is None:
                writer.kill()
        cache.close()
