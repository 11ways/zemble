"""Cross-process cache buyers never carry a preflight snapshot into a later write."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import zemble
from zemble.embedding.cache import BUSY_TIMEOUT_SECONDS, EmbeddingCache, text_hash

_BUY = """
import sys
import numpy as np
from pathlib import Path
from zemble.embedding.cache import EmbeddingCache, text_hash
cache = EmbeddingCache('fake:concurrency', Path(sys.argv[1]))
cache.put_many([(text_hash(sys.argv[2]), 3, np.array([1., 0., 0.], dtype=np.float32))])
cache.close()
"""
_WAITING_BUY = """
import sys
import numpy as np
from pathlib import Path
from zemble.embedding.cache import EmbeddingCache, text_hash
cache = EmbeddingCache('fake:concurrency', Path(sys.argv[1]))
print('opened', flush=True)
sys.stdin.readline()
print('writing', flush=True)
cache.put_many([(text_hash('waiting'), 3, np.array([1., 0., 0.], dtype=np.float32))])
cache.close()
print('committed', flush=True)
"""


def _env() -> dict:
    return {**os.environ, "PYTHONPATH": str(Path(zemble.__file__).resolve().parent.parent), "OPENBLAS_NUM_THREADS": "1"}


def _buy(directory: Path, text: str) -> None:
    subprocess.run([sys.executable, "-c", _BUY, str(directory), text], env=_env(), check=True, timeout=20)


def test_preflight_then_another_process_commit_then_embedding(tmp_path: Path) -> None:
    """A paid embedding may follow preflight after any other buyer committed in the same family."""
    cache = EmbeddingCache("fake:concurrency", tmp_path)
    try:
        # 1. Cache preflight discovers a miss without retaining a transaction.
        assert cache.pending(["ours"], 3) == ["ours"]
        assert not cache._connection.in_transaction, "step 1: no WAL read snapshot survives preflight"
        # 2. An independent buyer commits while this buyer is away at its provider.
        _buy(tmp_path, "other")
        # 3. Returning from the provider can write and stamp our result atomically.
        cache.put_many([(text_hash("ours"), 3, np.array([0.0, 1.0, 0.0], dtype=np.float32))])
        assert cache.covered([text_hash("ours"), text_hash("other")], 3) == {text_hash("ours"), text_hash("other")}
        assert not cache._connection.in_transaction, "step 3: lookup and store both leave the connection idle"
    finally:
        cache.close()


def test_writer_waits_natively_and_failed_batch_releases_its_lock(tmp_path: Path, monkeypatch) -> None:
    """Contending writers wait for a short batch; errors roll back vectors together with their stamps."""
    cache = EmbeddingCache("fake:concurrency", tmp_path)
    child = subprocess.Popen(
        [sys.executable, "-c", _WAITING_BUY, str(tmp_path)],
        env=_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "opened"
        assert cache._connection.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_SECONDS * 1000
        cache._connection.execute("BEGIN IMMEDIATE")
        child.stdin.write("go\n")
        child.stdin.flush()
        assert child.stdout.readline().strip() == "writing"
        assert child.poll() is None, "step 1: the other writer waits instead of failing or rerunning embedding"
        cache._connection.commit()
        output, _ = child.communicate(timeout=20)
        assert child.returncode == 0 and "committed" in output
        assert cache.get(text_hash("waiting"), 3) is not None

        # 2. A stamp failure cannot leave a partially persisted vector or a live write transaction.
        def fail(_digests):
            raise ValueError("stamp failed")

        monkeypatch.setattr(cache, "_stamp", fail)
        with pytest.raises(ValueError, match="stamp failed"):
            cache.put_many([(text_hash("rollback"), 3, np.ones(3, dtype=np.float32))])
        assert cache.get(text_hash("rollback"), 3) is None
        assert not cache._connection.in_transaction
        _buy(tmp_path, "after rollback")
    finally:
        cache._connection.rollback()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        cache.close()


def test_concurrent_first_openers_publish_one_valid_wal_family(tmp_path: Path) -> None:
    """The journal-mode transition and schema creation are serialized only during initialization."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda text: _buy(tmp_path, text), ["one", "two", "three", "four"]))
    cache = EmbeddingCache("fake:concurrency", tmp_path)
    try:
        assert cache._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert cache._connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert cache.covered([text_hash(text) for text in ("one", "two", "three", "four")], 3) == {
            text_hash(text) for text in ("one", "two", "three", "four")
        }
    finally:
        cache.close()
