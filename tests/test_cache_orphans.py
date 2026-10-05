"""Behaviour journey over `zemble clear orphans`: every kind of orphan found, sized, kept or removed."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_chunk
from zemble.cache import cache_key
from zemble.cache_orphans import OrphanKind, find_orphans
from zemble.cli import _cli_main
from zemble.embedding.cache import EmbeddingCache
from zemble.graph.store import build_graph, connect, graph_folder
from zemble.index.chunk_store import save_chunks

_DAY = 86400


def _index(cache_folder: Path, key: str, root: str, **extra: object) -> Path:
    """Save a minimal index folder whose metadata names a root and a voyage embedder."""
    folder = cache_folder / key / "index"
    save_chunks(folder / "chunks", [make_chunk("def f(): pass")])
    metadata = {"root_path": root, "embedder": "voyage:voyage-4-lite@8", **extra}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    return folder


def _age(path: Path, days: float) -> None:
    """Backdate a file's modification time."""
    stamp = time.time() - days * _DAY
    os.utime(path, (stamp, stamp))


def _exited_pid() -> int:
    """Return the pid of a process that has already exited."""
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


def _write(root: Path, relative: str, text: str) -> None:
    """Write one source file under a root."""
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _clear(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *flags: str) -> str:
    """Run `zemble clear orphans` with flags and return what it printed."""
    monkeypatch.setattr(sys, "argv", ["zemble", "clear", "orphans", *flags])
    _cli_main()
    return capsys.readouterr().out


def test_clear_orphans_finds_every_kind_and_removes_only_those(
    graph_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every kind of orphan is listed with sizes by a dry run and removed by a real one; live entries stay."""
    cache = graph_cache
    live_root = tmp_path / "live"
    live_root.mkdir()
    gone_root = tmp_path / "gone"

    # 1. Live entries: an index of an existing root, a fresh git-URL entry, the embedder file it uses.
    live_index = _index(cache, cache_key(str(live_root)), str(live_root))
    fresh_url = _index(cache, "c" * 64, str(tmp_path / "clone-a"), time=time.time() - 2 * _DAY)
    EmbeddingCache("voyage:voyage-4-lite", cache / "embeddings").close()

    # 2. Orphans: an index of a deleted root, built with exclude patterns (its key carries them).
    excluded_key = cache_key(str(gone_root), ["build/"])
    gone_index = _index(cache, excluded_key, str(gone_root), exclude=["build/"])
    # A git-URL entry built 40 days ago: its clone is long gone.
    stale_url = _index(cache, "d" * 64, str(tmp_path / "clone-b"), time=time.time() - 40 * _DAY)
    # An embedder file no index reads, untouched for 40 days; a recent one is kept.
    unused = EmbeddingCache("voyage:voyage-code-4", cache / "embeddings")
    unused.close()
    _age(unused.path, 40)
    # Merely opening it today leaves an empty WAL dated today, which dates an open, not a use.
    unused.path.with_name(unused.path.name + "-wal").write_bytes(b"")
    EmbeddingCache("openai:http://localhost:1/v1#recent", cache / "embeddings").close()
    # A file no index reads but whose vectors were served lately (its use stamps say so) is kept.
    served = EmbeddingCache("voyage:voyage-3", cache / "embeddings")
    served.put_many([("digest", 8, np.ones(8, dtype=np.float32))])
    served.close()
    _age(served.path, 40)
    # A column temp file a killed save left an hour and more ago; a fresh one may still be renamed.
    leftover = live_index / "semantic_index" / "vectors.npy.123.456.tmp.npy"
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b"x" * 128)
    _age(leftover, 1)
    fresh_temp = live_index / "semantic_index" / "vectors.npy.123.789.tmp.npy"
    fresh_temp.write_bytes(b"x")
    # The staging folder of a build whose process is gone, and a file named after a dead builder;
    # a staging folder of a build still running is kept however old it is.
    dead_pid = _exited_pid()
    dead_staging = live_index.parent / f".staging-index-{dead_pid}-abc123"
    (dead_staging / "chunks").mkdir(parents=True)
    (dead_staging / "chunks" / "content.bin").write_bytes(b"x" * 64)
    _age(dead_staging, 1)
    live_staging = live_index.parent / f".staging-index-{os.getpid()}-def456"
    live_staging.mkdir()
    _age(live_staging, 1)
    dead_builder = live_index / f"graph.sqlite.building-{dead_pid}"
    dead_builder.write_bytes(b"x" * 32)
    _age(dead_builder, 1)
    # A graph whose root was deleted, and a live graph with a version no reader holds any more.
    dead_ws = tmp_path / "dead-ws"
    _write(dead_ws, "p/A.java", "package p;\npublic class A {}\n")
    build_graph(str(dead_ws))
    dead_graph = graph_folder(str(dead_ws)).parent
    shutil.rmtree(dead_ws)
    live_ws = tmp_path / "live-ws"
    _write(live_ws, "p/B.java", "package p;\npublic class B {}\n")
    build_graph(str(live_ws))
    reader = connect(str(live_ws))
    build_graph(str(live_ws), force=True)
    reader.close()

    # 3. The scan names each orphan with its kind, and nothing live.
    found = {(orphan.kind, orphan.target) for orphan in find_orphans(cache)}
    assert found == {
        (OrphanKind.INDEX_ROOT_GONE, gone_index.parent),
        (OrphanKind.GIT_URL_STALE, stale_url.parent),
        (OrphanKind.EMBEDDER_UNUSED, unused.path),
        (OrphanKind.TEMP_LEFTOVER, leftover),
        (OrphanKind.TEMP_LEFTOVER, dead_builder),
        (OrphanKind.STAGING_LEFTOVER, dead_staging),
        (OrphanKind.GRAPH_ROOT_GONE, dead_graph),
        (OrphanKind.RETIRED_GRAPH, graph_folder(str(live_ws))),
    }, "step 3: exactly the eight orphans"

    # 4. A dry run prints each with its size and a total, and deletes nothing.
    out = _clear(monkeypatch, capsys, "--dry-run")
    assert out.count("Would clear") == 8 and "MB)" in out and "Would free" in out, f"step 4: sized listing: {out}"
    assert gone_index.exists() and unused.path.exists() and leftover.exists(), "step 4: nothing deleted"

    # 5. The real run removes the eight and keeps every live entry.
    out = _clear(monkeypatch, capsys)
    assert f"Cleared orphaned index for `{gone_root}`" in out, f"step 5: the familiar line for a gone root: {out}"
    assert not gone_index.exists() and not stale_url.exists() and not dead_graph.exists(), "step 5: folders gone"
    assert not unused.path.exists() and not leftover.exists(), "step 5: files gone"
    assert not dead_staging.exists() and not dead_builder.exists(), "step 5: a dead build's leftovers gone"
    assert live_staging.exists(), "step 5: a running build's staging folder stays"
    assert [name for name in os.listdir(graph_folder(str(live_ws))) if name.endswith(".sqlite")] == [
        "graph-2.sqlite"
    ], "step 5: only the current graph version is left"
    assert live_index.exists() and fresh_url.exists() and fresh_temp.exists(), "step 5: live entries stay"
    remaining = sorted(path.name for path in (cache / "embeddings").glob("*.sqlite"))
    assert remaining == [
        "openai-http-localhost-1-v1-recent.sqlite",
        "voyage-voyage-3.sqlite",
        "voyage-voyage-4-lite.sqlite",
    ], remaining

    # 6. A second run finds nothing more.
    assert "No orphaned indexes found" in _clear(monkeypatch, capsys), "step 6: idempotent"
