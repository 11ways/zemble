"""The cache's size bound: least recently used entries go first, and nothing a process holds ever goes."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from zemble.cache import USED_MARKER, cache_key, mark_root, mark_used
from zemble.cache_orphans import OrphanKind, over_bound, remove_orphan
from zemble.daemon import server

_HOUR = 3600


def _entry(cache: Path, root: Path, size: int, used_hours_ago: float | None) -> Path:
    """A key folder of `size` bytes for a live root, stamped as used that many hours ago (None: never stamped)."""
    root.mkdir(exist_ok=True)
    folder = cache / cache_key(str(root))
    folder.mkdir()
    (folder / "dupes-units.sqlite").write_bytes(b"x" * size)
    mark_root(folder, str(root))
    # Written well before the hour in which a write means "in use now".
    _age(folder / "dupes-units.sqlite", max(used_hours_ago or 0, 2))
    if used_hours_ago is not None:
        mark_used(folder)
        at = time.time() - used_hours_ago * _HOUR
        os.utime(folder / USED_MARKER, (at, at))
    return folder


def _age(path: Path, hours: float) -> None:
    at = time.time() - hours * _HOUR
    os.utime(path, (at, at))


def _bytes(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def test_the_cache_keeps_its_most_recently_used_entries_within_its_bound(tmp_path: Path) -> None:
    """Newest first within the bound; a held root, an open file and a fresh write each keep their entry."""
    cache = tmp_path / "cache"
    cache.mkdir()
    newest = _entry(cache, tmp_path / "newest", 100, 0)
    second = _entry(cache, tmp_path / "second", 100, 1)
    small_old = _entry(cache, tmp_path / "small-old", 10, 3)
    held_root = _entry(cache, tmp_path / "held", 100, 5)
    open_file = _entry(cache, tmp_path / "open", 100, 6)
    # Never stamped: its last use is its newest write, which a reader's shared-memory file does not move.
    unstamped = _entry(cache, tmp_path / "unstamped", 100, None)
    _age(unstamped / "dupes-units.sqlite", 4)
    (unstamped / "index").mkdir()
    (unstamped / "index" / "graph-1.sqlite-shm").write_bytes(b"")
    embeddings = cache / "embeddings"
    embeddings.mkdir()
    vectors = embeddings / "voyage-voyage-4-lite.sqlite"
    vectors.write_bytes(b"v" * 100)
    _age(vectors, 30 * 24)

    # 1. A bound that fits three entries: the held root and the entry a process holds a file of open count first,
    #    the newest fits, and every entry less recently used goes, however small (LRU, never best fit).
    bound = sum(_bytes(folder) for folder in (held_root, open_file, newest))
    with (open_file / "dupes-units.sqlite").open("rb"):
        doomed = over_bound(cache, max_bytes=bound, held_roots={str(tmp_path / "held")})
        assert {orphan.target.name for orphan in doomed} == {
            second.name,
            small_old.name,
            unstamped.name,
            vectors.name,
        }, "step 1: THE ASSERTION -- least recently used past the bound, held ones kept"
        assert all(orphan.kind is OrphanKind.LEAST_RECENTLY_USED for orphan in doomed), "step 1: the kind"
        assert sum(orphan.size for orphan in doomed) == sum(map(_bytes, (second, small_old, unstamped))) + 100, (
            "step 1: sized"
        )
    for orphan in doomed:
        assert remove_orphan(orphan), f"step 1: removed {orphan.target}"
    assert sorted(path.name for path in cache.iterdir() if path.is_dir()) == sorted(
        [newest.name, held_root.name, open_file.name, "embeddings"]
    ), "step 1: what stays"
    assert not vectors.exists(), "step 1: the embedder file went whole"

    # 2. An entry written within the hour is in use whatever its stamp: never evicted.
    _age(newest / USED_MARKER, 48)
    (newest / "dupes-units.sqlite").write_bytes(b"y" * 100)
    _age(held_root / USED_MARKER, 48)
    _age(open_file / USED_MARKER, 48)
    _age(open_file / "dupes-units.sqlite", 48)
    doomed = over_bound(cache, max_bytes=0, held_roots=set())
    assert {orphan.target.name for orphan in doomed} == {held_root.name, open_file.name}, "step 2: fresh write stays"


def test_a_use_is_stamped_at_most_once_an_hour(tmp_path: Path) -> None:
    """The stamp is the marker's time; a use within the hour of the last stamp writes nothing."""
    folder = tmp_path / "key"
    folder.mkdir()
    mark_used(folder)
    marker = folder / USED_MARKER
    _age(marker, 0.5)
    before = marker.stat().st_mtime
    mark_used(folder)
    assert marker.stat().st_mtime == before, "within the hour: untouched"
    _age(marker, 2)
    mark_used(folder)
    assert time.time() - marker.stat().st_mtime < 60, "after it: stamped now"
    mark_used(tmp_path / "missing")
    assert not (tmp_path / "missing").exists(), "a folder that is not there is never made"


@pytest.mark.anyio
async def test_the_daemon_stamps_what_a_read_used_and_holds_what_it_serves(tmp_path: Path) -> None:
    """A read stamps its root's key folder; the roots a daemon serves are the ones its bound never evicts."""
    from zemble.cache import resolve_cache_folder

    root = tmp_path / "served"
    root.mkdir()
    (root / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    daemon = server.Daemon(watch=False)
    from tests.conftest import FakeEmbedder

    daemon.cache._embedder = FakeEmbedder()
    daemon.cache._model_ready.set()
    await daemon._prepare_read("search", {"path": str(root)}, time.monotonic() + 60)
    folder = resolve_cache_folder() / cache_key(str(root))
    assert (folder / USED_MARKER).is_file(), "step 1: THE ASSERTION -- the read stamped its folder"
    assert str(root) in daemon.held_roots(), "step 2: a served root is held"


@pytest.mark.anyio
async def test_the_daemon_releases_idle_lanes_and_graphs_so_the_bound_can_evict_them(tmp_path: Path) -> None:
    """An overlay or graph unused past the idle release stops being held; a used one, and a resident root's graph, stay.

    2026-11-25: the daemon held every lane it had ever served, so its 15.4 GB cache had nothing the 12 GB bound could
    evict.
    """
    import asyncio

    daemon = server.Daemon(watch=False)
    base = (str(tmp_path / "base"), ("code",))
    idle = (str(tmp_path / "idle-lane"), ("code",))
    used = (str(tmp_path / "used-lane"), ("code",))
    now = time.time()
    for key, used_at in ((base, now - 10 * _HOUR), (idle, now - 5 * _HOUR), (used, now - 1 * _HOUR)):
        future = asyncio.get_running_loop().create_future()
        future.set_result(object())
        daemon.cache._tasks[key] = future
        daemon.cache.last_used[key] = used_at
    daemon.cache._lane_bases[idle] = base
    daemon.cache._lane_bases[used] = base
    gone_graph = str(tmp_path / "graph-only")
    for root, used_at in ((gone_graph, now - 5 * _HOUR), (base[0], now - 5 * _HOUR), (used[0], now)):
        daemon.graphs.ready.add(root)
        daemon.graphs.used[root] = used_at
    assert idle[0] in daemon.held_roots() and gone_graph in daemon.held_roots(), "step 1: both held before"

    released = daemon.release_idle(now=now)

    assert sorted(released) == sorted([idle[0], gone_graph]), f"step 2: THE ASSERTION -- the idle ones go: {released}"
    held = daemon.held_roots()
    assert idle[0] not in held and gone_graph not in held, "step 2: no longer held, so the bound may evict them"
    assert {base[0], used[0]} <= held, "step 3: a used lane and the base it overlays stay"
    assert base[0] in daemon.graphs.ready, "step 3: a resident root's graph stays, however old its last ask"

    graph_lane = (str(tmp_path / "graph-lane"), ("code",))
    future = asyncio.get_running_loop().create_future()
    future.set_result(object())
    daemon.cache._tasks[graph_lane] = future
    daemon.cache.last_used[graph_lane] = now - 5 * _HOUR
    daemon.cache._lane_bases[graph_lane] = base
    daemon.graphs.used[graph_lane[0]] = now
    assert daemon.release_idle(now=now) == [], "step 4: a lane asked only graph questions stays"
