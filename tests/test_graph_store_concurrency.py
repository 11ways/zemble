"""Behaviour journeys over readers and writers sharing one workspace's graph store."""

import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from zemble.graph import cli as graph_cli
from zemble.graph.provider import SqliteGraphProvider
from zemble.graph.store import (
    GRAPH_LOCK_NAME,
    GRAPH_POINTER_NAME,
    LEGACY_GRAPH_DB_NAME,
    _is_version_name,
    _read_pointer,
    build_graph,
    connect,
    graph_db_path,
    graph_exists,
    graph_folder,
    graph_present,
    refresh_graph,
)

_CIRCLE = "src/main/java/com/example/core/Circle.java"
#: Rows the held writer adds so its transaction spills far past sqlite's 2 MB page cache,
#: the point at which a rollback-journal writer locks every reader out.
_PADDING_ROWS = 20_000


def _workspace(graph_fixture_root: Path, tmp_path: Path) -> Path:
    """Copy the fixture workspace so a test can edit it."""
    return Path(shutil.copytree(graph_fixture_root, tmp_path / "ws"))


def _versions(folder: Path) -> list[str]:
    """List the version files on disk, sidecars excluded."""
    return sorted(entry for entry in os.listdir(folder) if _is_version_name(entry))


def _counts(path: str) -> tuple[int, int]:
    """Count the symbols and edges a reader sees."""
    connection = connect(path)
    try:
        return (
            connection.execute("SELECT COUNT(*) FROM symbols").fetchone()[0],
            connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        )
    finally:
        connection.close()


class _HeldWriter:
    """A refresh paused mid-transaction, after a write large enough to spill out of the page cache."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, run) -> None:
        """Patch the resolve pass to pause, and start `run` in a thread."""
        import zemble.graph.store as store

        self.writing = threading.Event()
        self.release = threading.Event()
        self.result: list[object] = []
        original = store._resolve_pass

        def held(connection: sqlite3.Connection, *args, **kwargs) -> None:
            original(connection, *args, **kwargs)
            connection.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)", [(f"pad{i}", "x" * 1000) for i in range(_PADDING_ROWS)]
            )
            self.writing.set()
            assert self.release.wait(60), "the test never released the writer"
            connection.execute("DELETE FROM meta WHERE key LIKE 'pad%'")

        monkeypatch.setattr(store, "_resolve_pass", held)
        self.thread = threading.Thread(target=lambda: self.result.append(run()), daemon=True)
        self.thread.start()
        assert self.writing.wait(60), "the writer never reached its write"

    def finish(self) -> object:
        """Let the writer commit and return what its build returned."""
        self.release.set()
        self.thread.join(60)
        assert not self.thread.is_alive(), "the writer did not finish"
        return self.result[0]


def test_a_reader_answers_during_a_long_write(
    graph_fixture_root: Path, graph_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refresh holding a spilled write transaction open never makes a reader wait."""
    workspace = _workspace(graph_fixture_root, tmp_path)
    path = str(workspace)
    build_graph(path)
    circle = workspace / _CIRCLE
    circle.write_text(circle.read_text().replace("double area()", "double surface()"), encoding="utf-8")

    # 1. A refresh renames a method and is held open after writing well past the page cache.
    writer = _HeldWriter(monkeypatch, lambda: build_graph(path, changed_paths=[circle]))

    # 2. Readers of every kind answer at once, from the last committed graph.
    started = time.monotonic()
    assert graph_exists(path), "step 2: the graph exists while it is being written"
    provider = SqliteGraphProvider(path)
    try:
        assert provider.definition("Circle.area"), "step 2: the reader sees the committed graph"
        assert not provider.definition("Circle.surface"), "step 2: and nothing uncommitted"
    finally:
        provider.close()
    elapsed = time.monotonic() - started
    assert elapsed < 2, f"step 2: no reader waited on the writer, took {elapsed:.1f} s"

    # 3. Once the writer commits, the next reader sees its work.
    stats = writer.finish()
    assert stats.extracted_files == 1, "step 3: the held refresh completed"
    provider = SqliteGraphProvider(path)
    try:
        assert provider.definition("Circle.surface"), "step 3: the committed rename is visible"
    finally:
        provider.close()


def test_a_second_refresh_skips_while_one_writes(
    graph_fixture_root: Path, graph_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Of two concurrent refreshes one writes; the other, in another process, skips and reads."""
    workspace = _workspace(graph_fixture_root, tmp_path)
    path = str(workspace)
    first = build_graph(path)
    provider = SqliteGraphProvider(path)
    circles = len(provider.definition("Circle"))
    provider.close()
    (workspace / _CIRCLE).touch()

    # 1. One refresh takes the writer lock and is held mid-transaction.
    writer = _HeldWriter(monkeypatch, lambda: refresh_graph(path))

    # 2. A refresh in another process finds the lock held, skips, and still reads the graph.
    script = textwrap.dedent(
        """
        import sys
        from zemble.graph.store import refresh_graph
        from zemble.graph.provider import SqliteGraphProvider
        print(refresh_graph(sys.argv[1]))
        provider = SqliteGraphProvider(sys.argv[1])
        print(len(provider.definition("Circle")))
        provider.close()
        """
    )
    other = subprocess.run(
        [sys.executable, "-c", script, path], capture_output=True, text=True, timeout=60, env=os.environ.copy()
    )
    assert other.returncode == 0, f"step 2: the second refresh failed: {other.stderr}"
    assert other.stdout.split() == ["None", str(circles)], f"step 2: it skipped and read, got {other.stdout!r}"

    # 3. The once-per-process refresh every query runs skips the same way rather than erroring.
    graph_cli._refreshed.discard(path)
    graph_cli.ensure_graph(path, allow_daemon=False)
    assert path in graph_cli._refreshed, "step 3: ensure_graph counted its refresh as done"

    # 4. The writer's refresh is the one that landed.
    stats = writer.finish()
    assert stats is not None and stats.extracted_files == 1, "step 4: the first refresh wrote"
    assert (stats.symbols, stats.edges) == (first.symbols, first.edges), "step 4: with the same graph"


def test_a_killed_full_build_leaves_the_previous_graph_current(
    graph_fixture_root: Path, graph_cache: Path, tmp_path: Path
) -> None:
    """A full build killed mid-write publishes nothing, holds no lock, and leaves only debris that is swept."""
    workspace = _workspace(graph_fixture_root, tmp_path)
    path = str(workspace)
    build_graph(path)
    folder = graph_folder(path)
    before = _read_pointer(folder)
    counts = _counts(path)

    # 1. A forced build is SIGKILLed after it has written its extraction into a new version.
    script = textwrap.dedent(
        """
        import os, signal, sys
        import zemble.graph.store as store
        store._resolve_pass = lambda *args, **kwargs: os.kill(os.getpid(), signal.SIGKILL)
        store.build_graph(sys.argv[1], force=True, workers=1)
        """
    )
    killed = subprocess.run([sys.executable, "-c", script, path], timeout=60, env=os.environ.copy())
    assert killed.returncode == -9, "step 1: the build died by SIGKILL"
    assert len(_versions(folder)) == 2, "step 1: it left a half-written version behind"

    # 2. The pointer never moved, and readers still get the whole previous graph.
    assert _read_pointer(folder) == before, "step 2: the previous version is still current"
    assert graph_exists(path), "step 2: and it is a graph"
    assert _counts(path) == counts, "step 2: with every row it had"

    # 3. The dead writer's lock died with it, and the next refresh sweeps its debris.
    stats = refresh_graph(path)
    assert stats is not None, "step 3: the next refresh took the lock"
    assert _versions(folder) == [before.current], "step 3: the half-written version is gone"
    leftovers = [entry for entry in os.listdir(folder) if entry.startswith("graph-2")]
    assert leftovers == [], f"step 3: and so are its sidecars, found {leftovers}"


def test_a_single_file_store_is_migrated_on_first_open(
    graph_fixture_root: Path, graph_cache: Path, tmp_path: Path
) -> None:
    """The store an older zemble wrote is copied into the versioned layout by its first reader, not rebuilt."""
    workspace = _workspace(graph_fixture_root, tmp_path)
    path = str(workspace)
    build_graph(path)
    counts = _counts(path)
    folder = graph_folder(path)

    # 1. Recreate the old layout: one rollback-journal `graph.sqlite` and nothing else.
    built = graph_db_path(path)
    assert built is not None
    source = sqlite3.connect(built)
    source.execute("VACUUM INTO ?", (str(folder / LEGACY_GRAPH_DB_NAME),))
    source.close()
    legacy = sqlite3.connect(folder / LEGACY_GRAPH_DB_NAME)
    assert legacy.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    legacy.close()
    for entry in os.listdir(folder):
        if entry.startswith("graph-") or entry in (GRAPH_POINTER_NAME, GRAPH_LOCK_NAME):
            (folder / entry).unlink()
    legacy_bytes = (folder / LEGACY_GRAPH_DB_NAME).read_bytes()
    assert graph_present(path), "step 1: the old layout counts as a present graph"

    # 2. The first reader migrates it and reads the same graph.
    assert _counts(path) == counts, "step 2: the first reader sees every row"
    pointer = _read_pointer(folder)
    assert pointer is not None and _versions(folder) == [pointer.current], "step 2: it became one version"
    connection = connect(path)
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal", "step 2: in WAL mode"
    connection.close()

    # 3. Nothing was rebuilt: the next refresh finds every file unchanged.
    stats = refresh_graph(path)
    assert stats is not None and stats.extracted_files == 0, "step 3: the migration kept the extraction"

    # 4. The old file is left untouched for processes still running older code.
    assert (folder / LEGACY_GRAPH_DB_NAME).read_bytes() == legacy_bytes, "step 4: the legacy file is untouched"


def test_a_forced_build_never_reads_a_legacy_store(graph_fixture_root: Path, graph_cache: Path, tmp_path: Path) -> None:
    """A forced build over the old layout builds from source instead of copying the old file first."""
    workspace = _workspace(graph_fixture_root, tmp_path)
    path = str(workspace)
    folder = graph_folder(path)
    folder.mkdir(parents=True)

    # 1. The old single-file store is unreadable, so any attempt to copy it would fail as corrupt.
    (folder / LEGACY_GRAPH_DB_NAME).write_bytes(b"not a database at all")

    # 2. The forced build ignores it: no corruption to recover from, one fresh version.
    stats = build_graph(path, force=True)
    assert not stats.rebuilt_from_corruption, "step 2: the legacy file was never opened"
    pointer = _read_pointer(folder)
    assert pointer is not None and _versions(folder) == [pointer.current], "step 2: one fresh version"
    assert (folder / LEGACY_GRAPH_DB_NAME).read_bytes() == b"not a database at all", "step 2: left untouched"


def test_old_versions_are_swept_but_never_current_or_previous(
    graph_fixture_root: Path, graph_cache: Path, tmp_path: Path
) -> None:
    """Each full build keeps exactly the version it replaced, and a reader on that one keeps reading."""
    workspace = _workspace(graph_fixture_root, tmp_path)
    path = str(workspace)
    build_graph(path)
    folder = graph_folder(path)
    counts = _counts(path)
    unrelated = folder / "chunks.npy"
    unrelated.write_bytes(b"index data")

    # 1. Debris from killed builds lies beside the store: a version and orphaned sidecars.
    for name in ("graph-7.sqlite", "graph-7.sqlite-wal", "graph-0.sqlite-shm"):
        (folder / name).write_bytes(b"debris")

    # 2. A reader is open on the current version when the next full build lands.
    first = _read_pointer(folder)
    reader = connect(path)
    build_graph(path, force=True)
    second = _read_pointer(folder)
    assert second is not None and second.previous == first.current, "step 2: the replaced version is kept"
    assert _versions(folder) == sorted([first.current, second.current]), "step 2: current and previous only"
    assert not any(entry.startswith(("graph-7", "graph-0")) for entry in os.listdir(folder)), "step 2: debris swept"
    assert reader.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == counts[0], "step 2: the reader reads on"

    # 3. Another full build retires the oldest; the reader's file is gone yet its open handle still reads.
    build_graph(path, force=True)
    third = _read_pointer(folder)
    assert third is not None and third.previous == second.current, "step 3: the newest replaced version is kept"
    assert _versions(folder) == sorted([second.current, third.current]), "step 3: the oldest is swept"
    assert reader.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == counts[0], "step 3: still readable"
    reader.close()

    # 4. Nothing outside the version files was touched, and readers see the same graph.
    assert unrelated.read_bytes() == b"index data", "step 4: an unrelated file in the folder survived"
    assert (folder / GRAPH_LOCK_NAME).is_file(), "step 4: the lock file is never deleted"
    assert _counts(path) == counts, "step 4: the graph is unchanged by its rebuilds"
