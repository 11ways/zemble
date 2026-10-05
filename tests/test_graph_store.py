"""Behaviour journeys over graph storage and incremental rebuilds."""

import json
import logging
import shutil
import sqlite3
from pathlib import Path

import pytest

from zemble.graph.model import Edge, EdgeKind, Resolution
from zemble.graph.store import (
    GRAPH_FORMAT_VERSION,
    GRAPH_POINTER_NAME,
    _compact_if_drifted,
    _Pointer,
    _publish,
    _read_pointer,
    build_graph,
    compact_stored_graphs,
    connect,
    edge_from_row,
    graph_db_path,
    graph_exists,
    graph_folder,
    insert_edges,
    open_db,
)


def _as_edge_table(db: Path) -> None:
    """Rewrite a store into the format-7 layout: one `edges` table spelling every id as text."""
    legacy = sqlite3.connect(db)
    legacy.executescript(
        "CREATE TABLE legacy AS SELECT * FROM edges; DROP VIEW edges; DROP TABLE edge_rows; DROP TABLE refs; "
        "ALTER TABLE legacy RENAME TO edges; UPDATE meta SET value = '7' WHERE key = 'format_version';"
    )
    legacy.commit()
    legacy.close()


def _copy_workspace(source: Path, destination: Path) -> Path:
    """Copy the fixture workspace so a test can edit it."""
    shutil.copytree(source, destination)
    return destination


def test_build_journey(graph_fixture_root: Path, graph_cache: Path) -> None:
    """A first build creates the database, records meta, and is a no-op when repeated."""
    # 1. The graph is buildable with no search index present.
    folder = graph_folder(str(graph_fixture_root))
    assert not (folder / GRAPH_POINTER_NAME).exists(), "step 1: no graph exists before the first build"
    stats = build_graph(str(graph_fixture_root))
    assert graph_exists(str(graph_fixture_root)), "step 1: the build creates the database"

    # 2. Every Java file was extracted and nothing else was.
    assert stats.extracted_files == 12, "step 2: all twelve fixture files are extracted"
    assert stats.skipped_by_language == {}, "step 2: the fixture holds no non-Java files"

    # 3. The format version and root are recorded.
    connection = connect(str(graph_fixture_root))
    meta = {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM meta")}
    connection.close()
    assert meta["format_version"] == str(GRAPH_FORMAT_VERSION), "step 3: the format version is stored"
    assert meta["root"] == str(graph_fixture_root), "step 3: the root is stored"

    # 4. Rebuilding without changes extracts nothing and re-resolves nothing.
    again = build_graph(str(graph_fixture_root))
    assert (again.extracted_files, again.reresolved_files) == (0, 0), "step 4: an unchanged tree is a no-op"

    # 5. The counts did not drift.
    assert (again.symbols, again.edges) == (stats.symbols, stats.edges), "step 5: a no-op build changes no counts"


def test_incremental_journey(graph_fixture_root: Path, graph_cache: Path, tmp_path: Path) -> None:
    """Edits, renames and deletions each move exactly what they should."""
    workspace = _copy_workspace(graph_fixture_root, tmp_path / "ws")
    path = str(workspace)
    first = build_graph(path)

    # 1. Touching a file re-extracts it but drags in no dependents, because no name moved.
    circle = workspace / "src/main/java/com/example/core/Circle.java"
    circle.touch()
    touched = build_graph(path)
    assert touched.extracted_files == 1, "step 1: only the touched file is re-extracted"
    assert touched.reresolved_files == 1, "step 1: an unchanged declaration set drags in no dependents"
    assert touched.edges == first.edges, "step 1: re-extraction does not duplicate edges"

    # 2. Renaming a method invalidates every file that wrote that name.
    circle.write_text(circle.read_text().replace("double area()", "double surface()"), encoding="utf-8")
    renamed = build_graph(path)
    assert renamed.reresolved_files > 1, "step 2: a renamed declaration re-resolves its users"

    # 3. The rename is visible: the old call no longer lands on Circle.
    connection = connect(path)
    landed = connection.execute(
        "SELECT dst_id FROM edges WHERE kind = 'calls' AND dst_name = 'area' AND src_id LIKE '%CircleTest%'"
    ).fetchall()
    assert all(row["dst_id"] is None or "core.Circle" not in row["dst_id"] for row in landed), (
        "step 3: the renamed method is no longer the call target"
    )
    connection.close()

    # 4. Restoring the name restores the edge.
    circle.write_text(circle.read_text().replace("double surface()", "double area()"), encoding="utf-8")
    build_graph(path)
    connection = connect(path)
    restored = connection.execute(
        "SELECT dst_id FROM edges WHERE kind = 'calls' AND dst_name = 'area' AND src_id LIKE '%CircleTest%'"
    ).fetchall()
    connection.close()
    assert any("core.Circle.area" in (row["dst_id"] or "") for row in restored), (
        "step 4: restoring the name restores the edge"
    )

    # 5. Deleting a file removes its symbols and its edges.
    (workspace / "src/main/java/com/example/util/Circle.java").unlink()
    deleted = build_graph(path)
    assert deleted.removed_files == 1, "step 5: the deleted file is reported"
    connection = connect(path)
    left = connection.execute("SELECT COUNT(*) AS n FROM symbols WHERE file_path LIKE '%util/Circle.java'").fetchone()[
        "n"
    ]
    connection.close()
    assert left == 0, "step 5: no symbol survives its file"

    # 6. With only one Circle left, the formerly ambiguous reference resolves.
    connection = connect(path)
    consumer = connection.execute(
        "SELECT resolution FROM edges WHERE src_id LIKE '%Consumer.measure%' AND dst_name = 'Circle'"
    ).fetchone()
    connection.close()
    assert consumer["resolution"] == "unique_name", "step 6: removing the twin resolves the ambiguity"


def test_graph_db_path_creates_its_folder(graph_cache: Path, tmp_path: Path) -> None:
    """The graph folder is created on demand, so no search index is required first."""
    target = tmp_path / "empty"
    target.mkdir()
    assert graph_db_path(str(target)) is None, "no version exists before the first build"
    assert graph_folder(str(target)).is_dir(), "the cache folder is created when the graph path is asked for"
    build_graph(str(target))
    path = graph_db_path(str(target))
    assert path is not None and path.name == "graph-1.sqlite", "the first build is version 1"


def test_files_without_a_grammar_are_counted_not_extracted(graph_cache: Path, tmp_path: Path) -> None:
    """Every language with a grammar is extracted; the rest is reported per language instead of failing."""
    workspace = tmp_path / "mixed"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src/Main.java").write_text("package p;\npublic class Main {}\n", encoding="utf-8")
    (workspace / "src/app.ts").write_text("export const x = 1;\n", encoding="utf-8")
    (workspace / "src/app.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "src/app.cob").write_text("IDENTIFICATION DIVISION.\n", encoding="utf-8")
    stats = build_graph(str(workspace))
    assert stats.extracted_files == 3, "Java, TypeScript and Python are all extracted"
    assert stats.skipped_by_language == {"cobol": 1}, "a language without a grammar is counted"


def test_change_set_refresh_journey(graph_fixture_root: Path, graph_cache: Path, tmp_path: Path) -> None:
    """A refresh driven by a change set updates the named files and looks at nothing else."""
    workspace = _copy_workspace(graph_fixture_root, tmp_path / "ws")
    path = str(workspace)
    build_graph(path)
    circle = workspace / "src/main/java/com/example/core/Circle.java"

    # 1. A named edit is picked up, and re-resolves the users of the name it moved.
    circle.write_text(circle.read_text().replace("double area()", "double surface()"), encoding="utf-8")
    named = build_graph(path, changed_paths=[circle])
    assert named.extracted_files == 1, "step 1: the named file was re-extracted"
    assert named.reresolved_files > 1, "step 1: and its dependents were re-resolved"
    connection = connect(path)
    renamed = connection.execute("SELECT 1 FROM symbols WHERE name = 'surface'").fetchone()
    connection.close()
    assert renamed is not None, "step 1: the new declaration is in the graph"

    # 2. An edit the change set does not name is not discovered: nothing walks the tree.
    other = workspace / "src/main/java/com/example/core/Point.java"
    other.write_text(other.read_text().replace("class Point", "class Point /* edited */"), encoding="utf-8")
    unnamed = build_graph(path, changed_paths=[circle])
    assert unnamed.extracted_files == 0, "step 2: only named paths are looked at"

    # 3. The same edit named explicitly is picked up, and a full walk agrees with the result.
    named_again = build_graph(path, changed_paths=[other])
    assert named_again.extracted_files == 1, "step 3: naming it re-extracts it"
    walked = build_graph(path)
    assert walked.extracted_files == 0, "step 3: a walk finds nothing left to do"
    assert (walked.symbols, walked.edges) == (named_again.symbols, named_again.edges), "step 3: same graph either way"

    # 4. A named file that was deleted is removed from the graph.
    other.unlink()
    removed = build_graph(path, changed_paths=[other])
    assert removed.removed_files == 1, "step 4: the deleted file left the graph"
    connection = connect(path)
    remaining = connection.execute(
        "SELECT 1 FROM symbols WHERE file_path = ?", ("src/main/java/com/example/core/Point.java",)
    ).fetchone()
    connection.close()
    assert remaining is None, "step 4: and so did its symbols"


def test_torn_store_journey(graph_fixture_root: Path, graph_cache: Path, tmp_path: Path, caplog) -> None:
    """A store torn by a killed build is refused loudly and rebuilt, never read past."""
    workspace = _copy_workspace(graph_fixture_root, tmp_path / "ws")
    path = str(workspace)
    first = build_graph(path)
    db = graph_db_path(path)
    assert db is not None

    # 1. The live store is written durably, so a kill leaves a log to recover from rather
    #    than a torn file: that is what `synchronous=OFF` + `journal_mode=MEMORY` discarded.
    connection = open_db(db)
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal", "step 1: the log is on disk"
    assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2, "step 1: and commits fsync"
    connection.close()

    # 2. A file cut short mid-write is exactly what an OOM-killed build left behind.
    intact = db.read_bytes()
    db.write_bytes(intact[: len(intact) // 2])
    assert not graph_exists(path), "step 2: a malformed store is not a graph"

    # 3. Building over it says so at ERROR and rebuilds from source instead of logging past it.
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="zemble.graph.store"):
        rebuilt = build_graph(path)
    assert rebuilt.rebuilt_from_corruption, "step 3: the build reports the rebuild it had to do"
    assert any("malformed" in record.getMessage() for record in caplog.records), "step 3: and says so out loud"

    # 4. What it rebuilt is the graph that was there before, and sqlite agrees it is sound.
    assert (rebuilt.symbols, rebuilt.edges) == (first.symbols, first.edges), "step 4: the same graph came back"
    connection = connect(path)
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "step 4: the file is sound"
    connection.close()

    # 5. The rebuild is a new version, and the malformed one is gone rather than kept as previous.
    pointer = _read_pointer(db.parent)
    assert pointer is not None and pointer.current != db.name, "step 5: the rebuild is a new version"
    assert pointer.previous is None and not db.exists(), "step 5: the malformed version was dropped"

    # 6. A refresh after the rebuild is an ordinary no-op again.
    again = build_graph(path)
    assert (again.extracted_files, again.rebuilt_from_corruption) == (0, False), "step 6: back to a plain refresh"


def test_a_format_6_store_keeps_only_candidate_counts(tmp_path: Path) -> None:
    """Stored candidate lists become counts: read as counts before migrating, migrated by the next writer."""
    db = tmp_path / "graph-1.sqlite"
    legacy = sqlite3.connect(db)
    legacy.execute(
        "CREATE TABLE edges (src_id TEXT, dst_id TEXT, dst_name TEXT, kind TEXT, line INTEGER, resolution TEXT, "
        "candidates TEXT, arity INTEGER, receiver TEXT, receiver_type TEXT, is_new INTEGER, file_path TEXT, "
        "source TEXT, origin_ref TEXT)"
    )
    ids = [f"a/B.java#B.id{n}()" for n in range(300)]
    legacy.executemany(
        "INSERT INTO edges (src_id, dst_name, kind, line, resolution, candidates, arity) VALUES (?,?,?,?,?,?,?)",
        [
            ("a/A.java#A.run()", "id", "calls", 3, "ambiguous", json.dumps(ids), 0),
            ("a/A.java#A.run()", "x", "calls", 4, "unresolved", None, 0),
        ],
    )
    legacy.commit()
    legacy.close()

    # 1. A reader of the unmigrated store already sees counts, never the list.
    reader = open_db(db, read_only=True)
    edges = [edge_from_row(row) for row in reader.execute("SELECT * FROM edges ORDER BY line")]
    reader.close()
    assert [edge.ambiguity() for edge in edges] == [300, 0], "step 1: a format-6 row is read as its count"
    assert all(edge.candidates == [] for edge in edges), "step 1: and the list itself is not materialised"

    # 2. The next writer replaces the lists by their counts and drops the column.
    writer = open_db(db)
    columns = {row["name"] for row in writer.execute("PRAGMA table_info(edges)")}
    edges = [edge_from_row(row) for row in writer.execute("SELECT * FROM edges ORDER BY line")]
    writer.close()
    assert "candidates" not in columns and "candidate_count" in columns, "step 2: the list column is gone"
    assert [edge.ambiguity() for edge in edges] == [300, 0], "step 2: the counts survived the migration"

    # 3. An edge resolved now stores its count the same way.
    fresh = Edge(src_id="a/A.java#A.go()", dst_name="id", kind=EdgeKind.CALLS, line=9, candidates=ids[:7])
    fresh.resolution = Resolution.AMBIGUOUS
    writer = open_db(db)
    insert_edges(writer, [fresh])
    stored = edge_from_row(writer.execute("SELECT * FROM edges WHERE line = 9").fetchone())
    writer.close()
    assert stored.ambiguity() == 7, "step 3: a new ambiguous edge keeps its count"


def test_compact_brings_every_stored_graph_to_the_current_format(graph_fixture_root: Path, graph_cache: Path) -> None:
    """`zemble graph compact` migrates a graph nobody writes to, and gives the freed space back."""
    path = str(graph_fixture_root)
    built = build_graph(path)
    db = graph_db_path(path)
    assert db is not None
    # The state a format-6 zemble left: an edges table where every edge carries a long candidate list.
    _as_edge_table(db)
    legacy = sqlite3.connect(db)
    legacy.execute("ALTER TABLE edges ADD COLUMN candidates TEXT")
    legacy.execute("UPDATE edges SET candidates = ?", (json.dumps([f"x/Y.java#Y.m{n}()" for n in range(10000)]),))
    legacy.execute("ALTER TABLE edges DROP COLUMN candidate_count")
    legacy.commit()
    legacy.close()

    # 1. Every graph is reported with its size before and after.
    reports = compact_stored_graphs(graph_cache)
    assert [report.folder for report in reports] == [db.parent], "step 1: the one stored graph is compacted"
    report = reports[0]
    assert report.skipped is None and report.size_after < report.size_before // 4, "step 1: and shrank"

    # 2. It is in the current format now, with every edge and its count intact.
    connection = connect(path)
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(edges)")}
    count = connection.execute("SELECT COUNT(*), MIN(candidate_count) FROM edges").fetchone()
    layout = connection.execute("SELECT type FROM sqlite_master WHERE name = 'edges'").fetchone()[0]
    spelled = connection.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
    distinct = connection.execute(
        "SELECT COUNT(*) FROM "
        "(SELECT src_id FROM edges UNION SELECT dst_id FROM edges UNION SELECT file_path FROM edges)"
    ).fetchone()[0]
    connection.close()
    assert "candidates" not in columns, "step 2: the list column is gone"
    assert layout == "view", "step 2: edges are read through the view over interned keys"
    assert spelled == distinct - 1, "step 2: every id and file is spelled once (the NULL dst_id is not one)"
    assert tuple(count) == (built.edges, 10000), "step 2: every edge kept its count"

    # 3. Running it again finds nothing left to do.
    again = compact_stored_graphs(graph_cache)[0]
    assert again.size_after == again.size_before, "step 3: a current, compact graph is left as it is"


def test_a_bloated_store_is_compacted(tmp_path: Path) -> None:
    """Deleted rows are returned to the filesystem as a new version, and a small store is left alone."""
    folder = tmp_path / "graph"
    folder.mkdir()
    db = folder / "graph-1.sqlite"
    connection = open_db(db)
    connection.executemany(
        "INSERT INTO symbols (id, name) VALUES (?, ?)", [(f"id{i}", "x" * 400) for i in range(20_000)]
    )
    connection.commit()
    connection.close()
    _publish(folder, _Pointer(db.name))
    grown = db.stat().st_size

    # 1. A store with nothing free is left exactly as it is.
    assert not _compact_if_drifted(folder, db.name), "step 1: nothing to reclaim"
    assert db.stat().st_size == grown, "step 1: and nothing was rewritten"

    # 2. Deleting most of it frees pages that sqlite keeps in the file.
    connection = open_db(db)
    connection.execute("DELETE FROM symbols WHERE id != 'id0'")
    connection.commit()
    connection.close()
    assert db.stat().st_size == grown, "step 2: a delete never shrinks the file on its own"

    # 3. The next build's compaction hands them back as the next version, with the rows intact.
    assert _compact_if_drifted(folder, db.name), "step 3: the drift is compacted"
    assert _read_pointer(folder) == _Pointer("graph-2.sqlite"), "step 3: the copy is current"
    assert not db.exists(), "step 3: and the version it replaced, which no reader held, is gone"
    compact = folder / "graph-2.sqlite"
    assert compact.stat().st_size < grown // 2, "step 3: and the file actually shrank"
    connection = open_db(compact, read_only=True)
    assert connection.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == 1, "step 3: the surviving row survived"
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal", "step 3: in the store's journal mode"
    connection.close()


def test_an_explicit_compact_reclaims_what_drift_waits_for(graph_cache: Path) -> None:
    """`zemble graph compact` gives back a free share too small for a build's own compaction."""
    folder = graph_cache / "some-root" / "index"
    folder.mkdir(parents=True)
    db = folder / "graph-1.sqlite"
    connection = open_db(db)
    connection.executemany(
        "INSERT INTO symbols (id, name) VALUES (?, ?)", [(f"id{i}", "x" * 400) for i in range(20_000)]
    )
    # Contiguous rows, so their pages come free whole: about a twentieth of the store.
    connection.execute("DELETE FROM symbols WHERE id LIKE 'id19%'")
    connection.commit()
    connection.close()
    _publish(folder, _Pointer(db.name))
    grown = db.stat().st_size

    # 1. A twentieth of the store free is below what a build compacts.
    assert not _compact_if_drifted(folder, db.name), "step 1: a build leaves it free"

    # 2. The explicit command reclaims it, rows intact.
    report = compact_stored_graphs(graph_cache)[0]
    assert report.size_before == grown and report.size_after < grown * 0.95, "step 2: the command reclaims it"
    connection = open_db(folder / _read_pointer(folder).current, read_only=True)
    assert connection.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == 18_889, "step 2: no row was lost"
    connection.close()


def test_a_failed_refresh_leaves_the_graph_it_started_from(
    graph_fixture_root: Path, graph_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refresh that fails mid-resolution rolls back and leaves no scratch behind."""
    import zemble.graph.store as store

    workspace = _copy_workspace(graph_fixture_root, tmp_path / "ws")
    path = str(workspace)
    built = build_graph(path)
    circle = workspace / "src/main/java/com/example/core/Circle.java"

    def _tables() -> tuple[int, int, int]:
        connection = connect(path)
        counts = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("symbols", "edges", "files")
        )  # noqa: S608
        connection.close()
        return counts

    # 1. An edit is refreshed in place, and resolution fails after its symbols and edges moved.
    before = _tables()
    circle.write_text(circle.read_text().replace("double area()", "double surface()"), encoding="utf-8")

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("resolution failed")

    resolve_pass = store._resolve_pass
    monkeypatch.setattr(store, "_resolve_pass", _fail)
    with pytest.raises(RuntimeError, match="resolution failed"):
        build_graph(path, changed_paths=[circle])

    # 2. The graph is exactly the one before, and the scratch database is gone.
    assert _tables() == before == (built.symbols, built.edges, before[2]), "step 2: nothing of the refresh was kept"
    assert not list(graph_folder(path).glob("graph-scratch.building-*")), "step 2: no scratch left"

    # 3. The next refresh picks the edit up as if nothing had happened.
    monkeypatch.setattr(store, "_resolve_pass", resolve_pass)
    refreshed = build_graph(path, changed_paths=[circle])
    assert refreshed.extracted_files == 1, "step 3: the edit is still pending and is refreshed now"
