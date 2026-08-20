"""Behaviour journeys over graph storage and incremental rebuilds."""

import shutil
from pathlib import Path

from zemble.graph.store import GRAPH_DB_NAME, build_graph, connect, graph_db_path, graph_exists, graph_folder


def _copy_workspace(source: Path, destination: Path) -> Path:
    """Copy the fixture workspace so a test can edit it."""
    shutil.copytree(source, destination)
    return destination


def test_build_journey(graph_fixture_root: Path, graph_cache: Path) -> None:
    """A first build creates the database, records meta, and is a no-op when repeated."""
    # 1. The graph is buildable with no search index present.
    folder = graph_folder(str(graph_fixture_root))
    assert not (folder / GRAPH_DB_NAME).exists(), "step 1: no graph exists before the first build"
    stats = build_graph(str(graph_fixture_root))
    assert graph_exists(str(graph_fixture_root)), "step 1: the build creates the database"

    # 2. Every Java file was extracted and nothing else was.
    assert stats.extracted_files == 12, "step 2: all twelve fixture files are extracted"
    assert stats.skipped_by_language == {}, "step 2: the fixture holds no non-Java files"

    # 3. The format version and root are recorded.
    connection = connect(str(graph_fixture_root))
    meta = {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM meta")}
    connection.close()
    assert meta["format_version"] == "1", "step 3: the format version is stored"
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
    path = graph_db_path(str(target))
    assert path.parent.is_dir(), "the cache folder is created when the graph path is asked for"
    assert path.name == GRAPH_DB_NAME, "the graph lives in graph.sqlite"


def test_non_java_files_are_counted_not_extracted(graph_cache: Path, tmp_path: Path) -> None:
    """A workspace with other languages reports them per language instead of failing."""
    workspace = tmp_path / "mixed"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src/Main.java").write_text("package p;\npublic class Main {}\n", encoding="utf-8")
    (workspace / "src/app.ts").write_text("export const x = 1;\n", encoding="utf-8")
    (workspace / "src/app.py").write_text("x = 1\n", encoding="utf-8")
    stats = build_graph(str(workspace))
    assert stats.extracted_files == 1, "only the Java file is extracted"
    assert stats.skipped_by_language == {"typescript": 1, "python": 1}, "the rest is counted per language"
