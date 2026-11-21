"""A graph an older zemble built keeps one derived override per method; its first write re-derives them once."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from tests.conftest import git
from zemble.graph.store import build_graph, connect, graph_db_path

_PIPELINE = "src/main/java/app/pipeline"
CORE = {
    f"{_PIPELINE}/GenerationHandle.java": (
        "package app.pipeline;\n\npublic interface GenerationHandle {\n    Object result();\n}\n"
    ),
    f"{_PIPELINE}/Resulting.java": "package app.pipeline;\n\npublic interface Resulting {\n    Object result();\n}\n",
    f"{_PIPELINE}/DefaultGenerationHandle.java": (
        "package app.pipeline;\n\npublic class DefaultGenerationHandle implements GenerationHandle, Resulting {\n"
        "    public Object result() { return null; }\n}\n"
    ),
}
#: A class of two interfaces whose compiler facts name only one override: the facts are the truth for it.
DUAL = "src/main/java/app/extra/Dual.java"
EXTRA = {
    DUAL: (
        "package app.extra;\n\nimport app.pipeline.GenerationHandle;\nimport app.pipeline.Resulting;\n\n"
        "public class Dual implements GenerationHandle, Resulting {\n    public Object result() { return null; }\n}\n"
    ),
}
#: The lane's own class of two interfaces, whose rows only the lane's overlay holds.
LANE_HANDLE = (
    "package app.pipeline;\n\npublic class LaneHandle implements GenerationHandle, Resulting {\n"
    "    public Object result() { return null; }\n}\n"
)


def _repo(directory: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "main", str(directory)], check=True)
    git(directory, "add", "-A")
    git(directory, "commit", "-q", "-m", "initial")


def _write_facts(extra: Path) -> None:
    lines = [
        {"zemble_facts": 1, "tool": "javac-facts", "tool_version": "0.1.0", "generated_at": "2026-11-21T09:00:00Z"}
        | {"language": "java", "root": "../.."},
        {"t": "file", "path": DUAL, "sha256": hashlib.sha256((extra / DUAL).read_bytes()).hexdigest()},
        {"t": "override", "from": "app.extra.Dual#result()", "to": "app.pipeline.GenerationHandle#result()"},
    ]
    facts = extra / "build/zemble/facts-compileJava.jsonl"
    facts.parent.mkdir(parents=True)
    facts.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def _overrides(root: Path) -> dict[str, set[tuple[str, str]]]:
    """Every OVERRIDES edge the graph answers, as overriding type -> (overridden type, source)."""
    connection = connect(str(root))
    try:
        rows = connection.execute("SELECT src_id, dst_id, source FROM edges WHERE kind = 'overrides'").fetchall()
    finally:
        connection.close()
    found: dict[str, set[tuple[str, str]]] = {}
    for src, dst, source in rows:
        found.setdefault(_type_of(src), set()).add((_type_of(dst), source or "tree-sitter"))
    return found


def _type_of(symbol_id: str) -> str:
    return symbol_id.split("#", 1)[1].rsplit(".", 1)[0]


def _as_an_older_zemble_left_it(db: Path, owner: str, *, base_db: Path | None = None) -> None:
    """Drop the override edge an older derivation never wrote (*owner* -> Resulting), and the record of the step."""
    connection = sqlite3.connect(db.as_uri(), uri=True)
    try:
        if base_db is not None:
            connection.execute("ATTACH DATABASE ? AS base", (f"{base_db.as_uri()}?mode=ro",))
        refs = ["main.refs", *(["base.refs"] if base_db is not None else [])]

        def keys(pattern: str) -> list[int]:
            return [
                key
                for table in refs
                for (key,) in connection.execute(f"SELECT key FROM {table} WHERE text LIKE ?", (pattern,))  # noqa: S608
            ]

        sources, targets = keys(f"%#app.pipeline.{owner}.result%"), keys("%#app.pipeline.Resulting.result%")
        dropped = connection.execute(
            f"DELETE FROM main.edge_rows WHERE kind = 'overrides' AND src IN ({','.join(map(str, sources))}) "  # noqa: S608
            f"AND dst IN ({','.join(map(str, targets))})"
        ).rowcount
        assert dropped == 1, f"the second-branch edge of {owner} was stored"
        connection.execute("DELETE FROM main.meta WHERE key = 'overrides_per_branch'")
        connection.commit()
    finally:
        connection.close()


@pytest.fixture
def workspace(tmp_path: Path, graph_cache: Path) -> tuple[Path, Path]:
    """A base of two checkouts (one with compiler facts) and a lane holding the first with a class of its own."""
    base = (tmp_path / "ws").resolve()
    _repo(base / "core", CORE)
    _repo(base / "extra", EXTRA)
    _write_facts(base / "extra")
    build_graph(str(base))
    lane = (tmp_path / "lane").resolve()
    lane.mkdir()
    git(base / "core", "worktree", "add", "-q", "--detach", str(lane / "core"))
    (lane / "core" / _PIPELINE / "LaneHandle.java").write_text(LANE_HANDLE, encoding="utf-8")
    build_graph(str(lane))
    return base, lane


def test_an_older_graph_gets_every_branchs_override_back_once_without_a_re_extract(
    workspace: tuple[Path, Path],
) -> None:
    """The first write of new code re-derives the tree-sitter overrides per branch, facts and lanes respected."""
    base, lane = workspace
    both = {("app.pipeline.GenerationHandle", "tree-sitter"), ("app.pipeline.Resulting", "tree-sitter")}
    facts_only = {("app.pipeline.GenerationHandle", "javac-facts")}
    full = _overrides(lane)
    assert full["app.pipeline.DefaultGenerationHandle"] == both and full["app.pipeline.LaneHandle"] == both, full
    assert _overrides(base)["app.extra.Dual"] == facts_only, "precondition: the compiler names one override of Dual"

    # 1. The stores as an older zemble left them: one edge per method, and no record of the step.
    base_db, lane_db = graph_db_path(str(base)), graph_db_path(str(lane))
    assert base_db is not None and lane_db is not None
    _as_an_older_zemble_left_it(Path(base_db), "DefaultGenerationHandle")
    _as_an_older_zemble_left_it(Path(lane_db), "LaneHandle", base_db=Path(base_db))
    old = _overrides(lane)
    assert ("app.pipeline.Resulting", "tree-sitter") not in old["app.pipeline.LaneHandle"] | old[
        "app.pipeline.DefaultGenerationHandle"
    ]

    # 2. The lane's next refresh re-derives its own rows only; the base's rows are the base's to fix.
    stats = build_graph(str(lane), changed_paths=[])
    assert stats.extracted_files == 0, "2: nothing is extracted"
    after = _overrides(lane)
    assert after["app.pipeline.LaneHandle"] == both, after
    assert ("app.pipeline.Resulting", "tree-sitter") not in after["app.pipeline.DefaultGenerationHandle"]

    # 3. The base's next refresh re-derives its own; the facts-covered class keeps exactly what the compiler said.
    stats = build_graph(str(base), changed_paths=[])
    assert stats.extracted_files == 0 and stats.reresolved_files == 0, "3: nothing is extracted or re-resolved"
    fixed = _overrides(base)
    assert fixed["app.pipeline.DefaultGenerationHandle"] == both, fixed
    assert fixed["app.extra.Dual"] == facts_only, "3: compiler facts are never re-derived"

    # 4. It runs once: an edge dropped again after the step stays dropped by the next refresh.
    connection = sqlite3.connect(base_db)
    try:
        connection.execute(
            "DELETE FROM edge_rows WHERE kind = 'overrides' AND dst IN "
            "(SELECT key FROM refs WHERE text LIKE '%#app.pipeline.Resulting.result%')"
        )
        connection.commit()
    finally:
        connection.close()
    build_graph(str(base), changed_paths=[])
    assert ("app.pipeline.Resulting", "tree-sitter") not in _overrides(base)["app.pipeline.DefaultGenerationHandle"]
