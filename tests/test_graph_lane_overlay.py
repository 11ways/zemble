"""A lane's graph is an overlay of its base's: it answers exactly what a whole graph with the lane's edits would."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.conftest import git
from zemble.graph import SqliteGraphProvider, build_graph
from zemble.graph.facts import TREE_SITTER_SOURCE
from zemble.graph.model import EdgeKind
from zemble.graph.store import connect, graph_db_path, graph_stale, resolve_graph_root
from zemble.types import ContentType

BASE = """package core;

import other.Remote;

public class Base {
    public int value() { return Remote.answer(); }
    public int doubled() { return value() * 2; }
}
"""
BASE_EDITED = """package core;

import other.Remote;

public class Base {
    public int value() { return Remote.answer(); }
    public int tripled() { return value() * 3; }
    public int laneOnly() { return tripled(); }
}
"""
ADDED = """package core;

import other.Remote;

public class Added {
    public int sum(Base base) { return base.tripled() + Remote.answer(); }
}
"""
USER = """package extra;

import core.Base;
import core.Gone;

public class User extends Base {
    @Override
    public int value() { return 3; }
    public int use(Base base) { return base.doubled() + Gone.gone(); }
}
"""
USER_EDITED = USER.replace("base.doubled() + Gone.gone()", "base.tripled() + base.laneOnly()")
REMOTE = """package other;

import core.Base;

public class Remote {
    public static int answer() { return 42; }
    public int callBack(Base base) { return base.doubled(); }
}
"""
#: Writes the name the lane moves (`doubled`) but means its own method, so it resolves as before.
LONELY = "package other;\n\npublic class Lonely {\n    void only() { doubled(); }\n    int doubled() { return 2; }\n}\n"
GONE = "package core;\n\npublic class Gone {\n    public static int gone() { return 1; }\n}\n"
#: The base renames a method the lane's own files and its unchanged ones call.
REMOTE_MOVED = REMOTE.replace("public static int answer()", "public static int answerTwo()")
WORKSPACE = {
    "core": {
        "src/main/java/core/Base.java": BASE,
        "src/main/java/core/Gone.java": GONE,
    },
    "extra": {
        "src/main/java/extra/User.java": USER,
        "src/test/java/extra/UserTest.java": (
            "package extra;\n\npublic class UserTest {\n    void test() { new User().use(null); }\n}\n"
        ),
    },
    "other": {
        "src/main/java/other/Remote.java": REMOTE,
        "src/main/java/other/Lonely.java": LONELY,
    },
}
HELD = ("core", "extra")


def _repo(directory: Path, files: dict[str, str]) -> None:
    """Write files into a fresh repository and commit them."""
    for name, text in files.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "main", str(directory)], check=True)
    git(directory, "add", "-A")
    git(directory, "commit", "-q", "-m", "initial")


def _lane_edits(root: Path) -> None:
    """The lane's edits: one file changed (a name moved), one added, one deleted."""
    (root / "core/src/main/java/core/Base.java").write_text(BASE_EDITED, encoding="utf-8")
    (root / "core/src/main/java/core/Added.java").write_text(ADDED, encoding="utf-8")
    (root / "core/src/main/java/core/Gone.java").unlink()


def _in_scope(path: str) -> bool:
    return path.split("/", 1)[0] in HELD


def _answers(path: Path) -> dict[str, object]:
    """Everything the graph says about the lane's checkouts, as comparable values."""
    provider = SqliteGraphProvider(str(path))
    try:
        symbols = sorted(
            (s.id, s.kind.value, s.name, s.qualified_name, s.start_line, s.end_line, s.container_id, s.signature)
            for s in provider.all_symbols()
            if _in_scope(s.file_path)
        )
        edges = sorted(
            (e.src_id, e.dst_id or "", e.dst_name or "", e.kind.value, e.line, e.resolution.value, e.source)
            for e in provider.outgoing_edges(list(EdgeKind))
            if _in_scope(e.src_id)
        )
        callers = {
            symbol: sorted(
                (hit.symbol.id, hit.line, hit.binding.value, hit.through.id if hit.through is not None else "")
                for hit in provider.callers(symbol)
                if _in_scope(hit.symbol.id)
            )
            for symbol, *_fields in symbols
        }
        measured = provider.file_measurements()
        files = sorted((relative, metrics.lines) for relative, metrics, _test in measured if _in_scope(relative))
        metrics = sorted((s.id, m.lines, m.complexity) for s, m in provider.all_metrics() if _in_scope(s.file_path))
        calls = sorted(
            (tally.caller_id, tally.callee_id, tally.resolution.value, tally.calls)
            for tally in provider.call_tallies()
            if _in_scope(tally.caller_id)
        )
        dependencies = sorted(
            (dependency.source_path, dependency.target_path, dependency.kind.value, dependency.edges)
            for dependency in provider.file_dependencies(list(EdgeKind))
            if _in_scope(dependency.source_path)
        )
        return {
            "symbols": symbols,
            "edges": edges,
            "callers": callers,
            "files": files,
            "metrics": metrics,
            "calls": calls,
            "dependencies": dependencies,
        }
    finally:
        provider.close()


@pytest.fixture
def lanes(tmp_path: Path, graph_cache: Path) -> tuple[Path, Path, Path]:
    """A base workspace of three checkouts, a lane holding two of them, and the reference copy of the base."""
    workspace = (tmp_path / "ws").resolve()
    for name, files in WORKSPACE.items():
        _repo(workspace / name, files)
    _write_facts(workspace)
    build_graph(str(workspace))
    lane = (tmp_path / "lane").resolve()
    lane.mkdir()
    for name in HELD:
        git(workspace / name, "worktree", "add", "-q", "--detach", str(lane / name))
    reference = (tmp_path / "reference").resolve()
    shutil.copytree(workspace, reference, symlinks=True)
    return workspace, lane, reference


def _write_facts(workspace: Path) -> None:
    """A build's facts about `User`, untracked as build output is, so a lane has none of its own."""
    user = "src/main/java/extra/User.java"
    lines = [
        {"zemble_facts": 1, "tool": "javac-facts", "tool_version": "0.1.0", "generated_at": "2026-11-04T09:00:00Z"}
        | {"language": "java", "root": "../.."},
        {"t": "file", "path": user, "sha256": hashlib.sha256((workspace / "extra" / user).read_bytes()).hexdigest()},
        {"t": "call", "from": "extra.User#use(core.Base)", "to": "core.Base#doubled()", "path": user, "line": 9},
        {"t": "call", "from": "extra.User#use(core.Base)", "to": "core.Gone#gone()", "path": user, "line": 9},
        {"t": "override", "from": "extra.User#value()", "to": "core.Base#value()"},
    ]
    facts = workspace / "extra/build/zemble/facts-compileJava.jsonl"
    facts.parent.mkdir(parents=True)
    facts.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def _own_rows(lane: Path) -> tuple[set[str], int]:
    """The files whose rows the lane's overlay store holds itself, and its symbol count."""
    db_path = graph_db_path(str(lane))
    assert db_path is not None
    connection = sqlite3.connect(db_path)
    try:
        files = {row[0] for row in connection.execute("SELECT path FROM files")}
        return files, connection.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    finally:
        connection.close()


def _own_edge_files(lane: Path) -> set[str]:
    """The files whose edges the lane's overlay store holds itself, their keys read through the layered `refs`."""
    connection = connect(str(lane))
    try:
        query = "SELECT DISTINCT r.text FROM main.edge_rows e JOIN refs r ON r.key = e.file"
        return {row[0] for row in connection.execute(query)}
    finally:
        connection.close()


def test_a_lane_overlay_answers_what_a_whole_graph_of_its_edits_would(lanes: tuple[Path, Path, Path]) -> None:
    """Overlay journey: first build, a lane edit, a base that moved, a base version that went away."""
    workspace, lane, reference = lanes

    # 1. The first build extracts the lane's edits only, stores only their rows, and answers like a whole graph.
    _lane_edits(lane)
    _lane_edits(reference)
    stats = build_graph(str(lane))
    assert stats.extracted_files == 2, "1: only the changed and the added file are extracted"
    assert stats.removed_files == 1, "1: the file the lane deleted is removed; the checkout it lacks is not"
    owned, own_symbols = _own_rows(lane)
    assert owned == {"core/src/main/java/core/Base.java", "core/src/main/java/core/Added.java"}, "1: own rows only"
    assert own_symbols < 10, "1: the overlay holds its own symbols, not its base's"
    assert stats.restored_files >= 1, "1: a dependent that resolves as before is left to the base"
    assert "extra/src/main/java/extra/User.java" in _own_edge_files(lane), "1: one that resolves anew is stored"
    assert "other/src/main/java/other/Lonely.java" not in _own_edge_files(lane), "1: and none of its rows stored"
    build_graph(str(reference))
    assert _answers(lane) == _answers(reference), "1: the overlay answers what a whole graph with the edits does"
    facts_edges = [edge for edge in _answers(lane)["edges"] if edge[6] != TREE_SITTER_SOURCE]
    assert facts_edges, "1: the base's facts still describe the lane's sources it holds unchanged"
    provider = SqliteGraphProvider(str(lane))
    try:
        assert not provider.definition("Lonely"), "1: a checkout the lane does not hold is out of its scope"
        assert provider.definition("laneOnly"), "1: the lane's edit is in its graph"
        (value,) = [symbol for symbol in provider.definition("User.value") if symbol.file_path.startswith("extra/")]
        inherited = {
            (hit.symbol.qualified_name, hit.binding.value, hit.through.qualified_name)
            for hit in provider.callers(value.id)
            if hit.through is not None
        }
        assert ("core.Base.tripled", "overridden", "core.Base.value") in inherited, (
            "1: a call the lane's edit binds to the overridden declaration reaches its override"
        )
        assert all(_in_scope(path) for path, _ in [(p[len(str(lane)) + 1 :], t) for p, t in provider.source_files()])
    finally:
        provider.close()
    base = SqliteGraphProvider(str(workspace))
    try:
        assert not base.definition("laneOnly") and base.definition("doubled"), "1: the base's graph is untouched"
    finally:
        base.close()

    # 2. A lane edit refreshes the overlay in place from the change set.
    for root in (lane, reference):
        (root / "extra/src/main/java/extra/User.java").write_text(USER_EDITED, encoding="utf-8")
    edited = [lane / "extra/src/main/java/extra/User.java"]
    stats = build_graph(str(lane), changed_paths=edited)
    assert stats.extracted_files == 1, "2: the edited file alone is extracted"
    build_graph(str(reference), changed_paths=[reference / "extra/src/main/java/extra/User.java"])
    assert _answers(lane) == _answers(reference), "2: and the overlay still answers like the whole graph"

    # 3. The base moving (refreshed in place) makes the overlay stale; it is rebuilt on the new base.
    assert not graph_stale(str(lane)), "3: an overlay on the current base is fresh"
    for root in (workspace, reference):
        (root / "other/src/main/java/other/Remote.java").write_text(REMOTE_MOVED, encoding="utf-8")
    build_graph(str(workspace), changed_paths=[workspace / "other/src/main/java/other/Remote.java"])
    assert graph_stale(str(lane)), "3: a base refreshed in place leaves the overlay stale"
    build_graph(str(reference), changed_paths=[reference / "other/src/main/java/other/Remote.java"])
    stats = build_graph(str(lane))
    assert stats.extracted_files == 3, "3: a rebuilt overlay extracts the lane's own edits again, nothing else"
    assert not graph_stale(str(lane)), "3: and is fresh"
    assert _answers(lane) == _answers(reference), "3: answering like the whole graph on the moved base"

    # 4. A base whose version is replaced and retired: the base answers for the lane until the overlay is rebuilt.
    build_graph(str(workspace), force=True)
    assert graph_stale(str(lane)), "4: a new base version leaves the overlay stale"
    meanwhile = SqliteGraphProvider(str(lane))
    try:
        assert meanwhile.definition("Lonely"), "4: the base's graph answers meanwhile, whole"
    finally:
        meanwhile.close()
    build_graph(str(lane))
    assert _answers(lane) == _answers(reference), "4: the rebuilt overlay answers like the whole graph again"
    assert resolve_graph_root(str(lane)) == (str(lane), None), "4: from the lane's own graph"


def test_a_lane_graph_an_older_zemble_copied_whole_is_replaced_by_an_overlay(lanes: tuple[Path, Path, Path]) -> None:
    """A lane graph that is a whole graph is stale and its next build replaces it with an overlay."""
    _workspace, lane, _reference = lanes
    folder = graph_db_path(str(lane))
    assert folder is None, "a fresh lane has no graph of its own"
    # The shape an older zemble left: a whole graph built under the lane's root.
    from zemble.graph import store

    lane_root = store._local_root(str(lane))
    store._build_into(lane_root, store._graph_dir(str(lane)), None, force=True, workers=1, changed_paths=None)
    assert graph_stale(str(lane)), "a whole lane graph is stale"
    build_graph(str(lane))
    assert not graph_stale(str(lane)), "its next build is an overlay"
    owned, _symbols = _own_rows(lane)
    assert owned == set(), "a lane without edits owns no rows"


def test_a_lane_answers_with_its_edits_before_its_dependents_and_a_stopped_pass_is_finished_later(
    lanes: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first pass publishes the lane's own files; dependents owed by a stopped worker are re-resolved next time."""
    from zemble.graph import overlay, store

    _workspace, lane, reference = lanes
    _lane_edits(lane)
    _lane_edits(reference)
    build_graph(str(reference))

    # 1. A worker stopped after publishing the lane's own files, before re-resolving their dependents.
    def stopped(**_kwargs: object) -> None:
        raise KeyboardInterrupt("worker stopped")

    resolve_owed = store._resolve_owed
    monkeypatch.setattr(store, "_resolve_owed", stopped)
    with pytest.raises(KeyboardInterrupt):
        build_graph(str(lane))
    monkeypatch.setattr(store, "_resolve_owed", resolve_owed)
    provider = SqliteGraphProvider(str(lane))
    try:
        assert provider.definition("laneOnly"), "1: the lane answers with its own edits already"
        owed = provider.meta().get(overlay.META_PENDING)
    finally:
        provider.close()
    assert owed and "doubled" in owed, "1: and records the moved names it still owes"

    # 2. The next refresh re-resolves what was owed and answers like the whole graph.
    stats = build_graph(str(lane))
    assert stats.extracted_files == 0, "2: nothing is extracted again"
    assert stats.published_seconds is not None and stats.reresolved_files >= 1, "2: the owed dependents are resolved"
    assert _answers(lane) == _answers(reference), "2: the overlay answers like the whole graph"
    provider = SqliteGraphProvider(str(lane))
    try:
        assert overlay.META_PENDING not in provider.meta(), "2: and owes nothing more"
    finally:
        provider.close()


def test_the_worker_reports_a_lane_overlays_first_pass_before_it_finishes(lanes: tuple[Path, Path, Path]) -> None:
    """The real graph worker prints the first pass's publish as its own line, ahead of the finished report."""
    import os
    import sys

    _workspace, lane, _reference = lanes
    _lane_edits(lane)
    completed = subprocess.run(
        [sys.executable, "-m", "zemble.daemon.graph_worker", str(lane), "2048"],
        input=json.dumps({"changed_paths": None}),
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "ZEMBLE_DAEMON": "0"},
    )
    published, finished = (json.loads(line) for line in completed.stdout.splitlines())
    assert published["report"] == "published" and finished["report"] == "finished", completed.stdout
    first = published["stats"]
    assert first["extracted_files"] == 2 and first["removed_files"] == 1, first
    assert first["owed_names"] >= 1 and first["published_seconds"] is not None, "the first pass owes the moved names"
    assert finished["stats"]["reresolved_files"] > first["reresolved_files"], "the second pass re-resolved dependents"


#: A lane edit that adds a documented migration: the architectural channel's evidence.
ADDED_DEPRECATED = ADDED.replace(
    "\n}\n",
    "\n    /** @deprecated use {@link #sum(Base)} instead. */\n"
    "    @Deprecated\n"
    "    public int total(Base base) { return sum(base); }\n}\n",
)


@pytest.fixture
def parked(tmp_path: Path, graph_cache: Path) -> tuple[Path, Path, Path]:
    """A lane parked inside its base (`<base>/lanes/lane`) whose root holds build tool files the base never walks."""
    workspace = (tmp_path / "ws").resolve()
    for name, files in WORKSPACE.items():
        _repo(workspace / name, files)
    _write_facts(workspace)
    # The real workspace's root ignore file keeps the run state out; a lane root is walked by it too.
    (workspace / ".gitignore").write_text(".zenit-dev/\n", encoding="utf-8")
    reference = (tmp_path / "reference").resolve()
    shutil.copytree(workspace, reference, symlinks=True)
    lane = workspace / "lanes" / "lane"
    lane.mkdir(parents=True)
    for name in HELD:
        git(workspace / name, "worktree", "add", "-q", "--detach", str(lane / name))
    # What zenit-dev writes at a lane's root, outside every checkout: the lane's, never the base's.
    (lane / ".zenit-dev").mkdir()
    (lane / ".zenit-dev/gradle-init.gradle").write_text("allprojects {\n}\n", encoding="utf-8")
    (lane / ".zenit-dev/lane.json").write_text('{"name": "lane"}\n', encoding="utf-8")
    for root in (lane, reference):
        _lane_edits(root)
        (root / "core/src/main/java/core/Added.java").write_text(ADDED_DEPRECATED, encoding="utf-8")
    build_graph(str(workspace))
    return workspace, lane, reference


def test_a_lane_parked_inside_its_base_is_its_own_overlay_never_a_sub_tree_of_the_base(
    parked: tuple[Path, Path, Path], graph_cache: Path
) -> None:
    """The base's graph holds none of the lane root's own files; the lane is answered as a lane, never a sub-tree."""
    from zemble.cache_orphans import OrphanKind, find_orphans
    from zemble.graph.cli import ensure_graph
    from zemble.graph.store import graph_covers, graph_present

    workspace, lane, reference = parked
    assert not graph_covers(str(workspace), "lanes/lane"), "the base's graph holds no file under the lane root"

    # 1. Before its overlay exists the base's graph answers for the lane whole, not as the lane root's sub-tree.
    assert resolve_graph_root(str(lane)) == (str(workspace), None), "1: the lane is a lane of its base"

    # 2. Ensuring the lane's graph builds the lane's overlay, which then answers like a whole graph of the edits.
    ensure_graph(str(lane), allow_daemon=False)
    assert graph_present(str(lane)), "2: the lane has a graph of its own"
    assert resolve_graph_root(str(lane)) == (str(lane), None), "2: and answers from it"
    build_graph(str(reference))
    assert _answers(lane) == _answers(reference), "2: what a whole graph with the lane's edits answers"

    # 3. A sub-path of the lane is a sub-tree of the lane's graph.
    assert resolve_graph_root(str(lane / "core")) == (str(lane), "core"), "3: a lane sub-path is the lane's"

    # 3b. The lane root's run state is out of the lane's graph, walked or named in a change set.
    assert not graph_covers(str(lane), ".zenit-dev"), "3b: the lane walk honours the base root's ignore file"
    (lane / ".zenit-dev/gradle-init-2.gradle").write_text("allprojects {\n}\n", encoding="utf-8")
    build_graph(str(lane), changed_paths=[lane / ".zenit-dev/gradle-init-2.gradle"])
    assert not graph_covers(str(lane), ".zenit-dev"), "3b: and so does a change set"

    # 4. The lane's graph is no orphan answered from the base, so a prune keeps it.
    orphans = [orphan for orphan in find_orphans(graph_cache) if orphan.kind is OrphanKind.GRAPH_UNDER_ANCESTOR]
    assert not orphans, "4: the lane's graph is kept"


def _candidates(payload: dict) -> set[tuple]:
    """The candidates among the lane's checkouts as comparable (evidence kind, member files and names) rows.

    A lane's graph hides the checkouts it does not hold, so a counterpart there is the whole graph's alone.
    """
    return {
        (candidate["evidence_kind"], tuple((m["file_path"], m["name"]) for m in candidate["members"]))
        for candidate in payload["candidates"]
        if all(_in_scope(member["file_path"]) for member in candidate["members"])
    }


@pytest.mark.anyio
async def test_a_fresh_lane_parked_inside_its_warm_base_answers_architectural_like_a_whole_graph(
    parked: tuple[Path, Path, Path],
) -> None:
    """A push hook's architectural ask of a fresh lane answers (never fails) and, once built, what the edits answer."""
    from tests.test_lanes import _daemon

    workspace, lane, reference = parked
    # A lane created a minute ago: one worktree and the build tool's state file, nothing the base's code index holds.
    fresh = workspace / "lanes" / "fresh"
    (fresh / ".zenit-dev").mkdir(parents=True)
    git(workspace / "core", "worktree", "add", "-q", "--detach", str(fresh / "core"))
    (fresh / ".zenit-dev/lane.properties").write_text("name=fresh\n", encoding="utf-8")
    daemon = _daemon()
    every = [content.value for content in ContentType]

    def request(path: Path, **args: object) -> dict:
        return {"id": 1, "cmd": "architectural", "args": {"path": str(path), **args}, "accepts_busy": True}

    # 0. The base is warm with every content type and holds none of the lane roots' own files.
    base_key, base = await daemon.index_for({"path": str(workspace), "content": every})
    assert base.subtree("lanes/lane") is None and base.subtree("lanes/fresh") is None, "0: no lane files"

    # 1. A fresh lane's first question answers from its overlay, never as the base's sub-tree of its root.
    answered_fresh = await daemon.handle(request(fresh))
    assert answered_fresh["ok"], f"1: a fresh lane answers its first question: {answered_fresh.get('error')}"
    lane_key, _served = await daemon.index_for({"path": str(lane)})
    assert lane_key[0] == str(lane) and daemon.cache.base_of(lane_key) == base_key, "1: served by the lane's overlay"
    first = await daemon.handle(request(lane))
    assert first["ok"], f"1: {first.get('error')}"
    job = daemon.graphs.jobs.get(str(lane))
    assert job is not None or str(lane) in daemon.graphs.ready, "1: the lane's own graph is built, not the base's"
    if job is not None:
        await job

    # 2. Once its overlay is published the lane answers what a whole graph of its edits answers.
    answered = await daemon.handle(request(lane))
    expected = await daemon.handle(request(reference, paths=list(HELD)))
    assert answered["ok"] and expected["ok"], (answered.get("error"), expected.get("error"))
    assert _candidates(answered["result"]), "2: the lane's documented migration is a candidate"
    assert _candidates(answered["result"]) == _candidates(expected["result"]), "2: as a whole graph answers"
    await daemon.graphs.close()


def test_a_lane_s_dependents_pass_keeps_one_batch_of_base_edges_in_its_scratch_database(
    lanes: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The base files a lane re-resolves are read a batch at a time, and those resolving as before leave at once."""
    from zemble.graph import store
    from zemble.graph.lookup import HIERARCHY_KINDS

    _workspace, lane, _reference = lanes
    _lane_edits(lane)
    monkeypatch.setattr(store, "_BATCH_FILES", 1)
    seen: list[tuple[set[str], set[str]]] = []
    staged = store._staged
    base_files = {"extra/src/main/java/extra/User.java", "other/src/main/java/other/Remote.java"}
    base_files |= {"other/src/main/java/other/Lonely.java", "extra/src/test/java/extra/UserTest.java"}

    def watched(connection: sqlite3.Connection, paths: list[str], *, hierarchy: bool) -> list:
        if not hierarchy:
            marks = ",".join("?" * len(HIERARCHY_KINDS))
            pending = connection.execute(
                f"SELECT DISTINCT file_path FROM scratch.pending WHERE kind NOT IN ({marks})", HIERARCHY_KINDS
            )
            resolved = connection.execute("SELECT DISTINCT file_path FROM scratch.resolved")
            seen.append(({row[0] for row in pending} & base_files, {row[0] for row in resolved} & base_files))
        return staged(connection, paths, hierarchy=hierarchy)

    monkeypatch.setattr(store, "_staged", watched)
    build_graph(str(lane))
    assert seen, "the dependents pass resolved its base files"
    assert all(not pending for pending, _resolved in seen), "no base file's edges wait in the scratch database"
    assert all("other/src/main/java/other/Lonely.java" not in resolved for _pending, resolved in seen), (
        "a base file resolving as before leaves the scratch database with its batch"
    )


def test_a_dependent_whose_ambiguous_calls_only_count_one_more_candidate_is_left_to_the_base(
    lanes: tuple[Path, Path, Path],
) -> None:
    """A declaration of a common name the lane adds changes no answer of a call that stays ambiguous."""
    workspace, lane, _reference = lanes
    goers = "package other;\n\npublic class {name} {{\n    public int go() {{ return 1; }}\n}}\n"
    for name in ("GoOne", "GoTwo"):
        (workspace / f"other/src/main/java/other/{name}.java").write_text(goers.format(name=name), encoding="utf-8")
    caller = "other/src/main/java/other/Caller.java"
    (workspace / caller).write_text(
        "package other;\n\npublic class Caller {\n    int call(Object some) { return some.go(); }\n}\n",
        encoding="utf-8",
    )
    build_graph(str(workspace))
    (lane / "core/src/main/java/core/GoThree.java").write_text(
        goers.format(name="GoThree").replace("package other;", "package core;"), encoding="utf-8"
    )
    build_graph(str(lane))
    assert caller not in _own_edge_files(lane), "a call ambiguous before and after is the base's to answer"
    connection = connect(str(workspace))
    try:
        (resolution,) = connection.execute(
            "SELECT resolution FROM edges WHERE file_path = ? AND dst_name = 'go'", (caller,)
        ).fetchone()
    finally:
        connection.close()
    assert resolution == "ambiguous", "the base answers it as the ambiguous call it is in the lane too"


def test_a_lane_s_own_facts_file_stores_the_symbol_facts_of_only_the_sources_it_changed(
    lanes: tuple[Path, Path, Path],
) -> None:
    """A lane's build rewrites a facts file its base has; the overlay keeps the base's rows of unchanged sources."""
    workspace, lane, _reference = lanes
    user, test = "src/main/java/extra/User.java", "src/test/java/extra/UserTest.java"

    def facts(root: Path, test_line: int, stamp: str) -> None:
        lines = [
            {"zemble_facts": 1, "tool": "javac-facts", "tool_version": "0.1.0", "generated_at": stamp}
            | {"language": "java", "root": "../.."},
            {"t": "file", "path": user, "sha256": hashlib.sha256((root / "extra" / user).read_bytes()).hexdigest()},
            {"t": "symbol", "ref": "extra.User", "path": user, "line": 9},
            {"t": "file", "path": test, "sha256": hashlib.sha256((root / "extra" / test).read_bytes()).hexdigest()},
            {"t": "symbol", "ref": "extra.UserTest", "path": test, "line": test_line},
        ]
        target = root / "extra/build/zemble/facts-compileJava.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    facts(workspace, 3, "2026-11-24T05:00:00Z")
    build_graph(str(workspace))
    facts(lane, 4, "2026-11-24T06:00:00Z")
    build_graph(str(lane))

    db_path = graph_db_path(str(lane))
    assert db_path is not None
    own = sqlite3.connect(db_path)
    try:
        stored = sorted(own.execute("SELECT ref, file_path, line FROM facts_symbols").fetchall())
    finally:
        own.close()
    assert stored == [("extra.UserTest", f"extra/{test}", 4)], "the overlay stores the changed source's facts only"
    connection = connect(str(lane))
    try:
        layered = sorted(tuple(row) for row in connection.execute("SELECT ref, file_path, line FROM facts_symbols"))
    finally:
        connection.close()
    assert layered == [("extra.User", f"extra/{user}", 9), ("extra.UserTest", f"extra/{test}", 4)], (
        "and the lane reads the base's facts of the source it did not change beside its own"
    )


def test_the_facts_files_an_overlay_holds_are_read_off_the_index_flat_as_the_base_s_facts_grow(
    lanes: tuple[Path, Path, Path],
) -> None:
    """A build asks which facts files hold `symbol` facts; over a lane overlay that must not scan the base's rows.

    A DISTINCT over the layered view judged both masks of every base row: 24 s per overlay build of the zenit
    workspace (1.09M rows), every time its base republished.
    """
    from zemble.graph import store

    workspace, lane, _reference = lanes
    build_graph(str(lane))
    facts_file = "extra/build/zemble/facts-compileJava.jsonl"
    base_db = graph_db_path(str(workspace))
    assert base_db is not None

    def grow_base(count: int, start: int) -> None:
        writer = sqlite3.connect(base_db)
        try:
            writer.executemany(
                "INSERT INTO facts_symbols (ref, file_path, line, facts_file, language) VALUES (?,?,?,?,?)",
                [(f"gen.G{i}", f"extra/src/gen/G{i}.java", 1, facts_file, "java") for i in range(start, start + count)],
            )
            writer.commit()
        finally:
            writer.close()

    def steps(query: Callable[[sqlite3.Connection], set[str]]) -> tuple[set[str], int]:
        connection = connect(str(lane))
        ticks = [0]

        def tick() -> int:
            ticks[0] += 1
            return 0

        try:
            connection.set_progress_handler(tick, 1000)
            return query(connection), ticks[0]
        finally:
            connection.close()

    def naive(connection: sqlite3.Connection) -> set[str]:
        return {row[0] for row in connection.execute("SELECT DISTINCT facts_file FROM facts_symbols")}

    grow_base(2_000, 0)
    small, small_steps = steps(store._stored_facts_files)
    naive_small, naive_small_steps = steps(naive)
    grow_base(38_000, 2_000)
    large, large_steps = steps(store._stored_facts_files)
    naive_large, naive_large_steps = steps(naive)

    assert small == naive_small and large == naive_large, "it answers what the DISTINCT over the view answers"
    assert facts_file in large, "the base's facts file is held through the overlay"
    assert naive_large_steps > 10 * max(1, naive_small_steps), "the DISTINCT over the view grows with the base's rows"
    assert large_steps <= small_steps + 5, (
        f"THE ASSERTION -- flat as the base's facts grow 20-fold: {small_steps} then {large_steps} thousand steps"
    )


GENERATED_FIXTURE = Path(__file__).parent / "fixtures" / "generated"
TEMPLATE = "src/common/templates/widget-display/markdown.hwk"
FUNCTIONS = "src/common/java/com/example/ui/WidgetFunctions.java"
GENERATED_ROOT = "build/generated-sources/hawkeye/common/java/be/elevenways/hawkeye/generated"
TPL = f"{GENERATED_ROOT}/zenitwidget/Tpl_WidgetDisplayMarkdown.java"
TPL_FROM = (
    "be.elevenways.hawkeye.generated.zenitwidget.Tpl_WidgetDisplayMarkdown"
    "#renderRoot(be.elevenways.hawkeye.common.render.RenderContext)"
)
TPL_BRANCH = (
    "be.elevenways.hawkeye.generated.zenitwidget.Tpl_WidgetDisplayMarkdown"
    "#branch2(be.elevenways.hawkeye.common.render.RenderContext)"
)


def _template_calls(root: Path) -> list[tuple]:
    """The template's call edges as one graph answers them."""
    connection = connect(str(root))
    try:
        rows = connection.execute(
            "SELECT dst_id, dst_name, line, resolution, source FROM edges WHERE file_path = ? AND kind = 'calls'",
            (f"mod/{TEMPLATE}",),
        ).fetchall()
    finally:
        connection.close()
    return sorted(tuple(row) for row in rows)


def test_a_lane_reads_the_build_output_it_lacks_where_its_base_s_facts_were_written(
    tmp_path: Path, graph_cache: Path
) -> None:
    """A lane that never built its module keeps the base's template facts, and stores none of their rows itself."""
    import os

    workspace = (tmp_path / "ws").resolve()
    module = workspace / "mod"
    shutil.copytree(GENERATED_FIXTURE, module)
    (module / ".gitignore").write_text("build/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "main", str(module)], check=True)
    git(module, "add", "-A")
    git(module, "commit", "-q", "-m", "initial")
    # The base's build generated the class after its template was written, and wrote facts about it.
    os.utime(module / TEMPLATE, (1_000_000, 1_000_000))
    os.utime(module / TPL, (2_000_000, 2_000_000))
    lines = [
        {"zemble_facts": 1, "tool": "zemble-javac-facts", "tool_version": "0.1.0"}
        | {"generated_at": "2026-11-24T05:00:00Z", "language": "java", "root": "../.."},
        {"t": "file", "path": TPL, "sha256": hashlib.sha256((module / TPL).read_bytes()).hexdigest()},
        {"t": "call", "from": TPL_FROM, "to": "com.example.ui.WidgetFunctions#render(java.lang.String)", "line": 74}
        | {"path": TPL},
        {"t": "call", "from": TPL_BRANCH, "to": "com.example.ui.WidgetFunctions#render(java.lang.String)"}
        | {"path": TPL, "line": 83},
    ]
    facts = module / "build/zemble/facts-compileJava.jsonl"
    facts.parent.mkdir(parents=True)
    facts.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    build_graph(str(workspace))
    in_base = _template_calls(workspace)
    assert in_base and {row[4] for row in in_base} == {"zemble-javac-facts"}, "the base maps the template's calls"

    # 1. A lane checked out later (its template newer than the base's generated class) with no build output of its
    # own moves a name the template calls, so the template is re-resolved in the lane.
    lane = (tmp_path / "lane").resolve()
    lane.mkdir()
    git(module, "worktree", "add", "-q", "--detach", str(lane / "mod"))
    functions = lane / "mod" / FUNCTIONS
    functions.write_text(
        functions.read_text(encoding="utf-8").replace(
            '\n    @HawkeyeFunction(name = "localizedConfig"',
            "\n    public static String render(String body, int times) {\n        return body;\n    }\n"
            '\n    @HawkeyeFunction(name = "localizedConfig"',
        ),
        encoding="utf-8",
    )
    assert not (lane / "mod" / TPL).exists(), "1: the lane holds no generated class"
    build_graph(str(lane))

    # 2. The lane answers the template's calls as its base does: the facts still describe its unchanged template.
    assert _template_calls(lane) == in_base, "2: the base's facts describe the lane's unchanged template"
    assert f"mod/{TEMPLATE}" not in _own_edge_files(lane), "2: and the lane stores none of the template's rows"

    # 3. A template the lane edits is newer than every class the base generated: its facts no longer describe it.
    template = lane / "mod" / TEMPLATE
    template.write_text(template.read_text(encoding="utf-8") + "\n<p>lane</p>\n", encoding="utf-8")
    build_graph(str(lane), changed_paths=[template])
    assert {row[4] for row in _template_calls(lane)} == {TREE_SITTER_SOURCE}, "3: an edited template is the lane's"
