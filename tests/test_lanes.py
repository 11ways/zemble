"""Lanes: a folder of linked worktrees of an indexed workspace is answered from that workspace's index."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest

from tests.conftest import FakeEmbedder, git
from zemble.cache import cache_key, find_index_from_cache_folder, mark_root
from zemble.cache_orphans import OrphanKind, find_orphans, remove_orphan
from zemble.daemon import client, server
from zemble.daemon.protocol import QUERY_WAIT_ENV, CommandBusy, QueryTimedOut
from zemble.index.create import tree_moves_anything
from zemble.index.index import ZembleIndex
from zemble.index.overlay import OverlayIndex, build_overlay, overlay_folder
from zemble.index_cache import IndexCache, compute_cache_key
from zemble.lane import LaneComparer, find_lane
from zemble.types import ContentType

ALPHA = {
    "src/shared.py": "def shared_untouched_helper():\n    return 'kept in the lane'\n",
    "src/edited.py": "def edited_function():\n    return 'original body'\n",
    "src/removed.py": "def removed_in_lane_function():\n    return 1\n",
}
BETA = {"lib/beta_only.py": "def beta_only_function():\n    return 'beta'\n"}


def _repo(directory: Path, files: dict[str, str]) -> None:
    """Write files into a fresh repository and commit them."""
    for name, text in files.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "main", str(directory)], check=True)
    git(directory, "add", "-A")
    git(directory, "commit", "-q", "-m", "initial")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace folder, no repository itself, holding two independent checkouts."""
    root = (tmp_path / "ws").resolve()
    _repo(root / "alpha", ALPHA)
    _repo(root / "beta", BETA)
    (root / "README.md").write_text("# workspace\n", encoding="utf-8")
    return root


def _lane(workspace: Path, name: str, repos: tuple[str, ...] = ("alpha",)) -> Path:
    """Make a lane beside the workspace holding a linked worktree of each named checkout."""
    lane = workspace.parent / name
    lane.mkdir()
    for repo in repos:
        git(workspace / repo, "worktree", "add", "-q", "--detach", str(lane / repo))
    return lane.resolve()


def _embedded(embedder: FakeEmbedder) -> list[str]:
    """Every document text embedded since the calls were last cleared."""
    return [text for call in embedder.document_calls for text in call]


def _overlay(lane: Path, embedder: FakeEmbedder) -> OverlayIndex:
    """Build a lane's overlay on its base's stored index, as the in-process path does."""
    found = find_lane(lane, lambda base: True)
    assert found is not None
    base = ZembleIndex.from_path(found[0].base, embedder=embedder)
    return build_overlay(found[0], base, overlay_folder(find_index_from_cache_folder(str(lane), (ContentType.CODE,))))


def test_lane_identity_journey(workspace: Path, mock_embedder: FakeEmbedder, tmp_path: Path) -> None:
    """A lane is recognised by its worktrees; the workspace, a clone and an unindexed base are not lanes."""
    lane = _lane(workspace, "ws-lane")
    indexed = {str(workspace)}

    # 1. The lane root and a path inside it resolve to the workspace, with the right prefix.
    found = find_lane(lane, indexed.__contains__)
    assert found is not None and found[0].base == workspace and found[1] is None, "1: the lane root is a lane"
    assert [pair.prefix for pair in found[0].repos] == ["alpha"], "1: only the checkout it holds is matched"
    inner = find_lane(lane / "alpha" / "src", indexed.__contains__)
    assert inner is not None and inner[0].root == lane and inner[1] == "alpha/src", "1: a sub-path is a sub-tree"

    # 2. The workspace itself and its own checkouts are no lane.
    assert find_lane(workspace, indexed.__contains__) is None, "2: the base is not a lane of itself"
    assert find_lane(workspace / "alpha", indexed.__contains__) is None, "2: a main checkout is not a lane"

    # 3. A plain clone shares an origin but no object store: not a lane.
    clone = tmp_path / "clone"
    clone.mkdir()
    subprocess.run(["git", "clone", "-q", str(workspace / "alpha"), str(clone / "alpha")], check=True)
    assert find_lane(clone, indexed.__contains__) is None, "3: a clone is not a worktree"

    # 4. A lane whose base holds no index is no lane: it fails closed to an index of its own.
    assert find_lane(lane, lambda base: False) is None, "4: an unindexed base overlays nothing"

    # 5. A lane root that is itself a worktree of a workspace repository is a lane too.
    git(workspace / "beta", "worktree", "add", "-q", "--detach", str(tmp_path / "beta-lane"))
    rooted = find_lane(tmp_path / "beta-lane", {str(workspace / "beta")}.__contains__)
    assert rooted is not None and rooted[0].base == workspace / "beta", "5: a worktree root pairs with its main"


def test_comparer_asks_git_and_falls_back_to_bytes(workspace: Path) -> None:
    """Tracked unchanged files are the base's; edits on either side and untracked files are compared for real."""
    lane = _lane(workspace, "ws-lane")
    (lane / "alpha/src/edited.py").write_text("def edited_function():\n    return 'lane body'\n", encoding="utf-8")
    (lane / "alpha/src/untracked.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "alpha/src/shared.py").write_text(ALPHA["src/shared.py"] + "# base edit\n", encoding="utf-8")
    found = find_lane(lane, lambda base: True)
    assert found is not None
    comparer = LaneComparer(found[0])
    assert not comparer.same("alpha/src/edited.py"), "a lane edit differs"
    assert not comparer.same("alpha/src/shared.py"), "a base edit not committed differs too"
    assert comparer.same("alpha/src/removed.py"), "an untouched tracked file is the base's"
    assert not comparer.same("alpha/src/untracked.py"), "a file the base lacks differs"
    (lane / "README.md").write_text("# workspace\n", encoding="utf-8")
    assert comparer.same("README.md"), "a file outside every checkout is compared by its bytes"


def test_overlay_answers_with_the_lanes_edits_and_the_bases_rows(workspace: Path, mock_embedder: FakeEmbedder) -> None:
    """Edited and new files come from the delta, untouched ones from the base, all under the lane's root."""
    ZembleIndex.from_path(workspace, embedder=mock_embedder)
    lane = _lane(workspace, "ws-lane")
    (lane / "alpha/src/edited.py").write_text("def edited_function():\n    return 'lane body'\n", encoding="utf-8")
    (lane / "alpha/src/added.py").write_text("def added_in_lane_function():\n    return 2\n", encoding="utf-8")
    (lane / "alpha/src/removed.py").unlink()

    # 1. Only the edited and the added file are chunked and embedded; nothing of the base is.
    mock_embedder.document_calls.clear()
    overlay = _overlay(lane, mock_embedder)
    embedded = _embedded(mock_embedder)
    assert any("lane body" in text for text in embedded) and any("added_in_lane" in text for text in embedded), "1"
    assert not any("shared_untouched_helper" in text for text in embedded), "1: an unchanged file is never embedded"
    assert sorted(overlay.plan.differing) == ["alpha/src/added.py", "alpha/src/edited.py"], "1: the delta"

    # 2. The overlay holds exactly the lane's files: no removed file, no checkout the lane lacks.
    assert overlay.indexed_paths() == ["alpha/src/added.py", "alpha/src/edited.py", "alpha/src/shared.py"], "2"

    # 3. An edit changes results; an unedited file resolves from the base, under the lane's root.
    edited = overlay.search("edited_function", top_k=3)
    assert edited and edited[0].chunk.file_path == "alpha/src/edited.py", "3: the edited file is found"
    assert "lane body" in edited[0].chunk.content and "original body" not in edited[0].chunk.content, "3: as edited"
    shared = overlay.search("shared_untouched_helper", top_k=1)
    assert shared[0].chunk.file_path == "alpha/src/shared.py", "3: an unchanged file is answered"
    assert overlay._root == lane, "3: and named under the lane's root"
    assert not overlay.search("beta_only_function", top_k=5) or all(
        not result.chunk.file_path.startswith("beta/") for result in overlay.search("beta_only_function", top_k=5)
    ), "3: a checkout the lane does not hold never answers"
    assert not any(r.chunk.file_path == "alpha/src/removed.py" for r in overlay.search("removed_in_lane", top_k=5))

    # 4. Semantic ranking is exactly what an index of the lane built on its own ranks.
    reference = ZembleIndex.from_path(lane, embedder=FakeEmbedder())
    for query in ("edited function body", "shared helper", "added in lane"):
        ours = [(r.chunk.file_path, r.chunk.content) for r in overlay.search(query, top_k=3, alpha=1.0, rerank=False)]
        alone = [
            (r.chunk.file_path, r.chunk.content) for r in reference.search(query, top_k=3, alpha=1.0, rerank=False)
        ]
        assert ours == alone, f"4: {query!r} ranks as a lane index of its own"

    # 5. A sub-path of the lane is a sub-tree view of the overlay.
    view = overlay.subtree("alpha/src")
    assert view is not None and "edited.py" in view.indexed_paths(), "5: lane sub-paths speak their own paths"


def test_a_second_overlay_reuses_the_previous_delta(workspace: Path, mock_embedder: FakeEmbedder) -> None:
    """Re-planning a lane embeds only what moved since the last delta, and a reverted file returns to the base."""
    ZembleIndex.from_path(workspace, embedder=mock_embedder)
    lane = _lane(workspace, "ws-lane")
    (lane / "alpha/src/edited.py").write_text("def edited_function():\n    return 'lane body'\n", encoding="utf-8")
    first = _overlay(lane, mock_embedder)
    (lane / "alpha/src/added.py").write_text("def added_later():\n    return 3\n", encoding="utf-8")
    found = find_lane(lane, lambda base: True)
    assert found is not None
    mock_embedder.document_calls.clear()
    second = build_overlay(
        found[0], first.base, overlay_folder(find_index_from_cache_folder(str(lane), first.content)), first
    )
    embedded = _embedded(mock_embedder)
    assert any("added_later" in text for text in embedded), "the new file is embedded"
    assert not any("lane body" in text for text in embedded), "the file already in the delta is copied, not embedded"
    (lane / "alpha/src/edited.py").write_text(ALPHA["src/edited.py"], encoding="utf-8")
    third = build_overlay(
        found[0], first.base, overlay_folder(find_index_from_cache_folder(str(lane), first.content)), second
    )
    assert third.plan.differing == ["alpha/src/added.py"], "a reverted file is the base's again"


def test_index_cache_serves_a_lane_as_an_overlay_outside_the_slot_count(
    workspace: Path, mock_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache builds an overlay on its resident base, counts no slot for it, and drops it with its base."""
    ZembleIndex.from_path(workspace, embedder=mock_embedder)
    lane = _lane(workspace, "ws-lane")
    cache = IndexCache(max_size=1)
    monkeypatch.setattr("zemble.index_cache.load_embedder", lambda: mock_embedder)

    async def journey() -> None:
        await cache.load_embedder_once()
        key, index = await cache.get_with_key(str(lane))
        assert isinstance(index, OverlayIndex), "1: a lane is answered by an overlay"
        base_key = compute_cache_key(str(workspace))
        assert cache.base_of(key) == base_key, "1: on the resident base"
        assert len(cache._full_keys()) == 1, "2: the overlay takes no slot"
        _key, view = await cache.get_with_key(str(lane / "alpha"))
        assert "src/shared.py" in view.indexed_paths(), "3: a lane sub-path is served from the overlay"
        cache.evict(base_key)
        assert key not in cache._tasks, "4: evicting the base drops its overlays"

    asyncio.run(journey())


def test_lane_graph_is_seeded_from_the_base_graph(graph_fixture_root: Path, graph_cache: Path, tmp_path: Path) -> None:
    """A lane reads its base's graph until its own is seeded; the seed extracts only the lane's edits."""
    from zemble.graph import SqliteGraphProvider, build_graph
    from zemble.graph.provider import open_provider
    from zemble.graph.store import resolve_graph_root

    workspace = (tmp_path / "ws").resolve()
    files = {
        str(path.relative_to(graph_fixture_root)): path.read_text(encoding="utf-8")
        for path in graph_fixture_root.rglob("*.java")
    }
    _repo(workspace / "core", files)
    base = build_graph(str(workspace))
    lane = _lane(workspace, "ws-lane", ("core",))

    # 1. Before its own graph exists, the lane is answered from the base's, spelled under the lane.
    assert resolve_graph_root(str(lane)) == (str(workspace), None), "1: the base's graph answers meanwhile"
    provider = open_provider(str(lane))
    try:
        assert provider.source_files()[0][0].startswith(str(lane)), "1: absolute paths name the lane's files"
    finally:
        provider.close()

    # 2. The lane's first build copies the base's graph and extracts only the edited file.
    point = lane / "core/src/main/java/com/example/core/Point.java"
    text = point.read_text(encoding="utf-8")
    point.write_text(
        text.rstrip().removesuffix("}") + "    public int laneOnlyMethod() { return 1; }\n}\n", encoding="utf-8"
    )
    seeded = build_graph(str(lane))
    assert seeded.extracted_files == 1, "2: only the lane's edit is extracted"
    assert seeded.symbols == base.symbols + 1, "2: and the graph holds the base's symbols plus the new one"
    assert resolve_graph_root(str(lane)) == (str(lane), None), "2: the lane now answers from its own graph"
    lane_graph = SqliteGraphProvider(str(lane))
    try:
        assert lane_graph.definition("laneOnlyMethod"), "2: the lane's edit is in its graph"
        assert lane_graph.meta()["root"] == str(lane), "2: recorded under the lane's root"
    finally:
        lane_graph.close()
    base_graph = SqliteGraphProvider(str(workspace))
    try:
        assert not base_graph.definition("laneOnlyMethod"), "2: the base's graph is untouched"
    finally:
        base_graph.close()


def _daemon() -> server.Daemon:
    """A daemon without watchers whose cache embeds with the deterministic test embedder."""
    daemon = server.Daemon(watch=False)
    daemon.cache._embedder = FakeEmbedder()
    daemon.cache._model_ready.set()
    return daemon


@pytest.mark.anyio
async def test_daemon_journey_lane_overlay_rebuilds_and_a_removed_lane_goes(workspace: Path) -> None:
    """The daemon serves a lane from its base, re-plans it on change, follows its base, and drops it once removed."""
    daemon = _daemon()
    base_key, _base = await daemon.index_for({"path": str(workspace)})
    lane = _lane(workspace, "ws-lane")

    # 1. A lane is served as an overlay of the resident base, and reported as one.
    key, served = await daemon.index_for({"path": str(lane)})
    assert daemon.cache.base_of(key) == base_key, "1: the lane overlays the resident base"
    status = await server._cmd_status(daemon, {})
    entry = next(item for item in status["indexes"] if item["root"] == str(lane))
    assert entry["overlay_of"] == str(workspace) and entry["delta_chunks"] == 0, "1: status names the base"

    # 2. An edit in the lane is re-planned into the delta, never a full build.
    (lane / "alpha/src/edited.py").write_text("def edited_function():\n    return 'lane body'\n", encoding="utf-8")
    result = await daemon.rebuild(key, java_changed=False, changed_paths=[lane / "alpha/src/edited.py"])
    assert result["changed"] == 1 and result["delta_chunks"] >= 1, "2: the edit lands in the delta"
    _key, replanned = await daemon.index_for({"path": str(lane)})
    hits = replanned.search("edited_function", top_k=1)
    assert "lane body" in hits[0].chunk.content, "2: and is served"

    # 3. A rebuild of the base re-plans its overlays.
    (workspace / "alpha/src/shared.py").write_text(ALPHA["src/shared.py"] + "# base moved\n", encoding="utf-8")
    await daemon.rebuild(base_key, java_changed=False, changed_paths=[workspace / "alpha/src/shared.py"])
    assert key in daemon.pending, "3: the overlay is queued for a re-plan"
    await daemon._watch_tasks[key]
    _key, followed = await daemon.index_for({"path": str(lane)})
    assert "alpha/src/shared.py" in followed.plan.differing, "3: a file the base changed is the lane's own now"

    # 4. A removed lane is evicted by its next rebuild, and its cache folder pruned.
    git(workspace / "alpha", "worktree", "remove", "--force", str(lane / "alpha"))
    lane.rmdir()
    result = await daemon.rebuild(key, java_changed=False)
    assert result == {"skipped": "root is gone"} and key not in daemon.cache._tasks, "4: the lane is dropped"
    removed = await daemon.prune_gone_roots()
    assert str(lane) in removed, "4: and its cache folder pruned"
    assert str(workspace) not in removed, "4: a live root's cache is kept"


def test_prune_names_folders_by_marker_or_by_the_daemon_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A key folder whose stores do not name their root is pruned once its marker or the logs name a gone root."""
    cache = tmp_path / "cache"
    cache.mkdir()
    live = tmp_path / "live"
    live.mkdir()
    gone_marked, gone_logged = str(tmp_path / "gone-marked"), str(tmp_path / "gone-logged")
    for root in (gone_marked, gone_logged, str(live)):
        (cache / cache_key(root)).mkdir()
        (cache / cache_key(root) / "dupes-units.sqlite").write_bytes(b"x")
    mark_root(cache / cache_key(gone_marked), gone_marked)
    mark_root(cache / cache_key(str(live)), str(live))
    (cache / "daemon.log").write_text(f"INFO zemble.daemon.server: watching {gone_logged} (299 extensions)\n")
    unnamed_old, unnamed_new = cache / ("a" * 64), cache / ("b" * 64)
    for folder in (unnamed_old, unnamed_new):
        folder.mkdir()
        (folder / "dupes-units.sqlite").write_bytes(b"x")
    long_ago = time.time() - 40 * 86400
    os.utime(unnamed_old / "dupes-units.sqlite", (long_ago, long_ago))
    os.utime(unnamed_old, (long_ago, long_ago))
    found = {orphan.detail or orphan.target.name: orphan.kind for orphan in find_orphans(cache)}
    assert found == {
        gone_marked: OrphanKind.NAMED_ROOT_GONE,
        gone_logged: OrphanKind.NAMED_ROOT_GONE,
        unnamed_old.name: OrphanKind.UNNAMED_STALE,
    }, "gone roots by marker or log, a nameless folder only once it is old"
    for orphan in find_orphans(cache):
        assert remove_orphan(orphan)
    assert sorted(path.name for path in cache.iterdir() if path.is_dir()) == sorted(
        [cache_key(str(live)), unnamed_new.name]
    )


def test_a_full_rescan_of_a_tree_at_rest_moves_nothing(workspace: Path, mock_embedder: FakeEmbedder) -> None:
    """The walk that guards a full rescan sees an untouched tree as unmoved, and an edit or deletion as moved."""
    index = ZembleIndex.from_path(workspace, embedder=mock_embedder)
    content = index.storage_content
    assert not tree_moves_anything(workspace, content, index._manifest), "an untouched tree"
    (workspace / "alpha/src/added.py").write_text("x = 1\n", encoding="utf-8")
    assert tree_moves_anything(workspace, content, index._manifest), "a new file"
    (workspace / "alpha/src/added.py").unlink()
    (workspace / "alpha/src/removed.py").unlink()
    assert tree_moves_anything(workspace, content, index._manifest), "a vanished file"


def test_a_query_stops_waiting_for_a_busy_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Busy is retried only within ZEMBLE_QUERY_WAIT_SECONDS, then reported with what to do."""
    monkeypatch.setenv(QUERY_WAIT_ENV, "0.5")
    calls: list[float | None] = []

    def busy(cmd: str, args: dict | None = None, *, timeout: float | None = None, **_kwargs: object) -> None:
        calls.append(timeout)
        raise CommandBusy("graph construction pending")

    monkeypatch.setattr(client, "call", busy)
    started = time.monotonic()
    with pytest.raises(QueryTimedOut, match="still busy after"):
        client.answer("home", {"path": "/x"})
    assert time.monotonic() - started < 3, "the wait is bounded"
    assert calls and all(timeout is not None and timeout <= 1.0 for timeout in calls), "each try carries the deadline"


def test_a_lanes_unit_index_reads_its_bases_rows_and_keeps_only_its_delta(workspace: Path) -> None:
    """A lane's first focused dupes extracts only what the base's unit index lacks (2026-10-29: 6+ minutes cold)."""
    from zemble.dedup.unitcache import INDEX_FILE_NAME, FileRow, UnitIndex, root_cache_folder

    # 1. The base holds rows for two files; a lane mirrors the base's layout.
    base = UnitIndex(workspace, "sig")
    base.store([FileRow(path="alpha/src/shared.py", digest="d-shared", failed=False),
                FileRow(path="alpha/src/edited.py", digest="d-original", failed=False)])
    base.close()
    base_bytes = (root_cache_folder(workspace) / INDEX_FILE_NAME).read_bytes()
    lane = _lane(workspace, "lane-units")

    # 2. The lane takes the base's row of an unchanged file, and none of a file it edited.
    index = UnitIndex(lane, "sig")
    found = index.load({"alpha/src/shared.py": "d-shared", "alpha/src/edited.py": "d-edited"})
    assert sorted(found) == ["alpha/src/shared.py"], "THE ASSERTION -- the base's unchanged row serves the lane"

    # 3. The lane stores its delta in its own index; the base's file is never written.
    index.store([FileRow(path="alpha/src/edited.py", digest="d-edited", failed=False)])
    assert sorted(index.load({"alpha/src/shared.py": "d-shared", "alpha/src/edited.py": "d-edited"})) == [
        "alpha/src/edited.py", "alpha/src/shared.py"]
    index.close()
    assert (root_cache_folder(workspace) / INDEX_FILE_NAME).read_bytes() == base_bytes, "the base is read-only"

    # 4. A root outside every lane reads only its own rows.
    assert UnitIndex(workspace, "sig").load({"alpha/src/edited.py": "d-edited"}) == {}
