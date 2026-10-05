"""One root keeps one index on disk: a request a wider stored index covers is answered from it."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from unittest.mock import patch

from tests.conftest import FakeEmbedder
from zemble.cache import (
    cached_index_compatible,
    find_index_from_cache_folder,
    has_cached_index,
    resolve_index_root,
    save_index_to_cache,
)
from zemble.cache_orphans import OrphanKind, find_orphans, remove_orphan
from zemble.graph.store import build_graph, graph_present, resolve_graph_root
from zemble.index import ZembleIndex
from zemble.index.types import PersistencePath
from zemble.index_cache import IndexCache
from zemble.types import ContentType

CODE = (ContentType.CODE,)
CODE_DOCS = (ContentType.CODE, ContentType.DOCS)


def _files(index: ZembleIndex) -> set[str]:
    """Return the files a (view of an) index answers from."""
    return {result.chunk.file_path for result in index.search("project token name config", top_k=50)} | set(
        index._file_mapping
    )


def test_one_root_keeps_one_index(tmp_project: Path, mock_embedder: FakeEmbedder) -> None:
    """A code+docs index replaces the code index of the same root and answers code requests itself."""
    root = str(tmp_project)
    code_folder = find_index_from_cache_folder(root, CODE)

    # 1. A code request on a fresh root stores a code index, next to which the graph lives.
    save_index_to_cache(ZembleIndex.from_path(tmp_project, content=CODE, embedder=mock_embedder), root)
    assert not PersistencePath.from_path(code_folder).non_existing(), "step 1: the code index is stored"
    graph_file = code_folder / "graph-1.sqlite"
    graph_file.write_bytes(b"graph")

    # 2. Storing code+docs for the same root removes the code index it covers, but not the graph.
    wide = ZembleIndex.from_path(tmp_project, content=CODE_DOCS, embedder=mock_embedder)
    save_index_to_cache(wide, root)
    assert PersistencePath.from_path(code_folder).non_existing(), "step 2: the covered code index is gone"
    assert graph_file.read_bytes() == b"graph", "step 2: the graph sharing its folder stays"

    # 3. A code request is answered from the code+docs index, narrowed to code, embedding nothing.
    mock_embedder.document_calls.clear()
    narrowed = ZembleIndex.from_path(tmp_project, content=CODE, embedder=mock_embedder)
    assert mock_embedder.document_calls == [], "step 3: nothing is embedded again"
    assert narrowed.content == CODE and narrowed.storage_content == CODE_DOCS, "step 3: stored wide, answers code"
    assert _files(narrowed) == {"auth.py", "utils.py"}, "step 3: the docs file never answers a code request"
    assert narrowed.stats.indexed_files == 2, "step 3: and stats describe the code it answers for"
    save_index_to_cache(narrowed, root)
    assert PersistencePath.from_path(code_folder).non_existing(), "step 3: answering never writes a code index back"

    # 4. A caller's own filter on the narrowed view keeps it narrowed.
    filtered = narrowed.filtered(exclude=["utils.py"])
    assert filtered is not None and _files(filtered) == {"auth.py"}, "step 4: a filter never brings docs back"
    sub = narrowed.subtree(".")
    assert sub is None or "README.md" not in _files(sub), "step 4: nor does a sub-tree view"

    # 5. Every lookup that asks "is this root indexed for code" sees the covering index.
    assert has_cached_index(root, CODE), "step 5: a covering index counts as an index of the root"
    assert cached_index_compatible(root, mock_embedder.model_id, CODE), "step 5: and as a compatible one"

    # 6. The shared index cache keys the root by what it stores and answers with what was asked.
    async def serve() -> tuple:
        cache = IndexCache()
        await cache.load_embedder_once()
        return await cache.get_with_key(root, content=CODE)

    with patch("zemble.index_cache.load_embedder", return_value=mock_embedder):
        key, served = asyncio.run(serve())
    assert key[1] == CODE_DOCS, "step 6: one resident index per root, the stored one"
    assert served.content == CODE and _files(served) == {"auth.py", "utils.py"}, "step 6: narrowed to code"


def test_a_covered_index_left_from_before_is_an_orphan(tmp_project: Path, mock_embedder: FakeEmbedder) -> None:
    """`zemble clear orphans` removes a code index a code+docs index of the same root covers."""
    root = str(tmp_project)
    cache_folder = find_index_from_cache_folder(root, CODE).parent.parent
    code_folder = find_index_from_cache_folder(root, CODE)
    save_index_to_cache(ZembleIndex.from_path(tmp_project, content=CODE, embedder=mock_embedder), root)
    kept = code_folder.parent / "kept-code-index"
    shutil.copytree(code_folder, kept)
    save_index_to_cache(ZembleIndex.from_path(tmp_project, content=CODE_DOCS, embedder=mock_embedder), root)
    # The state an older zemble left: both variants stored side by side, plus the graph.
    shutil.rmtree(code_folder)
    kept.rename(code_folder)
    (code_folder / "graph-1.sqlite").write_bytes(b"graph")

    # 1. The narrower index is reported, with the size of its own stores only.
    orphans = [orphan for orphan in find_orphans(cache_folder) if orphan.kind is OrphanKind.INDEX_COVERED]
    assert [orphan.target for orphan in orphans] == [code_folder], "step 1: the covered code index is an orphan"
    assert orphans[0].size > 0

    # 2. Removing it removes the index and keeps the graph.
    assert remove_orphan(orphans[0]), "step 2: it is removed"
    assert PersistencePath.from_path(code_folder).non_existing(), "step 2: its stores are gone"
    assert (code_folder / "graph-1.sqlite").read_bytes() == b"graph", "step 2: the graph stays"
    assert not [orphan for orphan in find_orphans(cache_folder) if orphan.kind is OrphanKind.INDEX_COVERED]


def test_a_sub_root_an_ancestor_answers_for_is_an_orphan(
    tmp_path: Path, mock_embedder: FakeEmbedder, graph_fixture_root: Path, graph_cache: Path
) -> None:
    """A sub-repo's own index and graph go once its workspace's index and graph answer for it."""
    workspace = tmp_path / "work"
    shutil.copytree(graph_fixture_root, workspace)
    sub = workspace / "src" / "main" / "java" / "com" / "example" / "core"
    save_index_to_cache(ZembleIndex.from_path(sub, content=CODE, embedder=mock_embedder), str(sub))
    build_graph(str(sub))

    # 1. Before the workspace has an index or graph of its own, the sub-root's are not orphans.
    kinds = {OrphanKind.INDEX_UNDER_ANCESTOR, OrphanKind.GRAPH_UNDER_ANCESTOR}
    assert not [orphan for orphan in find_orphans(graph_cache) if orphan.kind in kinds], "step 1: still needed"

    # 2. Once the workspace has both, the sub-root's are reported, naming who answers instead.
    save_index_to_cache(ZembleIndex.from_path(workspace, content=CODE, embedder=mock_embedder), str(workspace))
    build_graph(str(workspace))
    found = {orphan.kind: orphan for orphan in find_orphans(graph_cache) if orphan.kind in kinds}
    assert set(found) == kinds, "step 2: the sub-root's index and graph are both orphans"
    assert all(str(workspace) in orphan.detail for orphan in found.values()), "step 2: the answering root is named"

    # 3. Removing them leaves routing to the workspace, which still answers for the sub-path.
    assert all(remove_orphan(orphan) for orphan in found.values()), "step 3: both are removed"
    assert not graph_present(str(sub)), "step 3: the sub-root graph is gone"
    assert resolve_index_root(str(sub), mock_embedder.model_id, CODE) == (
        str(workspace),
        "src/main/java/com/example/core",
    )
    assert resolve_graph_root(str(sub)) == (str(workspace), "src/main/java/com/example/core"), "step 3: graph too"
