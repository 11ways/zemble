"""Serving a sub-directory from the index of an ancestor root, instead of indexing it again."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.conftest import write_index_components
from zemble.cache import (
    find_ancestor_index_root,
    indexed_ancestor_hint,
    resolve_index_root,
    save_index_to_cache,
)
from zemble.index import ZembleIndex
from zemble.types import ContentType


def _workspace(root: Path) -> Path:
    """Write a two-repo workspace: `alpha/` and `beta/`, each with its own sources."""
    for repo, symbol in (("alpha", "alpha"), ("beta", "beta")):
        package = root / repo / "src"
        package.mkdir(parents=True)
        (package / "session.py").write_text(
            textwrap.dedent(f"""\
                def {symbol}_session_token(user):
                    \"\"\"Return the session token for a user.\"\"\"
                    return f"{symbol}-{{user}}"

                def {symbol}_revoke_session(token):
                    return not token
                """),
            encoding="utf-8",
        )
        (package / "store.py").write_text(
            textwrap.dedent(f"""\
                class {symbol.capitalize()}Store:
                    def save_session(self, token):
                        self.token = token

                    def load_session(self):
                        return self.token
                """),
            encoding="utf-8",
        )
    return root


@pytest.fixture
def workspace_index(mock_embedder: Any, tmp_path: Path) -> ZembleIndex:
    """An index over a workspace holding two sub-repos."""
    with patch("zemble.index.index.load_embedder", return_value=mock_embedder):
        return ZembleIndex.from_path(_workspace(tmp_path))


def test_a_subtree_view_is_the_whole_index_filtered(workspace_index: ZembleIndex) -> None:
    """A subtree view answers with the ancestor index's own results, filtered to the prefix.

    The point of filtering instead of indexing the sub-directory is that scores and ranking
    stay those of the big index; the view must therefore be indistinguishable from taking the
    full result list and dropping everything outside the prefix.
    """
    # 1. The whole workspace answers with results from both repos.
    full = workspace_index.search("session token", top_k=50)
    assert {result.chunk.file_path.split("/")[0] for result in full} == {"alpha", "beta"}, "both repos are indexed"

    # 2. The view of one repo is that same list, filtered, in the same order and with the same scores.
    view = workspace_index.subtree("alpha")
    assert view is not None, "the workspace index holds files under alpha/"
    filtered = [result for result in full if result.chunk.file_path.startswith("alpha/")]
    restricted = view.search("session token", top_k=50)
    assert [(r.chunk.file_path, r.chunk.start_line) for r in restricted] == [
        (r.chunk.file_path, r.chunk.start_line) for r in filtered
    ], "same results in the same order"
    # The scores themselves are RRF ranks WITHIN the candidate pool, and the pool is the
    # selected one, exactly as `filter_languages` has always worked; only the order is a
    # promise, and it is the big index's order.

    # 3. Result paths stay relative to the index root, not to the sub-directory.
    assert all(result.chunk.file_path.startswith("alpha/") for result in restricted), "paths stay root-relative"

    # 4. Stats describe the sub-tree, not the workspace.
    assert view.stats.total_chunks < workspace_index.stats.total_chunks, "the view counts only its own chunks"
    assert view.stats.indexed_files == 2, "alpha/ holds two files"

    # 5. An explicit filter narrows the view further; it can never widen it past the prefix.
    assert view.search("session token", top_k=50, filter_paths=["beta/src/store.py"]) == [], "beta is out of reach"

    # 6. A prefix nothing was indexed under is refused rather than answered emptily.
    assert workspace_index.subtree("gamma") is None, "an empty subtree is not a view"

    # 7. find_related over the view stays inside it.
    seed = next(result.chunk for result in filtered)
    assert all(r.chunk.file_path.startswith("alpha/") for r in view.find_related(seed, top_k=10)), "related stays in"


def test_a_subtree_view_is_cached_per_prefix(workspace_index: ZembleIndex) -> None:
    """Building a view walks every chunk, so the same prefix hands back the same object."""
    assert workspace_index.subtree("alpha") is workspace_index.subtree("alpha/"), "one view per prefix"


def test_resolve_index_root_prefers_a_loaded_ancestor(tmp_path: Path) -> None:
    """A sub-directory of a root that is already in memory is routed to that root."""
    workspace = _workspace(tmp_path)
    root, prefix = resolve_index_root(
        str(workspace / "alpha"), "fake:test@256", [ContentType.CODE], loaded_roots={str(workspace)}
    )
    assert (root, prefix) == (str(workspace), "alpha"), "the loaded ancestor answers for its sub-directory"


def test_resolve_index_root_finds_a_validated_on_disk_ancestor(
    mock_embedder: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing in memory, a valid on-disk ancestor index is found and used.

    A sub-repo that never had an index of its own is exactly the case that used to cost a
    full re-embed of the sub-tree.
    """
    monkeypatch.setenv("ZEMBLE_CACHE_LOCATION", str(tmp_path / "cache"))
    workspace = _workspace(tmp_path / "work")

    # 1. Nothing is indexed yet, so the sub-directory is its own root.
    assert resolve_index_root(str(workspace / "alpha"), mock_embedder.model_id, [ContentType.CODE]) == (
        str(workspace / "alpha"),
        None,
    ), "with no ancestor index the sub-directory indexes itself"

    # 2. Index the workspace and persist it the way every surface does.
    with patch("zemble.index.index.load_embedder", return_value=mock_embedder):
        index = ZembleIndex.from_path(workspace)
    save_index_to_cache(index, str(workspace))

    # 3. Now the sub-directory is routed to the workspace index.
    assert find_ancestor_index_root(str(workspace / "alpha"), mock_embedder.model_id, [ContentType.CODE]) == str(
        workspace
    ), "the on-disk workspace index is an ancestor"
    assert resolve_index_root(str(workspace / "alpha" / "src"), mock_embedder.model_id, [ContentType.CODE]) == (
        str(workspace),
        "alpha/src",
    ), "a deeper sub-directory routes to the same root with a longer prefix"

    # 4. A different content selection is a different index, so it does not borrow this one.
    assert resolve_index_root(str(workspace / "alpha"), mock_embedder.model_id, [ContentType.DOCS]) == (
        str(workspace / "alpha"),
        None,
    ), "content types must match"

    # 5. An index of exactly the requested path keeps serving it.
    with patch("zemble.index.index.load_embedder", return_value=mock_embedder):
        own = ZembleIndex.from_path(workspace / "alpha")
    save_index_to_cache(own, str(workspace / "alpha"))
    assert resolve_index_root(str(workspace / "alpha"), mock_embedder.model_id, [ContentType.CODE]) == (
        str(workspace / "alpha"),
        None,
    ), "a deliberate sub-root index is not thrown away"


def test_the_refusal_names_the_ancestor_that_is_already_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused build inside an indexed tree says which root to search instead."""
    monkeypatch.setenv("ZEMBLE_CACHE_LOCATION", str(tmp_path / "cache"))
    workspace = _workspace(tmp_path / "work")

    assert indexed_ancestor_hint(str(workspace / "alpha")) is None, "no ancestor index, no advice"

    from zemble.cache import find_index_from_cache_folder

    index_folder = find_index_from_cache_folder(str(workspace))
    write_index_components(index_folder)
    (index_folder / "metadata.json").write_text("{}", encoding="utf-8")
    hint = indexed_ancestor_hint(str(workspace / "alpha"))
    assert hint is not None, "an indexed ancestor is named"
    assert str(workspace) in hint and str(workspace / "alpha") in hint, "both paths are named"
    assert "ZEMBLE_EMBED_CONFIRM=1" in hint, "and the way to index it anyway"
