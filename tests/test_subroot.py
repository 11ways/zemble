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
        (r.chunk.file_path.removeprefix("alpha/"), r.chunk.start_line) for r in filtered
    ], "same results in the same order"
    # The scores themselves are RRF ranks WITHIN the candidate pool, and the pool is the
    # selected one, exactly as `filter_languages` has always worked; only the order is a
    # promise, and it is the big index's order.

    # 3. Result paths are relative to the sub-directory, the root the caller named: the same
    #    spelling the symbol graph and `dupes` use for that directory, and joinable with it.
    assert all(result.chunk.file_path.startswith("src/") for result in restricted), "paths are sub-directory-relative"
    assert restricted[0].chunk.content == filtered[0].chunk.content, "only the path is rebased"

    # 4. Stats describe the sub-tree, not the workspace.
    assert view.stats.total_chunks < workspace_index.stats.total_chunks, "the view counts only its own chunks"
    assert view.stats.indexed_files == 2, "alpha/ holds two files"

    # 5. An explicit filter narrows the view further; it can never widen it past the prefix.
    #    It is spelled relative to the sub-directory too.
    assert view.search("session token", top_k=50, filter_paths=["beta/src/store.py"]) == [], "beta is out of reach"
    assert view.search("session token", top_k=50, filter_paths=["../beta/src/store.py"]) == [], "no escaping either"
    only_store = view.search("session token", top_k=50, filter_paths=["src/store.py"])
    assert only_store and all(r.chunk.file_path == "src/store.py" for r in only_store), "a sub-relative filter works"

    # 6. A prefix nothing was indexed under is refused rather than answered emptily.
    assert workspace_index.subtree("gamma") is None, "an empty subtree is not a view"

    # 7. find_related over the view stays inside it, takes a seed as the view spells it, and
    #    never hands the seed back as its own neighbour.
    seed = restricted[0].chunk
    related = view.find_related(seed, top_k=10)
    assert related, "the seed has neighbours inside alpha/"
    assert all(r.chunk.file_path.startswith("src/") for r in related), "related stays in, sub-relative"
    assert seed not in {r.chunk for r in related}, "the seed is excluded even though only its path was rebased"


def test_a_subtree_view_resolves_locations_in_its_own_spelling(workspace_index: ZembleIndex) -> None:
    """A location copied out of any answer for the sub-directory resolves against the view.

    The bug this guards: `dupes` and the graph name files relative to the directory the
    caller passed, while the view used to name them relative to the ancestor, so a path
    copied from one tool into `find_related` was reported as not indexed.
    """
    view = workspace_index.subtree("alpha")
    assert view is not None

    # Any line inside the chunk resolves, not only its first; the answer is sub-relative.
    first = view.chunk_at("src/session.py", 1)
    assert first is not None and first.file_path == "src/session.py"
    inside = view.chunk_at("src/session.py", 2)
    assert inside == first, "a line inside the chunk resolves to the same chunk as its first line"

    # The ancestor's spelling is NOT accepted silently: one vocabulary, not two.
    assert view.chunk_at("alpha/src/session.py", 1) is None
    # ...but the root index, which speaks that spelling, still resolves it.
    assert workspace_index.chunk_at("alpha/src/session.py", 1) is not None
    assert workspace_index.chunk_at("src/session.py", 1) is None

    # The view lists and describes only its own files, in its own spelling.
    assert view.indexed_paths() == ["src/session.py", "src/store.py"]
    assert [chunk.file_path for chunk in view.chunks_of("src/store.py")] == ["src/store.py"] * len(
        view.chunks_of("src/store.py")
    )
    assert view.chunks_of("beta/src/store.py") == []
    assert all(path.startswith(("alpha/", "beta/")) for path in workspace_index.indexed_paths())


def test_an_unresolved_location_is_reported_as_a_path_problem(workspace_index: ZembleIndex) -> None:
    """The error names the nearest indexed path and the spelling rule, never 'not indexed'."""
    from zemble.utils import describe_unresolved_location, nearest_indexed_path

    view = workspace_index.subtree("alpha")
    assert view is not None

    # A path in the ancestor's spelling maps to its sub-relative twin.
    message = describe_unresolved_location(view, "alpha/src/session.py", 1)
    assert "No indexed file matches 'alpha/src/session.py'" in message
    assert "Did you mean 'src/session.py'?" in message
    assert "relative to the repo you passed" in message

    # A known file with a line past its last chunk lists the spans it does have.
    message = describe_unresolved_location(view, "src/session.py", 999)
    assert message.startswith("'src/session.py' is indexed, but no chunk covers line 999; its chunks span 1-")

    # The nearest match counts shared trailing segments, needing at least the file name.
    paths = ["src/session.py", "src/store.py", "lib/other/store.py"]
    assert nearest_indexed_path(paths, "hawkeye/src/store.py") == "src/store.py"
    assert nearest_indexed_path(paths, "other/store.py") == "lib/other/store.py"
    assert nearest_indexed_path(paths, "store.py") == "lib/other/store.py", "a tie is broken by sorted order"
    assert nearest_indexed_path(paths, "nothing.py") is None
    assert nearest_indexed_path(paths, "") is None
    assert "Did you mean" not in describe_unresolved_location(view, "nothing.py", 1)


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

    # 6. ...unless the workspace index is already IN MEMORY: then the loaded ancestor answers,
    #    because a second resident index costs RAM and a build for nothing.
    assert resolve_index_root(
        str(workspace / "alpha"), mock_embedder.model_id, [ContentType.CODE], loaded_roots={str(workspace)}
    ) == (str(workspace), "alpha"), "a loaded ancestor beats an on-disk exact index"


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
