"""Query-time `paths` / `exclude` filtering, and the build-time exclusion that has its own key."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from zemble.cache import cache_key, exclude_digest, find_index_from_cache_folder
from zemble.index import ZembleIndex
from zemble.index.view import IndexView
from zemble.index_cache import compute_cache_key
from zemble.types import ContentType


def _workspace(root: Path, copies: int = 6) -> None:
    """Write the same session helper into `src/`, `vendor/` and `tests/`, so ranking has ties."""
    for area in ("src", "vendor", "tests"):
        directory = root / area
        directory.mkdir(parents=True)
        for index in range(copies):
            (directory / f"session_{index}.py").write_text(
                textwrap.dedent(f"""\
                    def {area}_session_token_{index}(user):
                        \"\"\"Return the session token for a user.\"\"\"
                        return f"{area}-{index}-{{user}}"
                    """),
                encoding="utf-8",
            )


@pytest.fixture()
def index(tmp_path: Path, mock_embedder) -> ZembleIndex:
    """An index over a three-area workspace."""
    root = tmp_path / "workspace"
    _workspace(root)
    return ZembleIndex.from_path(root, embedder=mock_embedder)


def test_view_keeps_what_it_says(tmp_path: Path) -> None:
    """The view predicate rebases onto the routing prefix and matches gitignore syntax."""
    view = IndexView.build(paths=["src"], exclude=["*.min.js"])
    assert view is not None
    assert view.keeps("src/app.py"), "a file under a kept path survives"
    assert not view.keeps("vendor/app.py"), "anything outside the kept paths is dropped"
    assert not view.keeps("src/app.min.js"), "an excluded pattern wins over a kept path"

    nested = IndexView.build(prefix="repo", paths=["src"], exclude=["vendor/"])
    assert nested is not None
    assert nested.keeps("repo/src/app.py"), "paths are relative to the requested repo, not the index root"
    assert not nested.keeps("src/app.py"), "a file outside the routed sub-tree is never in the view"
    assert not nested.keeps("repo/vendor/app.py"), "an exclude pattern is rebased the same way"
    assert IndexView.build() is None, "restricting nothing is not a view at all"


def test_query_time_filter_keeps_top_k_full(index: ZembleIndex) -> None:
    """Excluded files are dropped from the CANDIDATES, so top_k still comes back full."""
    unfiltered = index.search("session token", top_k=5)
    assert len(unfiltered) == 5, "the fixture has more than five matching chunks"
    leader = unfiltered[0].chunk.file_path.split("/")[0]

    view = index.filtered(exclude=[f"{leader}/"])
    assert view is not None
    filtered = view.search("session token", top_k=5)
    assert len(filtered) == 5, "the filter narrows the candidates before the truncation, not after"
    assert all(not result.chunk.file_path.startswith(f"{leader}/") for result in filtered), f"{leader} is gone"


def test_paths_and_exclude_combine(index: ZembleIndex) -> None:
    """`paths` keeps, `exclude` drops, and asking for both applies both."""
    view = index.filtered(paths=["src", "vendor"], exclude=["vendor/"])
    assert view is not None
    results = view.search("session token", top_k=10)
    assert results, "something survives"
    assert all(result.chunk.file_path.startswith("src/") for result in results), "only src is left"
    assert index.filtered(paths=["nowhere"]) is None, "a filter that keeps nothing says so"
    assert index.filtered() is index, "no filter is the index itself, not a copy"


def test_ranking_is_untouched_without_a_filter(index: ZembleIndex) -> None:
    """An absent filter must not move a single result: this is the bit-identity guarantee."""
    before = index.search("session token", top_k=8)
    after = index.filtered((), ()).search("session token", top_k=8)
    assert [(r.chunk.file_path, r.score) for r in before] == [(r.chunk.file_path, r.score) for r in after]


def test_default_cache_keys_are_unchanged(tmp_path: Path) -> None:
    """A build with no exclude keys exactly as it always did, on disk and in memory."""
    path = str(tmp_path)
    assert exclude_digest([]) == "", "nothing excluded is no digest"
    assert exclude_digest(["  "]) == "", "blank patterns are not patterns"
    assert cache_key(path) == cache_key(path, []), "the on-disk key ignores an empty exclude"
    assert cache_key(path) == cache_key(path, ("",)), "and ignores a blank one"
    assert compute_cache_key(path, None, (ContentType.CODE,)) == (str(Path(path).resolve()), (ContentType.CODE,)), (
        "the in-memory key is the two-element tuple it has always been"
    )
    assert find_index_from_cache_folder(path) == find_index_from_cache_folder(path, (ContentType.CODE,), ()), (
        "the default index folder is unchanged"
    )


def test_excluded_builds_key_separately(tmp_path: Path) -> None:
    """A pruned build never shares a cache entry with the plain index of the same root."""
    path = str(tmp_path)
    assert cache_key(path, ["vendor/"]) != cache_key(path), "the on-disk key differs"
    assert cache_key(path, ["vendor/"]) == cache_key(path, ["vendor/", "vendor/"]), "order and repeats are irrelevant"
    assert cache_key(path, ["a", "b"]) == cache_key(path, ["b", "a"]), "the digest is order-independent"
    keyed = compute_cache_key(path, None, (ContentType.CODE,), ["vendor/"])
    assert len(keyed) == 3 and keyed[2] == exclude_digest(["vendor/"]), "the digest rides along as a third element"


def test_build_time_exclusion_prunes_the_walk(tmp_path: Path, mock_embedder) -> None:
    """`exclude` on a first build keeps those files out of the index entirely."""
    root = tmp_path / "workspace"
    _workspace(root, copies=2)
    pruned = ZembleIndex.from_path(root, embedder=mock_embedder, exclude=["vendor/"])
    assert pruned.stats.indexed_files == 4, "src and tests only"
    assert all(not chunk.file_path.startswith("vendor/") for chunk in pruned.chunks), "vendor was never chunked"
    whole = ZembleIndex.from_path(root, embedder=mock_embedder)
    assert whole.stats.indexed_files == 6, "the plain build still sees the whole tree"
