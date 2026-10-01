"""Behaviour journeys over a sub-directory answered from its ancestor's symbol graph."""

import inspect
import os
import shutil
from pathlib import Path

import pytest

from zemble.graph import cli as graph_cli
from zemble.graph.cli import ensure_graph
from zemble.graph.model import Hit
from zemble.graph.provider import SqliteGraphProvider, SubtreeGraphProvider, open_provider
from zemble.graph.store import build_graph, graph_present, resolve_graph_root

_CORE = "src/main/java/com/example/core"
_QUERIES = (
    "callers",
    "callees",
    "references",
    "implementations",
    "supertypes",
    "overrides_of",
    "overridden_by",
    "tests_of",
    "neighbors",
)


def _hit_key(hit: Hit) -> tuple:
    """Everything a caller reads off a hit."""
    return (hit.symbol.id, hit.symbol.file_path, hit.edge_kind, hit.line, hit.resolution, hit.depth, hit.reason)


def _write(root: Path, relative: str, text: str) -> None:
    """Write one source file under a root."""
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


@pytest.fixture(autouse=True)
def _fresh_refresh_memory() -> None:
    """Forget which roots this process refreshed, so each journey starts cold."""
    graph_cli._refreshed.clear()


def test_a_sub_directory_is_answered_from_its_ancestors_graph(
    graph_fixture_root: Path, graph_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A covered sub-directory gets no graph of its own, and its answers equal the ones its own graph gives."""
    workspace = Path(shutil.copytree(graph_fixture_root, tmp_path / "ws"))
    sub = workspace / _CORE
    build_graph(str(workspace))

    # 1. Ensuring the sub-directory refreshes the workspace graph and builds nothing new.
    ensure_graph(str(sub), allow_daemon=False)
    assert not graph_present(str(sub)), "step 1: no graph was built for the sub-directory"
    assert resolve_graph_root(str(sub)) == (str(workspace), _CORE), "step 1: it routes to the workspace graph"

    # 2. The view speaks paths and ids relative to the sub-directory, as its own graph would.
    view = open_provider(str(sub))
    assert isinstance(view, SubtreeGraphProvider), "step 2: a view over the ancestor graph"
    circles = view.definition("Circle")
    assert circles and {(symbol.file_path, symbol.id.split("#")[0]) for symbol in circles} == {
        ("Circle.java", "Circle.java")
    }, "step 2: only the in-folder Circle, rebased"
    assert view.symbols_in_file("Circle.java"), "step 2: files are named relative to the sub-directory"
    assert view.symbol(circles[0].id) == circles[0], "step 2: a view id reads back the same symbol"

    # 3. Every answer equals the one a graph built for the sub-directory alone gives.
    monkeypatch.setenv("ZEMBLE_CACHE_LOCATION", str(tmp_path / "alone-cache"))
    build_graph(str(sub))
    alone = SqliteGraphProvider(str(sub))
    try:
        names = sorted({row["name"] for row in alone.connection.execute("SELECT name FROM symbols")})
        assert len(names) > 20, "step 3: the comparison covers the whole folder"
        for name in names:
            mine = alone.definition(name)
            assert view.definition(name) == mine, f"step 3: definition({name}) is unchanged"
            for symbol in mine:
                for query in _QUERIES:
                    expected = [_hit_key(hit) for hit in getattr(alone, query)(symbol.id)]
                    actual = [_hit_key(hit) for hit in getattr(view, query)(symbol.id)]
                    assert actual == expected, f"step 3: {query}({symbol.id}) is unchanged"
    finally:
        alone.close()
        view.close()


def test_a_chain_through_a_sibling_module_reaches_the_sub_directory(graph_cache: Path, tmp_path: Path) -> None:
    """The ancestor graph resolves what crosses a module boundary, which a graph of one module never sees."""
    workspace = tmp_path / "ws"
    _write(workspace, "a/src/p/Base.java", "package p;\npublic class Base {}\n")
    _write(workspace, "b/src/p/Middle.java", "package p;\npublic class Middle extends Base {}\n")
    _write(workspace, "a/src/p/Leaf.java", "package p;\npublic class Leaf extends Middle {}\n")
    build_graph(str(workspace))

    # 1. From the workspace graph, filtered to module `a`: Leaf is a subtype of Base through `b`.
    view = open_provider(str(workspace / "a"))
    try:
        base = view.definition("Base")[0]
        found = [(hit.symbol.qualified_name, hit.depth) for hit in view.implementations(base.id)]
        assert found == [("p.Leaf", 2)], "step 1: the in-folder subtype, found through the sibling module"
        assert not view.definition("Middle"), "step 1: the sibling module's own symbols stay out of the answer"
    finally:
        view.close()

    # 2. A graph of module `a` alone cannot see the link.
    alone_root = tmp_path / "alone" / "a"
    shutil.copytree(workspace / "a", alone_root)
    build_graph(str(alone_root))
    alone = SqliteGraphProvider(str(alone_root))
    try:
        assert alone.implementations(alone.definition("Base")[0].id) == [], "step 2: no chain without `b`"
    finally:
        alone.close()


def test_a_sub_directory_the_ancestor_does_not_cover_gets_its_own_graph(graph_cache: Path, tmp_path: Path) -> None:
    """A folder the ancestor's walk skips is not answered by an empty filter but by a graph of its own."""
    workspace = tmp_path / "ws"
    _write(workspace, "src/p/Kept.java", "package p;\npublic class Kept {}\n")
    _write(workspace, "vendored/p/Skipped.java", "package p;\npublic class Skipped {}\n")
    _write(workspace, ".gitignore", "vendored/\n")
    os.makedirs(workspace / ".git")
    build_graph(str(workspace))
    sub = workspace / "vendored"

    # 1. The workspace graph holds nothing under the ignored folder, so it does not answer for it.
    ensure_graph(str(sub), allow_daemon=False)
    assert graph_present(str(sub)), "step 1: the uncovered folder got a graph of its own"
    assert resolve_graph_root(str(sub)) == (str(sub), None), "step 1: and routes to it"

    # 2. That graph answers for it.
    provider = open_provider(str(sub))
    try:
        assert isinstance(provider, SqliteGraphProvider), "step 2: its own graph, unfiltered"
        assert [symbol.file_path for symbol in provider.definition("Skipped")] == ["p/Skipped.java"]
    finally:
        provider.close()


def test_the_graph_cli_answers_a_sub_directory_without_building_for_it(
    graph_fixture_root: Path, graph_cache: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`zemble graph build` and a query on a covered sub-directory both work on the ancestor's graph."""
    workspace = Path(shutil.copytree(graph_fixture_root, tmp_path / "ws"))
    sub = workspace / _CORE
    build_graph(str(workspace))
    parser = __import__("zemble.cli", fromlist=["_build_parser"])._build_parser()

    # 1. `graph build` of the sub-directory refreshes the workspace graph instead.
    assert graph_cli.run_graph(parser.parse_args(["graph", "build", str(sub), "--no-daemon"])) == 0
    assert f"Graph built for {workspace}" in capsys.readouterr().out, "step 1: the ancestor was built"
    assert not graph_present(str(sub)), "step 1: and no graph for the sub-directory"

    # 2. A query prints paths relative to the sub-directory.
    assert graph_cli.run_graph(parser.parse_args(["graph", "definition", str(sub), "Shape", "--no-daemon"])) == 0
    out = capsys.readouterr().out
    assert " Shape.java:" in out and _CORE not in out, f"step 2: rebased paths, got {out!r}"


def test_the_subtree_view_offers_every_provider_query() -> None:
    """Every public query of the sqlite provider has a filtering twin on the view, so none answers unfiltered."""
    public = {
        name for name, _ in inspect.getmembers(SqliteGraphProvider, inspect.isfunction) if not name.startswith("_")
    }
    missing = public - set(dir(SubtreeGraphProvider))
    assert not missing, f"SubtreeGraphProvider lacks {sorted(missing)}"
    for name in public:
        assert getattr(SubtreeGraphProvider, name) is not getattr(SqliteGraphProvider, name), name
