"""Behaviour journeys over the graph CLI and MCP surfaces."""

import asyncio
import json
from pathlib import Path

import pytest

from zemble.cli import _cli_main
from zemble.graph.cli import EXIT_AMBIGUOUS


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    """Run the zemble CLI with the given arguments and return its exit code."""
    from zemble.graph import cli as graph_cli

    graph_cli._refreshed.clear()
    monkeypatch.setattr("sys.argv", ["zemble", *argv])
    try:
        _cli_main()
    except SystemExit as exit_code:
        return int(exit_code.code or 0)
    return 0


def test_cli_journey(graph_fixture_root: Path, graph_cache: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Build, query, and hit the ambiguity and not-found exits."""
    root = str(graph_fixture_root)

    # 1. `graph build` reports what it built.
    assert _run(monkeypatch, "graph", "build", root) == 0, "step 1: a build succeeds"
    assert "symbols" in capsys.readouterr().out, "step 1: the build prints a summary"

    # 2. `--stats` adds the machine-readable statistics.
    _run(monkeypatch, "graph", "build", root, "--stats")
    assert '"resolution_counts"' in capsys.readouterr().out, "step 2: --stats prints the full statistics"

    # 3. A query prints the symbol it chose and one line per hit.
    _run(monkeypatch, "graph", "callers", root, "Helpers.twice")
    output = capsys.readouterr().out
    assert "callers: " in output, "step 3: the query names what it answered"
    assert "called from Circle.area" in output, "step 3: each hit carries its reason"

    # 4. `--json` produces machine output instead.
    _run(monkeypatch, "graph", "implementations", root, "Shape", "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["symbol"]["qualified_name"] == "com.example.core.Shape", "step 4: the chosen symbol is reported"
    assert any(hit["qualified_name"] == "com.example.core.Circle" for hit in payload["results"]), (
        "step 4: results carry qualified names"
    )

    # 5. An ambiguous name lists the candidates and exits 2.
    code = _run(monkeypatch, "graph", "callers", root, "Circle")
    assert code == EXIT_AMBIGUOUS, "step 5: ambiguity exits 2"
    assert "is ambiguous" in capsys.readouterr().err, "step 5: the candidates are listed"

    # 6. An unknown name exits 1 and says what the graph covers.
    assert _run(monkeypatch, "graph", "definition", root, "Nonexistent") == 1, "step 6: unknown names exit 1"
    assert "No symbol named" in capsys.readouterr().err, "step 6: and say so"

    # 7. `tests-of` reaches the naming-based test edge.
    _run(monkeypatch, "graph", "tests-of", root, "com.example.core.Circle")
    assert "tested by CircleTest" in capsys.readouterr().out, "step 7: tests-of finds the naming match"


def test_cli_builds_the_graph_on_first_query(
    graph_fixture_root: Path, graph_cache: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A query with no graph yet builds one instead of failing."""
    from zemble.graph.store import graph_exists

    assert not graph_exists(str(graph_fixture_root)), "no graph exists yet"
    assert _run(monkeypatch, "graph", "definition", str(graph_fixture_root), "Shape") == 0, "the query succeeds"
    capsys.readouterr()
    assert graph_exists(str(graph_fixture_root)), "the query built the graph"


def test_mcp_tools_journey(graph_fixture_root: Path, graph_cache: Path) -> None:
    """The MCP tools answer the same questions as the CLI, as payload objects."""
    from zemble.graph.mcp import answer as _answer

    root = str(graph_fixture_root)

    # 1. Definition returns every declaration.
    payload = _answer(root, "Shape", "definition")
    assert payload["results"][0]["qualified_name"] == "com.example.core.Shape", "step 1: the interface is found"

    # 2. Callers carry a reason and a resolution.
    payload = _answer(root, "Helpers.twice", "callers")
    assert payload["results"], "step 2: callers are found"
    assert all("resolution" in hit and "reason" in hit for hit in payload["results"]), (
        "step 2: each hit explains itself"
    )

    # 3. Ambiguity is reported as data, not as an exception.
    payload = _answer(root, "Circle", "callers")
    assert "ambiguous" in payload["error"], "step 3: ambiguity is reported"
    assert len(payload["candidates"]) == 2, "step 3: with both candidates"

    # 4. An empty answer explains what the graph covers.
    payload = _answer(root, "Marker", "tests_of")
    assert payload["results"] == [], "step 4: nothing tests Marker"
    assert "Java" in payload["note"], "step 4: and the note says why that may be"

    # 5. Neighbours accept the extra arguments.
    payload = _answer(root, "com.example.core.Circle", "neighbors", hops=2, kinds=None)
    assert len(payload["results"]) > 1, "step 5: a two-hop walk returns more than the origin"

    # 6. Every payload is an object; a tool that encoded it into a string would double-encode it.
    assert isinstance(payload, dict), "step 6: the answer is data, not a JSON string"


def test_mcp_server_registers_the_graph_tools() -> None:
    """The graph tools are added to the existing MCP server without replacing it."""
    from zemble.index_cache import IndexCache
    from zemble.mcp import create_server

    tools = {tool.name for tool in asyncio.run(create_server(IndexCache()).list_tools())}
    assert {"graph_definition", "graph_callers", "graph_implementations", "graph_tests_of", "graph_neighbors"} <= tools
    assert {"search", "find_related"} <= tools, "the existing tools are untouched"
