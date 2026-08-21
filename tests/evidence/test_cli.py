"""The `explain`, `outline` and `signatures` surfaces, on the CLI and over MCP."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import FakeEmbedder
from zemble.cli import _cli_main
from zemble.graph.cli import EXIT_AMBIGUOUS, EXIT_NOT_FOUND
from zemble.index_cache import IndexCache
from zemble.mcp import create_server
from zemble.types import Chunk, SearchResult

_CIRCLE = "src/main/java/com/example/core/Circle.java"


def _fake_index(root: Path) -> MagicMock:
    """An index that returns one real chunk of the fixture workspace."""
    content = (root / _CIRCLE).read_text(encoding="utf-8")
    chunk = Chunk(content=content, file_path=_CIRCLE, start_line=1, end_line=content.count("\n") + 1, language="java")
    index = MagicMock()
    # An unfiltered view of an index is the index itself, which is what the real one returns.
    index.filtered.return_value = index
    index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    index._root = root
    return index


def _run(monkeypatch: pytest.MonkeyPatch, root: Path, *argv: str) -> int:
    """Run the CLI with a fake index in place, returning its exit code."""
    from zemble.graph import cli as graph_cli

    graph_cli._refreshed.clear()
    monkeypatch.setattr(sys, "argv", ["zemble", *argv])
    with patch("zemble.cli.ZembleIndex.from_path", return_value=_fake_index(root)):
        try:
            _cli_main()
        except SystemExit as exit_code:
            return int(exit_code.code or 0)
    return 0


def test_cli_journey(
    graph_fixture_root: Path, graph_cache: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Walk explain, outline and signatures through their output modes and refusals."""
    root = str(graph_fixture_root)

    # 1. `explain` prints a markdown bundle that states its own cost.
    assert _run(monkeypatch, graph_fixture_root, "explain", root, "how is an area computed", "--budget", "1500") == 0, (
        "step 1: explain succeeds"
    )
    output = capsys.readouterr().out
    assert output.startswith("# Evidence for: how is an area computed"), "step 1: the query heads the bundle"
    assert "tokens." in output, "step 1: the bundle states what it cost"

    # 2. `--json` gives the same bundle as data.
    _run(monkeypatch, graph_fixture_root, "explain", root, "how is an area computed", "--budget", "1500", "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_tokens"] <= payload["budget_tokens"], "step 2: the budget is reported and respected"
    assert all(item["reason"] for item in payload["items"]), "step 2: every item carries a reason"

    # 3. `outline` renders a type without touching the index.
    assert _run(monkeypatch, graph_fixture_root, "outline", root, "Registry") == 0, "step 3: outline succeeds"
    assert "class Registry<T extends Shape>" in capsys.readouterr().out, "step 3: the declaration is shown"

    # 4. `--members` narrows it.
    _run(monkeypatch, graph_fixture_root, "outline", root, _CIRCLE, "--members", "scale")
    narrowed = capsys.readouterr().out
    assert "scale(double factor)" in narrowed and "label()" not in narrowed, "step 4: only matching members remain"

    # 5. An ambiguous outline target exits 2 and names the candidates.
    assert _run(monkeypatch, graph_fixture_root, "outline", root, "Circle") == EXIT_AMBIGUOUS, "step 5: ambiguity is 2"
    assert "is ambiguous" in capsys.readouterr().err, "step 5: the candidates are listed"

    # 6. `signatures` prints the declaration and its exact callers.
    assert _run(monkeypatch, graph_fixture_root, "signatures", root, "Helpers.twice") == 0, (
        "step 6: signatures succeeds"
    )
    signature_output = capsys.readouterr().out
    assert "method double twice(double value)" in signature_output, "step 6: the signature is first"
    assert "Circle.area" in signature_output, "step 6: callers are listed one per line"

    # 7. An unknown symbol exits 1 and says what the graph covers.
    assert _run(monkeypatch, graph_fixture_root, "signatures", root, "NotHere") == EXIT_NOT_FOUND, "step 7: exit 1"
    assert "No symbol named" in capsys.readouterr().err, "step 7: and says so"

    # 8. The bundle header states the intent it detected and the rule that decided it.
    _run(monkeypatch, graph_fixture_root, "explain", root, "how is an area computed", "--budget", "1500")
    assert "intent: architecture (rule: how-does; order: default)" in capsys.readouterr().out, (
        "step 8: the detection is printed beside the order that packed"
    )

    # 9. `--intent` overrides that detection, and says it was overridden.
    _run(
        monkeypatch,
        graph_fixture_root,
        "explain",
        root,
        "how is an area computed",
        "--budget",
        "1500",
        "--intent",
        "consumer",
    )
    assert "intent: consumer (rule: override; order: consumer)" in capsys.readouterr().out, (
        "step 9: the override is honest"
    )


@pytest.fixture()
def cache(mock_embedder: FakeEmbedder) -> IndexCache:
    """An index cache backed by the deterministic test embedder."""
    prepared = IndexCache()
    prepared._embedder = mock_embedder
    prepared._model_ready.set()
    return prepared


async def _call(cache: IndexCache, root: Path, tool: str, args: dict[str, Any]) -> str:
    """Invoke one MCP tool with a fake index in place and return its text result."""
    with patch("zemble.mcp.ZembleIndex.from_path", return_value=_fake_index(root)):
        server = create_server(cache)
        result = await server.call_tool(tool, args)
    return result[0].text


def test_mcp_tools(graph_fixture_root: Path, graph_cache: Path, cache: IndexCache) -> None:
    """The three tools are registered and answer over the fixture workspace."""
    import asyncio

    from zemble.graph import cli as graph_cli

    graph_cli._refreshed.clear()
    root = str(graph_fixture_root)

    # 1. `explain` returns markdown, budgeted.
    bundle = asyncio.run(_call(cache, graph_fixture_root, "explain", {"query": "area", "repo": root, "budget": 900}))
    assert bundle.startswith("# Evidence for: area"), "step 1: the bundle is markdown"

    # 2. `outline` returns JSON entries.
    outline_payload = json.loads(
        asyncio.run(_call(cache, graph_fixture_root, "outline", {"target": "Registry", "repo": root}))
    )
    assert outline_payload["package"] == "com.example.core", "step 2: the package is reported"
    assert any(entry["name"] == "anonymousShape" for entry in outline_payload["entries"]), "step 2: members are listed"

    # 3. `signatures` returns the exact callers.
    signature_payload = json.loads(
        asyncio.run(_call(cache, graph_fixture_root, "signatures", {"symbol": "Helpers.twice", "repo": root}))
    )
    assert signature_payload["signature"] == "double twice(double value)", "step 3: the signature is reported"
    assert signature_payload["callers"], "step 3: its callers come with it"

    # 4. An ambiguous name is an error payload, not a failure.
    ambiguous = json.loads(asyncio.run(_call(cache, graph_fixture_root, "outline", {"target": "Circle", "repo": root})))
    assert "ambiguous" in ambiguous["error"], "step 4: ambiguity is reported as data"
    assert len(ambiguous["candidates"]) == 2, "step 4: with every candidate named"


def test_cli_explain_prefers_the_daemon_then_falls_back(
    graph_fixture_root: Path, graph_cache: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`explain` asks the daemon first, and answers in this process with one notice when it cannot."""
    from zemble.daemon import client
    from zemble.daemon.protocol import DaemonUnavailable
    from zemble.evidence import cli as evidence_cli

    root = str(graph_fixture_root)

    # 1. A daemon answer is rendered as-is, without scanning the workspace here.
    warm = {"bundle": {"items": [{"reason": "from the daemon"}]}, "markdown": "# Evidence for: warm"}
    monkeypatch.setattr(client, "call", lambda cmd, args, **kwargs: warm)
    monkeypatch.setattr(evidence_cli, "ensure_graph", lambda *args, **kwargs: pytest.fail("no in-process scan"))
    assert _run(monkeypatch, graph_fixture_root, "explain", root, "warm") == 0, "step 1: the daemon answer is a success"
    assert capsys.readouterr().out.startswith("# Evidence for: warm"), "step 1: printed verbatim"

    # 2. An unreachable daemon is one stderr line, then the real in-process bundle.
    monkeypatch.undo()

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise DaemonUnavailable("not running (ENOENT)")

    monkeypatch.setattr(client, "call", _refuse)
    assert _run(monkeypatch, graph_fixture_root, "explain", root, "how is an area computed", "--budget", "1200") == 0, (
        "step 2: the fallback still answers"
    )
    captured = capsys.readouterr()
    assert "daemon unavailable (not running (ENOENT)); running in-process" in captured.err, "step 2: one honest line"
    assert captured.out.startswith("# Evidence for: how is an area computed"), "step 2: built here instead"


def test_mcp_explain_prefers_the_daemon(
    graph_fixture_root: Path, graph_cache: Path, cache: IndexCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MCP tool takes the daemon's bundle rather than building a second index of its own."""
    import asyncio

    from zemble.daemon import client

    asked: list[str] = []

    def _answer(cmd: str, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        asked.append(cmd)
        return {"bundle": {"items": [{"reason": "from the daemon"}]}, "markdown": "# Evidence for: warm"}

    monkeypatch.setattr(client, "call", _answer)
    with patch("zemble.mcp.ZembleIndex.from_path", side_effect=AssertionError("must not index here")):
        server = create_server(cache)
        result = asyncio.run(server.call_tool("explain", {"query": "warm", "repo": str(graph_fixture_root)}))
    assert result[0].text == "# Evidence for: warm", "the daemon's markdown is what the tool returns"
    assert asked == ["explain"], "and it asked for exactly that command"
