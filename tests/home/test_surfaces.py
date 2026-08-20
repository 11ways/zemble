"""The `home` surfaces: the CLI, the MCP tool, and what they do without a daemon."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import FakeEmbedder
from zemble.cli import _cli_main
from zemble.graph.cli import EXIT_NOT_FOUND
from zemble.index_cache import IndexCache
from zemble.mcp import create_server
from zemble.types import Chunk, SearchResult

_CIRCLE = "src/main/java/com/example/core/Circle.java"

_CONFIG = """
    order = ["core", "util", "app"]

    [modules]
    core = "src/main/java/com/example/core/**"
    util = "src/main/java/com/example/util/**"
    app = "src/main/java/com/example/app/**"

    [[forbidden]]
    from = "core"
    to = "app"
    why = "the core never reaches into an application"

    [skills]
    core = ["core-shapes"]

    [[rules]]
    text = "Nothing lands without a wired consumer and a test"
"""


@pytest.fixture
def workspace(tmp_path: Path, graph_fixture_root: Path, graph_cache: Path) -> Path:
    """A copy of the graph fixture workspace that declares its own modules."""
    root = tmp_path / "workspace"
    shutil.copytree(graph_fixture_root, root)
    (root / ".zemble").mkdir()
    (root / ".zemble" / "home.toml").write_text(textwrap.dedent(_CONFIG).strip() + "\n", encoding="utf-8")
    return root


def _fake_index(root: Path) -> MagicMock:
    """An index that returns one real chunk of the workspace."""
    content = (root / _CIRCLE).read_text(encoding="utf-8")
    chunk = Chunk(content=content, file_path=_CIRCLE, start_line=1, end_line=content.count("\n") + 1, language="java")
    index = MagicMock()
    index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    index.find_related.return_value = []
    index.content = ()
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


def test_cli_journey(workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Walk `zemble home` through markdown, JSON, a nothing-found answer and a bad config."""
    root = str(workspace)

    # 1. The default rendering is the markdown answer.
    assert _run(monkeypatch, workspace, "home", root, "compute the area of a shape") == 0, "step 1: it succeeds"
    output = capsys.readouterr().out
    assert output.startswith("# Home for: compute the area of a shape"), "step 1: the description heads the answer"
    for heading in ("## Existing mechanisms", "## Candidate homes", "## Verdict", "## Checklist"):
        assert heading in output, f"step 1: {heading} is rendered"
    assert "core-shapes" in output, "step 1: the workspace's skill for the candidate is echoed"

    # 2. `--json` gives the same answer as data.
    _run(monkeypatch, workspace, "home", root, "compute the area of a shape", "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] in ("EXTEND_EXISTING", "NEW_MECHANISM", "UNCERTAIN"), "step 2: a verdict is carried"
    assert payload["candidates"][0]["module"] == "core", "step 2: the module comes from the declared globs"
    assert payload["mechanisms"], "step 2: the symbol behind the hit is described"

    # 3. Nothing found is exit 1, not a crash.
    def _empty(*args: Any, **kwargs: Any) -> MagicMock:
        index = _fake_index(workspace)
        index.search.return_value = []
        return index

    from zemble.graph import cli as graph_cli

    graph_cli._refreshed.clear()
    monkeypatch.setattr(sys, "argv", ["zemble", "home", root, "quantum teleportation"])
    with patch("zemble.cli.ZembleIndex.from_path", side_effect=_empty):
        try:
            _cli_main()
        except SystemExit as exit_code:
            assert int(exit_code.code or 0) == EXIT_NOT_FOUND, "step 3: an empty answer exits 1"
    assert "UNCERTAIN" in capsys.readouterr().out, "step 3: and says it is uncertain"

    # 4. A malformed config refuses loudly instead of answering generically.
    (workspace / ".zemble" / "home.toml").write_text("[modules]\ncore = 12\n", encoding="utf-8")
    assert _run(monkeypatch, workspace, "home", root, "compute the area of a shape") == 1, "step 4: a bad config fails"
    assert "must be a string or a list of strings" in capsys.readouterr().err, "step 4: with the reason"


@pytest.fixture
def cache(mock_embedder: FakeEmbedder) -> IndexCache:
    """An index cache backed by the deterministic test embedder."""
    prepared = IndexCache()
    prepared._embedder = mock_embedder
    prepared._model_ready.set()
    return prepared


async def _call(cache: IndexCache, root: Path, args: dict[str, Any]) -> str:
    """Invoke the `home` MCP tool with a fake index in place and return its text."""
    with patch("zemble.mcp.ZembleIndex.from_path", return_value=_fake_index(root)):
        server = create_server(cache)
        result = await server.call_tool("home", args)
    return result[0][0].text


def test_mcp_tool_answers_in_process(workspace: Path, cache: IndexCache) -> None:
    """The tool is registered and answers with the markdown, daemon or no daemon."""
    from zemble.graph import cli as graph_cli

    graph_cli._refreshed.clear()
    answer = asyncio.run(_call(cache, workspace, {"description": "compute an area", "repo": str(workspace)}))
    assert answer.startswith("# Home for: compute an area"), "the tool returns the rendered answer"
    assert "## Verdict" in answer, "verdict included"


def test_mcp_tool_reports_a_broken_config(workspace: Path, cache: IndexCache) -> None:
    """A config that cannot be trusted comes back as the reason, not as a stack trace."""
    from zemble.graph import cli as graph_cli

    graph_cli._refreshed.clear()
    (workspace / ".zemble" / "home.toml").write_text("[[forbidden]]\nfrom = 'core'\n", encoding="utf-8")
    answer = asyncio.run(_call(cache, workspace, {"description": "compute an area", "repo": str(workspace)}))
    assert "needs a non-empty 'to'" in answer, "the tool explains what is wrong with the config"


def test_cli_prefers_the_daemon_and_falls_back_when_it_is_gone(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A daemon answer is printed verbatim; an unavailable daemon is one line and an answer."""
    from zemble.daemon import client
    from zemble.daemon.protocol import DaemonUnavailable
    from zemble.home import cli as home_cli

    monkeypatch.delenv("ZEMBLE_DAEMON", raising=False)
    monkeypatch.setattr(client, "_disabled_reason", None)

    # 1. When the daemon answers, nothing is built in this process.
    served = {
        "home": {"verdict": "NEW_MECHANISM", "mechanisms": [], "candidates": [{"module": "core"}]},
        "markdown": "# Home for: served by the daemon\n",
    }
    real_in_process = home_cli._in_process
    monkeypatch.setattr(client, "call", lambda cmd, args, **kwargs: served)
    monkeypatch.setattr(home_cli, "_in_process", lambda args: pytest.fail("must not answer in-process"))
    assert _run(monkeypatch, workspace, "home", str(workspace), "anything") == 0, "step 1: the daemon answer is used"
    assert "served by the daemon" in capsys.readouterr().out, "step 1: printed verbatim"

    # 2. When it is gone, one honest stderr line and the in-process answer.
    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise DaemonUnavailable("not running (ENOENT)")

    monkeypatch.setattr(client, "call", _refuse)
    monkeypatch.setattr(home_cli, "_in_process", real_in_process)
    assert _run(monkeypatch, workspace, "home", str(workspace), "compute an area") == 0, "step 2: it still answers"
    captured = capsys.readouterr()
    assert "daemon unavailable (not running (ENOENT)); running in-process" in captured.err, "step 2: one honest line"
    assert captured.out.startswith("# Home for: compute an area"), "step 2: answered here instead"
