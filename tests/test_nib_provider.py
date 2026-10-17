"""nib provider mode: the parts each tool family lays out, and `zemble nib-provider` over real stdio."""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import zemble
from tests.conftest import FakeEmbedder, make_chunk
from tests.test_mcp import _nib_part_complaint
from zemble.evidence.answers import explain_payload
from zemble.evidence.bundle import split_truncation, truncate_lines
from zemble.evidence.mcp import explain_parts
from zemble.graph import SqliteGraphProvider, build_graph
from zemble.index import ZembleIndex
from zemble.index_cache import IndexCache
from zemble.mcp import _results_parts, create_server
from zemble.nib.parts import NibAnswer, nib_answer, source_part
from zemble.types import SearchResult
from zemble.utils import format_results

_SOURCE = textwrap.dedent(
    """\
    def first():
        a = 1
        b = 2
        c = 3
        return a + b + c


    def second():
        return 2
    """
)


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    """A small Python file whose lines the parts point at."""
    path = tmp_path / "pkg" / "mod.py"
    path.parent.mkdir()
    path.write_text(_SOURCE, encoding="utf-8")
    return path


def _file_lines(path: Path, start: int, end: int) -> str:
    """Return lines `start`..`end` of a file joined by newlines."""
    return "\n".join(path.read_text(encoding="utf-8").split("\n")[start - 1 : end])


def test_source_part_widens_a_chunk_cut_inside_its_lines(source_file: Path, tmp_path: Path) -> None:
    """A chunk that starts after a line's indentation still reports the whole lines it sits on."""
    part = source_part(str(tmp_path), "pkg/mod.py", 2, "a = 1\n    b = 2\n    c")
    assert part == {
        "path": str(source_file),
        "start": 2,
        "end": 4,
        "indexed": _file_lines(source_file, 2, 4),
    }, "step 1: absolute path, whole lines"

    source_file.write_text(_SOURCE.replace("b = 2", "b = 20"), encoding="utf-8")
    stale = source_part(str(tmp_path), "pkg/mod.py", 2, "a = 1\n    b = 2\n    c")
    assert stale is not None and stale["indexed"] == "a = 1\n    b = 2\n    c", (
        "step 2: a changed file gets zemble's text"
    )
    assert source_part(str(tmp_path), "pkg/mod.py", 2, "") is None, "step 3: no text, no location"


@pytest.mark.parametrize(("max_snippet_lines", "end"), [(None, 5), (2, 2), (0, None)], ids=["full", "two", "none"])
def test_search_parts_honour_max_snippet_lines(
    source_file: Path, tmp_path: Path, max_snippet_lines: int | None, end: int | None
) -> None:
    """The located span is the snippet the text form shows: whole chunk, first N lines, or no lines."""
    chunk = make_chunk(_SOURCE.split("\n\n\n")[0], "pkg/mod.py")
    payload = format_results("first", [SearchResult(chunk=chunk, score=1.0)], max_snippet_lines)
    parts = _results_parts(str(tmp_path), payload)
    assert parts is not None
    texts = [part["text"] for part in parts if "text" in part]
    assert texts == ["first: 1 result(s)", "#1  pkg/mod.py:1-5  score 100"], "prose names the hit, never its code"
    locations = [part for part in parts if "path" in part]
    if end is None:
        assert locations == [], "max_snippet_lines=0 shows no lines"
        return
    assert locations == [
        {"path": str(source_file), "start": 1, "end": end, "indexed": _file_lines(source_file, 1, end)}
    ]


def test_results_parts_leave_an_error_as_text() -> None:
    """A payload without results has no layout, so it travels as text only."""
    assert _results_parts("/tmp", {"error": "No results found."}) is None
    assert _results_parts("/tmp", "Refusing to index /x") is None


def test_nib_answer_is_plain_outside_nib_mode_and_for_remote_repos(tmp_path: Path) -> None:
    """The default server and a git URL repo get the plain payload back, unchanged."""
    default = create_server(IndexCache())
    nib = create_server(IndexCache(), nib=True)
    payload = {"query": "q", "results": []}
    assert nib_answer(default, str(tmp_path), payload, _results_parts) is payload, "step 1: default mode"
    assert nib_answer(nib, "https://github.com/org/repo", payload, _results_parts) is payload, "step 2: remote repo"
    wrapped = nib_answer(nib, str(tmp_path), payload, _results_parts)
    assert isinstance(wrapped, NibAnswer) and wrapped.plain is payload, "step 3: nib mode wraps the same payload"
    assert nib_answer(nib, None, "report") == NibAnswer("report", None), "step 4: no builder means text only"


def test_split_truncation_inverts_truncate_lines() -> None:
    """The marker `truncate_lines` appends is split off again, with the count it names."""
    text = "\n".join(f"line {n}" for n in range(1, 11))
    assert split_truncation(truncate_lines(text, 4)) == ("line 1\nline 2\nline 3\nline 4", 6)
    assert split_truncation(text) == (text, 0), "an uncut text comes back whole"


def _item(**fields: Any) -> dict[str, Any]:
    """An explain item as the wire carries it, verbatim content by default."""
    return {
        "kind": "chunk",
        "file_path": "pkg/mod.py",
        "start_line": 1,
        "end_line": 5,
        "reason": "search hit #1",
        "text": "",
        "tokens": 10,
        "presentation": "content",
        "tier": 0,
        "verbatim": True,
        **fields,
    }


def test_explain_parts_show_file_lines_as_locations_and_renderings_as_prose(source_file: Path, tmp_path: Path) -> None:
    """Verbatim items are located (a truncation as a note), the rest stay prose."""
    first = _SOURCE.split("\n\n\n")[0]
    items = [
        _item(text=truncate_lines(first, 2), presentation="truncated"),
        _item(kind="outline", text="class Thing\n  method f()  L1-5", verbatim=False, reason="outline"),
        _item(kind="caller", text="", presentation="location", reason="caller"),
        {key: value for key, value in _item(text=first, reason="old daemon").items() if key != "verbatim"},
    ]
    bundle = {
        "query": "first",
        "intent": "unknown",
        "intent_rule": "empty",
        "order": "default",
        "total_tokens": 40,
        "unlisted_omissions": 0,
        "items": items,
        "omitted": [{"kind": "test", "file_path": "t.py", "start_line": 3, "end_line": 4, "reason": "no room"}],
    }
    parts = explain_parts(str(tmp_path), {"bundle": bundle, "markdown": ""})
    assert parts[0] == {
        "text": "# Evidence for: first\n\nintent: unknown (rule: empty; order: default)\n4 item(s), ~40 tokens."
    }
    assert parts[1] == {"text": "## pkg/mod.py:1-5  (search hit #1)\n(showing 2 lines; 3 more not shown)"}
    assert parts[2] == {"path": str(source_file), "start": 1, "end": 2, "indexed": _file_lines(source_file, 1, 2)}
    assert parts[3] == {"text": "## pkg/mod.py:1-5  (outline)\nclass Thing\n  method f()  L1-5"}
    assert parts[4] == {"text": "## pkg/mod.py:1-5  (caller)\n(location only)"}
    assert parts[5] == {"text": "## pkg/mod.py:1-5  (old daemon)"}, "an item of unknown kind of text shows no text"
    assert parts[6] == {"text": "## Not included (locations only)\n\n- t.py:3-4  (no room)"}
    assert len(parts) == 7


def test_a_real_bundle_marks_only_file_lines_verbatim(graph_fixture_root: Path, graph_cache: Path) -> None:
    """Over a real index and graph, outline-style items are prose and every location matches its file."""
    root = str(graph_fixture_root)
    build_graph(root)
    graph = SqliteGraphProvider(root)
    try:
        index = ZembleIndex.from_path(graph_fixture_root, embedder=FakeEmbedder())
        payload = explain_payload(index, graph, "circle area", 4000, 20)
    finally:
        graph.close()
    items = payload["bundle"]["items"]
    shown = [item for item in items if item["presentation"] != "location"]
    assert {item["kind"] for item in shown if not item["verbatim"]} <= {"outline", "implementation", "supertype"}
    assert any(item["verbatim"] for item in shown), "the fixture yields located evidence"
    parts = explain_parts(root, payload)
    assert any("path" in part for part in parts)
    assert [c for c in map(_nib_part_complaint, parts) if c is not None] == []


_PROVIDER_LAUNCHER = """
import sys

from zemble.cli import main

sys.argv = ["zemble", "nib-provider", "--no-daemon"]
main()
"""


@pytest.mark.anyio
async def test_nib_provider_answers_over_stdio(tmp_path: Path, graph_fixture_root: Path) -> None:
    """`zemble nib-provider` run in a workspace lists the tools and answers a call with nib's parts."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    launcher = tmp_path / "run_provider.py"
    launcher.write_text(_PROVIDER_LAUNCHER, encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(zemble.__file__).resolve().parent.parent),
        "HF_HUB_OFFLINE": "1",
        "ZEMBLE_CACHE_LOCATION": str(tmp_path / "cache"),
    }
    env.pop("ZEMBLE_DAEMON", None)  # `--no-daemon` is the explicit opt-out a subprocess needs
    params = StdioServerParameters(command=sys.executable, args=[str(launcher)], env=env, cwd=str(graph_fixture_root))
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        names = {tool.name for tool in (await session.list_tools()).tools}
        assert {"search", "graph_definition", "outline", "explain"} <= names
        result = await session.call_tool("graph_definition", {"symbol": "Shape"})
    assert not result.isError
    structured = result.structuredContent
    assert structured is not None and structured["nib"] == 1
    shape = graph_fixture_root / "src/main/java/com/example/core/Shape.java"
    located = [part for part in structured["parts"] if "path" in part]
    assert located and located[0]["path"] == str(shape), "the default repo is the working directory, made absolute"
    assert "interface Shape" in _file_lines(shape, located[0]["start"], located[0]["end"])
