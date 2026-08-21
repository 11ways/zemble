import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import zemble
from tests.conftest import FakeEmbedder, make_chunk
from zemble.daemon import client as daemon_client
from zemble.daemon.protocol import CommandRefused, DaemonUnavailable
from zemble.index.chunk_store import resolve_chunk
from zemble.index_cache import CACHE_MAX_SIZE, IndexCache
from zemble.mcp import create_server, serve
from zemble.types import Chunk, ContentType, SearchResult
from zemble.utils import format_results, is_git_url


def _tool_text(result: Any) -> str:
    """Extract the text string from a FastMCP call_tool result."""
    return result[0].text


async def _call_tool(
    cache: IndexCache,
    tool: str,
    args: dict[str, Any],
    *,
    index_method: str,
    index_return: list[SearchResult],
    index_chunks: list[Chunk] | None = None,
) -> str:
    """Patch ZembleIndex.from_path with a fake index and invoke the tool, returning the text."""
    fake_index = MagicMock()
    # An unfiltered view of an index is the index itself, which is what the real one returns.
    fake_index.filtered.return_value = fake_index
    getattr(fake_index, index_method).return_value = index_return
    if index_chunks is not None:
        fake_index.chunks = index_chunks
        fake_index.chunk_at.side_effect = lambda file_path, line: resolve_chunk(index_chunks, file_path, line)
        fake_index.chunks_of.side_effect = lambda file_path: [c for c in index_chunks if c.file_path == file_path]
        fake_index.indexed_paths.return_value = sorted({c.file_path for c in index_chunks})
    with patch("zemble.mcp.ZembleIndex.from_path", return_value=fake_index):
        server = create_server(cache)
        result = await server.call_tool(tool, args)
    return _tool_text(result)


@pytest.fixture()
def cache(mock_embedder: FakeEmbedder) -> IndexCache:
    """An IndexCache backed by a stub embedder."""
    c = IndexCache()
    c._embedder = mock_embedder
    c._model_ready.set()
    return c


def test_resolve_chunk() -> None:
    """_resolve_chunk returns the correct chunk and handles boundary and miss cases."""
    interior = make_chunk("line1\nline2\nline3", "src/a.py")  # start=1, end=3
    boundary = make_chunk("last line", "src/a.py")  # start=1, end=1 (single-line)

    # Line strictly inside a multi-line chunk hits the early-return path.
    assert resolve_chunk([interior], "src/a.py", 2) is interior

    # Line equal to end_line of a single-line chunk hits the fallback path.
    assert resolve_chunk([boundary], "src/a.py", 1) is boundary

    # Unknown file returns None.
    assert resolve_chunk([interior], "src/other.py", 1) is None

    # Line out of range returns None.
    assert resolve_chunk([interior], "src/a.py", 99) is None

    # Separator mismatch (e.g. backslash-stored path, forward-slash query) still matches.
    backslash_chunk = make_chunk("line1\nline2\nline3", "src\\a.py")
    assert resolve_chunk([backslash_chunk], "src/a.py", 2) is backslash_chunk


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("https://github.com/org/repo", True),
        ("http://github.com/org/repo", True),
        ("git://github.com/org/repo", True),
        ("ssh://git@github.com/org/repo", True),
        ("git+ssh://git@github.com/org/repo", True),
        ("file:///tmp/repo", True),
        ("git@github.com:org/repo", True),  # scp-like
        ("/local/path/to/repo", False),
        ("./relative/path", False),
        ("repo_name", False),
    ],
)
def test_is_git_url(path: str, expected: bool) -> None:
    """Remote git URLs are detected; local paths are not."""
    assert is_git_url(path) is expected


@pytest.mark.parametrize(
    ("max_snippet_lines", "has_content", "content_key"),
    [
        (None, True, "content"),
        (3, True, "content"),
        (0, False, None),
    ],
    ids=["full", "truncated", "location_only"],
)
def test_format_results(max_snippet_lines: int | None, has_content: bool, content_key: str | None) -> None:
    """format_results: consistent flat schema regardless of max_snippet_lines."""
    empty_out = format_results("query", [], max_snippet_lines)
    assert empty_out == {"query": "query", "results": []}

    chunks = [make_chunk(f"line1\nline2\nline3\nline4\ndef fn_{i}(): pass", f"f{i}.py") for i in range(3)]
    results = [SearchResult(chunk=c, score=round(0.1 * (i + 1), 3)) for i, c in enumerate(chunks)]
    out = format_results("foo", results, max_snippet_lines)
    assert out["query"] == "foo"
    for entry in out["results"]:
        assert "file_path" in entry
        assert "start_line" in entry
        assert "end_line" in entry
        assert "score" in entry
        assert "chunk" not in entry
        if has_content:
            assert content_key in entry
            if max_snippet_lines is not None:
                assert entry[content_key].count("\n") < max_snippet_lines
        else:
            assert "content" not in entry


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("source", "patch_target"),
    [
        ("local_tmp_path", "from_path"),
        ("https://github.com/org/repo", "from_git"),
    ],
    ids=["local_path", "git_url"],
)
async def test_index_cache_builds_and_caches(cache: IndexCache, tmp_path: Path, source: str, patch_target: str) -> None:
    """IndexCache.get() builds via the correct ZembleIndex.* entrypoint and caches subsequent calls."""
    resolved_source = str(tmp_path) if source == "local_tmp_path" else source
    fake_index = MagicMock()
    with (
        patch(f"zemble.mcp.ZembleIndex.{patch_target}", return_value=fake_index) as mock_build,
        patch("zemble.index_cache.save_index_to_cache") as mock_save,
        patch("zemble.index_cache.get_validated_cache", return_value=Path("/fake/cache")),
    ):
        first = await cache.get(resolved_source)
        second = await cache.get(resolved_source)
        docs_first = await cache.get(resolved_source, content=(ContentType.DOCS,))
        docs_second = await cache.get(resolved_source, content=(ContentType.DOCS,))
    assert first is fake_index
    assert second is fake_index
    assert docs_first is fake_index
    assert docs_second is fake_index
    assert [call.kwargs["content"] for call in mock_build.call_args_list] == [
        (ContentType.CODE,),
        (ContentType.DOCS,),
    ]
    assert mock_save.call_count == 2


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("source", "patch_target", "expected_build_calls", "validate_called"),
    [
        ("local_tmp_path", "from_path", 2, True),
        ("https://github.com/org/repo", "from_git", 1, False),
    ],
    ids=["local_path_rebuilds_when_stale", "git_url_skips_revalidation"],
)
async def test_index_cache_staleness_check_scope(
    cache: IndexCache,
    tmp_path: Path,
    source: str,
    patch_target: str,
    expected_build_calls: int,
    validate_called: bool,
) -> None:
    """Local paths are revalidated (and rebuilt when stale) on every get(); git URLs never are."""
    resolved_source = str(tmp_path) if source == "local_tmp_path" else source
    with (
        patch(f"zemble.mcp.ZembleIndex.{patch_target}", return_value=MagicMock()) as mock_build,
        patch("zemble.index_cache.save_index_to_cache"),
        patch("zemble.index_cache.get_validated_cache", return_value=None) as mock_validate,
        # Disable the cooldown: real build duration (here, just thread-dispatch overhead) would
        # otherwise sometimes exceed the gap between the two get() calls below, flaking the test.
        patch("zemble.index_cache.MIN_REVALIDATE_FACTOR", 0),
    ):
        await cache.get(resolved_source)
        await cache.get(resolved_source)
    assert mock_build.call_count == expected_build_calls
    assert mock_validate.called is validate_called


@pytest.mark.anyio
async def test_index_cache_skips_staleness_check_during_cooldown(cache: IndexCache, tmp_path: Path) -> None:
    """A slow-to-build local path is not revalidated again until its cooldown elapses."""
    cache_key = cache._compute_cache_key(str(tmp_path))
    cache._tasks[cache_key] = asyncio.create_task(_succeed())
    await asyncio.sleep(0)  # let the task finish
    cache._revalidate_after[cache_key] = time.monotonic() + 30.0  # a build that took 10s, just finished
    with patch("zemble.index_cache.get_validated_cache") as mock_validate:
        await cache._evict_if_stale(cache_key)
    mock_validate.assert_not_called()


async def _succeed() -> MagicMock:
    return MagicMock()


@pytest.mark.anyio
async def test_index_cache_skips_staleness_check_for_failed_task(cache: IndexCache, tmp_path: Path) -> None:
    """A cached entry that finished with an exception is not revalidated; it is left for the normal retry path."""

    async def _raise() -> MagicMock:
        raise RuntimeError("boom")

    cache_key = cache._compute_cache_key(str(tmp_path))
    cache._tasks[cache_key] = asyncio.create_task(_raise())
    await asyncio.sleep(0)  # let the task finish
    with patch("zemble.index_cache.get_validated_cache") as mock_validate:
        await cache._evict_if_stale(cache_key)
    mock_validate.assert_not_called()


@pytest.mark.anyio
async def test_index_cache_does_not_evict_entry_replaced_during_validation(cache: IndexCache, tmp_path: Path) -> None:
    """If a concurrent caller already replaced a stale entry, _evict_if_stale must not evict the new one."""
    cache_key = cache._compute_cache_key(str(tmp_path))
    cache._tasks[cache_key] = asyncio.create_task(_succeed())
    await asyncio.sleep(0)
    cache._revalidate_after[cache_key] = 0.0  # cooldown already elapsed

    replacement_task = object()

    def _replace_entry_then_report_stale(*args: object, **kwargs: object) -> None:
        # Simulate a concurrent get() winning the race and installing a fresh task first.
        cache._tasks[cache_key] = replacement_task  # type: ignore[assignment]
        return None

    with patch("zemble.index_cache.get_validated_cache", side_effect=_replace_entry_then_report_stale):
        await cache._evict_if_stale(cache_key)
    assert cache._tasks.get(cache_key) is replacement_task


@pytest.mark.anyio
async def test_index_cache_evicts_on_failure(cache: IndexCache, tmp_path: Path) -> None:
    """A failed build evicts the entry so the next call can retry."""
    call_count = 0

    def _failing_then_ok(path: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("build failed")
        return MagicMock()

    with patch("zemble.mcp.ZembleIndex.from_path", side_effect=_failing_then_ok):
        with pytest.raises(RuntimeError, match="build failed"):
            await cache.get(str(tmp_path))
        result = await cache.get(str(tmp_path))
    assert result is not None
    assert call_count == 2


@pytest.mark.anyio
async def test_index_cache_ignores_cache_save_failure(cache: IndexCache, tmp_path: Path) -> None:
    """A cache save failure must not fail the MCP request."""
    fake_index = MagicMock()
    with (
        patch("zemble.mcp.ZembleIndex.from_path", return_value=fake_index),
        patch("zemble.index_cache.save_index_to_cache", side_effect=RuntimeError("save failed")),
    ):
        assert await cache.get(str(tmp_path)) is fake_index


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("search", {"query": "foo", "repo": "https://github.com/x/y"}),
        ("find_related", {"file_path": "src/foo.py", "line": 1, "repo": "https://github.com/x/y"}),
    ],
)
async def test_tool_index_failure(cache: IndexCache, tool: str, args: dict[str, object]) -> None:
    """Both tools return a friendly error message when indexing fails."""
    with patch("zemble.mcp.ZembleIndex.from_git", side_effect=RuntimeError("clone failed")):
        server = create_server(cache)
        result = await server.call_tool(tool, args)
    text = _tool_text(result)
    assert "Failed to index" in text
    assert "clone failed" in text


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "args", "method", "results", "chunks", "expected_substrings"),
    [
        pytest.param(
            "search",
            {"query": "bar", "repo": "/some/path"},
            "search",
            [SearchResult(chunk=make_chunk("def bar(): pass", "src/bar.py"), score=0.9)],
            None,
            ["bar", "0.9"],
            id="search_with_results",
        ),
        pytest.param(
            "search",
            {"query": "nothing", "repo": "/some/path"},
            "search",
            [],
            None,
            ["No results found"],
            id="search_no_results",
        ),
        pytest.param(
            "find_related",
            {"file_path": "src/foo.py", "line": 1, "repo": "/some/path"},
            "find_related",
            [SearchResult(chunk=make_chunk("class Foo: pass", "src/foo.py"), score=0.8)],
            [make_chunk("class Foo: pass", "src/foo.py")],
            ["src/foo.py:1", "0.8"],
            id="find_related_with_results",
        ),
        pytest.param(
            "find_related",
            {"file_path": "src/foo.py", "line": 1, "repo": "/some/path"},
            "find_related",
            [],
            [make_chunk("class Foo: pass", "src/foo.py")],
            ["No related chunks found"],
            id="find_related_no_results",
        ),
        pytest.param(
            "find_related",
            {"file_path": "src/foo.py", "line": 40, "repo": "/some/path"},
            "find_related",
            [],
            [make_chunk("class Foo: pass", "src/foo.py")],
            ["'src/foo.py' is indexed, but no chunk covers line 40", "1-1"],
            id="find_related_line_outside_every_chunk",
        ),
        pytest.param(
            "find_related",
            {"file_path": "workspace/src/foo.py", "line": 1, "repo": "/some/path"},
            "find_related",
            [],
            [make_chunk("class Foo: pass", "src/foo.py")],
            ["No indexed file matches 'workspace/src/foo.py'", "Did you mean 'src/foo.py'?", "relative to the repo"],
            id="find_related_unknown_file",
        ),
    ],
)
async def test_tool_output(
    cache: IndexCache,
    tool: str,
    args: dict[str, Any],
    method: str,
    results: list[SearchResult],
    chunks: list[Chunk] | None,
    expected_substrings: list[str],
) -> None:
    """Search and find_related format results (or an empty-state message) through the server."""
    text = await _call_tool(cache, tool, args, index_method=method, index_return=results, index_chunks=chunks)
    for substring in expected_substrings:
        assert substring in text


@pytest.mark.anyio
async def test_search_builds_exact_content_indexes(
    cache: IndexCache,
    mock_embedder: FakeEmbedder,
    tmp_project: Path,
) -> None:
    """MCP search lazily builds the exact requested content index."""
    (tmp_project / "settings.toml").write_text("project = 'zemble'\n")
    expected = [
        (None, {".py"}),
        ("docs", {".md"}),
        ("config", {".toml"}),
        ("all", {".md", ".py", ".toml"}),
    ]

    with (
        patch("zemble.index.index.load_embedder", return_value=mock_embedder),
        patch("zemble.index_cache.save_index_to_cache"),
    ):
        server = create_server(cache)
        for content, expected_suffixes in expected:
            args = {"query": "project", "repo": str(tmp_project), "top_k": 20}
            if content is not None:
                args["content"] = content
            result = await server.call_tool("search", args)
            payload = json.loads(_tool_text(result))
            assert {Path(item["file_path"]).suffix for item in payload["results"]} == expected_suffixes


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("load_err", "stdio_yields"),
    [
        (None, True),
        (RuntimeError("boom"), True),
        (None, False),
    ],
    ids=["model_loads", "model_load_fails", "cancel_pending_init"],
)
async def test_serve_runs_stdio(
    load_err: Exception | None,
    stdio_yields: bool,
) -> None:
    """serve() runs stdio and handles all background init outcomes without raising."""

    async def fake_stdio() -> None:
        if stdio_yields:
            await asyncio.sleep(0.05)  # let the background init task run

    load_kwargs = {"side_effect": load_err} if load_err else {"return_value": FakeEmbedder()}
    with (
        patch("zemble.index_cache.load_embedder", **load_kwargs),
        patch("mcp.server.fastmcp.FastMCP.run_stdio_async", side_effect=fake_stdio) as mock_run,
    ):
        await serve()

    mock_run.assert_called_once()


@pytest.mark.anyio
async def test_serve_opens_stdio_before_model_loads() -> None:
    """Stdio must open before load_embedder() finishes."""
    stdio_opened = threading.Event()

    def blocking_load_model() -> FakeEmbedder:
        assert stdio_opened.wait(timeout=1.0), "stdio did not open"
        return FakeEmbedder()

    async def fake_run_stdio() -> None:
        stdio_opened.set()
        await asyncio.sleep(0.05)

    with (
        patch("zemble.index_cache.load_embedder", side_effect=blocking_load_model),
        patch("mcp.server.fastmcp.FastMCP.run_stdio_async", side_effect=fake_run_stdio),
    ):
        await serve()


@pytest.mark.anyio
async def test_index_cache_awaits_model(tmp_path: Path) -> None:
    """get() blocks until the model is installed, then proceeds."""
    cache = IndexCache()  # no model yet
    fake_index = MagicMock()
    with patch("zemble.mcp.ZembleIndex.from_path", return_value=fake_index):
        get_task = asyncio.create_task(cache.get(str(tmp_path)))
        await asyncio.sleep(0.01)
        assert not get_task.done(), "get() must block until the model is installed"
        cache._embedder = FakeEmbedder()
        cache._model_ready.set()
        result = await asyncio.wait_for(get_task, timeout=1.0)
    assert result is fake_index


@pytest.mark.anyio
async def test_index_cache_propagates_model_error(tmp_path: Path) -> None:
    """If model load fails, awaiting tool calls re-raise the original exception."""
    cache = IndexCache()
    get_task = asyncio.create_task(cache.get(str(tmp_path)))
    await asyncio.sleep(0.01)
    assert not get_task.done()
    cache._model_error = RuntimeError("HF download failed")
    cache._model_ready.set()
    with pytest.raises(RuntimeError, match="HF download failed"):
        await asyncio.wait_for(get_task, timeout=1.0)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("repo", "tool", "extra_args"),
    [
        ("file:///home/user/secret", "search", {"query": "foo"}),
        ("ssh://internal-host/repo", "search", {"query": "foo"}),
        ("git@github.com:org/repo", "search", {"query": "foo"}),
        ("file:///home/user/secret", "find_related", {"file_path": "src/foo.py", "line": 1}),
        ("ssh://internal-host/repo", "find_related", {"file_path": "src/foo.py", "line": 1}),
    ],
    ids=["file_search", "ssh_search", "scp_search", "file_find_related", "ssh_find_related"],
)
async def test_tool_rejects_unsafe_repo(cache: IndexCache, repo: str, tool: str, extra_args: dict[str, object]) -> None:
    """Both tools reject unsafe git transport schemes (ssh://, file://, SCP-form) supplied as repo."""
    server = create_server(cache)
    result = await server.call_tool(tool, {**extra_args, "repo": repo})
    assert "Only https://" in _tool_text(result)


@pytest.mark.anyio
async def test_index_cache_lru_eviction(cache: IndexCache, tmp_path: Path) -> None:
    """IndexCache evicts the least-recently-used entry when the cache is full."""
    dirs = [tmp_path / str(i) for i in range(CACHE_MAX_SIZE + 1)]
    for d in dirs:
        d.mkdir()
    with patch("zemble.mcp.ZembleIndex.from_path", return_value=MagicMock()):
        for d in dirs[:CACHE_MAX_SIZE]:
            await cache.get(str(d))
        first_key = cache._compute_cache_key(str(dirs[0]))
        assert first_key in cache._tasks
        await cache.get(str(dirs[CACHE_MAX_SIZE]))
    assert first_key not in cache._tasks
    assert len(cache._tasks) == CACHE_MAX_SIZE


def test_cache_evict(cache: IndexCache, tmp_path: Path) -> None:
    """evict() removes an existing exact cache entry."""
    key = cache._compute_cache_key(str(tmp_path))
    cache._tasks[key] = MagicMock()
    cache.evict(key)
    assert key not in cache._tasks


def test_cache_evict_missing(cache: IndexCache, tmp_path: Path) -> None:
    """evict() on an unknown key is a no-op."""
    cache.evict(cache._compute_cache_key(str(tmp_path)))  # should not raise


@pytest.mark.anyio
async def test_status_tool_returns_the_runtime_identity(cache: IndexCache) -> None:
    """`status` answers with an object naming the code this server runs, plus its staleness."""
    from zemble.runtime import identity

    server = create_server(cache)
    content = await server.call_tool("status", {})
    payload = json.loads(content[0].text)
    assert isinstance(payload, dict), "the tool hands back an object, encoded once as text"
    assert payload["identity"]["pid"] == identity().pid, "the answering process names itself"
    assert payload["identity"]["zemble_version"] == identity().zemble_version
    assert payload["stale"] is False, "a server started from the current checkout is not stale"
    assert content, "and it renders as content too"


@pytest.mark.anyio
async def test_tool_calls_probe_for_staleness(cache: IndexCache, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every tool call runs the throttled staleness probe before answering."""
    from zemble.runtime import mcp as runtime_mcp

    probes: list[float | None] = []
    monkeypatch.setattr(runtime_mcp, "warn_if_stale", lambda now=None: probes.append(now))
    server = create_server(cache)
    await server.call_tool("status", {})
    assert probes == [None], "the call went through the staleness-aware server"


def _double_encoding_complaint(name: str, result: Any) -> str | None:
    """Return why one tool result is not handed over exactly once, or None when it is clean.

    Two shapes are hunted: text that is itself a JSON string, and the SDK's structured
    twin. FastMCP wraps any non-object return annotation as ``{"result": value}`` in
    `structuredContent`, and a client that prefers that lane then renders formatted text
    as a JSON object; every tool is registered with ``structured_output=False`` so the
    text lane is the only one.
    """
    content, structured = result if isinstance(result, tuple) else (result, None)
    text = content[0].text
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        decoded = None
    if isinstance(decoded, str):
        return f"{name}: the text content is a JSON string holding {decoded[:40]!r}"
    if structured is not None:
        return f"{name}: answered with structured content {str(structured)[:40]!r} next to the text"
    return None


@pytest.mark.anyio
async def test_no_tool_returns_json_inside_a_json_string(
    cache: IndexCache,
    mock_embedder: FakeEmbedder,
    graph_fixture_root: Path,
    graph_cache: Path,
) -> None:
    """Every registered tool hands its answer over once: text stays text, JSON stays an object."""
    bad = ([MagicMock(text=json.dumps(json.dumps({"root": "x"})))], None)
    assert _double_encoding_complaint("stub", bad) is not None, "the check itself recognises JSON-in-JSON"
    wrapped = ([MagicMock(text="plain report")], {"result": "plain report"})
    assert _double_encoding_complaint("stub", wrapped) is not None, "the check itself recognises the SDK twin"
    assert _double_encoding_complaint("stub", ([MagicMock(text="plain report")], None)) is None

    root = str(graph_fixture_root)
    dupes_root = str(Path(__file__).parent / "fixtures" / "dedup_lanes")
    calls: list[tuple[str, dict[str, Any]]] = [
        ("status", {}),
        ("search", {"query": "circle", "repo": root}),
        ("find_related", {"file_path": "src/main/java/com/example/core/Circle.java", "line": 5, "repo": root}),
        ("graph_definition", {"symbol": "Shape", "repo": root}),
        ("graph_callers", {"symbol": "Helpers.twice", "repo": root}),
        ("graph_implementations", {"symbol": "Shape", "repo": root}),
        ("graph_tests_of", {"symbol": "com.example.core.Circle", "repo": root}),
        ("graph_neighbors", {"symbol": "com.example.core.Circle", "repo": root}),
        ("outline", {"target": "Shape", "repo": root}),
        ("signatures", {"symbol": "Helpers.twice", "repo": root}),
        ("explain", {"query": "circle area", "repo": root, "budget": 400}),
        ("home", {"description": "compute the area of a shape", "repo": root}),
        ("dupes", {"repo": dupes_root, "kind": "exact"}),
        ("dupes", {"repo": dupes_root, "kind": "exact", "format": "json"}),
    ]

    with (
        patch("zemble.index.index.load_embedder", return_value=mock_embedder),
        patch("zemble.index_cache.save_index_to_cache"),
    ):
        server = create_server(cache)
        listed = await server.list_tools()
        registered = {tool.name for tool in listed}
        assert registered <= {name for name, _ in calls}, "every registered tool is covered by this test"
        schemas = {tool.name for tool in listed if tool.outputSchema is not None}
        assert schemas == set(), "a tool advertising an output schema gets its text wrapped as {'result': ...}"
        complaints = []
        for name, args in calls:
            result = await server.call_tool(name, args)
            assert result[0], f"{name} answered with no content at all"
            complaint = _double_encoding_complaint(name, result)
            if complaint is not None:
                complaints.append(complaint)
    assert complaints == [], "no tool may make its client parse JSON out of JSON"


_MCP_LAUNCHER = """
import sys

from zemble.cli import main

sys.argv = ["zemble"]
main()
"""


def _processes_running(needle: str) -> list[int]:
    """Return the pids of every process whose command line mentions this string."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if needle in cmdline:
            found.append(int(entry.name))
    return found


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the survivor scan reads /proc")
def test_mcp_server_exits_on_stdin_eof(tmp_path: Path) -> None:
    """A stdio server whose stdin is already at EOF leaves, and leaves nothing running behind it."""
    launcher = tmp_path / "run_mcp.py"
    launcher.write_text(_MCP_LAUNCHER, encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(zemble.__file__).resolve().parent.parent),
        "HF_HUB_OFFLINE": "1",
        "ZEMBLE_DAEMON": "0",
    }
    process = subprocess.Popen(
        [sys.executable, str(launcher)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        process.communicate(timeout=60)
    except subprocess.TimeoutExpired:  # pragma: no cover - the regression this test guards
        process.kill()
        process.communicate()
        pytest.fail("the MCP server did not exit on stdin EOF")
    assert process.returncode == 0, "an EOF exit is a clean exit"
    survivors = _processes_running(str(launcher))
    assert survivors == [], f"the server left processes behind: {survivors}"


@pytest.mark.anyio
async def test_a_daemon_refusal_is_reported_once(cache: IndexCache, monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal is the answer: the tool reports it and never rebuilds the index it just refused."""

    def _refuse(cmd: str, args: dict[str, Any], **kwargs: Any) -> Any:
        raise CommandRefused("Refusing to index /some/path: too big. Exclude paths with .zembleignore")

    monkeypatch.setattr(daemon_client, "call", _refuse)
    monkeypatch.setattr(
        "zemble.mcp.ZembleIndex.from_path", lambda *args, **kwargs: pytest.fail("must not build after a refusal")
    )
    server = create_server(cache)
    text = _tool_text(await server.call_tool("search", {"query": "anything", "repo": "/some/path"}))
    assert "Refusing to index" in text, "the refusal reaches the caller"
    assert ".zembleignore" in text, "with its remedies"


@pytest.mark.anyio
async def test_a_daemon_outage_still_falls_back(cache: IndexCache, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable daemon is an outage, not an answer, so this process still answers."""

    def _unavailable(cmd: str, args: dict[str, Any], **kwargs: Any) -> Any:
        raise DaemonUnavailable("not running (ENOENT)")

    monkeypatch.setattr(daemon_client, "call", _unavailable)
    fake_index = MagicMock()
    fake_index.filtered.return_value = fake_index
    fake_index.search.return_value = []
    with patch("zemble.mcp.ZembleIndex.from_path", return_value=fake_index):
        server = create_server(cache)
        result = await server.call_tool("search", {"query": "anything", "repo": "/some/path"})
    assert "No results found" in _tool_text(result), "the in-process answer came back"


@pytest.mark.anyio
async def test_search_forwards_paths_and_exclude(cache: IndexCache, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both filters ride the daemon protocol, and reach the index when this process answers."""
    sent: dict[str, Any] = {}

    def _record(cmd: str, args: dict[str, Any], **kwargs: Any) -> Any:
        sent.update(args)
        return {"query": "q", "results": []}

    monkeypatch.setattr(daemon_client, "call", _record)
    server = create_server(cache)
    await server.call_tool("search", {"query": "q", "repo": "/some/path", "paths": ["src"], "exclude": ["vendor/"]})
    assert sent["paths"] == ["src"] and sent["exclude"] == ["vendor/"], "the filters went over the wire"

    monkeypatch.setattr(daemon_client, "call", lambda *args, **kwargs: (_ for _ in ()).throw(DaemonUnavailable("off")))
    fake_index = MagicMock()
    fake_index.filtered.return_value = fake_index
    fake_index.search.return_value = []
    with patch("zemble.mcp.ZembleIndex.from_path", return_value=fake_index):
        server = create_server(cache)
        await server.call_tool("search", {"query": "q", "repo": "/some/path", "paths": ["src"], "exclude": ["v/"]})
    fake_index.filtered.assert_called_once_with(["src"], ["v/"])
