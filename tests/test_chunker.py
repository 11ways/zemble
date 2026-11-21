import logging
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from semble_grammars import UnsupportedPlatformError

from zemble.chunking.chunking import _DESIRED_CHUNK_LENGTH_CHARS, Chunk, chunk_lines, chunk_source
from zemble.chunking.core import ChunkBoundary, _cached_get_parser, chunk, chunk_with_tree
from zemble.index.files import detect_language, read_file_text

_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clear_parser_cache():
    """Clear the parser cache in between tests."""
    _cached_get_parser.cache_clear()
    yield
    _cached_get_parser.cache_clear()


def test_chunk_lines() -> None:
    """chunk_lines: empty input → []; real input → non-empty chunks starting at line 1."""
    assert chunk_lines("", 23) == []

    content = "\n".join(f"line {i}" for i in range(10))
    chunks = chunk_lines(content, 10)
    assert len(chunks) >= 2
    assert chunks[0].start == 0


def test_chunk_source_empty_string() -> None:
    """chunk_source returns [] for whitespace-only input."""
    assert chunk_source("   \n\n", "foo.py", "python") == []


def test_chunk_source_language() -> None:
    """Check that chunking defaults to line splitting with non-existent and None languages.

    A line-chunked file still gets a capsule; with no parse tree it is the path alone.
    """
    with patch("zemble.chunking.chunking.chunk_lines", wraps=chunk_lines) as chunk_line_spy:
        assert chunk_source("hello", "foo.loki", "loki") == [
            Chunk(
                content="hello",
                file_path="foo.loki",
                start_line=1,
                end_line=1,
                language="loki",
                context="foo.loki foo loki",
            )
        ]
        chunk_line_spy.assert_called_once()
    with patch("zemble.chunking.chunking.chunk_lines", wraps=chunk_lines) as chunk_line_spy:
        assert chunk_source("1+1=3", "foo.json", None) == [
            Chunk(
                content="1+1=3",
                file_path="foo.json",
                start_line=1,
                end_line=1,
                language=None,
                context="foo.json foo json",
            )
        ]
        chunk_line_spy.assert_called_once()


def test_core_chunk_empty_input() -> None:
    """core.chunk returns [] for whitespace-only input."""
    assert chunk("   \n", "python", 100) == []


def test_core_chunk_lines_merges_small_lines() -> None:
    """core.chunk_lines merges adjacent small lines that fit within desired_length."""
    # Each line is 2 chars ("a\n"), desired_length=10 allows merging up to 5 lines
    content = "a\nb\nc\nd\ne\nf\n"
    chunks = chunk_lines(content, 10)
    # 6 lines x 2 chars = 12 total; first 5 merge (10 chars), last 1 alone (2 chars)
    assert len(chunks) == 2
    assert chunks[0] == ChunkBoundary(start=0, end=10)
    assert chunks[1] == ChunkBoundary(start=10, end=12)


def test_core_chunk_recursive_split_and_break() -> None:
    """core.chunk recursively splits large nodes and breaks when siblings exceed the limit."""
    code = "x = 1\ndef foo():\n    a = 1\n    b = 2\n    c = 3\ny = 2\n"
    chunks = chunk(code, "python", 10)
    assert chunks is not None
    assert len(chunks) >= 3
    assert chunks[0].start == 0
    # Non-overlapping and within bounds
    for i, c in enumerate(chunks):
        assert 0 <= c.start < c.end <= len(code)
        if i > 0:
            assert c.start >= chunks[i - 1].end


def test_core_chunk_leaf_node_exceeds_desired_length() -> None:
    """core.chunk handles leaf tokens (e.g. a very long identifier) that exceed desired_length."""
    from zemble.chunking.core import chunk

    long_var = "x" * 100
    code = f"{long_var} = 1\n"
    chunks = chunk(code, "python", 50)
    assert chunks is not None
    assert len(chunks) >= 1
    assert chunks[0].start == 0
    for c in chunks:
        assert 0 <= c.start < c.end <= len(code)


def test_get_parser(caplog: pytest.LogCaptureFixture) -> None:
    """Test that get parser only logs the first time."""
    _cached_get_parser.cache_clear()
    with caplog.at_level(logging.WARNING, logger="zemble.chunking.core"):
        _cached_get_parser("hello")
        assert len(caplog.records) == 1
        assert "not found" in caplog.records[0].message
        assert "hello" in caplog.records[0].message

        caplog.clear()
        _cached_get_parser("hello")
        assert len(caplog.records) == 0

    with patch("zemble.chunking.core.get_parser", side_effect=UnsupportedPlatformError):
        with caplog.at_level(logging.WARNING, logger="zemble.chunking.core"):
            _cached_get_parser("Python")
            assert len(caplog.records) == 1
            assert "No bundled grammars for this platform" in caplog.records[0].message

            caplog.clear()
            _cached_get_parser("Python")
            assert len(caplog.records) == 0

    with patch("zemble.chunking.core.get_parser", side_effect=ValueError):
        with caplog.at_level(logging.WARNING, logger="zemble.chunking.core"):
            _cached_get_parser("Ruby")
            assert len(caplog.records) == 1
            assert "Uncaught exception" in caplog.records[0].message

            caplog.clear()
            _cached_get_parser("Ruby")
            assert len(caplog.records) == 0


def test_chunks_is_none() -> None:
    """Test that chunk returns None when parser is not available."""
    with patch("zemble.chunking.core._cached_get_parser", lambda x: None):
        chunks = chunk("x = 1", "python", 10)
        assert chunks is None


def test_unsupported_platform_error() -> None:
    """Test that chunk returns None when the current platform has no bundled grammars."""
    with patch("zemble.chunking.core.get_parser", side_effect=UnsupportedPlatformError):
        chunks = chunk("x = 1", "python", 10)
        assert chunks is None


def test_chunker_deep_string(caplog: pytest.LogCaptureFixture) -> None:
    """Test that chunking works with a very deep string."""
    deep_string = "abs(0)\n"
    for _ in range(10000):
        deep_string = f"abs({deep_string})\n"
    with caplog.at_level(logging.WARNING, logger="zemble.chunking.core"):
        chunks = chunk_source(deep_string, "deep_string.py", "python")
        assert chunks is not None
        assert len(caplog.records) == 1
        assert "Recursion depth exceeded in chunk." in caplog.records[0].message


def _counted_lines(source: str, language: str | None) -> list[tuple[int, int]]:
    """The (start, end) line of every chunk as chunk_source numbered them by counting newlines in each prefix."""
    if not source.strip():
        return []
    boundaries = None
    if language is not None:
        parsed = chunk_with_tree(source, language, _DESIRED_CHUNK_LENGTH_CHARS)
        if parsed is not None:
            boundaries = parsed[0]
    if boundaries is None:
        boundaries = chunk_lines(source, _DESIRED_CHUNK_LENGTH_CHARS)
    lines = []
    for boundary in boundaries:
        end_index = max(boundary.end - 1, boundary.start)
        lines.append((source[: boundary.start].count("\n") + 1, source[:end_index].count("\n") + 1))
    return lines


_MULTIBYTE = 'name = "caf\u00e9 \u4e2d\u6587 \U0001f600"\n# \u00fcber \u2603\n'

#: Sources whose line numbering a running index could plausibly get wrong; offsets are characters.
_EDGE_CASES = [
    ("", "python"),
    ("\n\n\n", "python"),
    ("x = 1", "python"),
    ("def f():\n    return 1", "python"),
    ("def f():\r\n    return 1\r\n\r\nclass A:\r\n    pass\r\n" * 40, "python"),
    ("one\rtwo\rthree\r" * 300, None),
    ("\n\n\ndef late():\n    pass\n" * 60, "python"),
    ((_MULTIBYTE + "def g():\n    return '\u00e9'\n\n") * 80, "python"),
    ((_MULTIBYTE * 200).rstrip("\n"), None),
    (('class A {\n    String s = "\u00e9\u4e2d\U0001f600";\n}\n') * 60, "java"),
    ("line without end " * 400, None),
]


def test_chunk_lines_equal_the_newline_count_over_the_repo_and_edge_cases() -> None:
    """Every chunk's start and end line is the newline count of its prefix, over this repo's files and odd texts."""
    tracked = subprocess.run(
        ["git", "ls-files", "src", "tests", "javac-facts"], cwd=_REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    sources = [(read_file_text(_REPO / name), detect_language(Path(name))) for name in tracked] + _EDGE_CASES
    assert len(sources) > 300 and any(language == "java" for _, language in sources)
    chunks = 0
    for source, language in sources:
        found = [(chunk.start_line, chunk.end_line) for chunk in chunk_source(source, "f", language)]
        assert found == _counted_lines(source, language)
        chunks += len(found)
    assert chunks > 3000


def test_numbering_a_large_files_chunks_is_linear() -> None:
    """A 2.9 MB file chunks in seconds: numbering each chunk by counting its whole prefix took 16 s."""
    source = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(80000))
    started = time.perf_counter()
    chunks = chunk_source(source, "big.py", "python")
    assert time.perf_counter() - started < 8.0
    assert len(chunks) > 1000
