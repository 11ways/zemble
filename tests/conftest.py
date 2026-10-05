import hashlib
import os
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from zemble.types import Chunk


def write_index_components(path: Path) -> None:
    """Create empty stand-ins for every component PersistencePath checks for."""
    path.mkdir(parents=True, exist_ok=True)
    for component in ("chunks", "bm25_index", "semantic_index", "symbols"):
        (path / component).write_text("")


def write_bm25(directory: Path, documents: dict[str, list[str]]) -> Any:
    """Write a BM25 index over *documents*, in order, and load it."""
    from zemble.index.bm25 import BM25, BM25Writer

    writer = BM25Writer(directory)
    for chunk_id, tokens in documents.items():
        writer.add(chunk_id, tokens)
    writer.finish()
    return BM25.load(directory)


def make_chunk(content: str, file_path: str = "src/module.py") -> Chunk:
    """Create a minimal Chunk for use in tests."""
    return Chunk(
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=content.count("\n") + 1,
        language="python",
    )


@pytest.fixture(autouse=True)
def no_real_daemon(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep the suite off the user's real daemon and real cache: a test owns neither.

    The daemon's own tests point the socket at a temporary directory and re-enable it there,
    and the `cache`/`graph_cache` fixtures point ZEMBLE_CACHE_LOCATION somewhere of their own;
    this is the floor under every other test. Without it `stats.py` appended a row per tool
    call to the developer's real ~/.cache/zemble/savings.jsonl - measured at +11 rows from one
    test file - so a suite run silently edited the numbers `zemble savings` reports.
    """
    monkeypatch.setenv("ZEMBLE_DAEMON", "0")
    # Local unit journeys are explicit opt-outs, never simulations of an automatic fallback.
    monkeypatch.setattr("zemble.daemon.client._disabled_reason", "--no-daemon")
    # Never read the developer's real ~/.config/zemble/env (keys, hosted providers) in a test.
    monkeypatch.setenv("ZEMBLE_ENV_FILE", str(tmp_path_factory.mktemp("userenv") / "absent"))
    monkeypatch.setenv("ZEMBLE_CACHE_LOCATION", str(tmp_path_factory.mktemp("zemble-cache")))
    # In-thread socket tests must not change the test runner's process allocation limit.
    monkeypatch.setattr("zemble.daemon.server.allocation_backstop", lambda budget: budget)
    yield
    # Checked before monkeypatch restores anything: a `monkeypatch.undo()` inside a test drops this
    # floor too, and the rest of that test then wrote the developer's real cache.
    assert os.environ.get("ZEMBLE_CACHE_LOCATION"), "the test dropped the cache isolation (monkeypatch.undo()?)"


@pytest.fixture
def tmp_py_file(tmp_path: Path) -> Path:
    """A simple Python file with two functions."""
    code = textwrap.dedent(
        """\
        def add(a, b):
            \"\"\"Add two numbers.\"\"\"
            return a + b

        def subtract(a, b):
            return a - b

        X = 42
        """
    )
    f = tmp_path / "math_utils.py"
    f.write_text(code)
    return f


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """A small project with a few Python files."""
    (tmp_path / "auth.py").write_text(
        textwrap.dedent(
            """\
            def authenticate(token):
                \"\"\"Verify an auth token.\"\"\"
                return token == "secret"

            def login(username, password):
                return authenticate(password)
            """
        )
    )
    (tmp_path / "utils.py").write_text(
        textwrap.dedent(
            """\
            def format_name(first, last):
                return f"{first} {last}"

            class Config:
                debug = False
                host = "localhost"
            """
        )
    )
    (tmp_path / "README.md").write_text("# Test project\n")
    return tmp_path


class FakeEmbedder:
    """A deterministic embedder for tests: same text in, same unit vector out.

    Vectors are derived from a hash of the text, so a chunk keeps its vector across
    processes and the dimension is free to be anything (256 is not special).
    """

    def __init__(self, dimensions: int = 256, model_id: str = "fake:test") -> None:
        """Initialise the fake.

        :param dimensions: The vector width to produce.
        :param model_id: The spec string the index will record.
        """
        self._dimensions = dimensions
        self._model_id = f"{model_id}@{dimensions}"
        self.document_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        """The normalized spec string."""
        return self._model_id

    @property
    def dimensions(self) -> int:
        """The vector width."""
        return self._dimensions

    def _vectors(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Return one deterministic unit vector per text."""
        if not texts:
            return np.empty((0, self._dimensions), dtype=np.float32)
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            vector = np.random.default_rng(seed).standard_normal(self._dimensions).astype(np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed documents."""
        self.document_calls.append(list(texts))
        return self._vectors(texts)

    def embed_queries(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed queries."""
        self.query_calls.append(list(texts))
        return self._vectors(texts)


@pytest.fixture
def mock_embedder() -> FakeEmbedder:
    """A deterministic 256-dimensional embedder."""
    return FakeEmbedder()


@pytest.fixture
def graph_fixture_root() -> Path:
    """The miniature Java workspace used by the symbol graph tests."""
    return Path(__file__).parent / "fixtures" / "graph"


@pytest.fixture
def graph_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the zemble cache at a throwaway folder so graph builds are isolated."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("ZEMBLE_CACHE_LOCATION", str(cache))
    return cache


@pytest.fixture
def built_graph(graph_fixture_root: Path, graph_cache: Path) -> Any:
    """A freshly built graph over the fixture workspace, with a provider open on it."""
    from zemble.graph import SqliteGraphProvider, build_graph

    build_graph(str(graph_fixture_root))
    provider = SqliteGraphProvider(str(graph_fixture_root))
    yield provider
    provider.close()
