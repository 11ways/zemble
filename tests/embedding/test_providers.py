from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from zemble import ZembleIndex
from zemble.embedding import http as embedding_http
from zemble.embedding.cache import CachingEmbedder
from zemble.embedding.http import EmbeddingRequestError, batched
from zemble.embedding.openai_compat import OpenAICompatibleEmbedder
from zemble.embedding.registry import parse_embedder_spec
from zemble.embedding.voyage import VoyageEmbedder


class FakeProvider:
    """A localhost embeddings endpoint that records requests and replays scripted responses."""

    def __init__(self) -> None:
        """Start the server on an ephemeral port."""
        self.requests: list[dict[str, Any]] = []
        self.responses: list[tuple[int, dict[str, Any]]] = []
        self.dimensions = 4
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                provider.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
                status, payload = provider._next(body)
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args: Any) -> None:
                """Silence the default stderr access log."""

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def _next(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Return the next scripted response, or a well-formed default."""
        if self.responses:
            return self.responses.pop(0)
        inputs = body["input"]
        dimensions = body.get("output_dimension") or body.get("dimensions") or self.dimensions
        data = [
            {"object": "embedding", "index": i, "embedding": [float(i + 1)] * dimensions} for i in range(len(inputs))
        ]
        return 200, {"object": "list", "data": data, "usage": {"total_tokens": 7 * len(inputs)}}

    @property
    def base_url(self) -> str:
        """The base URL an OpenAI-compatible embedder should be pointed at."""
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def close(self) -> None:
        """Shut the server down."""
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def provider() -> Iterator[FakeProvider]:
    """A running fake embeddings endpoint."""
    fake = FakeProvider()
    yield fake
    fake.close()


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make backoff cost no wall time."""
    monkeypatch.setattr(embedding_http, "_sleep", lambda seconds: None)


def _voyage(provider: FakeProvider, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> VoyageEmbedder:
    """Build a Voyage embedder pointed at the fake server with a key in the environment."""
    monkeypatch.setattr("zemble.embedding.voyage.VOYAGE_URL", f"{provider.base_url}/embeddings")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    return VoyageEmbedder(**kwargs)


def test_voyage_request_shape(provider: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """Voyage gets the documented body, the bearer header, and asymmetric input_type."""
    embedder = _voyage(provider, monkeypatch, model="voyage-code-4", dimensions=4)

    documents = embedder.embed_documents(["def a(): pass", "def b(): pass"])
    assert documents.shape == (2, 4)
    body = provider.requests[0]["body"]
    assert body == {
        "input": ["def a(): pass", "def b(): pass"],
        "model": "voyage-code-4",
        "truncation": True,
        "input_type": "document",
        "output_dimension": 4,
    }
    assert provider.requests[0]["headers"]["Authorization"] == "Bearer test-key"
    assert embedder.total_tokens == 14

    embedder.embed_queries(["where is a"])
    assert provider.requests[1]["body"]["input_type"] == "query"


def test_voyage_returns_normalized_rows_in_input_order(provider: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows come back L2-normalized and re-sorted by the provider's index field."""
    embedder = _voyage(provider, monkeypatch, model="voyage-code-4", dimensions=4)
    provider.responses.append(
        (
            200,
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0, 0.0, 0.0]},
                    {"index": 0, "embedding": [3.0, 0.0, 0.0, 0.0]},
                ],
                "usage": {"total_tokens": 2},
            },
        )
    )
    result = embedder.embed_documents(["first", "second"])
    np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result[1], [0.0, 1.0, 0.0, 0.0], atol=1e-6)


def test_voyage_missing_key_is_loud(provider: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent VOYAGE_API_KEY is refused by name before any request is made."""
    monkeypatch.setattr("zemble.embedding.voyage.VOYAGE_URL", f"{provider.base_url}/embeddings")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(EmbeddingRequestError, match="VOYAGE_API_KEY is not set"):
        VoyageEmbedder("voyage-code-4", 4).embed_documents(["x"])
    assert provider.requests == []


def test_batching_splits_on_count_and_size(provider: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """A list larger than the per-request ceiling is split, and every row still comes back once."""
    embedder = _voyage(provider, monkeypatch, model="voyage-code-4", dimensions=4)
    monkeypatch.setattr(embedder, "_max_texts", 3)
    texts = [f"text-{i}" for i in range(7)]
    result = embedder.embed_documents(texts)
    assert result.shape == (7, 4)
    assert [len(request["body"]["input"]) for request in provider.requests] == [3, 3, 1]
    assert embedder.request_count == 3


def test_retry_on_429_then_success(provider: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rate-limited request is retried and the eventual success is returned."""
    embedder = _voyage(provider, monkeypatch, model="voyage-code-4", dimensions=4)
    provider.responses.append((429, {"detail": "rate limit exceeded"}))
    provider.responses.append((503, {"detail": "upstream busy"}))
    result = embedder.embed_documents(["x"])
    assert result.shape == (1, 4)
    assert len(provider.requests) == 3


def test_gives_up_after_five_tries(provider: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent 429s stop after the documented maximum and report the last error."""
    embedder = _voyage(provider, monkeypatch, model="voyage-code-4", dimensions=4)
    provider.responses.extend([(429, {"detail": "still rate limited"})] * 6)
    with pytest.raises(EmbeddingRequestError, match="gave up after 5 attempts"):
        embedder.embed_documents(["x"])
    assert len(provider.requests) == 5


def test_4xx_surfaces_the_provider_message(provider: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 400 is raised immediately, carrying the API's own wording."""
    embedder = _voyage(provider, monkeypatch, model="voyage-code-4", dimensions=4)
    provider.responses.append((400, {"detail": "output_dimension 7 is not supported"}))
    with pytest.raises(EmbeddingRequestError, match="output_dimension 7 is not supported"):
        embedder.embed_documents(["x"])
    assert len(provider.requests) == 1


def test_openai_error_shape_is_understood(provider: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """The OpenAI ``{"error": {"message": ...}}`` body is surfaced as well as Voyage's ``detail``."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedder = OpenAICompatibleEmbedder(provider.base_url, "nomic-embed-text", 4)
    provider.responses.append((404, {"error": {"message": "model 'nomic-embed-text' not found"}}))
    with pytest.raises(EmbeddingRequestError, match="model 'nomic-embed-text' not found"):
        embedder.embed_documents(["x"])


def test_openai_request_shape_has_no_input_type(provider: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """The OpenAI schema carries no input_type, and an unset key means no Authorization header."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedder = OpenAICompatibleEmbedder(provider.base_url, "nomic-embed-text", 768)
    provider.dimensions = 768
    embedder.embed_documents(["def a(): pass"])
    embedder.embed_queries(["where is a"])
    for request in provider.requests:
        assert "input_type" not in request["body"]
        assert request["path"] == "/v1/embeddings"
        assert "Authorization" not in request["headers"]
    assert provider.requests[0]["body"] == {"input": ["def a(): pass"], "model": "nomic-embed-text", "dimensions": 768}


def test_openai_sends_key_when_present(provider: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured key env var becomes a bearer header."""
    monkeypatch.setenv("MY_KEY", "sk-test")
    embedder = OpenAICompatibleEmbedder(provider.base_url, "text-embedding-3-small", 4, api_key_env="MY_KEY")
    embedder.embed_documents(["x"])
    assert provider.requests[0]["headers"]["Authorization"] == "Bearer sk-test"


def test_dimensions_are_probed_when_unknown(provider: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unspecified width is discovered with a single probe request and then remembered."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider.dimensions = 11
    embedder = OpenAICompatibleEmbedder(provider.base_url, "mystery-model")
    assert embedder.dimensions == 11
    assert embedder.dimensions == 11
    assert len(provider.requests) == 1
    assert embedder.model_id == f"openai:{provider.base_url}#mystery-model@11"


def test_wrong_count_from_provider_is_refused(provider: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """A response with fewer embeddings than inputs is an error, not a silently short matrix."""
    embedder = _voyage(provider, monkeypatch, model="voyage-code-4", dimensions=4)
    provider.responses.append((200, {"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}]}))
    with pytest.raises(EmbeddingRequestError, match="returned 1 embeddings for 2 inputs"):
        embedder.embed_documents(["a", "b"])


@pytest.mark.parametrize(
    ("texts", "max_texts", "max_chars", "expected"),
    [
        ([], 3, 100, []),
        (["a", "b"], 5, 100, [(0, ["a", "b"])]),
        (["a", "b", "c"], 2, 100, [(0, ["a", "b"]), (2, ["c"])]),
        (["aaa", "bbb"], 5, 4, [(0, ["aaa"]), (1, ["bbb"])]),
        (["x" * 50, "y"], 5, 4, [(0, ["x" * 50]), (1, ["y"])]),
    ],
)
def test_batched(texts: list[str], max_texts: int, max_chars: int, expected: list[tuple[int, list[str]]]) -> None:
    """Batches respect both ceilings, and an oversized single text still gets sent."""
    assert list(batched(texts, max_texts, max_chars)) == expected


def test_http_provider_indexes_and_reuses_the_cache(
    provider: FakeProvider, monkeypatch: pytest.MonkeyPatch, tmp_project: Path
) -> None:
    """A real index over an HTTP provider costs requests once; the second build costs zero."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider.dimensions = 12
    spec = f"openai:{provider.base_url}#fake-embed@12"

    # 1. Cold build: chunks are embedded over HTTP and the vectors land in the index.
    first = parse_embedder_spec(spec)
    assert isinstance(first, CachingEmbedder), "step 1: remote providers must be cached by default"
    with patch("zemble.index.index.load_embedder", return_value=first):
        index = ZembleIndex.from_path(tmp_project)
    cold_requests = len(provider.requests)
    assert cold_requests > 0, "step 1: a cold build must call the provider"
    assert index._semantic_index.vectors.shape[1] == 12
    assert index.stats.embedder == spec

    # 2. Search still reaches the provider: queries are deliberately not cached.
    index.search("authenticate token", top_k=3)
    assert len(provider.requests) == cold_requests + 1, "step 2: a query must reach the provider"
    query_requests = len(provider.requests)

    # 3. Warm build with a fresh embedder over the same cache: not one request.
    second = parse_embedder_spec(spec)
    with patch("zemble.index.index.load_embedder", return_value=second):
        ZembleIndex.from_path(tmp_project)
    assert len(provider.requests) == query_requests, "step 3: every chunk must come from the sqlite cache"
    assert second.inner.request_count == 0
