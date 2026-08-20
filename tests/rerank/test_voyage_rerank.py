from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from zemble.embedding import http as embedding_http
from zemble.embedding.http import EmbeddingRequestError
from zemble.rerank.voyage import VoyageReranker


class FakeRerankService:
    """A localhost rerank endpoint that records requests and replays scripted responses."""

    def __init__(self) -> None:
        """Start the server on an ephemeral port."""
        self.requests: list[dict[str, Any]] = []
        self.responses: list[tuple[int, dict[str, Any]]] = []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                service.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
                status, payload = service._next(body)
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
        """Return the next scripted response, or a default that ranks the last document first."""
        if self.responses:
            return self.responses.pop(0)
        documents = body["documents"]
        data = [
            {"index": index, "relevance_score": round(index / max(len(documents) - 1, 1), 4)}
            for index in reversed(range(len(documents)))
        ]
        return 200, {"object": "list", "data": data, "usage": {"total_tokens": 11 * len(documents)}}

    @property
    def url(self) -> str:
        """The rerank endpoint URL."""
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1/rerank"

    def close(self) -> None:
        """Shut the server down."""
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def service() -> Iterator[FakeRerankService]:
    """A running fake rerank endpoint."""
    fake = FakeRerankService()
    yield fake
    fake.close()


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make backoff cost no wall time."""
    monkeypatch.setattr(embedding_http, "_sleep", lambda seconds: None)


def _reranker(
    service: FakeRerankService, monkeypatch: pytest.MonkeyPatch, model: str = "rerank-2.5-lite"
) -> VoyageReranker:
    """Build a Voyage reranker pointed at the fake server with a key in the environment."""
    monkeypatch.setattr("zemble.rerank.voyage.VOYAGE_RERANK_URL", service.url)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    return VoyageReranker(model)


def test_request_shape_and_score_order(service: FakeRerankService, monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented body is sent, and scores come back in input order however they arrive."""
    reranker = _reranker(service, monkeypatch)

    scores = reranker.score("where is auth", ["def a(): ...", "def b(): ...", "def c(): ..."])

    assert service.requests[0]["body"] == {
        "query": "where is auth",
        "documents": ["def a(): ...", "def b(): ...", "def c(): ..."],
        "model": "rerank-2.5-lite",
        "truncation": True,
    }
    assert service.requests[0]["headers"]["Authorization"] == "Bearer test-key"
    assert scores == [0.0, 0.5, 1.0], "re-sorted by the index field, not by arrival order"
    assert reranker.total_tokens == 33
    assert reranker.request_count == 1


def test_long_candidate_lists_are_split(service: FakeRerankService, monkeypatch: pytest.MonkeyPatch) -> None:
    """More documents than one request allows are split, and every score keeps its slot."""
    monkeypatch.setattr("zemble.rerank.voyage.MAX_DOCUMENTS_PER_REQUEST", 2)
    reranker = _reranker(service, monkeypatch)

    scores = reranker.score("q", ["a", "b", "c", "d", "e"])

    assert reranker.request_count == 3
    assert [r["body"]["documents"] for r in service.requests] == [["a", "b"], ["c", "d"], ["e"]]
    assert len(scores) == 5


def test_empty_candidate_list_costs_no_request(service: FakeRerankService, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing to score means nothing is sent."""
    reranker = _reranker(service, monkeypatch)
    assert reranker.score("q", []) == []
    assert service.requests == []


def test_missing_key_is_refused_before_any_request(service: FakeRerankService, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset API key names the variable instead of failing at the socket."""
    reranker = _reranker(service, monkeypatch)
    monkeypatch.delenv("VOYAGE_API_KEY")
    with pytest.raises(EmbeddingRequestError, match="VOYAGE_API_KEY is not set"):
        reranker.score("q", ["a"])
    assert service.requests == []


def test_wrong_score_count_is_refused(service: FakeRerankService, monkeypatch: pytest.MonkeyPatch) -> None:
    """A response holding fewer scores than documents is an error, not a silent zero."""
    reranker = _reranker(service, monkeypatch)
    service.responses.append((200, {"data": [{"index": 0, "relevance_score": 0.9}]}))
    with pytest.raises(EmbeddingRequestError, match="returned 1 scores for 2 documents"):
        reranker.score("q", ["a", "b"])


def test_server_errors_are_retried(service: FakeRerankService, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 500 is retried through the shared HTTP helper; a 400 is not."""
    reranker = _reranker(service, monkeypatch)
    service.responses.append((500, {"detail": "busy"}))
    assert reranker.score("q", ["a"]) == [0.0]
    assert len(service.requests) == 2

    service.responses.append((400, {"detail": "bad model"}))
    with pytest.raises(EmbeddingRequestError, match="bad model"):
        reranker.score("q", ["a"])
