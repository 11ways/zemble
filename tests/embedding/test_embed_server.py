"""The shared embedding server, end to end over real HTTP on localhost, with a fake provider behind it."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import stat
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from tests.conftest import make_chunk
from tests.embedding.test_embedding_gc import _index, _set
from zemble.cache import resolve_cache_folder
from zemble.chunking.capsule import embedding_text
from zemble.daemon import server as daemon_server
from zemble.embedding import server as http_server
from zemble.embedding.cache import EmbeddingCache, cache_root, text_hash, today
from zemble.embedding.gc import collect_unused
from zemble.embedding.http import EmbeddingRequestError
from zemble.embedding.pricing import EmbeddingBudgetExceeded, require_affordable_bill
from zemble.embedding.referenced import REPORTED_FILE, REPORTED_TODAY, FamilyReport, distinct_batches, report_referenced
from zemble.embedding.registry import build_embedder
from zemble.embedding.served import ServerClient, ServerEmbedder, ServerReranker
from zemble.embedding.server import KeyRing, add_key, make_server
from zemble.embedding.server_cli import run_embed_server
from zemble.embedding.service import EmbeddingService
from zemble.embedding.wire import DOCUMENTS_PER_REQUEST, REFERENCED_ROUTE, SERVER_ENV, SERVER_KEY_ENV, server_settings
from zemble.rerank.registry import load_reranker

SPEC = "voyage:voyage-4-lite@8"
FAMILY = "voyage:voyage-4-lite"
RERANKER = "voyage:rerank-2.5-lite"


class FakeProvider:
    """A paid provider stand-in that records every text it was made to embed."""

    def __init__(self, dimensions: int = 8, delay: float = 0.0, fail_first: int = 0) -> None:
        """Build vectors from the text alone; optionally slow, optionally failing at first."""
        self._dimensions = dimensions
        self.delay = delay
        self.fail_first = fail_first
        self.document_batches: list[list[str]] = []
        self.query_batches: list[list[str]] = []
        self.total_tokens = 0
        self._lock = threading.Lock()

    model_id = "voyage:voyage-4-lite@8"
    semantic_weight_bonus = 0.15
    is_remote = True

    @property
    def dimensions(self) -> int:
        """The vector width."""
        return self._dimensions

    @property
    def declared_dimensions(self) -> int:
        """The width, known without a request."""
        return self._dimensions

    def vectors(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Return one deterministic unit vector per text."""
        rows = []
        for text in texts:
            seed = int(text_hash(text)[:8], 16)
            vector = np.random.default_rng(seed).standard_normal(self._dimensions).astype(np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows, dtype=np.float32).reshape(len(texts), self._dimensions)

    def embed_documents(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed documents, recording the batch, failing while `fail_first` lasts."""
        with self._lock:
            if self.fail_first > 0:
                self.fail_first -= 1
                raise EmbeddingRequestError("provider says: quota exceeded")
            self.document_batches.append(list(texts))
            self.total_tokens += sum(len(text) for text in texts)
        time.sleep(self.delay)
        return self.vectors(texts)

    def embed_queries(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed queries, recording the batch."""
        self.query_batches.append(list(texts))
        return self.vectors(texts)

    def bought(self) -> list[str]:
        """Every document text the provider was paid for, in order."""
        return [text for batch in self.document_batches for text in batch]


class FakeReranker:
    """A hosted reranker stand-in: longer passages score higher."""

    model_id = RERANKER

    def __init__(self) -> None:
        """Record every call."""
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score by length."""
        self.calls.append((query, list(passages)))
        return [float(len(passage)) for passage in passages]


class Running:
    """A server on a free localhost port, plus what a test needs to inspect it."""

    def __init__(self, tmp_path: Path, provider: FakeProvider) -> None:
        """Start serving in a background thread."""
        self.provider = provider
        self.reranker = FakeReranker()
        self.keys_file = tmp_path / "server-keys"
        self.key = add_key(self.keys_file, "test-client")
        self.service = EmbeddingService(
            tmp_path / "server-data",
            [SPEC],
            [RERANKER],
            provider=lambda spec: provider,
            reranker=lambda spec: self.reranker,
        )
        self.server = make_server(self.service, KeyRing(self.keys_file), "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop serving and release the port."""
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a client's retry backoff cost no wall time."""
    monkeypatch.setattr("zemble.embedding.http._sleep", lambda seconds: None)


def _serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> Iterator[Running]:
    running = Running(tmp_path, provider)
    monkeypatch.setenv(SERVER_ENV, running.url)
    monkeypatch.setenv(SERVER_KEY_ENV, running.key)
    try:
        yield running
    finally:
        running.stop()


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_backoff: None) -> Iterator[Running]:
    """A running server with an instant fake provider, and this process configured as its client."""
    yield from _serve(tmp_path, monkeypatch, FakeProvider())


def test_server_journey(served: Running) -> None:
    """Two clients share one cache: each text is bought once, queries never stored, reranks proxied."""
    provider = served.provider

    # 1. With a server configured, the registry builds a client embedder whose identity is the provider's own.
    first = build_embedder(SPEC).embedder
    assert isinstance(first, ServerEmbedder), "step 1: a paid family must go through the server"
    assert first.model_id == "voyage:voyage-4-lite@8", "step 1: the model id must stay the provider's"
    assert first.family == FAMILY

    # 2. A cold request buys every text once and returns the provider's vectors.
    vectors = first.embed_documents(["alpha", "beta"])
    assert provider.bought() == ["alpha", "beta"], "step 2: both texts must be bought exactly once"
    np.testing.assert_allclose(vectors, provider.vectors(["alpha", "beta"]), atol=1e-6)

    # 3. A second machine's request is served from the server's cache; only the new text is bought.
    second = build_embedder(SPEC).embedder
    mixed = second.embed_documents(["beta", "gamma", "alpha"])
    assert provider.bought() == ["alpha", "beta", "gamma"], "step 3: only the unseen text may be bought"
    np.testing.assert_allclose(mixed, provider.vectors(["beta", "gamma", "alpha"]), atol=1e-6)

    # 4. Duplicates inside one request share one provider slot.
    second.embed_documents(["delta", "delta"])
    assert provider.bought()[-1:] == ["delta"] and provider.bought().count("delta") == 1, "step 4"

    # 5. The budget guard's question is answered by the server: what is still unpaid, each text once.
    assert second.pending_documents(["alpha", "epsilon", "epsilon"]) == ["epsilon"], "step 5"
    assert second.stored_digests([text_hash("alpha"), text_hash("zeta")]) == {text_hash("alpha")}, "step 5"

    # 6. Queries always reach the provider and are never stored as documents.
    second.embed_queries(["only a query"])
    assert provider.query_batches == [["only a query"]], "step 6: a query must reach the provider"
    assert second.pending_documents(["only a query"]) == ["only a query"], "step 6: a query vector is never stored"

    # 7. A hosted reranker is served too, so the client needs no provider key at all.
    reranker = load_reranker(RERANKER)
    assert isinstance(reranker, ServerReranker), "step 7: a hosted reranker must go through the server"
    assert reranker.score("q", ["a", "abc"]) == [1.0, 3.0], "step 7: scores come from the served reranker"
    assert served.reranker.calls == [("q", ["a", "abc"])]

    # 8. A family the server was not configured to pay for is refused by name.
    with pytest.raises(EmbeddingRequestError, match="does not serve 'voyage:voyage-code-4@8'"):
        build_embedder("voyage:voyage-code-4@8").embedder.embed_documents(["alpha"])

    # 9. The status report counts what happened.
    status = first.client.status()
    assert status["families"][0]["vectors"] == 4, "step 9: alpha, beta, gamma, delta are stored"
    assert status["documents_missed"] == 4
    assert status["queries"] == 1 and status["rerank_passages"] == 2


def test_a_request_spanning_several_batches_keeps_its_order(served: Running) -> None:
    """Texts sent in several requests come back as one matrix in input order."""
    many = [f"text {n}" for n in range(DOCUMENTS_PER_REQUEST * 2 + 3)]
    vectors = build_embedder(SPEC).embedder.embed_documents(many)
    np.testing.assert_allclose(vectors, served.provider.vectors(many), atol=1e-6)


def test_an_unknown_key_is_refused(served: Running, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong key gets nothing, and the refusal says why instead of retrying."""
    monkeypatch.setenv(SERVER_KEY_ENV, "not-a-key")
    with pytest.raises(EmbeddingRequestError, match="401: missing or unknown API key"):
        build_embedder(SPEC).embedder.embed_documents(["alpha"])
    assert served.provider.bought() == []


def test_a_server_without_a_key_needs_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Naming a server without a key is a configuration error, not an unauthenticated client."""
    monkeypatch.setenv(SERVER_ENV, "http://127.0.0.1:1")
    monkeypatch.delenv(SERVER_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=SERVER_KEY_ENV):
        build_embedder(SPEC)


def test_concurrent_buyers_pay_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_backoff: None) -> None:
    """Machines asking for the same uncached texts at the same moment buy each text once between them."""
    for running in _serve(tmp_path, monkeypatch, FakeProvider(delay=0.3)):
        texts = [f"text {number}" for number in range(6)]
        results: list[npt.NDArray[np.float32]] = []
        errors: list[BaseException] = []

        def ask(subset: list[str]) -> None:
            try:
                results.append(build_embedder(SPEC).embedder.embed_documents(subset))
            except BaseException as exc:  # pragma: no cover - surfaced by the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=ask, args=(subset,)) for subset in (texts, texts[2:], texts[::-1])]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert not errors, errors
        assert sorted(running.provider.bought()) == sorted(texts), "every text must be bought exactly once"
        assert len(results) == 3


def test_a_failed_purchase_is_reported_once_and_releases_its_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    """A provider failure reaches the client with the provider's words, without five client retries."""
    for running in _serve(tmp_path, monkeypatch, FakeProvider(fail_first=1)):
        embedder = build_embedder(SPEC).embedder
        with pytest.raises(EmbeddingRequestError, match="quota exceeded"):
            embedder.embed_documents(["alpha"])
        assert running.provider.fail_first == 0, "the client must not retry a provider failure"
        # The failed claim was released: the next request buys the text instead of waiting forever.
        embedder.embed_documents(["alpha"])
        assert running.provider.bought() == ["alpha"]


def test_the_bill_guard_asks_the_server(served: Running, monkeypatch: pytest.MonkeyPatch) -> None:
    """The budget guard counts only what the server would buy, so a cached build never prompts."""
    embedder = build_embedder(SPEC).embedder
    texts = ["x" * 4000 for _ in range(1)] + [f"chunk {n} " * 200 for n in range(20)]
    monkeypatch.setenv("ZEMBLE_EMBED_BUDGET_USD", "0.0000001")
    with pytest.raises(EmbeddingBudgetExceeded):
        require_affordable_bill(embedder, texts)
    embedder.embed_documents(texts)
    require_affordable_bill(embedder, texts)  # all paid for on the server: nothing to refuse


def test_preflight_reads_the_servers_coverage(served: Running, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`embed-status` judges the bill against the server's cache and names the server as the store."""
    from zemble.embedding.preflight import embed_status
    from zemble.types import ContentType

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "one.py").write_text("def one():\n    return 1\n")
    (tree / "two.py").write_text("def two():\n    return 2\n")
    monkeypatch.setenv("ZEMBLE_EMBEDDER", SPEC)
    before = embed_status(tree, (ContentType.CODE,))
    assert before.cache_path == served.url
    assert before.cached == 0 and before.uncached == before.chunks_total > 0

    from zemble.chunking.capsule import embedding_text
    from zemble.index.create import plan_files

    root = tree.resolve()
    texts = [embedding_text(chunk) for planned in plan_files(root, display_root=root) for chunk in planned.chunks]
    build_embedder(SPEC).embedder.embed_documents(texts)
    after = embed_status(tree, (ContentType.CODE,))
    assert after.cached == after.chunks_total and after.uncached == 0


def test_push_moves_a_machines_vectors_to_the_server(served: Running, tmp_path: Path) -> None:
    """`push` uploads what this machine already paid for, keeps the server's own, and can drop the local file."""
    local = EmbeddingCache(FAMILY, cache_root())
    vectors = served.provider.vectors(["old one", "old two"])
    local.put_many([(text_hash("old one"), 8, vectors[0]), (text_hash("old two"), 8, vectors[1])])
    local.close()
    path = cache_root() / "voyage-voyage-4-lite.sqlite"

    # 1. A pushed vector is served without the provider ever being asked.
    args = argparse.Namespace(embed_server_action="push", files=[], family=None, remove_local=True)
    assert run_embed_server(args) == 0
    served_vectors = build_embedder(SPEC).embedder.embed_documents(["old two", "old one"])
    np.testing.assert_allclose(served_vectors, vectors[::-1], atol=0)
    assert served.provider.bought() == [], "step 1: pushed vectors must not be bought again"

    # 2. --remove-local deleted the local file once the server held all of it.
    assert not path.exists(), "step 2: the local file must be gone"

    # 3. A foreign file under any name is pushed with an explicit family, and nothing is overwritten.
    foreign_dir = tmp_path / "elsewhere"
    foreign = EmbeddingCache(FAMILY, foreign_dir)
    foreign.put_many([(text_hash("old one"), 8, np.ones(8, dtype=np.float32)), (text_hash("third"), 8, vectors[0])])
    foreign.close()
    renamed = foreign_dir / "seed.sqlite"
    (foreign_dir / "voyage-voyage-4-lite.sqlite").rename(renamed)
    args = argparse.Namespace(embed_server_action="push", files=[renamed], family=FAMILY, remove_local=False)
    assert run_embed_server(args) == 0
    again = build_embedder(SPEC).embedder.embed_documents(["old one", "third"])
    np.testing.assert_allclose(again[0], vectors[0], atol=0, err_msg="step 3: a stored vector is never replaced")
    np.testing.assert_allclose(again[1], vectors[0], atol=0)
    assert renamed.exists(), "step 3: without --remove-local the file stays"


def test_store_refuses_malformed_rows(served: Running) -> None:
    """A row whose bytes do not match its width is refused as a bad request, storing nothing."""
    client = build_embedder(SPEC).embedder.client
    with pytest.raises(EmbeddingRequestError, match="400: malformed vector row"):
        client.store(FAMILY, [(text_hash("short"), 8, b"\x00" * 12)])


def test_key_ring_reads_labels_and_reloads(tmp_path: Path) -> None:
    """Keys carry labels, comments are ignored, and a key added later is accepted without a restart."""
    path = tmp_path / "keys"
    first = add_key(path, "aeor")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600, "the key file must be private"
    ring = KeyRing(path)
    assert ring.label_for(first) == "aeor"
    assert ring.label_for("guess") is None
    time.sleep(0.01)
    second = add_key(path, "dalaran box")
    assert ring.label_for(second) == "dalaran-box"
    assert ring.count() == 2


def test_the_server_refuses_to_start_without_a_key(tmp_path: Path) -> None:
    """No key file, no server: an open endpoint would spend the provider key for anyone."""
    service = EmbeddingService(tmp_path / "data", [SPEC], provider=lambda spec: FakeProvider())
    with pytest.raises(ValueError, match="lists no API key"):
        make_server(service, KeyRing(tmp_path / "missing"), "127.0.0.1", 0)


def test_server_gc_sweeps_only_what_nobody_used(tmp_path: Path) -> None:
    """The server keeps every vector used within its grace period and sweeps the rest."""
    directory = tmp_path / "data"
    cache = EmbeddingCache(FAMILY, directory)
    provider = FakeProvider()
    cache.put_many([(text_hash("fresh"), 8, provider.vectors(["fresh"])[0])])
    cache.put_many([(text_hash("stale"), 8, provider.vectors(["stale"])[0])])
    with cache._lock:
        cache._connection.execute("UPDATE used SET day = day - 400 WHERE text_sha256 = ?", (text_hash("stale"),))
    cache.close()

    dry = collect_unused(directory, grace_days=180, dry_run=True)
    assert [(report.rows, report.swept) for report in dry] == [(2, 1)]
    swept = collect_unused(directory, grace_days=180, dry_run=False)
    assert swept[0].refused is None and swept[0].swept == 1
    remaining = EmbeddingCache(FAMILY, directory)
    try:
        assert remaining.covered([text_hash("fresh"), text_hash("stale")], 8) == {text_hash("fresh")}
    finally:
        remaining.close()


def _server_days(running: Running) -> dict[str, int]:
    """Every use stamp the server's cache file holds."""
    connection = sqlite3.connect(running.service.directory / "voyage-voyage-4-lite.sqlite")
    try:
        return dict(connection.execute("SELECT text_sha256, day FROM used").fetchall())
    finally:
        connection.close()


def test_a_covered_hit_is_stamped_on_the_server(served: Running) -> None:
    """A pre-flight's coverage question proves its hits are in a live tree, so the server stamps them."""
    embedder = build_embedder(SPEC).embedder
    embedder.embed_documents(["alpha", "beta"])
    _set(served.service.directory / "voyage-voyage-4-lite.sqlite", "UPDATE used SET day = ?", today() - 400)
    assert embedder.stored_digests([text_hash("alpha"), text_hash("gamma")]) == {text_hash("alpha")}
    assert _server_days(served) == {text_hash("alpha"): today(), text_hash("beta"): today() - 400}


def test_distinct_batches_send_a_chunk_shared_by_indexes_once(tmp_path: Path) -> None:
    """Indexes of one workspace share most chunks; the report sends each shared hash once, in bounded batches."""
    shared = [make_chunk(f"def shared_{i}(): pass") for i in range(5)]
    first = _index(tmp_path, "a" * 64, [*shared, make_chunk("def own_a(): pass")])
    second = _index(tmp_path, "b" * 64, [make_chunk("def own_b(): pass"), *shared[::-1]], variant="overlay")
    broken = _index(tmp_path, "c" * 64, shared)
    (broken / "chunks" / "chunks.json").write_text("not json")
    report = FamilyReport(FAMILY, 3)
    batches = list(distinct_batches([first, broken, second], 2, report))
    sent = [digest for batch in batches for digest in batch]
    assert all(len(batch) <= 2 for batch in batches)
    assert len(sent) == len(set(sent)) == 7, "each distinct hash exactly once"
    assert set(sent) == {text_hash(embedding_text(chunk)) for chunk in [*shared]} | {
        text_hash(embedding_text(make_chunk("def own_a(): pass"))),
        text_hash(embedding_text(make_chunk("def own_b(): pass"))),
    }
    assert report.unreadable == 1, "an unreadable index is passed over and counted"


def test_referenced_report_journey(
    served: Running,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A client's saved indexes and overlays keep their server vectors alive, once a day, quietly past an old server."""
    cache_folder = resolve_cache_folder()
    chunks = [make_chunk(f"def chunk_{i}(): pass") for i in range(5)]
    _index(cache_folder, "a" * 64, chunks, embedder=SPEC)
    _index(cache_folder, "b" * 64, [*chunks[:2], make_chunk("def lane(): pass")], embedder=SPEC, variant="overlay-code")
    _index(cache_folder, "c" * 64, [make_chunk("def free(): pass")], embedder="model2vec:minishlab/potion-code-16M")
    held = [text_hash(embedding_text(chunk)) for chunk in chunks[:4]]
    build_embedder(SPEC).embedder.embed_documents([embedding_text(chunk) for chunk in chunks[:4]])
    _set(served.service.directory / "voyage-voyage-4-lite.sqlite", "UPDATE used SET day = ?", today() - 400)
    settings = server_settings()
    assert settings is not None
    client = ServerClient(settings)

    # 1. The paid family's index and overlay are reported; the server stamps what it holds, and only that.
    [report] = report_referenced(cache_folder, client)
    assert (report.family, report.indexes, report.sent, report.held, report.skipped) == (FAMILY, 2, 6, 4, None)
    assert _server_days(served) == dict.fromkeys(held, today()), "step 1: no row for a hash the server never stored"
    assert served.service.counters.referenced_reported == 6

    # 2. The same day it is not sent again, across processes: the day is kept in the cache folder.
    assert (cache_folder / REPORTED_FILE).exists()
    [again] = report_referenced(cache_folder, client)
    assert again.skipped == REPORTED_TODAY and served.service.counters.referenced_reported == 6, "step 2"

    # 3. The command line sends one report by hand, `--force` past today's.
    assert run_embed_server(argparse.Namespace(embed_server_action="report", force=True)) == 0
    out = capsys.readouterr().out
    assert f"{FAMILY}: 6 distinct hashes from 2 index(es) reported to {served.url}, 4 held there" in out, out

    # 4. A server older than the route answers 404: logged once, the day recorded, nothing raised.
    original = http_server._route

    def old_route(service: EmbeddingService, route: str, payload: dict) -> dict:
        if route == REFERENCED_ROUTE:
            raise http_server._HttpError(404, f"unknown route {route}")
        return original(service, route, payload)

    monkeypatch.setattr(http_server, "_route", old_route)
    tomorrow = today() + 1
    monkeypatch.setattr("zemble.embedding.referenced.today", lambda: tomorrow)
    with caplog.at_level(logging.INFO, logger="zemble.embedding.referenced"):
        [old] = report_referenced(cache_folder, client)
    assert old.skipped is not None and "predates" in old.skipped, "step 4"
    assert sum("no referenced-vectors route" in record.message for record in caplog.records) == 1
    assert report_referenced(cache_folder, client)[0].skipped == REPORTED_TODAY, "step 4: retried only the next day"

    # 5. The next day, against an upgraded server, it goes out again.
    monkeypatch.setattr(http_server, "_route", original)
    monkeypatch.setattr("zemble.embedding.referenced.today", lambda: tomorrow + 1)
    assert report_referenced(cache_folder, client)[0].held == 4, "step 5"


@pytest.mark.anyio
async def test_the_daemon_reports_referenced_vectors_in_the_background(
    served: Running, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daemon's loop sends the report after its first delay, and its env knob turns it off."""
    cache_folder = resolve_cache_folder()
    _index(cache_folder, "a" * 64, [make_chunk("def kept(): pass")], embedder=SPEC)
    monkeypatch.setattr(daemon_server, "_REFERENCED_FIRST_DELAY_SECONDS", 0.0)
    daemon = daemon_server.Daemon(watch=False)
    loop = asyncio.create_task(daemon.referenced_loop())
    try:
        for _ in range(500):
            if served.service.counters.referenced_reported:
                break
            await asyncio.sleep(0.01)
    finally:
        loop.cancel()
    assert served.service.counters.referenced_reported == 1, "the loop reported the one chunk"

    monkeypatch.setenv(daemon_server.REFERENCED_HOURS_ENV, "0")
    await asyncio.wait_for(daemon.referenced_loop(), timeout=5)
