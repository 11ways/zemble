from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.embedding.test_pricing import PricedEmbedder
from zemble.chunking.capsule import embedding_text
from zemble.embedding.cache import EmbeddingCache, text_hash
from zemble.embedding.preflight import embed_status
from zemble.embedding.registry import ResolvedEmbedder
from zemble.index.create import plan_files
from zemble.types import ContentType

FAMILY = "voyage:voyage-4-lite"


@pytest.fixture
def paid_embedder(monkeypatch: pytest.MonkeyPatch) -> PricedEmbedder:
    """Resolve every spec to one remote-looking embedder, so no provider is ever contacted."""
    embedder = PricedEmbedder(dimensions=8)
    monkeypatch.setattr(
        "zemble.embedding.preflight.build_embedder",
        lambda spec: ResolvedEmbedder(spec=spec, embedder=embedder, scheme="voyage", family=FAMILY),
    )
    return embedder


def chunk_texts(root: Path) -> list[str]:
    """Return the exact texts a build over this tree would embed."""
    return [
        embedding_text(chunk)
        for planned in plan_files(root, (ContentType.CODE,), display_root=root)
        for chunk in planned.chunks
    ]


def seed(texts: list[str], dims: int) -> None:
    """Store a vector for each text at a width, as a paid build would have."""
    cache = EmbeddingCache(FAMILY)
    try:
        cache.put_many([(text_hash(text), dims, np.ones(dims, dtype=np.float32) / np.sqrt(dims)) for text in texts])
    finally:
        cache.close()


def test_embed_status_journey(tmp_project: Path, paid_embedder: PricedEmbedder) -> None:
    """A pre-flight walks cold -> partly cached -> matryoshka-covered -> fully reusable, embedding nothing."""
    (tmp_project / "billing.py").write_text("def charge(amount):\n    return amount * 2\n")
    texts = chunk_texts(tmp_project)
    assert len(texts) >= 3, "the fixture project must chunk into enough files to split up"

    # 1. Cold: nothing cached, nothing reusable, and the whole tree is pending.
    status = embed_status(tmp_project)
    assert (status.chunks_total, status.reusable, status.cached) == (len(texts), 0, 0), "step 1: everything is pending"
    assert status.uncached == len(texts), "step 1: an empty cache means every chunk is uncached"
    assert status.estimated_tokens > 0, "step 1: pending chunks cost tokens"
    assert status.price_per_million_usd == 0.02, "step 1: voyage-4-lite is priced from the table"
    assert status.estimated_usd == pytest.approx(status.estimated_tokens * 0.02 / 1_000_000), "step 1: cost is derived"
    assert not status.would_refuse, "step 1: a tiny tree is far under the default budget"
    assert paid_embedder.document_batches == [], "step 1: a pre-flight never embeds"

    # 2. Seeding one chunk's vector at the requested width moves it from uncached to cached.
    seed(texts[:1], 8)
    status = embed_status(tmp_project)
    assert (status.cached, status.uncached) == (1, len(texts) - 1), "step 2: the seeded chunk is a cache hit"
    assert status.cache_path is not None and status.cache_path.endswith(".sqlite"), "step 2: the cache file is named"

    # 3. A wider vector counts too: the cache slices it, so it is not a chunk anyone pays for again.
    seed(texts[1:2], 16)
    status = embed_status(tmp_project)
    assert status.cached == 2, "step 3: a matryoshka-wider vector counts as cached"

    # 4. With an index on disk, unchanged files are reusable and are never even looked up.
    from zemble.cache import save_index_to_cache
    from zemble.index import ZembleIndex

    index = ZembleIndex.from_path(tmp_project, embedder=paid_embedder)
    save_index_to_cache(index, str(tmp_project))
    status = embed_status(tmp_project)
    assert status.reusable == len(texts), "step 4: every unchanged file is reused from the previous index"
    assert (status.cached, status.uncached, status.estimated_tokens) == (0, 0, 0), "step 4: a warm build is free"
    assert not status.would_refuse, "step 4: a free build is never refused"

    # 5. Touching one file puts exactly that file's chunks back in the pending set.
    target = tmp_project / "auth.py"
    target.write_text(target.read_text() + "\n\ndef logout(token):\n    return None\n")
    status = embed_status(tmp_project)
    assert status.reusable < len(texts), "step 5: the changed file is no longer reusable"
    assert status.uncached > 0, "step 5: its chunks are pending again"
    assert status.chunks_total == status.reusable + status.cached + status.uncached, "step 5: the counts add up"


def test_embed_status_reports_a_refusal(tmp_project: Path, paid_embedder: PricedEmbedder, monkeypatch) -> None:
    """The report says whether a build would be refused, using the same budget the guard reads."""
    monkeypatch.setenv("ZEMBLE_EMBED_BUDGET_TOKENS", "1")
    assert embed_status(tmp_project).would_refuse, "one token of budget cannot pay for a whole tree"
    monkeypatch.setenv("ZEMBLE_EMBED_CONFIRM", "1")
    assert not embed_status(tmp_project).would_refuse, "a confirmed caller is not refused"


def test_embed_status_of_a_local_embedder(tmp_project: Path) -> None:
    """A local embedder is free, uses no vector cache, and can never be refused."""
    status = embed_status(tmp_project, embedder_spec="model2vec:minishlab/potion-code-16M-v2")
    assert not status.remote, "model2vec runs here"
    assert status.price_per_million_usd == 0.0, "a local embedder is free"
    assert status.estimated_usd == 0.0, "free stays free however many chunks there are"
    assert status.cache_path is None, "the sqlite vector cache is for paid providers only"
    assert not status.would_refuse, "a local build is never budget-gated"


def test_embed_status_json_shape(tmp_project: Path, paid_embedder: PricedEmbedder) -> None:
    """The JSON payload carries every number the text report shows."""
    payload = embed_status(tmp_project).to_dict()
    assert json.loads(json.dumps(payload)) == payload, "the payload must be JSON-encodable"
    assert set(payload) == {
        "path",
        "embedder",
        "family",
        "remote",
        "dimensions",
        "content",
        "chunks_total",
        "reusable",
        "cached",
        "uncached",
        "estimated_tokens",
        "price_per_million_usd",
        "estimated_usd",
        "cache_path",
        "budget_tokens",
        "would_refuse",
        "chunk_seconds",
        "cache_lookup_seconds",
    }, "the JSON shape is a contract; add a key deliberately"


def test_embed_status_missing_path(tmp_path: Path) -> None:
    """A root that does not exist is refused by name."""
    with pytest.raises(FileNotFoundError):
        embed_status(tmp_path / "absent")


def test_cli_embed_status(
    tmp_project: Path,
    paid_embedder: PricedEmbedder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`zemble embed-status` prints the report as text, and as JSON when asked."""
    from zemble.cli import _cli_main

    monkeypatch.setattr(sys, "argv", ["zemble", "embed-status", str(tmp_project)])
    with pytest.raises(SystemExit) as raised:
        _cli_main()
    assert raised.value.code == 0, "a readable tree reports successfully"
    out = capsys.readouterr().out
    for fragment in ("root", "embedder", "chunks", "tokens", "cost", "budget"):
        assert fragment in out, f"the human report must mention {fragment}"

    monkeypatch.setattr(sys, "argv", ["zemble", "embed-status", str(tmp_project), "--json"])
    with pytest.raises(SystemExit):
        _cli_main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == str(tmp_project.resolve()), "the JSON report names the root it walked"
    assert paid_embedder.document_batches == [], "the CLI journey embeds nothing"


def test_cli_embed_status_missing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing root exits non-zero with the message on stderr."""
    from zemble.cli import _cli_main

    monkeypatch.setattr(sys, "argv", ["zemble", "embed-status", str(tmp_path / "absent")])
    with pytest.raises(SystemExit) as raised:
        _cli_main()
    assert raised.value.code == 1, "a missing root is an error"
    assert "does not exist" in capsys.readouterr().err, "and it says why"
