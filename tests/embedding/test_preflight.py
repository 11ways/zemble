from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.embedding.test_pricing import PricedEmbedder
from zemble.chunking.capsule import embedding_text
from zemble.embedding.cache import CachingEmbedder, EmbeddingCache, text_hash
from zemble.embedding.preflight import embed_status
from zemble.embedding.registry import ResolvedEmbedder
from zemble.index import ScopeRefused, ZembleIndex
from zemble.index.create import plan_files
from zemble.index.scope import WORK_LIMIT_ENV
from zemble.types import ContentType

FAMILY = "voyage:voyage-4-lite"


def resolve_to(monkeypatch: pytest.MonkeyPatch, provider: PricedEmbedder, scheme: str) -> None:
    """Resolve every spec to this provider behind the cache, the way the registry wraps a paid one."""
    monkeypatch.setattr(
        "zemble.embedding.preflight.build_embedder",
        lambda spec: ResolvedEmbedder(
            spec=spec, embedder=CachingEmbedder(provider, FAMILY), scheme=scheme, family=FAMILY
        ),
    )


@pytest.fixture
def paid_embedder(monkeypatch: pytest.MonkeyPatch) -> PricedEmbedder:
    """Resolve every spec to one remote-looking embedder, so no provider is ever contacted."""
    embedder = PricedEmbedder(dimensions=8)
    resolve_to(monkeypatch, embedder, "voyage")
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
    ZembleIndex.from_path(tmp_project, embedder=paid_embedder)
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
    """The report says whether a build would be refused, reading both ceilings a build passes."""
    # 1. The bill guard: one token of budget cannot pay for a whole tree.
    monkeypatch.setenv("ZEMBLE_EMBED_BUDGET_TOKENS", "1")
    assert embed_status(tmp_project).would_refuse, "step 1: the report reads the same budget the guard reads"
    monkeypatch.setenv("ZEMBLE_EMBED_CONFIRM", "1")
    assert not embed_status(tmp_project).would_refuse, "step 1: a confirmed caller is not refused"

    monkeypatch.delenv("ZEMBLE_EMBED_BUDGET_TOKENS")
    monkeypatch.delenv("ZEMBLE_EMBED_CONFIRM")

    # 2. The work guard: a build nobody would be billed much for is still refused as too much work.
    vendored = tmp_project / "vendored"
    vendored.mkdir()
    for index in range(12):
        (vendored / f"copy_{index}.py").write_text("x = 1\n" * 16_000, encoding="utf-8")
    status = embed_status(tmp_project)
    assert not status.would_refuse, "step 2: a megabyte of source is ordinary work"
    assert status.source_bytes > 1_000_000, "step 2: and the report says how much work a build is"
    monkeypatch.setenv(WORK_LIMIT_ENV, "1")
    assert embed_status(tmp_project).would_refuse, "step 2: past the work ceiling the report says REFUSED"


def test_embed_status_models_the_exclude_that_recovers_a_refused_build(
    tmp_path: Path, paid_embedder: PricedEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`exclude` is the sanctioned way out of a refusal, so the report has to be able to model it.

    It could not: the report passed no exclude to the walk and left it out of the manifest
    lookup, so it answered REFUSED for exactly the call that succeeds.
    """
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
    (root / "vendored").mkdir()
    for index in range(12):
        (root / "vendored" / f"copy_{index}.py").write_text("x = 1\n" * 16_000, encoding="utf-8")
    monkeypatch.setenv(WORK_LIMIT_ENV, "1")

    # 1. The plain root is refused, by the report and by a build alike.
    assert embed_status(root).would_refuse, "step 1: the fat tree is over the work ceiling"
    with pytest.raises(ScopeRefused):
        ZembleIndex.from_path(root, embedder=paid_embedder)

    # 2. With the fat directory excluded the report says allowed, and reports only the small tree.
    pruned = embed_status(root, exclude=("vendored/",))
    assert not pruned.would_refuse, "step 2: the report models the recovery it advertises"
    assert pruned.source_bytes < 1_000_000, f"step 2: and it measures the pruned walk, got {pruned.source_bytes}"

    # 3. The build agrees, which is the whole point: one answer, two surfaces.
    index = ZembleIndex.from_path(root, embedder=paid_embedder, exclude=["vendored/"])
    assert index.stats.indexed_files == 1, "step 3: the pruned build holds only the small tree"


def test_embed_status_reads_the_cache_of_a_model_whose_width_needs_a_probe(
    tmp_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A width nobody declared is not a reason to bill every chunk again.

    An OpenAI-compatible endpoint with no `@dims` declares no width, so the report skipped the
    cache lookup entirely and priced the whole tree - while the build probes the width once and
    then reads that very cache. The file itself records the width, and asks nobody.
    """

    class UndeclaredWidth(PricedEmbedder):
        """A remote embedder whose width only a provider request would tell."""

        declared_dimensions = None

    embedder = UndeclaredWidth(dimensions=8)
    resolve_to(monkeypatch, embedder, "openai")

    # 1. Nothing is declared, and cold it is honestly the whole tree.
    cold = embed_status(tmp_project)
    assert cold.dimensions is None, "step 1: the width is unknown without a request"
    assert cold.uncached > 0 and cold.estimated_tokens > 0, "step 1: an empty cache bills everything"

    # 2. With every chunk already bought at one width, the report finds them, having asked nobody.
    seed(chunk_texts(tmp_project), 8)
    warm = embed_status(tmp_project)
    assert warm.cached == cold.uncached, f"step 2: every bought chunk is a hit, got {warm.cached}"
    assert (warm.uncached, warm.estimated_tokens) == (0, 0), "step 2: nothing left to buy is nothing to bill"
    assert warm.cache_path is not None, "step 2: and the report names the file it read"
    assert embedder.document_batches == [], "step 2: a pre-flight never contacts a provider"

    # 3. A file holding two widths cannot say which one a build would read, so it bills again.
    seed(chunk_texts(tmp_project)[:1], 16)
    mixed = embed_status(tmp_project)
    assert mixed.uncached == cold.uncached, f"step 3: an ambiguous cache over-bills rather than guess, got {mixed}"


def test_embed_status_of_a_local_embedder(tmp_project: Path) -> None:
    """A local embedder is free, uses no vector cache, and can never be refused."""
    status = embed_status(tmp_project, embedder_spec="model2vec:minishlab/potion-code-16M-v2")
    assert not status.remote, "model2vec runs here"
    assert status.price_per_million_usd == 0.0, "a local embedder is free"
    assert status.estimated_usd == 0.0, "free stays free however many chunks there are"
    assert status.cache_path is None, "the sqlite vector cache is for paid providers only"
    assert not status.would_refuse, "a tiny tree is far under the local work ceiling"


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
        "source_bytes",
        "work_limit_bytes",
        "budget_usd",
        "budget_tokens",
        "would_refuse",
        "chunk_seconds",
        "cache_lookup_seconds",
        "dupes",
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
    for fragment in ("root", "embedder", "chunks", "tokens", "cost", "work", "budget", "verdict"):
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


def test_embed_status_estimates_a_dupes_run(tmp_path: Path, paid_embedder: PricedEmbedder) -> None:
    """The texts a dupes run would buy are estimated from the same builders the run embeds, then cached away."""
    from zemble.dedup.detect import DupeOptions, embedding_texts
    from zemble.dedup.model import CloneKind

    for name in ("Mailer", "Courier"):
        (tmp_path / f"{name}.java").write_text(
            f"class {name} {{\n    /** @return the value, or null when blank */\n"
            "    static String blankToNull(String value) {\n"
            "        String text = value.strip();\n        return text.isEmpty() ? null : text;\n    }\n}\n"
        )
    texts = embedding_texts(tmp_path, DupeOptions(kinds=(CloneKind.REIMPLEMENTS,)))
    assert texts, "the fixture yields bodies and intents to embed"

    # 1. Cold: every body and intent text of the run is pending, priced like a build's chunks.
    status = embed_status(tmp_path, dupes=(CloneKind.REIMPLEMENTS, CloneKind.HOLED)).dupes
    assert status is not None and status.kinds == ["reimplements"], "step 1: only a kind that embeds is estimated"
    assert (status.texts, status.uncached) == (len(texts), len(texts)), "step 1: nothing cached yet"
    assert status.estimated_tokens > 0 and not status.would_refuse, "step 1: a small run is affordable"
    assert paid_embedder.document_batches == [], "step 1: the estimate embeds nothing"

    # 2. Once the run's texts are cached, the run is free.
    seed(texts, 8)
    status = embed_status(tmp_path, dupes=(CloneKind.REIMPLEMENTS,)).dupes
    assert status is not None and (status.uncached, status.estimated_tokens) == (0, 0), "step 2: all cached"

    # 3. Near miss: kinds that embed nothing add no dupes line at all.
    assert embed_status(tmp_path, dupes=(CloneKind.HOLED,)).dupes is None, "step 3: holed embeds nothing"
