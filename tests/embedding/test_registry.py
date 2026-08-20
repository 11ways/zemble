from __future__ import annotations

import pytest

from zemble.embedding.cache import CachingEmbedder
from zemble.embedding.model2vec import Model2VecEmbedder
from zemble.embedding.openai_compat import OpenAICompatibleEmbedder
from zemble.embedding.registry import (
    DEFAULT_EMBEDDER_SPEC,
    EmbedderSpecError,
    caching_enabled,
    parse_embedder_spec,
    resolve_embedder_spec,
)
from zemble.embedding.voyage import VoyageEmbedder


def _unwrap(spec: str) -> object:
    """Parse a spec and return the underlying provider, unwrapping the cache if present."""
    embedder = parse_embedder_spec(spec)
    return embedder.inner if isinstance(embedder, CachingEmbedder) else embedder


def test_model2vec_spec_is_lazy_and_normalized() -> None:
    """A model2vec spec builds an unwrapped, uncached embedder that has not loaded anything yet."""
    embedder = parse_embedder_spec("model2vec:minishlab/potion-code-16M-v2")
    assert isinstance(embedder, Model2VecEmbedder)
    assert embedder.model_id == "model2vec:minishlab/potion-code-16M-v2"
    assert embedder._model is None, "constructing an embedder must not download a model"


@pytest.mark.parametrize(
    ("spec", "model", "dimensions", "model_id"),
    [
        ("voyage:voyage-code-4", "voyage-code-4", 1024, "voyage:voyage-code-4@1024"),
        ("voyage:voyage-code-4@256", "voyage-code-4", 256, "voyage:voyage-code-4@256"),
        ("voyage:voyage-4-lite@512", "voyage-4-lite", 512, "voyage:voyage-4-lite@512"),
    ],
)
def test_voyage_spec(spec: str, model: str, dimensions: int, model_id: str) -> None:
    """Voyage specs resolve model and dimensions, defaulting to the documented 1024."""
    embedder = _unwrap(spec)
    assert isinstance(embedder, VoyageEmbedder)
    assert embedder._model == model
    assert embedder.dimensions == dimensions
    assert embedder.model_id == model_id


def test_openai_spec() -> None:
    """An openai spec splits base URL, model and dimensions, and drops the trailing slash."""
    embedder = _unwrap("openai:http://localhost:11434/v1/#nomic-embed-text@768")
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder._base_url == "http://localhost:11434/v1"
    assert embedder._model == "nomic-embed-text"
    assert embedder.dimensions == 768
    assert embedder.model_id == "openai:http://localhost:11434/v1#nomic-embed-text@768"
    assert embedder._endpoint() == "http://localhost:11434/v1/embeddings"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("", "Empty embedder spec"),
        ("   ", "Empty embedder spec"),
        ("minishlab/potion-code-16M-v2", "Unknown embedder spec"),
        ("cohere:embed-v4", "Unknown embedder spec"),
        ("voyage:", "names no model"),
        ("voyage:voyage-code-4@abc", "Invalid dimensions"),
        ("voyage:voyage-code-4@0", "must be positive"),
        ("voyage:@256", "names no model"),
        ("model2vec:some/model@256", "width is fixed by the model"),
        ("openai:http://localhost:11434/v1", "must be openai:<base_url>#<model>"),
        ("openai:#model", "must be openai:<base_url>#<model>"),
        ("openai:http://x/v1#", "must be openai:<base_url>#<model>"),
    ],
)
def test_bad_specs_are_loud(spec: str, message: str) -> None:
    """Every malformed spec raises EmbedderSpecError naming what is wrong."""
    with pytest.raises(EmbedderSpecError, match=message):
        parse_embedder_spec(spec)


def test_remote_providers_are_cached_by_default() -> None:
    """Voyage and openai are wrapped in the content-hash cache; model2vec is not."""
    assert isinstance(parse_embedder_spec("voyage:voyage-code-4@256"), CachingEmbedder)
    assert isinstance(parse_embedder_spec("openai:http://localhost:1234/v1#m@8"), CachingEmbedder)
    assert not isinstance(parse_embedder_spec("model2vec:some/model"), CachingEmbedder)


def test_cache_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZEMBLE_EMBED_CACHE=0 removes the sqlite wrapper entirely."""
    monkeypatch.setenv("ZEMBLE_EMBED_CACHE", "0")
    assert not caching_enabled()
    assert isinstance(parse_embedder_spec("voyage:voyage-code-4@256"), VoyageEmbedder)


def test_spec_resolution_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit spec beats ZEMBLE_EMBEDDER, which beats the legacy ZEMBLE_MODEL_NAME, which beats the default."""
    monkeypatch.delenv("ZEMBLE_EMBEDDER", raising=False)
    monkeypatch.delenv("ZEMBLE_MODEL_NAME", raising=False)
    assert resolve_embedder_spec() == DEFAULT_EMBEDDER_SPEC

    monkeypatch.setenv("ZEMBLE_MODEL_NAME", "legacy/model")
    assert resolve_embedder_spec() == "model2vec:legacy/model"

    monkeypatch.setenv("ZEMBLE_EMBEDDER", "voyage:voyage-code-4@256")
    assert resolve_embedder_spec() == "voyage:voyage-code-4@256"
    assert resolve_embedder_spec("model2vec:explicit/model") == "model2vec:explicit/model"


def test_legacy_env_accepts_a_full_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ZEMBLE_MODEL_NAME that already carries a scheme is used verbatim, not double-prefixed."""
    monkeypatch.delenv("ZEMBLE_EMBEDDER", raising=False)
    monkeypatch.setenv("ZEMBLE_MODEL_NAME", "voyage:voyage-code-4@256")
    assert resolve_embedder_spec() == "voyage:voyage-code-4@256"
