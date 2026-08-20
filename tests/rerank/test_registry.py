from __future__ import annotations

import pytest

from zemble.rerank.cross_encoder import CrossEncoderReranker
from zemble.rerank.registry import (
    PassageMode,
    RerankerSpecError,
    RerankSettings,
    load_reranker,
    parse_reranker_spec,
    resolve_reranker_spec,
)
from zemble.rerank.voyage import VoyageReranker


def test_none_spec_builds_no_reranker() -> None:
    """The default spec turns the pass off without building anything."""
    assert parse_reranker_spec("none") is None
    assert parse_reranker_spec("NONE") is None


def test_cross_spec_builds_a_cross_encoder() -> None:
    """A cross: spec names a Hugging Face model and does not load it yet."""
    reranker = parse_reranker_spec("cross:cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_id == "cross:cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert reranker._model is None


def test_voyage_spec_builds_a_voyage_client() -> None:
    """A voyage: spec names a rerank model."""
    reranker = parse_reranker_spec("voyage:rerank-2.5-lite")
    assert isinstance(reranker, VoyageReranker)
    assert reranker.model_id == "voyage:rerank-2.5-lite"


@pytest.mark.parametrize(
    "spec",
    ["", "   ", "cross", "cross:", "local:model", "bert:base", ":model"],
)
def test_unknown_or_incomplete_specs_are_refused(spec: str) -> None:
    """An unknown scheme or a missing model is a loud error, never a silent no-op."""
    with pytest.raises(RerankerSpecError):
        parse_reranker_spec(spec)


def test_environment_supplies_the_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZEMBLE_RERANKER is read when no explicit spec is given, and an argument wins over it."""
    monkeypatch.delenv("ZEMBLE_RERANKER", raising=False)
    assert resolve_reranker_spec() == "none"
    assert load_reranker() is None

    monkeypatch.setenv("ZEMBLE_RERANKER", "voyage:rerank-2.5")
    assert resolve_reranker_spec() == "voyage:rerank-2.5"
    assert resolve_reranker_spec("none") == "none"
    reranker = load_reranker()
    assert reranker is not None
    assert reranker.model_id == "voyage:rerank-2.5"


def test_settings_default_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset knobs give the documented defaults."""
    for name in ("ZEMBLE_RERANK_K", "ZEMBLE_RERANK_ALPHA", "ZEMBLE_RERANK_PASSAGE"):
        monkeypatch.delenv(name, raising=False)
    assert RerankSettings.from_env() == RerankSettings(top_k=50, alpha=1.0, passage=PassageMode.CONTEXT)


def test_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every knob is settable, and the passage mode is a closed vocabulary."""
    monkeypatch.setenv("ZEMBLE_RERANK_K", "20")
    monkeypatch.setenv("ZEMBLE_RERANK_ALPHA", "0.7")
    monkeypatch.setenv("ZEMBLE_RERANK_PASSAGE", "content")
    assert RerankSettings.from_env() == RerankSettings(top_k=20, alpha=0.7, passage=PassageMode.CONTENT)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ZEMBLE_RERANK_K", "0"),
        ("ZEMBLE_RERANK_K", "-3"),
        ("ZEMBLE_RERANK_K", "many"),
        ("ZEMBLE_RERANK_ALPHA", "1.5"),
        ("ZEMBLE_RERANK_ALPHA", "high"),
        ("ZEMBLE_RERANK_PASSAGE", "capsule"),
    ],
)
def test_unusable_knobs_are_refused(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    """A knob set to something unusable fails loudly instead of falling back to the default."""
    monkeypatch.setenv(name, value)
    with pytest.raises(RerankerSpecError):
        RerankSettings.from_env()
