"""Tiny, structurally different adapters remain discoverable as reviewable counterparts."""

from types import SimpleNamespace
from unittest.mock import Mock

from zemble.dedup.architectural import architectural_candidates
from zemble.evidence.counterparts import counterparts, declaration
from zemble.evidence.related import related_payload
from zemble.graph import SqliteGraphProvider, build_graph
from zemble.types import Chunk, SearchResult


def test_documented_counterpart_survives_tiny_body_and_header_flooding(tmp_path):
    """Source-backed migration evidence survives structural differences without proving equality."""
    # 1. A deprecated adapter explicitly names an implementation with a different body.
    target_text = "package example;\nclass Canonical {\n static String normalize(String s) { return s.trim(); }\n}\n"
    old_text = """package example;
class Adapter {
 /** @deprecated Use {@link Canonical#normalize(String)}. */
 @Deprecated
 static String old(String value) { throw new UnsupportedOperationException(); }
 static String unrelated() { return "other"; }
}
"""
    (tmp_path / "Canonical.java").write_text(target_text)
    (tmp_path / "Adapter.java").write_text(old_text)
    build_graph(str(tmp_path), workers=1)
    graph = SqliteGraphProvider(str(tmp_path))
    try:
        owner = declaration(graph, "Adapter.java", 5)
        assert owner.name == "old", "step 1: seed resolves to the callable, not its containing type"
        found = counterparts(graph, tmp_path, owner)
        assert any(c.symbol.qualified_name == "example.Canonical.normalize" for c in found), "step 1: doc link resolves"
        # 2. Imports and same-file context dominate raw neighbors but cannot flood the answer.
        target = Chunk(target_text, "Canonical.java", 1, 4, "java")
        old = Chunk(old_text, "Adapter.java", 1, 7, "java")
        index = SimpleNamespace(
            chunk_at=Mock(return_value=old),
            chunks_of=lambda file: [target] if file == "Canonical.java" else [old],
            find_related=Mock(return_value=[SearchResult(old, 0.999)] * 40),
            search=Mock(return_value=[]),
            _reranker=None,
        )
        payload = related_payload(index, graph, tmp_path, "Adapter.java", 5)
        assert payload["results"][0]["file_path"] == "Canonical.java", "step 2: canonical source outranks own context"
        assert payload["results"][0]["evidence"], "step 2: the answer explains its source evidence"
        # 3. Architecture reports candidates without claiming token equality or equivalence.
        report = architectural_candidates(tmp_path, ["Adapter.java", "Canonical.java"], graph)
        assert report["candidates"], "step 3: the tiny adapter is an architectural candidate"
        assert all(not c["equivalence_proven"] for c in report["candidates"]), "step 3: review remains explicit"
    finally:
        graph.close()


def test_reranker_resolves_existing_seam_and_preserves_direct_evidence(monkeypatch):
    """Optional reranking orders semantic candidates without demoting documented counterparts."""
    from zemble.evidence.related import _rerank

    monkeypatch.setenv("ZEMBLE_RELATED_RERANK", "1")
    direct = SearchResult(Chunk("direct", "core.java", 1, 2), 2.0)
    first = SearchResult(Chunk("first", "a.java", 1, 2), .9)
    second = SearchResult(Chunk("second", "b.java", 1, 2), .8)
    scorer = SimpleNamespace(score=Mock(return_value=[.1, .99]))
    resolver = Mock(return_value=scorer)
    index = SimpleNamespace(_resolve_reranker=resolver)
    ranked = _rerank(index, [direct, first, second], "behavior")
    assert [result.chunk.file_path for result in ranked] == ["core.java", "b.java", "a.java"]
    resolver.assert_called_once_with(None)
    scorer.score.assert_called_once_with("behavior", ["first", "second"])
