import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import orjson
import pytest

from zemble.index import bm25 as bm25_module
from zemble.index.bm25 import BM25, BM25Writer


def _write(directory: Path, docs: dict[str, list[str]], previous: BM25 | None = None) -> BM25:
    """Write *docs* from scratch (or reusing nothing) and load the result."""
    writer = BM25Writer(directory, previous)
    for chunk_id, tokens in docs.items():
        writer.add(chunk_id, tokens)
    writer.finish()
    return BM25.load(directory)


def _reference_scores(docs: dict[str, list[str]], query: list[str]) -> np.ndarray:
    """Score *query* with the textbook Lucene BM25 over *docs*, one document at a time."""
    corpus = list(docs.values())
    avgdl = sum(len(tokens) for tokens in corpus) / len(corpus)
    scores = np.zeros(len(corpus), dtype=np.float64)
    for term, query_tf in Counter(query).items():
        df = sum(1 for tokens in corpus if term in tokens)
        if not df:
            continue
        idf = math.log(1 + (len(corpus) - df + 0.5) / (df + 0.5))
        for row, tokens in enumerate(corpus):
            tf = tokens.count(term)
            if tf:
                scores[row] += query_tf * idf * tf / (1.5 * (1 - 0.75 + 0.75 * len(tokens) / avgdl) + tf)
    return scores


def _random_corpus(rng: random.Random, vocabulary: list[str], count: int, prefix: str) -> dict[str, list[str]]:
    """Build a corpus of documents whose tokens are drawn from *vocabulary*."""
    return {
        f"{prefix}{doc}.py:{doc}": [rng.choice(vocabulary) for _ in range(rng.randint(0, 40))] for doc in range(count)
    }


def _postings(index: BM25) -> list[list[tuple[int, int]]]:
    """Every term's (document, frequency) pairs, sorted."""
    frozen = index._frozen
    offsets = np.asarray(frozen.posting_offsets)
    docs, tfs = np.asarray(frozen.posting_docs), np.asarray(frozen.posting_tf)
    return [
        sorted(zip(docs[offsets[row] : offsets[row + 1]].tolist(), tfs[offsets[row] : offsets[row + 1]].tolist()))
        for row in range(frozen.n_terms)
    ]


def test_scoring_matches_lucene_formula(tmp_path: Path) -> None:
    """BM25 scores use the Lucene term-frequency formula."""
    index = _write(tmp_path, {"a": ["authenticate", "token"], "b": ["login", "password"]})
    scores = index.get_scores(["authenticate"])
    np.testing.assert_allclose(scores[0], math.log(1 + 1.5 / 1.5) / 2.5)
    assert scores[1] == 0


@pytest.mark.parametrize(
    ("mask", "expected_nonzero"),
    [
        (None, [0, 1]),
        (np.array([True, False]), [0]),
    ],
)
def test_weight_mask_zeroes_masked_docs(tmp_path: Path, mask: np.ndarray | None, expected_nonzero: list[int]) -> None:
    """weight_mask zeroes out scores for masked-out positions, by global chunk order."""
    index = _write(tmp_path, {"a": ["shared"], "b": ["shared"]})
    scores = index.get_scores(["shared"], weight_mask=mask)
    assert [i for i, s in enumerate(scores) if s > 0] == expected_nonzero


@pytest.mark.parametrize("query", [[], ["zzznonexistent"]])
def test_unmatched_queries_return_all_zero(tmp_path: Path, query: list[str]) -> None:
    """Empty and unknown queries return an all-zero array sized to the corpus."""
    index = _write(tmp_path, {"a": ["foo"], "b": ["bar"]})
    scores = index.get_scores(query)
    assert scores.shape == (2,)
    assert np.all(scores == 0)


def test_an_index_of_empty_documents_loads_and_scores(tmp_path: Path) -> None:
    """Documents without a token still occupy their row, and an index with no posting at all loads."""
    index = _write(tmp_path, {"empty": [], "also-empty": []})
    assert index.doc_order == ["empty", "also-empty"]
    assert np.all(index.get_scores(["anything"]) == 0)


def test_load_rejects_inconsistent_document_state(tmp_path: Path) -> None:
    """Persisted columns that disagree about how many documents they hold are refused."""
    _write(tmp_path, {"a": ["authenticate"]})
    meta_path = tmp_path / "postings.json"
    meta = orjson.loads(meta_path.read_bytes())
    meta["n_docs"] += 1
    meta_path.write_bytes(orjson.dumps(meta))

    with pytest.raises(ValueError, match="document state"):
        BM25.load(tmp_path)


def test_written_scores_match_the_reference_formula(tmp_path: Path) -> None:
    """A written index scores a random corpus exactly like the formula applied document by document."""
    rng = random.Random(20260820)
    vocabulary = [f"term{i}" for i in range(60)]
    corpus = _random_corpus(rng, vocabulary, 120, "file")
    queries = [[rng.choice(vocabulary) for _ in range(rng.randint(1, 5))] for _ in range(40)]

    # 1. The document order is the order documents were written in.
    index = _write(tmp_path, corpus)
    assert index.doc_order == list(corpus)

    # 2. Every score agrees with the reference.
    for query in queries:
        np.testing.assert_allclose(index.get_scores(query), _reference_scores(corpus, query), rtol=1e-5)

    # 3. Every term's postings name each document once, which makes the vectorized accumulate a true sum.
    frozen = index._frozen
    offsets = np.asarray(frozen.posting_offsets)
    for row in range(frozen.n_terms):
        documents = np.asarray(frozen.posting_docs[offsets[row] : offsets[row + 1]])
        assert len(np.unique(documents)) == len(documents), "postings hold each document once"


@pytest.mark.parametrize("block", [3, 1 << 20])
def test_an_incremental_write_equals_a_write_from_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block: int
) -> None:
    """Carrying documents over from a previous index writes exactly what writing them anew would.

    This is the whole contract of reuse: nobody can tell, from the files or from a score, which
    documents came from the previous generation. A tiny merge block forces every term range
    across several merge steps.
    """
    monkeypatch.setattr(bm25_module, "_BLOCK_POSTINGS", block)
    rng = random.Random(20260821)
    vocabulary = [f"term{i}" for i in range(60)]
    corpus = _random_corpus(rng, vocabulary, 200, "file")
    previous = _write(tmp_path / "previous", corpus)

    # 1. One document is replaced, one dropped, the rest carried over, and two arrive with a new term.
    ids = list(corpus)
    live: dict[str, list[str]] = {}
    writer = BM25Writer(tmp_path / "next", previous)
    for row, chunk_id in enumerate(ids):
        if chunk_id == "file8.py:8":
            continue
        if chunk_id == "file7.py:7":
            live[chunk_id] = ["singularmarkerterm", "term1"]
            writer.add(chunk_id, live[chunk_id])
            continue
        live[chunk_id] = corpus[chunk_id]
        writer.reuse(row, 1)
    for chunk_id in ("new1.py:0", "new2.py:0"):
        live[chunk_id] = [rng.choice(vocabulary) for _ in range(rng.randint(1, 40))] + ["brandnewterm"]
        writer.add(chunk_id, live[chunk_id])
    writer.finish()
    reused = BM25.load(tmp_path / "next")
    fresh = _write(tmp_path / "fresh", live)

    # 2. The two generations hold the same documents, vocabulary and postings; within a term the
    #    carried postings come first, which no score can see because each document appears once.
    assert reused.doc_order == fresh.doc_order == list(live)
    for name in ("posting_offsets.npy", "doc_lengths.npy", "terms.bin", "postings.json"):
        assert (tmp_path / "next" / name).read_bytes() == (tmp_path / "fresh" / name).read_bytes(), name
    assert _postings(reused) == _postings(fresh)

    # 3. And score alike: the replaced document by its new tokens, the dropped one not at all.
    for query in (["singularmarkerterm"], ["brandnewterm"], ["term1", "term2"], *[[term] for term in vocabulary]):
        np.testing.assert_array_equal(reused.get_scores(query), fresh.get_scores(query))
    assert reused.get_scores(["singularmarkerterm"]).sum() > 0

    # 4. Writing the next generation never touched the previous one.
    np.testing.assert_allclose(previous.get_scores(["term1"]), _reference_scores(corpus, ["term1"]), rtol=1e-5)


def test_a_term_whose_documents_are_all_dropped_leaves_the_vocabulary(tmp_path: Path) -> None:
    """The vocabulary holds only terms some live document carries, as a build from scratch would."""
    previous = _write(tmp_path / "previous", {"a.py:0": ["kept", "doomed"], "b.py:0": ["kept"]})
    writer = BM25Writer(tmp_path / "next", previous)
    writer.reuse(1, 1)
    writer.finish()
    index = BM25.load(tmp_path / "next")
    assert index._frozen.terms.to_list() == ["kept"]
    assert index.doc_order == ["b.py:0"]


def test_term_frequencies_too_large_for_sixteen_bits_widen_the_column(tmp_path: Path) -> None:
    """A document repeating a term more than 65535 times keeps its exact count."""
    index = _write(tmp_path, {"big.py:0": ["x"] * 70_000, "small.py:0": ["x"]})
    frozen = index._frozen
    assert frozen.posting_tf.dtype == np.int32
    assert sorted(np.asarray(frozen.posting_tf).tolist()) == [1, 70_000]
