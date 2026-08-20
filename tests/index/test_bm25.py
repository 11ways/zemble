import math
import random
from pathlib import Path

import numpy as np
import orjson
import pytest

from zemble.index.bm25 import BM25


def _build(docs: dict[str, list[str]]) -> BM25:
    index = BM25()
    for chunk_id, tokens in docs.items():
        index.add_document(chunk_id, tokens)
    index.set_doc_order(list(docs))
    return index


def test_scoring_matches_lucene_formula() -> None:
    """BM25 scores use the Lucene term-frequency formula."""
    index = _build({"a": ["authenticate", "token"], "b": ["login", "password"]})
    scores = index.get_scores(["authenticate"])
    np.testing.assert_allclose(scores[0], math.log(1 + 1.5 / 1.5) / 2.5)
    assert scores[1] == 0


def test_removed_and_unordered_documents_stop_scoring() -> None:
    """Only documents retained in the current order contribute scores."""
    index = _build({"a": ["authenticate"], "b": ["login"]})
    index.remove_document("missing")
    index.set_doc_order(["b"])
    assert np.all(index.get_scores(["authenticate"]) == 0)

    index.remove_document("a")
    index.set_doc_order(["a", "b"])
    assert np.all(index.get_scores(["authenticate"]) == 0)


def test_duplicate_add_document_raises() -> None:
    """Re-adding an already-indexed chunk_id raises, catching caller bugs."""
    index = _build({"a": ["x"]})
    with pytest.raises(ValueError, match="already indexed"):
        index.add_document("a", ["y"])


@pytest.mark.parametrize(
    ("mask", "expected_nonzero"),
    [
        (None, [0, 1]),
        (np.array([True, False]), [0]),
    ],
)
def test_weight_mask_zeroes_masked_docs(mask: np.ndarray | None, expected_nonzero: list[int]) -> None:
    """weight_mask zeroes out scores for masked-out positions, by global chunk order."""
    index = _build({"a": ["shared"], "b": ["shared"]})
    scores = index.get_scores(["shared"], weight_mask=mask)
    nonzero = [i for i, s in enumerate(scores) if s > 0]
    assert nonzero == expected_nonzero


@pytest.mark.parametrize("query", [[], ["zzznonexistent"]])
def test_unmatched_queries_return_all_zero(query: list[str]) -> None:
    """Empty and unknown queries return an all-zero array sized to the corpus."""
    index = _build({"a": ["foo"], "b": ["bar"]})
    scores = index.get_scores(query)
    assert scores.shape == (2,)
    assert np.all(scores == 0)


def test_save_load_preserves_scores_and_doc_order(tmp_path: Path) -> None:
    """save/load roundtrips postings and doc_order, producing identical scores for a fixed query."""
    index = _build({"empty": [], "a": ["authenticate", "token"], "b": ["login", "password"]})
    index.save(tmp_path)

    loaded = BM25.load(tmp_path)
    assert loaded.doc_order == index.doc_order
    np.testing.assert_array_equal(loaded.get_scores(["authenticate"]), index.get_scores(["authenticate"]))


def test_load_rejects_inconsistent_document_order(tmp_path: Path) -> None:
    """Persisted columns that disagree about how many documents they hold are refused."""
    index = _build({"a": ["authenticate"]})
    index.save(tmp_path)
    meta_path = tmp_path / "postings.json"
    meta = orjson.loads(meta_path.read_bytes())
    meta["n_docs"] += 1
    meta_path.write_bytes(orjson.dumps(meta))

    with pytest.raises(ValueError, match="document state"):
        BM25.load(tmp_path)


def test_save_refuses_a_document_set_that_the_order_does_not_describe(tmp_path: Path) -> None:
    """A mutable index whose documents and order disagree cannot be persisted."""
    index = _build({"a": ["authenticate"]})
    index.set_doc_order(["a", "b"])

    with pytest.raises(ValueError, match="inconsistent"):
        index.save(tmp_path)


def test_frozen_scores_match_the_mutable_implementation(tmp_path: Path) -> None:
    """A loaded index scores a random corpus exactly like the in-memory one it was saved from.

    The columnar postings replace a per-document dict walk, so this walks one corpus through
    build, save and load and asserts the scores never move.
    """
    rng = random.Random(20260820)
    vocabulary = [f"term{i}" for i in range(60)]
    corpus = {f"file{doc}.py:{doc}": [rng.choice(vocabulary) for _ in range(rng.randint(0, 40))] for doc in range(120)}

    # 1. The mutable index is the implementation the columnar one has to reproduce.
    built = _build(corpus)
    queries = [[rng.choice(vocabulary) for _ in range(rng.randint(1, 5))] for _ in range(40)]
    expected = [built.get_scores(query) for query in queries]

    # 2. Saving and loading must not move a single score.
    built.save(tmp_path)
    loaded = BM25.load(tmp_path)
    assert loaded.doc_order == built.doc_order, "the document order survives the roundtrip"
    for query, want in zip(queries, expected):
        np.testing.assert_allclose(loaded.get_scores(query), want, atol=1e-6)
        np.testing.assert_array_equal(loaded.get_scores(query), want)

    # 3. Every term's postings name each document once, which is what makes the vectorized
    #    accumulate in the frozen scorer a true sum rather than a last-write-wins scatter.
    frozen = loaded._frozen
    assert frozen is not None
    offsets = np.asarray(frozen.posting_offsets)
    for row in range(frozen.n_terms):
        documents = np.asarray(frozen.posting_docs[offsets[row] : offsets[row + 1]])
        assert len(np.unique(documents)) == len(documents), "postings hold each document once"

    # 4. A mutation thaws the columns back into the dictionaries, and scoring still agrees.
    loaded.remove_document(built.doc_order[0])
    assert loaded._frozen is None
    reference = _build({key: tokens for key, tokens in corpus.items() if key != built.doc_order[0]})
    reference.set_doc_order(built.doc_order)
    np.testing.assert_array_equal(loaded.get_scores(queries[0]), reference.get_scores(queries[0]))
