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

    # 4. A mutation keeps the columns and records a delta beside them; scoring still agrees.
    loaded.remove_document(built.doc_order[0])
    assert loaded._frozen is frozen, "the base is kept, not rebuilt into dictionaries"
    reference = _build({key: tokens for key, tokens in corpus.items() if key != built.doc_order[0]})
    reference.set_doc_order(built.doc_order)
    loaded.set_doc_order(built.doc_order)
    np.testing.assert_allclose(loaded.get_scores(queries[0]), reference.get_scores(queries[0]), atol=1e-6)


def _random_corpus(rng: random.Random, vocabulary: list[str], count: int, prefix: str) -> dict[str, list[str]]:
    """Build a corpus of documents whose tokens are drawn from *vocabulary*."""
    return {
        f"{prefix}{doc}.py:{doc}": [rng.choice(vocabulary) for _ in range(rng.randint(0, 40))] for doc in range(count)
    }


def test_delta_scoring_matches_a_freshly_built_index(tmp_path: Path) -> None:
    """A loaded index that is updated in place scores exactly like one built from scratch.

    This is the whole contract of the delta: the caller cannot tell whether a document
    arrived in the persisted columns or afterwards, and neither can a score.
    """
    rng = random.Random(20260821)
    vocabulary = [f"term{i}" for i in range(60)]
    corpus = _random_corpus(rng, vocabulary, 200, "file")
    queries = [[rng.choice(vocabulary) for _ in range(rng.randint(1, 5))] for _ in range(40)]

    _build(corpus).save(tmp_path)
    loaded = BM25.load(tmp_path)
    base = loaded._frozen
    assert base is not None

    # 1. One file's documents are replaced, one is deleted and two arrive: an ordinary rebuild.
    live = dict(corpus)
    for chunk_id in ("file7.py:7", "file8.py:8"):
        loaded.remove_document(chunk_id)
        live.pop(chunk_id)
    for chunk_id in ("file7.py:7", "new1.py:0", "new2.py:0"):
        tokens = [rng.choice(vocabulary) for _ in range(rng.randint(0, 40))]
        loaded.add_document(chunk_id, tokens)
        live[chunk_id] = tokens
    order = list(live)
    loaded.set_doc_order(order)

    reference = _build(live)
    for query in queries:
        np.testing.assert_allclose(loaded.get_scores(query), reference.get_scores(query), atol=1e-6)

    # 2. The removed-and-re-added document is scored once, from its new tokens.
    marker = ["singularmarkerterm"]
    loaded.remove_document("file7.py:7")
    loaded.add_document("file7.py:7", marker)
    live["file7.py:7"] = marker
    loaded.set_doc_order(list(live))
    reference = _build(live)
    np.testing.assert_allclose(loaded.get_scores(marker), reference.get_scores(marker), atol=1e-6)
    assert loaded.get_scores(marker).sum() > 0, "the re-added document is findable by its new tokens"

    # 3. Folding turns base plus delta back into one base, and moves no score.
    folded = loaded.fold()
    assert folded._frozen is not None and folded.delta_documents == 0, "a folded index carries no delta"
    assert folded.doc_order == loaded.doc_order, "the document order survives the fold"
    for query in [*queries, marker]:
        np.testing.assert_allclose(folded.get_scores(query), loaded.get_scores(query), atol=1e-6)

    # 4. The folded index persists and reloads unchanged, so a cold start pays for no delta.
    folded.save(tmp_path)
    reloaded = BM25.load(tmp_path)
    assert reloaded.doc_order == folded.doc_order, "the persisted order is the folded one"
    for query in [*queries, marker]:
        np.testing.assert_allclose(reloaded.get_scores(query), folded.get_scores(query), atol=1e-6)

    # 5. Nothing wrote into the base the whole time: it still holds exactly what it was loaded with.
    assert loaded._frozen is base, "the delta never replaced the base"
    assert base.n_docs == len(corpus) and base.ids[7] == "file7.py:7", "the base still describes the saved corpus"


def test_for_update_leaves_the_index_it_came_from_serving(tmp_path: Path) -> None:
    """A rebuild works on a derived index; the one answering queries never changes."""
    rng = random.Random(4242)
    vocabulary = [f"term{i}" for i in range(30)]
    corpus = _random_corpus(rng, vocabulary, 40, "src")
    _build(corpus).save(tmp_path)
    served = BM25.load(tmp_path)
    before = {tuple(served.get_scores([term])) for term in vocabulary}

    updating = served.for_update()
    updating.remove_document("src3.py:3")
    updating.add_document("added.py:0", ["term1", "term1", "brandnewtoken"])
    order = [chunk_id for chunk_id in corpus if chunk_id != "src3.py:3"] + ["added.py:0"]
    updating.set_doc_order(order)

    assert updating._frozen is served._frozen, "both share one immutable base"
    assert {tuple(served.get_scores([term])) for term in vocabulary} == before, "the served index did not move"
    assert served.document_count == len(corpus), "and still holds every document it was loaded with"
    assert updating.document_count == len(corpus), "one gone, one added"
    assert updating.get_scores(["brandnewtoken"]).sum() > 0, "the update is visible in the derived index"
    assert served.get_scores(["brandnewtoken"]).sum() == 0, "and only there"


def test_a_grown_delta_is_folded_instead_of_carried(tmp_path: Path) -> None:
    """for_update folds once the delta is large enough that carrying it costs more than a fold."""
    from zemble.index import bm25 as bm25_module

    rng = random.Random(99)
    corpus = _random_corpus(rng, ["alpha", "beta", "gamma"], 40, "big")
    _build(corpus).save(tmp_path)
    loaded = BM25.load(tmp_path)
    small = loaded.for_update()
    assert small._frozen is loaded._frozen, "a fresh index is derived, not folded"

    for chunk_id in list(corpus)[:20]:
        small.remove_document(chunk_id)
    small.set_doc_order([chunk_id for chunk_id in corpus if chunk_id not in list(corpus)[:20]])
    scores = small.get_scores(["alpha"])

    monkeypatched = bm25_module._MIN_DELTA_DOCUMENTS
    try:
        bm25_module._MIN_DELTA_DOCUMENTS = 1
        grown = small.for_update()
    finally:
        bm25_module._MIN_DELTA_DOCUMENTS = monkeypatched
    assert grown._frozen is not small._frozen, "the delta was folded into a new base"
    assert grown.delta_documents == 0, "and the derived index starts clean"
    np.testing.assert_allclose(grown.get_scores(["alpha"]), scores, atol=1e-6)


def test_saving_replaces_columns_instead_of_truncating_them(tmp_path: Path) -> None:
    """A save never truncates a file a warm index still has mapped."""
    corpus = {"a.py:0": ["alpha", "beta"], "b.py:0": ["beta"]}
    _build(corpus).save(tmp_path)
    mapped = BM25.load(tmp_path)
    before = mapped.get_scores(["beta"]).copy()
    inode = (tmp_path / "posting_docs.npy").stat().st_ino

    updated = mapped.for_update()
    updated.add_document("c.py:0", ["beta", "gamma"])
    updated.set_doc_order([*corpus, "c.py:0"])
    updated.save(tmp_path)

    assert (tmp_path / "posting_docs.npy").stat().st_ino != inode, "the column was replaced, not rewritten"
    np.testing.assert_array_equal(
        mapped.get_scores(["beta"]), before, err_msg="the mapped index still reads its own data"
    )
    assert not list(tmp_path.glob("*.tmp*")), "no temporary file is left behind"
