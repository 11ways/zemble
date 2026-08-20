from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt
import orjson

from zemble.index.columnar import StringTable

_K1 = 1.5  # Term-frequency saturation
_B = 0.75  # Document length normalization

#: Bumped when the columnar on-disk layout changes shape.
_POSTINGS_FORMAT = 1

_META_NAME = "postings.json"
_TERMS_TABLE = "terms"
_DOC_IDS_TABLE = "doc_ids"
_POSTING_OFFSETS_NAME = "posting_offsets.npy"
_POSTING_DOCS_NAME = "posting_docs.npy"
_POSTING_TF_NAME = "posting_tf.npy"
_DOC_LENGTHS_NAME = "doc_lengths.npy"

_FILE_NAMES = (
    _META_NAME,
    _POSTING_OFFSETS_NAME,
    _POSTING_DOCS_NAME,
    _POSTING_TF_NAME,
    _DOC_LENGTHS_NAME,
    *StringTable.file_names(_TERMS_TABLE),
    *StringTable.file_names(_DOC_IDS_TABLE),
)


class _Frozen:
    """Columnar postings loaded from disk: the scoring source of truth until a mutation thaws them."""

    def __init__(self, path: Path) -> None:
        """Memory-map every column of a persisted index."""
        meta = orjson.loads((path / _META_NAME).read_bytes())
        if meta.get("format") != _POSTINGS_FORMAT:
            raise ValueError(f"Unsupported BM25 postings format {meta.get('format')!r}; expected {_POSTINGS_FORMAT}")
        self.n_docs: int = meta["n_docs"]
        self.n_terms: int = meta["n_terms"]
        self.total_doc_length: int = meta["total_doc_length"]
        self.terms = StringTable.load(path, _TERMS_TABLE)
        self.doc_ids = StringTable.load(path, _DOC_IDS_TABLE)
        self.posting_offsets = np.load(path / _POSTING_OFFSETS_NAME, mmap_mode="r")
        self.posting_docs = np.load(path / _POSTING_DOCS_NAME, mmap_mode="r")
        self.posting_tf = np.load(path / _POSTING_TF_NAME, mmap_mode="r")
        self.doc_lengths = np.load(path / _DOC_LENGTHS_NAME, mmap_mode="r")


class BM25:
    """BM25 inverted index supporting incremental document updates."""

    def __init__(self) -> None:
        """Create an empty index."""
        self._documents: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._total_doc_length = 0
        self.postings: dict[str, dict[str, int]] = {}
        self._doc_order: list[str] = []
        self._positions: dict[str, int] = {}
        self._frozen: _Frozen | None = None

    @property
    def doc_order(self) -> list[str]:
        """The current global chunk-list order that get_scores' output is aligned to."""
        if self._frozen is not None and not self._doc_order:
            self._doc_order = self._frozen.doc_ids.to_list()
        return self._doc_order

    @property
    def document_count(self) -> int:
        """The number of documents in the current document order, without materializing their IDs."""
        return self._frozen.n_docs if self._frozen is not None else len(self._doc_order)

    def _thaw(self) -> None:
        """Rebuild the mutable dictionaries from the frozen columns, dropping them as scoring source."""
        frozen = self._frozen
        if frozen is None:
            return
        doc_order = self.doc_order
        self._frozen = None

        posting_offsets = np.asarray(frozen.posting_offsets)
        posting_docs = np.asarray(frozen.posting_docs)
        posting_tf = np.asarray(frozen.posting_tf)

        documents: dict[str, Counter[str]] = {chunk_id: Counter() for chunk_id in doc_order}
        postings: dict[str, dict[str, int]] = {}
        for term_row, term in enumerate(frozen.terms.to_list()):
            start, end = posting_offsets[term_row], posting_offsets[term_row + 1]
            docs = posting_docs[start:end]
            counts = posting_tf[start:end]
            term_postings: dict[str, int] = {}
            for doc_index, count in zip(docs.tolist(), counts.tolist()):
                chunk_id = doc_order[doc_index]
                term_postings[chunk_id] = count
                documents[chunk_id][term] = count
            postings[term] = term_postings

        self._documents = documents
        self.postings = postings
        self._doc_lengths = dict(zip(doc_order, np.asarray(frozen.doc_lengths).tolist()))
        self._total_doc_length = frozen.total_doc_length
        self._positions = {chunk_id: i for i, chunk_id in enumerate(doc_order)}

    def add_document(self, chunk_id: str, tokens: list[str]) -> None:
        """Index one document, rejecting duplicate IDs."""
        self._thaw()
        if chunk_id in self._documents:
            raise ValueError(f"chunk_id already indexed: {chunk_id}")
        counts = Counter(tokens)
        self._documents[chunk_id] = counts
        self._doc_lengths[chunk_id] = len(tokens)
        self._total_doc_length += len(tokens)
        for term, count in counts.items():
            self.postings.setdefault(term, {})[chunk_id] = count

    def remove_document(self, chunk_id: str) -> None:
        """Remove a document's postings; no-op if chunk_id is not indexed."""
        self._thaw()
        counts = self._documents.pop(chunk_id, None)
        if counts is None:
            return
        self._total_doc_length -= self._doc_lengths.pop(chunk_id)
        for term in counts:
            docs = self.postings[term]
            docs.pop(chunk_id, None)
            if not docs:
                del self.postings[term]

    def set_doc_order(self, chunk_ids: list[str]) -> None:
        """Set the current global chunk-list order that get_scores' output is aligned to."""
        self._thaw()
        self._doc_order = chunk_ids
        self._positions = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}

    def get_scores(
        self, tokens: list[str], weight_mask: npt.NDArray[np.bool_] | None = None
    ) -> npt.NDArray[np.float32]:
        """Calculate BM25 scores for a tokenized query.

        :param tokens: Tokenized search query.
        :param weight_mask: Optional boolean mask aligned with doc_order.
        :return: Scores aligned with doc_order.
        """
        frozen = self._frozen
        if frozen is not None:
            scores = self._frozen_scores(frozen, tokens)
        else:
            scores = self._mutable_scores(tokens)

        if weight_mask is not None:
            scores = scores * weight_mask
        return scores

    def _frozen_scores(self, frozen: _Frozen, tokens: list[str]) -> npt.NDArray[np.float32]:
        """Score against the columnar postings; identical arithmetic to :meth:`_mutable_scores`."""
        scores: npt.NDArray[np.float32] = np.zeros(frozen.n_docs, dtype=np.float32)
        corpus_size = frozen.n_docs
        if not tokens or corpus_size == 0:
            return scores

        avgdl = frozen.total_doc_length / corpus_size
        for term, query_tf in Counter(tokens).items():
            term_row = frozen.terms.index_of(term)
            if term_row is None:
                continue
            start, end = frozen.posting_offsets[term_row], frozen.posting_offsets[term_row + 1]
            df = int(end - start)
            if df == 0:
                continue
            idf = math.log(1 + (corpus_size - df + 0.5) / (df + 0.5))
            doc_indices = np.asarray(frozen.posting_docs[start:end])
            tf = np.asarray(frozen.posting_tf[start:end], dtype=np.float64)
            dl = np.asarray(frozen.doc_lengths[doc_indices], dtype=np.float64)
            tfc = tf / (_K1 * (1 - _B + _B * dl / avgdl) + tf)
            contribution = np.float32((query_tf * idf) * tfc)
            # A term's document indices are unique, so this fancy-index add is a true accumulate.
            # The contribution is rounded to float32 before the sum because the scalar path adds a
            # Python float to a float32 element, which NumPy also rounds first (NEP 50 weak scalars).
            scores[doc_indices] = scores[doc_indices] + contribution
        return scores

    def _mutable_scores(self, tokens: list[str]) -> npt.NDArray[np.float32]:
        """Score against the mutable dictionaries used while building an index."""
        output_size = len(self._doc_order)
        corpus_size = len(self._documents)
        scores: npt.NDArray[np.float32] = np.zeros(output_size, dtype=np.float32)
        if not tokens or corpus_size == 0:
            return scores

        avgdl = self._total_doc_length / corpus_size
        for term, query_tf in Counter(tokens).items():
            docs = self.postings.get(term)
            if not docs:
                continue
            df = len(docs)
            idf = math.log(1 + (corpus_size - df + 0.5) / (df + 0.5))
            for chunk_id, tf in docs.items():
                idx = self._positions.get(chunk_id)
                if idx is None:
                    continue
                dl = self._doc_lengths[chunk_id]
                tfc = tf / (_K1 * (1 - _B + _B * dl / avgdl) + tf)
                scores[idx] += query_tf * idf * tfc

        return scores

    def save(self, path: Path) -> None:
        """Persist the index as columnar term/posting/document arrays.

        :param path: Directory to write the columns into.
        :raises ValueError: If the document set and the document order disagree.
        """
        path.mkdir(parents=True, exist_ok=True)
        if self._frozen is not None:
            self._save_frozen(path, self._frozen)
            return

        doc_order = self._doc_order
        if len(doc_order) != len(set(doc_order)) or set(self._documents) != set(doc_order):
            raise ValueError("BM25 document state is inconsistent with its document order")

        positions = self._positions
        term_list = sorted(self.postings)
        posting_counts = np.fromiter(
            (len(self.postings[term]) for term in term_list), dtype=np.int64, count=len(term_list)
        )
        total_postings = int(posting_counts.sum())
        doc_indices = np.fromiter(
            (positions[chunk_id] for term in term_list for chunk_id in self.postings[term]),
            dtype=np.int32,
            count=total_postings,
        )
        term_frequencies = np.fromiter(
            (tf for term in term_list for tf in self.postings[term].values()),
            dtype=np.int64,
            count=total_postings,
        )
        tf_dtype = np.uint16 if total_postings and int(term_frequencies.max()) <= np.iinfo(np.uint16).max else np.int32

        posting_offsets = np.zeros(len(term_list) + 1, dtype=np.int64)
        np.cumsum(posting_counts, out=posting_offsets[1:])

        doc_lengths = np.fromiter(
            (self._doc_lengths[chunk_id] for chunk_id in doc_order), dtype=np.int32, count=len(doc_order)
        )

        StringTable.save(path, _TERMS_TABLE, term_list)
        StringTable.save(path, _DOC_IDS_TABLE, doc_order)
        np.save(path / _POSTING_OFFSETS_NAME, posting_offsets)
        np.save(path / _POSTING_DOCS_NAME, doc_indices)
        np.save(path / _POSTING_TF_NAME, term_frequencies.astype(tf_dtype))
        np.save(path / _DOC_LENGTHS_NAME, doc_lengths)
        (path / _META_NAME).write_bytes(
            orjson.dumps(
                {
                    "format": _POSTINGS_FORMAT,
                    "n_docs": len(doc_order),
                    "n_terms": len(term_list),
                    "total_doc_length": self._total_doc_length,
                }
            )
        )

    @staticmethod
    def _save_frozen(path: Path, frozen: _Frozen) -> None:
        """Copy an unmutated frozen index straight back out, column for column."""
        StringTable.save(path, _TERMS_TABLE, frozen.terms.to_list())
        StringTable.save(path, _DOC_IDS_TABLE, frozen.doc_ids.to_list())
        np.save(path / _POSTING_OFFSETS_NAME, np.asarray(frozen.posting_offsets))
        np.save(path / _POSTING_DOCS_NAME, np.asarray(frozen.posting_docs))
        np.save(path / _POSTING_TF_NAME, np.asarray(frozen.posting_tf))
        np.save(path / _DOC_LENGTHS_NAME, np.asarray(frozen.doc_lengths))
        (path / _META_NAME).write_bytes(
            orjson.dumps(
                {
                    "format": _POSTINGS_FORMAT,
                    "n_docs": frozen.n_docs,
                    "n_terms": frozen.n_terms,
                    "total_doc_length": frozen.total_doc_length,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "BM25":
        """Load an index from its columnar files, memory-mapping the postings.

        :param path: Directory the index was saved to.
        :return: An index whose scoring reads the mapped columns directly.
        :raises ValueError: If the persisted columns disagree about their shapes.
        """
        index = cls()
        frozen = _Frozen(path)
        if (
            len(frozen.terms) != frozen.n_terms
            or len(frozen.posting_offsets) != frozen.n_terms + 1
            or len(frozen.doc_ids) != frozen.n_docs
            or len(frozen.doc_lengths) != frozen.n_docs
            or len(frozen.posting_docs) != len(frozen.posting_tf)
            or (frozen.n_terms and int(frozen.posting_offsets[-1]) != len(frozen.posting_docs))
        ):
            raise ValueError("Persisted BM25 document state is inconsistent")
        index._frozen = frozen
        return index

    @staticmethod
    def persisted_files(path: Path) -> list[Path]:
        """Return every file a persisted index is made of."""
        return [path / name for name in _FILE_NAMES]
