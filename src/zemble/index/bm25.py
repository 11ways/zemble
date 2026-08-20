from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt
import orjson

from zemble.index.columnar import StringTable, atomic_bytes, atomic_save

_K1 = 1.5  # Term-frequency saturation
_B = 0.75  # Document length normalization

#: Bumped when the columnar on-disk layout changes shape.
_POSTINGS_FORMAT = 1

#: A delta holding more than this share of the base's documents is folded instead of grown.
_MAX_DELTA_RATIO = 0.1
#: ... but a small index may always grow this many documents before folding is worth it.
_MIN_DELTA_DOCUMENTS = 2_000

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
    """Immutable columnar postings: the scoring base, never mutated once built.

    A base is shared by every index generation derived from it, so nothing here may be
    written to; an update is a delta beside it (see :class:`BM25`), and a fold turns
    base plus delta into a new base.
    """

    def __init__(
        self,
        n_docs: int,
        n_terms: int,
        total_doc_length: int,
        terms: StringTable,
        doc_ids: StringTable,
        posting_offsets: npt.NDArray[np.int64],
        posting_docs: npt.NDArray[np.int32],
        posting_tf: npt.NDArray[np.integer],
        doc_lengths: npt.NDArray[np.int32],
    ) -> None:
        """Hold the columns; callers build them with :meth:`load` or :meth:`BM25._columns`."""
        self.n_docs = n_docs
        self.n_terms = n_terms
        self.total_doc_length = total_doc_length
        self.terms = terms
        self.doc_ids = doc_ids
        self.posting_offsets = posting_offsets
        self.posting_docs = posting_docs
        self.posting_tf = posting_tf
        self.doc_lengths = doc_lengths
        self._ids: list[str] | None = None
        self._rows: dict[str, int] | None = None

    @classmethod
    def load(cls, path: Path) -> "_Frozen":
        """Memory-map every column of a persisted index.

        :param path: Directory the columns were written to.
        :return: The mapped base.
        :raises ValueError: If the stored format is not the one this zemble reads.
        """
        meta = orjson.loads((path / _META_NAME).read_bytes())
        if meta.get("format") != _POSTINGS_FORMAT:
            raise ValueError(f"Unsupported BM25 postings format {meta.get('format')!r}; expected {_POSTINGS_FORMAT}")
        return cls(
            n_docs=meta["n_docs"],
            n_terms=meta["n_terms"],
            total_doc_length=meta["total_doc_length"],
            terms=StringTable.load(path, _TERMS_TABLE),
            doc_ids=StringTable.load(path, _DOC_IDS_TABLE),
            posting_offsets=np.load(path / _POSTING_OFFSETS_NAME, mmap_mode="r"),
            posting_docs=np.load(path / _POSTING_DOCS_NAME, mmap_mode="r"),
            posting_tf=np.load(path / _POSTING_TF_NAME, mmap_mode="r"),
            doc_lengths=np.load(path / _DOC_LENGTHS_NAME, mmap_mode="r"),
        )

    @property
    def ids(self) -> list[str]:
        """The stored document IDs, materialized once and cached for every derived index."""
        if self._ids is None:
            self._ids = self.doc_ids.to_list()
        return self._ids

    @property
    def rows(self) -> dict[str, int]:
        """Document ID to base row, built once: the ID table is in document order, not sorted."""
        if self._rows is None:
            self._rows = {chunk_id: row for row, chunk_id in enumerate(self.ids)}
        return self._rows

    def save(self, path: Path) -> None:
        """Write the columns out, replacing each file atomically."""
        path.mkdir(parents=True, exist_ok=True)
        StringTable.save(path, _TERMS_TABLE, self.terms)
        StringTable.save(path, _DOC_IDS_TABLE, self.doc_ids)
        atomic_save(path / _POSTING_OFFSETS_NAME, np.asarray(self.posting_offsets))
        atomic_save(path / _POSTING_DOCS_NAME, np.asarray(self.posting_docs))
        atomic_save(path / _POSTING_TF_NAME, np.asarray(self.posting_tf))
        atomic_save(path / _DOC_LENGTHS_NAME, np.asarray(self.doc_lengths))
        atomic_bytes(
            path / _META_NAME,
            orjson.dumps(
                {
                    "format": _POSTINGS_FORMAT,
                    "n_docs": self.n_docs,
                    "n_terms": self.n_terms,
                    "total_doc_length": self.total_doc_length,
                }
            ),
        )


class BM25:
    """BM25 inverted index: an immutable columnar base plus the delta of its pending updates.

    An index built from nothing keeps everything in the mutable dictionaries. An index
    loaded from disk keeps its base memory-mapped and never rewrites it: an added document
    lands in the dictionaries, a removed one in a set of base rows, and scoring adds the two
    together. That is what makes an update cheap enough to do beside a live index instead of
    inside it - :meth:`for_update` hands out a new object sharing this one's base.
    """

    def __init__(self) -> None:
        """Create an empty index."""
        self._documents: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._total_doc_length = 0
        self.postings: dict[str, dict[str, int]] = {}
        self._doc_order: list[str] = []
        self._positions: dict[str, int] = {}
        self._frozen: _Frozen | None = None
        self._removed_rows: set[int] = set()
        self._removed_length = 0
        self._reordered = False
        self._views: tuple[npt.NDArray[np.bool_], npt.NDArray[np.int32]] | None = None

    @property
    def doc_order(self) -> list[str]:
        """The current global chunk-list order that get_scores' output is aligned to."""
        if self._frozen is not None and not self._doc_order:
            self._doc_order = list(self._frozen.ids)
        return self._doc_order

    @property
    def document_count(self) -> int:
        """The number of live documents, without materializing their IDs."""
        if self._frozen is None:
            return len(self._doc_order)
        return self._frozen.n_docs - len(self._removed_rows) + len(self._documents)

    @property
    def delta_documents(self) -> int:
        """How many documents the delta has added or removed since the base was built."""
        return len(self._documents) + len(self._removed_rows)

    @property
    def _pristine(self) -> bool:
        """Whether scoring can read the base alone: nothing added, removed or reordered."""
        return self._frozen is not None and not self._documents and not self._removed_rows and not self._reordered

    def for_update(self) -> "BM25":
        """Return an index to apply changes to, leaving this one untouched and servable.

        The result shares this index's immutable base and owns a copy of its delta, so a
        rebuild never writes into the object answering queries. A delta that has grown past
        :data:`_MAX_DELTA_RATIO`, and an index with no base at all, are folded first.

        :return: A new index carrying the same documents as this one.
        """
        frozen = self._frozen
        if frozen is None or self.delta_documents > max(_MIN_DELTA_DOCUMENTS, frozen.n_docs * _MAX_DELTA_RATIO):
            return self.fold()
        clone = BM25()
        clone._frozen = frozen
        clone._documents = dict(self._documents)
        clone._doc_lengths = dict(self._doc_lengths)
        clone._total_doc_length = self._total_doc_length
        clone.postings = {term: dict(docs) for term, docs in self.postings.items()}
        clone._doc_order = list(self._doc_order)
        clone._positions = dict(self._positions)
        clone._removed_rows = set(self._removed_rows)
        clone._removed_length = self._removed_length
        clone._reordered = self._reordered
        return clone

    def fold(self) -> "BM25":
        """Return an equivalent index whose documents are all in one freshly built base."""
        folded = BM25()
        folded._frozen = self._columns()
        return folded

    @staticmethod
    def _positions_of(chunk_ids: list[str]) -> dict[str, int]:
        """Return the position of every chunk ID in a document order."""
        return {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}

    def add_document(self, chunk_id: str, tokens: list[str]) -> None:
        """Index one document, rejecting duplicate IDs."""
        if chunk_id in self._documents or self._base_row(chunk_id) is not None:
            raise ValueError(f"chunk_id already indexed: {chunk_id}")
        counts = Counter(tokens)
        self._documents[chunk_id] = counts
        self._doc_lengths[chunk_id] = len(tokens)
        self._total_doc_length += len(tokens)
        for term, count in counts.items():
            self.postings.setdefault(term, {})[chunk_id] = count
        self._views = None

    def remove_document(self, chunk_id: str) -> None:
        """Remove a document's postings; no-op if chunk_id is not indexed."""
        counts = self._documents.pop(chunk_id, None)
        if counts is not None:
            self._total_doc_length -= self._doc_lengths.pop(chunk_id)
            for term in counts:
                docs = self.postings[term]
                docs.pop(chunk_id, None)
                if not docs:
                    del self.postings[term]
            self._views = None
            return
        row = self._base_row(chunk_id)
        if row is None:
            return
        self._removed_rows.add(row)
        assert self._frozen is not None
        self._removed_length += int(self._frozen.doc_lengths[row])
        self._views = None

    def _base_row(self, chunk_id: str) -> int | None:
        """Return the base row of a live document, or None when the base does not hold one."""
        if self._frozen is None:
            return None
        row = self._frozen.rows.get(chunk_id)
        return None if row is None or row in self._removed_rows else row

    def set_doc_order(self, chunk_ids: list[str]) -> None:
        """Set the current global chunk-list order that get_scores' output is aligned to."""
        self._doc_order = chunk_ids
        self._positions = self._positions_of(chunk_ids)
        # Re-declaring the base's own order changes nothing, and keeps the vectorized fast path.
        self._reordered = self._frozen is not None and chunk_ids != self._frozen.ids
        self._views = None

    def _build_views(self) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int32]]:
        """Return (alive, remap): which base rows still count, and where each one now scores.

        ``remap`` is -1 for a base row that was removed or that the current document order
        does not carry, which is what keeps a re-added document from being scored twice.
        """
        frozen = self._frozen
        assert frozen is not None
        alive: npt.NDArray[np.bool_] = np.ones(frozen.n_docs, dtype=np.bool_)
        if self._removed_rows:
            alive[np.fromiter(self._removed_rows, dtype=np.int64, count=len(self._removed_rows))] = False
        remap: npt.NDArray[np.int32] = np.full(frozen.n_docs, -1, dtype=np.int32)
        positions = self._positions
        for row, chunk_id in enumerate(frozen.ids):
            if alive[row]:
                position = positions.get(chunk_id)
                if position is not None:
                    remap[row] = position
        return alive, remap

    def _current_views(self) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int32]]:
        """Return the cached alive/remap views, building them if a mutation invalidated them."""
        if self._views is None:
            self._views = self._build_views()
        return self._views

    def get_scores(
        self, tokens: list[str], weight_mask: npt.NDArray[np.bool_] | None = None
    ) -> npt.NDArray[np.float32]:
        """Calculate BM25 scores for a tokenized query.

        :param tokens: Tokenized search query.
        :param weight_mask: Optional boolean mask aligned with doc_order.
        :return: Scores aligned with doc_order.
        """
        frozen = self._frozen
        if frozen is None:
            scores = self._mutable_scores(tokens)
        elif self._pristine:
            scores = self._frozen_scores(frozen, tokens)
        else:
            scores = self._delta_scores(frozen, tokens)

        if weight_mask is not None:
            scores = scores * weight_mask
        return scores

    def _frozen_scores(self, frozen: _Frozen, tokens: list[str]) -> npt.NDArray[np.float32]:
        """Score against an unchanged base; identical arithmetic to :meth:`_mutable_scores`."""
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

    def _delta_scores(self, frozen: _Frozen, tokens: list[str]) -> npt.NDArray[np.float32]:
        """Score the base and the delta together, as one corpus.

        Corpus size, average length and every term's document frequency count the live
        documents - base minus removed, plus added - so a query cannot tell whether a
        document arrived in the base or in the delta.
        """
        scores: npt.NDArray[np.float32] = np.zeros(len(self.doc_order), dtype=np.float32)
        corpus_size = self.document_count
        if not tokens or corpus_size == 0:
            return scores

        alive, remap = self._current_views()
        total_doc_length = frozen.total_doc_length - self._removed_length + self._total_doc_length
        avgdl = total_doc_length / corpus_size
        for term, query_tf in Counter(tokens).items():
            added = self.postings.get(term)
            doc_indices: npt.NDArray[np.int32] | None = None
            tf: npt.NDArray[np.float64] | None = None
            df = 0
            term_row = frozen.terms.index_of(term)
            if term_row is not None:
                start, end = frozen.posting_offsets[term_row], frozen.posting_offsets[term_row + 1]
                doc_indices = np.asarray(frozen.posting_docs[start:end])
                tf = np.asarray(frozen.posting_tf[start:end], dtype=np.float64)
                df = int(np.count_nonzero(alive[doc_indices]))
            df += len(added) if added else 0
            if df == 0:
                continue
            idf = math.log(1 + (corpus_size - df + 0.5) / (df + 0.5))

            if doc_indices is not None and tf is not None:
                positions = remap[doc_indices]
                keep = positions >= 0
                if keep.any():
                    kept_tf = tf[keep]
                    dl = np.asarray(frozen.doc_lengths[doc_indices[keep]], dtype=np.float64)
                    tfc = kept_tf / (_K1 * (1 - _B + _B * dl / avgdl) + kept_tf)
                    contribution = np.float32((query_tf * idf) * tfc)
                    target = positions[keep]
                    scores[target] = scores[target] + contribution
            if added:
                for chunk_id, count in added.items():
                    index = self._positions.get(chunk_id)
                    if index is None:
                        continue
                    length = self._doc_lengths[chunk_id]
                    tfc_value = count / (_K1 * (1 - _B + _B * length / avgdl) + count)
                    scores[index] += query_tf * idf * tfc_value
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

        An index carrying a delta is folded on the way out, so a cold load never pays for
        the updates that happened while it was warm; folding is what refuses a document
        set its document order does not describe.

        :param path: Directory to write the columns into.
        """
        path.mkdir(parents=True, exist_ok=True)
        (self._frozen if self._pristine else self._columns()).save(path)  # type: ignore[union-attr]

    def _columns(self) -> _Frozen:
        """Build one base holding every live document, in the current document order.

        :return: The folded base.
        :raises ValueError: If the document set and the document order disagree.
        """
        doc_order = self.doc_order
        if len(doc_order) != len(set(doc_order)) or self.document_count != len(doc_order):
            raise ValueError("BM25 document state is inconsistent with its document order")
        if self._frozen is None:
            if set(self._documents) != set(doc_order):
                raise ValueError("BM25 document state is inconsistent with its document order")
            return self._columns_from_documents(doc_order)
        return self._columns_from_delta(self._frozen, doc_order)

    def _columns_from_documents(self, doc_order: list[str]) -> _Frozen:
        """Build a base from the mutable dictionaries alone."""
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
        doc_lengths = np.fromiter(
            (self._doc_lengths[chunk_id] for chunk_id in doc_order), dtype=np.int32, count=len(doc_order)
        )
        return self._assemble(doc_order, term_list, posting_counts, doc_indices, term_frequencies, doc_lengths)

    def _columns_from_delta(self, frozen: _Frozen, doc_order: list[str]) -> _Frozen:
        """Fold a base and its delta into one base, without ever walking a document's terms.

        The base's postings are remapped in one vectorized pass: a removed document's rows
        drop out, everything else moves to its new position, and only the delta's own
        postings are visited one at a time.
        """
        _alive, remap = self._current_views()
        offsets = np.asarray(frozen.posting_offsets)
        base_docs = np.asarray(frozen.posting_docs)
        moved = remap[base_docs] if len(base_docs) else np.empty(0, dtype=np.int32)
        kept = moved >= 0
        counts = (
            np.add.reduceat(kept.astype(np.int64), offsets[:-1])
            if frozen.n_terms and len(base_docs)
            else np.zeros(frozen.n_terms, dtype=np.int64)
        )

        base_terms = frozen.terms.to_list()
        term_list = sorted(set(base_terms) | {term for term, docs in self.postings.items() if docs})
        row_of_term = {term: row for row, term in enumerate(term_list)}
        base_term_rows = np.fromiter((row_of_term[term] for term in base_terms), dtype=np.int64, count=len(base_terms))

        added_rows: list[int] = []
        added_docs: list[int] = []
        added_tf: list[int] = []
        positions = self._positions
        for term, docs in self.postings.items():
            row = row_of_term[term]
            for chunk_id, count in docs.items():
                position = positions.get(chunk_id)
                if position is None:
                    continue
                added_rows.append(row)
                added_docs.append(position)
                added_tf.append(count)

        term_rows = np.concatenate([np.repeat(base_term_rows, counts), np.asarray(added_rows, dtype=np.int64)])
        doc_indices = np.concatenate([moved[kept], np.asarray(added_docs, dtype=np.int32)]).astype(np.int32)
        term_frequencies = np.concatenate(
            [np.asarray(frozen.posting_tf)[kept], np.asarray(added_tf, dtype=np.int64)]
        ).astype(np.int64)

        order = np.argsort(term_rows, kind="stable")
        posting_counts = np.bincount(term_rows, minlength=len(term_list))
        doc_lengths = np.fromiter(
            (self._length_of(frozen, chunk_id) for chunk_id in doc_order), dtype=np.int32, count=len(doc_order)
        )
        # A term whose every posting was removed is dropped: the saved vocabulary holds only
        # terms some live document still carries, which is what a rebuilt-from-scratch index has.
        surviving = posting_counts > 0
        if not bool(surviving.all()):
            term_list = [term for term, count in zip(term_list, posting_counts) if count]
            posting_counts = posting_counts[surviving]
        return self._assemble(
            doc_order, term_list, posting_counts, doc_indices[order], term_frequencies[order], doc_lengths
        )

    def _length_of(self, frozen: _Frozen, chunk_id: str) -> int:
        """Return a live document's token count, from the delta or from the base.

        :param frozen: The base to read a base document's length from.
        :param chunk_id: The document to measure.
        :return: The document's token count.
        :raises ValueError: If the document order names something this index does not hold.
        """
        length = self._doc_lengths.get(chunk_id)
        if length is not None:
            return length
        row = frozen.rows.get(chunk_id)
        if row is None or row in self._removed_rows:
            raise ValueError("BM25 document state is inconsistent with its document order")
        return int(frozen.doc_lengths[row])

    def _assemble(
        self,
        doc_order: list[str],
        term_list: list[str],
        posting_counts: npt.NDArray[np.int64],
        doc_indices: npt.NDArray[np.int32],
        term_frequencies: npt.NDArray[np.int64],
        doc_lengths: npt.NDArray[np.int32],
    ) -> _Frozen:
        """Turn per-term posting counts and flat posting columns into a base."""
        posting_offsets = np.zeros(len(term_list) + 1, dtype=np.int64)
        np.cumsum(posting_counts, out=posting_offsets[1:])
        total = len(term_frequencies)
        tf_dtype = np.uint16 if total and int(term_frequencies.max()) <= np.iinfo(np.uint16).max else np.int32
        return _Frozen(
            n_docs=len(doc_lengths),
            n_terms=len(term_list),
            total_doc_length=int(doc_lengths.sum()),
            terms=StringTable.of(term_list),
            doc_ids=StringTable.of(doc_order),
            posting_offsets=posting_offsets,
            posting_docs=doc_indices,
            posting_tf=term_frequencies.astype(tf_dtype),
            doc_lengths=doc_lengths,
        )

    @classmethod
    def load(cls, path: Path) -> "BM25":
        """Load an index from its columnar files, memory-mapping the postings.

        :param path: Directory the index was saved to.
        :return: An index whose scoring reads the mapped columns directly.
        :raises ValueError: If the persisted columns disagree about their shapes.
        """
        index = cls()
        frozen = _Frozen.load(path)
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
