"""BM25 over memory-mapped columnar postings, and the writer that builds them in bounded memory."""

from __future__ import annotations

import math
from array import array
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import orjson

from zemble.index.columnar import BlobWriter, StringTable

_K1 = 1.5  # Term-frequency saturation
_B = 0.75  # Document length normalization

#: Bumped when the columnar on-disk layout changes shape.
_POSTINGS_FORMAT = 1

#: How many postings one merge step reads at a time; the writer never holds a whole posting column.
_BLOCK_POSTINGS = 1 << 20

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
    """Immutable columnar postings, memory-mapped; nothing ever writes into them."""

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
        """Hold the mapped columns."""
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

    @classmethod
    def load(cls, path: Path) -> _Frozen:
        """Memory-map every column of a persisted index.

        :param path: Directory the columns were written to.
        :return: The mapped columns.
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
        """The stored document IDs, materialized once."""
        if self._ids is None:
            self._ids = self.doc_ids.to_list()
        return self._ids


class BM25:
    """A persisted BM25 index, scored straight from its memory-mapped columns.

    Built only by :class:`BM25Writer`: an index is never updated in place, a rebuild writes the
    next generation beside the one that is serving.
    """

    def __init__(self, frozen: _Frozen) -> None:
        """Wrap loaded columns; use :meth:`load`."""
        self._frozen = frozen

    @property
    def doc_order(self) -> list[str]:
        """The document IDs, in the chunk order scores are aligned to."""
        return self._frozen.ids

    @property
    def document_count(self) -> int:
        """The number of documents, without materializing their IDs."""
        return self._frozen.n_docs

    def get_scores(
        self, tokens: list[str], weight_mask: npt.NDArray[np.bool_] | None = None
    ) -> npt.NDArray[np.float32]:
        """Calculate BM25 scores for a tokenized query.

        :param tokens: Tokenized search query.
        :param weight_mask: Optional boolean mask aligned with doc_order.
        :return: Scores aligned with doc_order.
        """
        frozen = self._frozen
        scores: npt.NDArray[np.float32] = np.zeros(frozen.n_docs, dtype=np.float32)
        corpus_size = frozen.n_docs
        if tokens and corpus_size:
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
                scores[doc_indices] = scores[doc_indices] + contribution
        if weight_mask is not None:
            scores = scores * weight_mask
        return scores

    @classmethod
    def load(cls, path: Path) -> BM25:
        """Load an index from its columnar files, memory-mapping the postings.

        :param path: Directory the index was written to.
        :return: An index whose scoring reads the mapped columns directly.
        :raises ValueError: If the persisted columns disagree about their shapes.
        """
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
        return cls(frozen)

    @staticmethod
    def persisted_files(path: Path) -> list[Path]:
        """Return every file a persisted index is made of."""
        return [path / name for name in _FILE_NAMES]


class BM25Writer:
    """Write a BM25 index document by document, in chunk order, in memory bounded by what is new.

    A new document's postings become three flat integers each, never a dict per document; a
    previous index's documents are carried over by row, and :meth:`finish` merges the two term
    by term into the new columns in blocks of :data:`_BLOCK_POSTINGS`.
    """

    def __init__(self, directory: Path, previous: BM25 | None = None) -> None:
        """Start writing into *directory*, reusing documents from *previous* when it is given."""
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory
        self._previous = previous._frozen if previous is not None else None
        self._ids = BlobWriter(directory, _DOC_IDS_TABLE)
        self._lengths = array("i")
        self._remap: npt.NDArray[np.int32] | None = (
            np.full(self._previous.n_docs, -1, dtype=np.int32) if self._previous is not None else None
        )
        self._vocabulary: dict[str, int] = {}
        self._terms = array("i")
        self._docs = array("i")
        self._frequencies = array("i")
        self._count = 0

    def add(self, chunk_id: str, tokens: list[str]) -> None:
        """Index one new document as the next one."""
        position = self._count
        self._ids.append(chunk_id.encode("utf-8"))
        self._lengths.append(len(tokens))
        vocabulary = self._vocabulary
        for term, frequency in Counter(tokens).items():
            self._terms.append(vocabulary.setdefault(term, len(vocabulary)))
            self._docs.append(position)
            self._frequencies.append(frequency)
        self._count += 1

    def reuse(self, start: int, count: int) -> None:
        """Carry the previous index's documents *start* to *start* + *count* over as the next ones."""
        previous, remap = self._previous, self._remap
        if previous is None or remap is None:
            raise ValueError("this writer has no previous index to reuse documents from")
        end = start + count
        self._ids.append_range(previous.doc_ids._blob, previous.doc_ids._offsets, start, end)  # noqa: SLF001
        self._lengths.frombytes(np.asarray(previous.doc_lengths[start:end], dtype=np.int32).tobytes())
        remap[start:end] = np.arange(self._count, self._count + count, dtype=np.int32)
        self._count += count

    def finish(self) -> None:
        """Merge the carried-over and the new postings into the columns, and write them out."""
        self._ids.finish()
        directory = self._directory
        doc_lengths = np.frombuffer(self._lengths, dtype=np.int32) if self._lengths else np.empty(0, np.int32)
        np.save(directory / _DOC_LENGTHS_NAME, doc_lengths)

        new_terms = list(self._vocabulary)
        new_term_ids = np.frombuffer(self._terms, dtype=np.int32) if self._terms else np.empty(0, np.int32)
        new_counts = np.bincount(new_term_ids, minlength=len(new_terms)).astype(np.int64)
        previous = self._previous
        previous_terms = previous.terms.to_list() if previous is not None else []
        kept = self._kept_counts() if previous is not None else np.empty(0, dtype=np.int64)

        # A term every posting of which was dropped leaves the vocabulary, exactly as a build from
        # scratch would never have had it.
        vocabulary = sorted({term for term, count in zip(previous_terms, kept) if count} | set(new_terms))
        row_of = {term: row for row, term in enumerate(vocabulary)}
        previous_rows = np.fromiter(
            (row_of.get(term, -1) for term in previous_terms), dtype=np.int64, count=len(previous_terms)
        )
        new_rows = np.fromiter((row_of[term] for term in new_terms), dtype=np.int64, count=len(new_terms))
        base_counts = np.zeros(len(vocabulary), dtype=np.int64)
        carried = previous_rows >= 0
        base_counts[previous_rows[carried]] = kept[carried]
        counts = base_counts.copy()
        counts[new_rows] += new_counts
        offsets = np.zeros(len(vocabulary) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        total = int(offsets[-1])

        new_frequencies = (
            np.frombuffer(self._frequencies, dtype=np.int32) if self._frequencies else np.empty(0, np.int32)
        )
        narrow = (previous is None or previous.posting_tf.dtype == np.uint16) and (
            not len(new_frequencies) or int(new_frequencies.max()) <= np.iinfo(np.uint16).max
        )
        tf_dtype = np.uint16 if narrow else np.int32
        if total:
            docs_out = np.lib.format.open_memmap(directory / _POSTING_DOCS_NAME, "w+", np.int32, (total,))
            tf_out = np.lib.format.open_memmap(directory / _POSTING_TF_NAME, "w+", tf_dtype, (total,))
            if previous is not None:
                self._write_carried(previous_rows, offsets, docs_out, tf_out)
            self._write_new(new_term_ids, new_rows, offsets[:-1] + base_counts, docs_out, tf_out)
            docs_out.flush()
            tf_out.flush()
            del docs_out, tf_out
        else:
            np.save(directory / _POSTING_DOCS_NAME, np.empty(0, dtype=np.int32))
            np.save(directory / _POSTING_TF_NAME, np.empty(0, dtype=tf_dtype))
        np.save(directory / _POSTING_OFFSETS_NAME, offsets)
        StringTable.save(directory, _TERMS_TABLE, vocabulary)
        (directory / _META_NAME).write_bytes(
            orjson.dumps(
                {
                    "format": _POSTINGS_FORMAT,
                    "n_docs": self._count,
                    "n_terms": len(vocabulary),
                    "total_doc_length": int(doc_lengths.sum(dtype=np.int64)),
                }
            )
        )

    def _blocks(self) -> Iterator[tuple[int, int]]:
        """Yield previous term ranges whose postings fit one merge step (a huge term gets its own)."""
        previous = self._previous
        assert previous is not None
        offsets = previous.posting_offsets
        start = 0
        while start < previous.n_terms:
            limit = int(offsets[start]) + _BLOCK_POSTINGS
            end = min(previous.n_terms, max(start + 1, int(np.searchsorted(offsets, limit, side="right")) - 1))
            yield start, end
            start = end

    def _kept_counts(self) -> npt.NDArray[np.int64]:
        """Return how many of each previous term's postings belong to a carried-over document."""
        previous, remap = self._previous, self._remap
        assert previous is not None and remap is not None
        offsets = np.asarray(previous.posting_offsets)
        kept = np.zeros(previous.n_terms, dtype=np.int64)
        for start, end in self._blocks():
            first = int(offsets[start])
            alive = remap[np.asarray(previous.posting_docs[first : offsets[end]])] >= 0
            running = np.concatenate(([0], np.cumsum(alive, dtype=np.int64)))
            local = offsets[start : end + 1] - first
            kept[start:end] = running[local[1:]] - running[local[:-1]]
        return kept

    def _write_carried(
        self,
        previous_rows: npt.NDArray[np.int64],
        offsets: npt.NDArray[np.int64],
        docs_out: npt.NDArray[np.int32],
        tf_out: npt.NDArray[np.integer],
    ) -> None:
        """Write every carried-over posting first in its term's range, in its previous order."""
        previous, remap = self._previous, self._remap
        assert previous is not None and remap is not None
        previous_offsets = np.asarray(previous.posting_offsets)
        for start, end in self._blocks():
            first, last = int(previous_offsets[start]), int(previous_offsets[end])
            moved = remap[np.asarray(previous.posting_docs[first:last])]
            keep = moved >= 0
            if not keep.any():
                continue
            lengths = np.diff(previous_offsets[start : end + 1])
            term_of = np.repeat(np.arange(start, end), lengths)
            running = np.cumsum(keep, dtype=np.int64)
            before_term = np.repeat(np.concatenate(([0], running))[previous_offsets[start:end] - first], lengths)
            rank = running - keep - before_term
            destination = offsets[previous_rows[term_of]] + rank
            docs_out[destination[keep]] = moved[keep]
            tf_out[destination[keep]] = np.asarray(previous.posting_tf[first:last])[keep]

    def _write_new(
        self,
        term_ids: npt.NDArray[np.int32],
        new_rows: npt.NDArray[np.int64],
        cursor: npt.NDArray[np.int64],
        docs_out: npt.NDArray[np.int32],
        tf_out: npt.NDArray[np.integer],
    ) -> None:
        """Write the new postings after the carried ones of their term, in document order."""
        if not len(term_ids):
            return
        cursor = cursor.copy()
        docs = np.frombuffer(self._docs, dtype=np.int32)
        frequencies = np.frombuffer(self._frequencies, dtype=np.int32)
        for first in range(0, len(term_ids), _BLOCK_POSTINGS):
            last = min(len(term_ids), first + _BLOCK_POSTINGS)
            rows = new_rows[term_ids[first:last]]
            order = np.argsort(rows, kind="stable")
            sorted_rows = rows[order]
            rank = np.arange(len(sorted_rows)) - np.searchsorted(sorted_rows, sorted_rows, side="left")
            destination = cursor[sorted_rows] + rank
            docs_out[destination] = docs[first:last][order]
            tf_out[destination] = frequencies[first:last][order]
            cursor += np.bincount(rows, minlength=len(cursor))
