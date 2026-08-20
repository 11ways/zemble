from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Collection, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import orjson

from zemble.cache import get_validated_cache, load_previous_for_incremental
from zemble.chunking.capsule import CapsuleOptions, embedding_text
from zemble.embedding.base import Embedder
from zemble.embedding.registry import load_embedder
from zemble.index.bm25 import BM25
from zemble.index.chunk_store import file_paths_of, languages_of, load_chunks, save_chunks
from zemble.index.create import create_index_from_path
from zemble.index.dense import SelectableBasicBackend
from zemble.index.files import read_file_text
from zemble.index.symbols import SymbolDefinitions, save_symbol_definitions
from zemble.index.types import CACHE_FORMAT_VERSION, FileManifestEntry, PersistencePath
from zemble.rerank.base import Reranker
from zemble.rerank.registry import RerankSettings, load_reranker, resolve_reranker_spec
from zemble.search import _search_semantic, search
from zemble.stats import save_search_stats
from zemble.types import CallType, Chunk, ContentType, IndexStats, SearchResult

_GIT_CLONE_TIMEOUT = int(os.environ.get("ZEMBLE_CLONE_TIMEOUT", 60))
_DEFAULT_CONTENT: tuple[ContentType, ...] = (ContentType.CODE,)
_ALL_CONTENT: tuple[ContentType, ...] = (ContentType.CODE, ContentType.DOCS, ContentType.CONFIG)
_INCLUDE_TEXT_FILES_DEPRECATION_MSG = (
    "include_text_files is deprecated and will be removed in a future version. "
    "Use content=(ContentType.CODE, ContentType.DOCS, ContentType.CONFIG) instead."
)


def _apply_include_text_files(
    content: ContentType | Sequence[ContentType], include_text_files: bool | None
) -> tuple[ContentType, ...]:
    """Apply the deprecated include_text_files override, emitting a DeprecationWarning."""
    if include_text_files is None:
        return (content,) if isinstance(content, ContentType) else tuple(content)
    warnings.warn(
        _INCLUDE_TEXT_FILES_DEPRECATION_MSG,
        DeprecationWarning,
        stacklevel=3,
    )
    return _ALL_CONTENT if include_text_files else _DEFAULT_CONTENT


def resolve_embedder(embedder: Embedder | str | None, model_path: str | None = None) -> Embedder:
    """Resolve the embedder to use from an embedder, a spec string, or the legacy model path.

    :param embedder: An embedder instance, a spec string, or None.
    :param model_path: Legacy bare Model2Vec model name; used only when `embedder` is None.
    :return: The embedder to index or search with.
    """
    if embedder is not None and not isinstance(embedder, str):
        return embedder
    if embedder is None and model_path is not None:
        return load_embedder(f"model2vec:{model_path}" if ":" not in model_path else model_path)
    return load_embedder(embedder)


class LazyFileSizes(dict):
    """File path -> character count, read on first use.

    Reading every indexed file up front costs more than a whole query at repository scale, and
    only the handful of paths a result set mentions is ever asked for. Membership answers the
    same question the eager dict did: an indexed path whose file could be read.
    """

    def __init__(self, root: Path, indexed_paths: Collection[str]) -> None:
        """Remember where to read from; read nothing yet."""
        super().__init__()
        self._root = root
        self._indexed_paths = indexed_paths

    def __missing__(self, key: str) -> int:
        """Read the file behind *key* and remember its size.

        :param key: Repo-relative file path.
        :return: The file's character count.
        :raises KeyError: If the path is not indexed or cannot be read.
        """
        if key not in self._indexed_paths:
            raise KeyError(key)
        try:
            size = len(read_file_text(self._root / key))
        except OSError:
            raise KeyError(key) from None
        super().__setitem__(key, size)
        return size

    def __contains__(self, key: object) -> bool:
        """Return whether *key* names an indexed, readable file."""
        if super().__contains__(key):
            return True
        try:
            self[key]
        except (KeyError, TypeError):
            return False
        return True


class ZembleIndex:
    """Fast local code index with hybrid search."""

    def __init__(
        self,
        embedder: Embedder,
        bm25_index: BM25,
        semantic_index: SelectableBasicBackend,
        chunks: Sequence[Chunk],
        root: Path | None = None,
        content: ContentType | Sequence[ContentType] = _DEFAULT_CONTENT,
        loaded_from_disk: bool = False,
        manifest: dict[str, FileManifestEntry] | None = None,
        capsules: CapsuleOptions | None = None,
        definitions: SymbolDefinitions | None = None,
        subtree_prefix: str | None = None,
    ) -> None:
        """Initialize a ZembleIndex. Should be created with from_path or from_git.

        :param embedder: Embedder used to build and query this index.
        :param bm25_index: The bm25 index.
        :param semantic_index: The semantic index.
        :param chunks: The found chunks.
        :param root: Root directory used to read file sizes for token-savings stats.
        :param content: Content type used when indexing; controls the search pipeline.
        :param loaded_from_disk: Whether the index was loaded from disk (cache hit); controls CLI messaging.
        :param manifest: File modification times and chunk ranges used for incremental reindexing.
        :param capsules: The context-capsule configuration this index's chunks were built with.
        :param definitions: Persisted symbol-definition lookup used by the rerank pass.
        :param subtree_prefix: Restrict every answer to chunks whose file path starts with this
            prefix; the stores stay whole, so scores and ranking are the full index's.
        """
        self.embedder = embedder
        self.chunks: Sequence[Chunk] = chunks
        self._bm25_index: BM25 = bm25_index
        self._semantic_index: SelectableBasicBackend = semantic_index
        self._root: Path | None = root
        self._content: tuple[ContentType, ...] = (content,) if isinstance(content, ContentType) else tuple(content)
        self._subtree_prefix: str | None = subtree_prefix
        self._file_mapping, self._language_mapping = self._populate_mapping()
        self._subtree_selector: npt.NDArray[np.int_] | None = (
            np.unique([index for indices in self._file_mapping.values() for index in indices])
            if subtree_prefix is not None
            else None
        )
        self._file_sizes: dict[str, int] = LazyFileSizes(root, self._file_mapping) if root else {}
        self.loaded_from_disk: bool = loaded_from_disk
        self._manifest: dict[str, FileManifestEntry] = manifest or {}
        self._capsules: CapsuleOptions = CapsuleOptions.resolve(capsules)
        self._definitions: SymbolDefinitions | None = definitions
        self._reranker_cache: tuple[str, Reranker | None] | None = None
        #: Subtree views built from this index, cached per prefix: the mapping pass is O(chunks).
        self._subtree_views: dict[str, ZembleIndex] = {}

    def _resolve_reranker(self, override: Reranker | None) -> Reranker | None:
        """Return the reranker to apply: an explicit one, else the environment's, built once per spec.

        :param override: A reranker passed to the call, or None to read the environment.
        :return: The reranker, or None when reranking is off.
        """
        if override is not None:
            return override
        spec = resolve_reranker_spec()
        if self._reranker_cache is None or self._reranker_cache[0] != spec:
            self._reranker_cache = (spec, load_reranker(spec))
        return self._reranker_cache[1]

    def _populate_mapping(self) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        """Build (file → chunk indices, language → chunk indices) mappings, in that order.

        A subtree view maps only the files under its prefix; the chunk list itself stays whole
        because every index into it is also a row of the vector matrix and a BM25 document.
        """
        language_to_id = defaultdict(list)
        file_to_id = defaultdict(list)
        prefix = self._subtree_prefix
        for i, (file_path, language) in enumerate(zip(file_paths_of(self.chunks), languages_of(self.chunks))):
            if prefix is not None and not file_path.replace("\\", "/").startswith(prefix):
                continue
            if language:
                language_to_id[language].append(i)
            file_to_id[file_path].append(i)

        return dict(file_to_id), dict(language_to_id)

    @property
    def stats(self) -> IndexStats:
        """Stats of an index."""
        language_counts = {language: len(indices) for language, indices in self._language_mapping.items()}

        return IndexStats(
            indexed_files=len(self._file_mapping),
            total_chunks=len(self.chunks) if self._subtree_selector is None else len(self._subtree_selector),
            languages=language_counts,
            embedder=self.embedder.model_id,
            dimensions=self.embedder.dimensions,
        )

    @property
    def content(self) -> tuple[ContentType, ...]:
        """Return the content types covered by this index."""
        return self._content

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        content: ContentType | Sequence[ContentType] = _DEFAULT_CONTENT,
        include_text_files: bool | None = None,
        model_path: str | None = None,
        embedder: Embedder | str | None = None,
        capsules: CapsuleOptions | None = None,
    ) -> ZembleIndex:
        """Create and index a ZembleIndex from a directory.

        :param path: Root directory to index.
        :param content: Content types to index, e.g. ContentType.CODE or [ContentType.CODE, ContentType.DOCS].
        :param include_text_files: Deprecated. Pass a content sequence directly instead.
        :param model_path: Legacy alias for ``embedder="model2vec:<name>"``.
        :param embedder: An embedder or spec string. If None, the environment default is used.
        :param capsules: Context-capsule knobs; None resolves the environment override, else the defaults.
        :return: An indexed ZembleIndex. Chunk file paths are relative to ``path``.
        :raises FileNotFoundError: If `path` does not exist.
        :raises NotADirectoryError: If `path` exists but is not a directory.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        normalized = _apply_include_text_files(content, include_text_files)
        resolved = resolve_embedder(embedder, model_path)
        resolved_capsules = CapsuleOptions.resolve(capsules)
        cache_path = get_validated_cache(str(path), resolved.model_id, normalized, resolved_capsules)
        if cache_path:
            return cls.load_from_disk(cache_path, embedder=resolved)

        path = path.resolve()
        previous = load_previous_for_incremental(str(path), resolved.model_id, normalized, resolved_capsules)
        bm25_index, semantic_index, chunks, manifest = create_index_from_path(
            path,
            embedder=resolved,
            content=normalized,
            display_root=path,
            previous=previous,
            capsules=resolved_capsules,
        )

        return ZembleIndex(
            resolved,
            bm25_index,
            semantic_index,
            chunks,
            root=path,
            content=normalized,
            manifest=manifest,
            capsules=resolved_capsules,
        )

    @classmethod
    def from_git(
        cls,
        url: str,
        ref: str | None = None,
        model_path: str | None = None,
        content: ContentType | Sequence[ContentType] = _DEFAULT_CONTENT,
        include_text_files: bool | None = None,
        embedder: Embedder | str | None = None,
    ) -> ZembleIndex:
        """Clone a git repository and index it.

        The repository is cloned into a temporary directory that is removed once
        indexing finishes. Chunk content is preserved in-memory, but
        chunk.file_path will not point to a readable file after this call
        returns — it is a repo-relative label, not a filesystem path.

        :param url: URL of the git repository to clone (any git provider).
        :param ref: Branch or tag to check out. Defaults to the remote HEAD.
        :param model_path: Legacy alias for ``embedder="model2vec:<name>"``.
        :param content: Content types to index, e.g. (ContentType.CODE,) or (ContentType.CODE, ContentType.DOCS).
        :param include_text_files: Deprecated. Pass content=(ContentType.CODE, ContentType.DOCS, ...) instead.
        :param embedder: An embedder or spec string. If None, the environment default is used.
        :return: An indexed ZembleIndex. Chunk file paths are repo-relative (e.g. ``src/foo.py``).
        :raises RuntimeError: If git is not on PATH, the clone fails, or times out.
        """
        normalized = _apply_include_text_files(content, include_text_files)
        resolved = resolve_embedder(embedder, model_path)
        resolved_capsules = CapsuleOptions.resolve()
        cache_key = f"{url}@{ref}" if ref else url
        cache_path = get_validated_cache(cache_key, resolved.model_id, normalized, resolved_capsules)
        if cache_path:
            return cls.load_from_disk(cache_path, embedder=resolved)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # `--` prevents `url` from being interpreted as a git option (e.g. `--upload-pack=...`).
            cmd = ["git", "clone", "--depth", "1", *(["--branch", ref] if ref else []), "--", url, tmp_dir]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=_GIT_CLONE_TIMEOUT
                )
            except FileNotFoundError:
                raise RuntimeError("git is not installed or not on PATH") from None
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"git clone timed out for {url!r} (limit: {_GIT_CLONE_TIMEOUT} s)") from None
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed for {url!r}:\n{result.stderr.strip()}")

            resolved_path = Path(tmp_dir).resolve()
            bm25_index, semantic_index, chunks, manifest = create_index_from_path(
                resolved_path,
                embedder=resolved,
                content=normalized,
                display_root=resolved_path,
                capsules=resolved_capsules,
            )

            return ZembleIndex(
                resolved,
                bm25_index,
                semantic_index,
                chunks,
                root=resolved_path,
                content=normalized,
                manifest=manifest,
                capsules=resolved_capsules,
            )

    def find_related(
        self, source: Chunk | SearchResult, *, top_k: int = 5, max_snippet_lines: int | None = None
    ) -> list[SearchResult]:
        """Return chunks semantically similar to the given chunk or search result.

        :param source: A SearchResult or Chunk to use as the seed.
        :param top_k: Number of similar chunks to return.
        :param max_snippet_lines: Lines of content to count for savings stats. None = full chunk.
        :return: Ranked list of SearchResult objects, most similar first.
        """
        target = source.chunk if isinstance(source, SearchResult) else source
        selector = self._get_selector_vector(filter_languages=[target.language]) if target.language else None
        # The seed is embedded exactly as the indexed chunks were, capsule included, so the
        # comparison stays inside one text convention.
        results = _search_semantic(
            embedding_text(target), self.embedder, self._semantic_index, self.chunks, top_k + 1, selector
        )
        results = [r for r in results if r.chunk != target][:top_k]
        save_search_stats(results, CallType.FIND_RELATED, self._file_sizes, max_snippet_lines)
        return results

    def _get_selector_vector(
        self, filter_languages: list[str] | None = None, filter_paths: list[str] | None = None
    ) -> npt.NDArray[np.int_] | None:
        """Create a vector of chunk indices to restrict retrieval to.

        A subtree view's prefix is the floor: an explicit filter narrows it further, it never
        widens it, so a view can only ever answer from inside its own sub-tree. A filter that
        was asked for and matched nothing selects nothing, which is what it says; it used to
        fall back to selecting everything.
        """
        selector = []
        for language in filter_languages or []:
            selector.extend(self._language_mapping.get(language, []))
        for filename in filter_paths or []:
            selector.extend(self._file_mapping.get(filename, []))

        if selector:
            chosen = np.unique(selector)
        elif filter_languages or filter_paths:
            chosen = np.empty(0, dtype=np.int_)
        else:
            chosen = None
        if self._subtree_selector is None:
            return chosen
        return self._subtree_selector if chosen is None else np.intersect1d(chosen, self._subtree_selector)

    def subtree(self, prefix: str) -> ZembleIndex | None:
        """Return a view of this index restricted to one sub-directory, or None if it holds nothing there.

        The view shares this index's chunks, vectors and postings, so a search through it
        scores and ranks exactly as the whole index does and is then filtered to the prefix;
        result paths therefore stay relative to THIS index's root, not to the sub-directory.

        :param prefix: A root-relative directory path; a trailing slash is added when missing.
        :return: The restricted view, or None when no indexed file lives under the prefix.
        """
        normalized = prefix.replace("\\", "/").strip("/")
        if not normalized:
            return self
        normalized += "/"
        cached = self._subtree_views.get(normalized)
        if cached is not None:
            return cached
        view = ZembleIndex(
            self.embedder,
            self._bm25_index,
            self._semantic_index,
            self.chunks,
            root=self._root,
            content=self._content,
            loaded_from_disk=self.loaded_from_disk,
            manifest=self._manifest,
            capsules=self._capsules,
            definitions=self._definitions,
            subtree_prefix=normalized,
        )
        if not view._file_mapping:
            return None
        self._subtree_views[normalized] = view
        return view

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float | None = None,
        filter_languages: list[str] | None = None,
        filter_paths: list[str] | None = None,
        rerank: bool | None = None,
        max_snippet_lines: int | None = None,
        reranker: Reranker | None = None,
        rerank_settings: RerankSettings | None = None,
    ) -> list[SearchResult]:
        """Search the index and return the top-k most relevant chunks.

        :param query: Natural-language or keyword query string.
        :param top_k: Maximum number of results to return.
        :param alpha: Blend weight for hybrid score combination; 1.0 = full semantic
            weight, 0.0 = full BM25 weight. None auto-detects from query type.
        :param filter_languages: Optional list of language codes; if set, only chunks in
            these languages are returned.
        :param filter_paths: Optional list of repo-relative file paths; if set, only
            chunks from these files are returned.
        :param rerank: Apply code-tuned reranking (file boost, identifier boost, path penalties).
            Defaults to True when ContentType.CODE was indexed.
        :param max_snippet_lines: Lines of content to count for savings stats. None = full chunk.
        :param reranker: Pairwise reranker for the head of the ranked list; None reads ``ZEMBLE_RERANKER``.
        :param rerank_settings: Window, blend weight and passage shape; None reads the environment.
        :return: Ranked list of SearchResult objects, best match first.
        """
        if not self.chunks or not query.strip():
            return []

        resolved_rerank = (ContentType.CODE in self._content) if rerank is None else rerank

        selector = self._get_selector_vector(filter_languages, filter_paths)
        results = search(
            query,
            self.embedder,
            self._semantic_index,
            self._bm25_index,
            self.chunks,
            top_k,
            alpha=alpha,
            selector=selector,
            rerank=resolved_rerank,
            definitions=self._definitions,
            reranker=self._resolve_reranker(reranker),
            rerank_settings=rerank_settings or RerankSettings.from_env(),
        )
        save_search_stats(results, CallType.SEARCH, self._file_sizes, max_snippet_lines)
        return results

    @classmethod
    def load_from_disk(cls: type[ZembleIndex], path: Path | str, embedder: Embedder | str | None = None) -> ZembleIndex:
        """Load the index from disk.

        :param path: Directory holding the persisted index.
        :param embedder: An already-built embedder or spec; None rebuilds it from the stored ``embedder`` id.
        :return: The loaded index.
        :raises FileNotFoundError: If the index or one of its files is missing.
        :raises ValueError: If the stored format is not the current one or the components disagree.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index not found at {path}")
        persistence_paths = PersistencePath.from_path(path)
        non_existent = persistence_paths.non_existing()
        if non_existent:
            missing = ", ".join(str(p) for p in non_existent)
            raise FileNotFoundError(f"Index not found at {path}. Missing: {missing}")

        with open(persistence_paths.metadata, "rb") as f:
            metadata = orjson.loads(f.read())
        found_version = metadata.get("cache_version")
        if found_version != CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported index format {found_version!r}; expected {CACHE_FORMAT_VERSION}. "
                "Rebuild it with ZembleIndex.from_path(<source directory>) before searching again."
            )

        bm25_index = BM25.load(persistence_paths.bm25_index)
        semantic_index = SelectableBasicBackend.load(persistence_paths.semantic_index)
        chunks = load_chunks(persistence_paths.chunks)
        definitions = SymbolDefinitions.load(persistence_paths.symbols)
        if not (len(chunks) == bm25_index.document_count == semantic_index.vectors.shape[0]):
            raise ValueError("Persisted index components have inconsistent document counts")
        root_path = metadata["root_path"]
        stored_embedder = metadata["embedder"]
        content = tuple(ContentType(s) for s in metadata.get("content_type", ["code"]))
        manifest = {
            indexed_path: FileManifestEntry(**entry) for indexed_path, entry in metadata.get("files", {}).items()
        }
        if root_path:
            root_path = Path(root_path)

        resolved = resolve_embedder(embedder if embedder is not None else stored_embedder)

        return cls(
            resolved,
            bm25_index,
            semantic_index,
            chunks,
            root=root_path,
            content=content,
            loaded_from_disk=True,
            manifest=manifest,
            capsules=CapsuleOptions.from_key(metadata.get("capsules", "")),
            definitions=definitions,
        )

    def save(self, path: Path | str) -> None:
        """Save the index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        persistence_paths = PersistencePath.from_path(path)

        self._bm25_index.save(persistence_paths.bm25_index)
        self._semantic_index.save(persistence_paths.semantic_index)
        save_chunks(persistence_paths.chunks, self.chunks)
        save_symbol_definitions(persistence_paths.symbols, self.chunks)
        from zemble.chunking.chunking import _DESIRED_CHUNK_LENGTH_CHARS  # avoid circular import at module level

        root_str = None if self._root is None else str(self._root)
        metadata = {
            "root_path": root_str,
            "time": datetime.now().timestamp(),
            "embedder": self.embedder.model_id,
            "dimensions": self.embedder.dimensions,
            "content_type": list(x.value for x in self._content),
            "chunk_size": _DESIRED_CHUNK_LENGTH_CHARS,
            "cache_version": CACHE_FORMAT_VERSION,
            "capsules": self._capsules.key,
            "files": self._manifest,
        }
        with open(persistence_paths.metadata, "wb") as f:
            data = orjson.dumps(metadata)
            f.write(data)
