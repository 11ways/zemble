"""Phase-level timing for a cold and a warm zemble query against one workspace.

Cold = fresh process: imports, cache validation, index load, then the query.
Warm = the same process querying an already-loaded index a second time.
"""

import argparse
import json
import sys
import time
from pathlib import Path

_T0 = time.perf_counter()

import numpy as np  # noqa: E402

from zemble.cache import get_validated_cache  # noqa: E402
from zemble.index.index import ZembleIndex  # noqa: E402
from zemble.ranking import apply_query_boost, boost_multi_chunk_files, rerank_topk, resolve_alpha  # noqa: E402
from zemble.search import _rrf_scores, _search_bm25  # noqa: E402
from zemble.types import ContentType, SearchResult  # noqa: E402
from zemble.utils import format_results  # noqa: E402

_IMPORT_MS = (time.perf_counter() - _T0) * 1000


class Timer:
    """Collect named phase durations in milliseconds."""

    def __init__(self) -> None:
        """Start with no recorded phases."""
        self.phases: dict[str, float] = {}

    def time(self, name: str, fn):  # noqa: ANN001, ANN201
        """Run fn, record its duration under name, and return its result."""
        start = time.perf_counter()
        result = fn()
        self.phases[name] = (time.perf_counter() - start) * 1000
        return result


def _load_phases(path: str, timer: Timer) -> ZembleIndex:
    """Load a cached index, timing each persisted component separately."""
    import orjson

    from zemble.embedding.registry import load_embedder
    from zemble.index.bm25 import BM25
    from zemble.index.chunk_store import load_chunks
    from zemble.index.dense import SelectableBasicBackend
    from zemble.index.symbols import SymbolDefinitions
    from zemble.index.types import FileManifestEntry, PersistencePath

    embedder_id = timer.time("load_model", lambda: load_embedder().model_id)
    cache_path = timer.time("cache_validate", lambda: get_validated_cache(path, embedder_id, (ContentType.CODE,)))
    if cache_path is None:
        raise SystemExit(f"No valid cached index for {path}; build one first.")
    pp = PersistencePath.from_path(cache_path)

    metadata = timer.time("load_metadata", lambda: orjson.loads(pp.metadata.read_bytes()))
    bm25_index = timer.time("load_bm25", lambda: BM25.load(pp.bm25_index))
    semantic_index = timer.time("load_dense", lambda: SelectableBasicBackend.load(pp.semantic_index))
    chunks = timer.time("load_chunks", lambda: load_chunks(pp.chunks))
    definitions = timer.time("load_symbols", lambda: SymbolDefinitions.load(pp.symbols))
    embedder = load_embedder(metadata["embedder"])
    manifest = {p: FileManifestEntry(**e) for p, e in metadata.get("files", {}).items()}
    index = timer.time(
        "build_index_object",
        lambda: ZembleIndex(
            embedder,
            bm25_index,
            semantic_index,
            chunks,
            root=Path(metadata["root_path"]) if metadata["root_path"] else None,
            content=tuple(ContentType(s) for s in metadata.get("content_type", ["code"])),
            loaded_from_disk=True,
            manifest=manifest,
            definitions=definitions,
        ),
    )
    return index


def _query_phases(index: ZembleIndex, query: str, top_k: int, timer: Timer) -> list[SearchResult]:
    """Run the search pipeline stage by stage, mirroring zemble.search.search."""
    alpha = timer.time("resolve_alpha", lambda: resolve_alpha(query, None))
    candidate_count = top_k * 5
    chunks = index.chunks

    embedding = timer.time("embed_query", lambda: index.embedder.embed_queries([query]))
    dense_hits = timer.time(
        "dense_search",
        lambda: index._semantic_index.query(embedding, k=candidate_count, selector=None)[0],
    )
    semantic = [SearchResult(chunk=chunks[i], score=1.0 - float(d)) for i, d in zip(dense_hits[0], dense_hits[1])]
    bm25 = timer.time("bm25_search", lambda: _search_bm25(query, index._bm25_index, chunks, candidate_count, None))

    def _fuse() -> list[tuple]:
        semantic_scores = {r.chunk: r.score for r in semantic}
        bm25_scores = {r.chunk: r.score for r in bm25 if r.score}
        ns = _rrf_scores(semantic_scores)
        nb = _rrf_scores(bm25_scores)
        candidates = sorted({*ns, *nb}, key=lambda c: c.start_line)
        combined = {c: alpha * ns.get(c, 0.0) + (1.0 - alpha) * nb.get(c, 0.0) for c in candidates}
        combined = {c: s for c, s in combined.items() if s}
        boost_multi_chunk_files(combined)
        combined = apply_query_boost(combined, query, chunks, index._definitions)
        return rerank_topk(combined, top_k, penalise_paths=alpha < 1.0)

    ranked = timer.time("fuse_rank", _fuse)
    results = [SearchResult(chunk=c, score=s) for c, s in ranked]
    timer.time("format_output", lambda: json.dumps(format_results(query, results, 10)))
    return results


def main() -> None:
    """Profile one cold and several warm queries and print a JSON report."""
    parser = argparse.ArgumentParser(description="Profile zemble query phases.")
    parser.add_argument("path")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--warm-runs", type=int, default=5)
    args = parser.parse_args()

    timer = Timer()
    timer.phases["imports"] = _IMPORT_MS
    index = _load_phases(args.path, timer)
    cold_query = Timer()
    results = _query_phases(index, args.query, args.top_k, cold_query)
    timer.phases.update({f"cold_{k}": v for k, v in cold_query.phases.items()})

    warm: dict[str, list[float]] = {}
    for _ in range(args.warm_runs):
        run = Timer()
        _query_phases(index, args.query, args.top_k, run)
        for k, v in run.phases.items():
            warm.setdefault(k, []).append(v)

    report = {
        "path": args.path,
        "query": args.query,
        "top_hit": results[0].chunk.location if results else None,
        "cold": timer.phases,
        "warm_median": {k: float(np.median(v)) for k, v in warm.items()},
    }
    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
