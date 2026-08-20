import argparse
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from benchmarks.data import (
    RepoSpec,
    Task,
    add_filter_args,
    add_repo_source_args,
    grouped_tasks,
    load_filtered_tasks,
    results_label,
    save_results,
)
from benchmarks.metrics import HIT_CUTOFFS, hit_at_k, ndcg_at_k, target_rank
from zemble import ZembleIndex
from zemble.embedding.registry import resolve_embedder_spec
from zemble.rerank.base import Reranker
from zemble.rerank.registry import RerankSettings, load_reranker, resolve_reranker_spec
from zemble.types import SearchResult

_LATENCY_RUNS = 5
_DIRECT_TOP_K = 10

#: Hit-rate keys, in report order; the one place the cutoffs become column names.
_HIT_KEYS = tuple(f"hit{k}" for k in HIT_CUTOFFS)
_HIT_LABELS = {f"hit{k}": f"hit@{k}" for k in HIT_CUTOFFS}


@dataclass(frozen=True)
class QueryResult:
    """One query's outcome, kept so a run can be re-analysed without re-running it."""

    query: str
    kind: str | None
    category: str
    relevant: list[str]
    relevant_ranks: list[int]
    ndcg10: float
    hit1: bool
    hit5: bool
    hit10: bool


@dataclass(frozen=True)
class EvalOutcome:
    """Everything one repo's evaluation produced, aggregates and per-query detail alike."""

    ndcg5: float
    ndcg10: float
    latencies: list[float]
    by_category: dict[str, float]
    by_kind: dict[str, float]
    tokens: int
    queries: list[QueryResult]
    hits: dict[str, float]
    hits_by_category: dict[str, dict[str, float]]
    hits_by_kind: dict[str, dict[str, float]]


@dataclass(frozen=True)
class RepoResult:
    """Per-repo benchmark result."""

    repo: str
    language: str
    mode: str
    chunks: int
    tokens: int
    ndcg5: float
    ndcg10: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    index_ms: float
    by_category: dict[str, float] = field(default_factory=dict)
    by_kind: dict[str, float] = field(default_factory=dict)
    hits: dict[str, float] = field(default_factory=dict)
    hits_by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    hits_by_kind: dict[str, dict[str, float]] = field(default_factory=dict)
    queries: list[QueryResult] = field(default_factory=list)


def _hit_rates(queries: list[QueryResult]) -> dict[str, float]:
    """Return the fraction of queries with a relevant result at or above each cutoff."""
    if not queries:
        return {f"hit{k}": 0.0 for k in HIT_CUTOFFS}
    return {f"hit{k}": sum(1 for q in queries if hit_at_k(q.relevant_ranks, k)) / len(queries) for k in HIT_CUTOFFS}


def _grouped_hit_rates(groups: dict[str, list[QueryResult]]) -> dict[str, dict[str, float]]:
    """Return per-group hit rates, in group-name order."""
    return {name: _hit_rates(members) for name, members in sorted(groups.items())}


def evaluate(
    index: ZembleIndex,
    tasks: list[Task],
    *,
    verbose: bool = False,
    alpha: float | None = None,
    rerank: bool = True,
    latency_runs: int = _LATENCY_RUNS,
    reranker: Reranker | None = None,
) -> EvalOutcome:
    """Score every task, returning the aggregates plus the per-query detail behind them.

    :param latency_runs: How many times each query is re-issued for its latency median; the
        ranking - and therefore every quality metric - comes from the last run either way,
        so lowering this only widens the latency error bars (and, for a hosted reranker,
        divides the bill).
    :param reranker: An explicit pairwise reranker; None lets the index read ``ZEMBLE_RERANKER``.
    """
    ndcg5_sum = 0.0
    ndcg10_sum = 0.0
    latencies: list[float] = []
    category_ndcg10: dict[str, list[float]] = defaultdict(list)
    kind_ndcg10: dict[str, list[float]] = defaultdict(list)
    by_category_queries: dict[str, list[QueryResult]] = defaultdict(list)
    by_kind_queries: dict[str, list[QueryResult]] = defaultdict(list)
    queries: list[QueryResult] = []
    tokens = 0

    for task in tasks:
        query_latencies: list[float] = []
        results: list[SearchResult] = []
        for _ in range(latency_runs):
            started = time.perf_counter()
            results = index.search(task.query, top_k=_DIRECT_TOP_K, alpha=alpha, rerank=rerank, reranker=reranker)
            query_latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(float(np.median(query_latencies)))
        tokens += sum(len(r.chunk.content) // 4 for r in results)

        relevant_ranks = [rank for t in task.all_relevant if (rank := target_rank(results, t)) is not None]
        n_relevant = len(task.all_relevant)
        q_ndcg5 = ndcg_at_k(relevant_ranks, n_relevant, 5)
        q_ndcg10 = ndcg_at_k(relevant_ranks, n_relevant, _DIRECT_TOP_K)
        ndcg5_sum += q_ndcg5
        ndcg10_sum += q_ndcg10
        category = task.category or "unknown"
        category_ndcg10[category].append(q_ndcg10)
        if task.kind:
            kind_ndcg10[task.kind].append(q_ndcg10)

        query_result = QueryResult(
            query=task.query,
            kind=task.kind,
            category=category,
            relevant=[t.path for t in task.relevant],
            relevant_ranks=sorted(relevant_ranks),
            ndcg10=round(q_ndcg10, 4),
            hit1=hit_at_k(relevant_ranks, 1),
            hit5=hit_at_k(relevant_ranks, 5),
            hit10=hit_at_k(relevant_ranks, _DIRECT_TOP_K),
        )
        queries.append(query_result)
        by_category_queries[category].append(query_result)
        if task.kind:
            by_kind_queries[task.kind].append(query_result)

        if verbose:
            targets_str = ", ".join(
                t.path if not t.start_line else f"{t.path}:{t.start_line}-{t.end_line}" for t in task.all_relevant
            )
            top_files = [r.chunk.file_path for r in results[:5]]
            print(
                f"  [{category:<12}] ndcg@10={q_ndcg10:.3f}  ranks={relevant_ranks}"
                f"  n_rel={n_relevant}  q={task.query!r}",
                file=sys.stderr,
            )
            print(f"               targets: {targets_str}", file=sys.stderr)
            print(f"               top-5:   {top_files}", file=sys.stderr)

    total = len(tasks)
    by_category = {cat: sum(vals) / len(vals) for cat, vals in sorted(category_ndcg10.items())}
    by_kind = {kind: sum(vals) / len(vals) for kind, vals in sorted(kind_ndcg10.items())}
    return EvalOutcome(
        ndcg5=ndcg5_sum / total,
        ndcg10=ndcg10_sum / total,
        latencies=latencies,
        by_category=by_category,
        by_kind=by_kind,
        tokens=tokens // total,
        queries=queries,
        hits=_hit_rates(queries),
        hits_by_category=_grouped_hit_rates(by_category_queries),
        hits_by_kind=_grouped_hit_rates(by_kind_queries),
    )


def _print_summary(results: list[RepoResult]) -> None:
    """Print per-language and overall benchmark summary to stderr."""
    languages = sorted({result.language for result in results})
    by_language = {lang: [r for r in results if r.language == lang] for lang in languages}
    columns = ["Avg", *[lang.title() for lang in languages]]

    language_ndcg10 = [
        sum(r.ndcg10 for r in language_results) / len(language_results) for language_results in by_language.values()
    ]
    language_tokens = [
        sum(r.tokens for r in language_results) / len(language_results) for language_results in by_language.values()
    ]
    language_p50 = [
        sum(r.p50_ms for r in language_results) / len(language_results) for language_results in by_language.values()
    ]
    language_p90 = [
        sum(r.p90_ms for r in language_results) / len(language_results) for language_results in by_language.values()
    ]
    language_p95 = [
        sum(r.p95_ms for r in language_results) / len(language_results) for language_results in by_language.values()
    ]
    language_p99 = [
        sum(r.p99_ms for r in language_results) / len(language_results) for language_results in by_language.values()
    ]
    language_index = [
        sum(r.index_ms for r in language_results) / len(language_results) for language_results in by_language.values()
    ]
    language_hits = {
        key: [
            sum(r.hits.get(key, 0.0) for r in language_results) / len(language_results)
            for language_results in by_language.values()
        ]
        for key in _HIT_KEYS
    }
    avg_ndcg10 = sum(language_ndcg10) / len(language_ndcg10)
    avg_tokens = sum(language_tokens) / len(language_tokens)
    avg_p50 = sum(language_p50) / len(language_p50)
    avg_p90 = sum(language_p90) / len(language_p90)
    avg_p95 = sum(language_p95) / len(language_p95)
    avg_p99 = sum(language_p99) / len(language_p99)
    avg_index = sum(language_index) / len(language_index)

    print(file=sys.stderr)
    print("By language", file=sys.stderr)
    for language, grouped in by_language.items():
        print(
            f"  {language}: repos={len(grouped)}"
            + f"  ndcg@5={sum(r.ndcg5 for r in grouped) / len(grouped):.3f}"
            + f"  tokens={sum(r.tokens for r in grouped) / len(grouped):.0f}"
            + f"  ndcg@10={sum(r.ndcg10 for r in grouped) / len(grouped):.3f}"
            + f"  p50={sum(r.p50_ms for r in grouped) / len(grouped):.2f}ms"
            + f"  p90={sum(r.p90_ms for r in grouped) / len(grouped):.2f}ms"
            + f"  p95={sum(r.p95_ms for r in grouped) / len(grouped):.2f}ms"
            + f"  p99={sum(r.p99_ms for r in grouped) / len(grouped):.2f}ms"
            + f"  index={sum(r.index_ms for r in grouped) / len(grouped):.0f}ms",
            file=sys.stderr,
        )

    print(file=sys.stderr)
    print(f"{'=' * 104}", file=sys.stderr)
    print("Hybrid benchmark by language", file=sys.stderr)
    print(f"{'=' * 104}", file=sys.stderr)
    print(f"\n  {'Metric':<28}  " + "  ".join(f"{column:>9}" for column in columns), file=sys.stderr)
    print(f"  {'-' * 28}  " + "  ".join(f"{'-' * 9:>9}" for _ in columns), file=sys.stderr)

    ndcg_row = [f"{avg_ndcg10:>9.3f}"]
    tokens_row = [f"{avg_tokens:>9.0f}"]
    p50_row = [f"{avg_p50:>8.2f}ms"]
    p90_row = [f"{avg_p90:>8.2f}ms"]
    p95_row = [f"{avg_p95:>8.2f}ms"]
    p99_row = [f"{avg_p99:>8.2f}ms"]
    index_row = [f"{avg_index:>7.0f}ms"]
    for language, language_results in by_language.items():
        ndcg_row.append(f"{sum(r.ndcg10 for r in language_results) / len(language_results):>9.3f}")
        tokens_row.append(f"{sum(r.tokens for r in language_results) / len(language_results):>9.0f}")
        p50_row.append(f"{sum(r.p50_ms for r in language_results) / len(language_results):>8.2f}ms")
        p90_row.append(f"{sum(r.p90_ms for r in language_results) / len(language_results):>8.2f}ms")
        p95_row.append(f"{sum(r.p95_ms for r in language_results) / len(language_results):>8.2f}ms")
        p99_row.append(f"{sum(r.p99_ms for r in language_results) / len(language_results):>8.2f}ms")
        index_row.append(f"{sum(r.index_ms for r in language_results) / len(language_results):>7.0f}ms")

    print(f"  {'NDCG@10':<28}  " + "  ".join(ndcg_row), file=sys.stderr)
    for key in _HIT_KEYS:
        row = [f"{sum(language_hits[key]) / len(language_hits[key]):>9.3f}"]
        for language_results in by_language.values():
            row.append(f"{sum(r.hits.get(key, 0.0) for r in language_results) / len(language_results):>9.3f}")
        print(f"  {_HIT_LABELS[key]:<28}  " + "  ".join(row), file=sys.stderr)
    print(f"  {'tokens':<28}  " + "  ".join(tokens_row), file=sys.stderr)
    print(f"  {'q-p50':<28}  " + "  ".join(p50_row), file=sys.stderr)
    print(f"  {'q-p90':<28}  " + "  ".join(p90_row), file=sys.stderr)
    print(f"  {'q-p95':<28}  " + "  ".join(p95_row), file=sys.stderr)
    print(f"  {'q-p99':<28}  " + "  ".join(p99_row), file=sys.stderr)
    print(f"  {'index':<28}  " + "  ".join(index_row), file=sys.stderr)

    _print_group(results, "category", lambda r: r.by_category)
    _print_group(results, "kind", lambda r: r.by_kind)
    _print_hit_group(results, "category", lambda r: r.hits_by_category)
    _print_hit_group(results, "kind", lambda r: r.hits_by_kind)


def _print_group(results: list[RepoResult], label: str, pick: Callable[[RepoResult], dict[str, float]]) -> None:
    """Print the mean NDCG@10 per group value, skipping groups no repo reported."""
    names = sorted({name for r in results for name in pick(r)})
    if not names:
        return
    print(file=sys.stderr)
    print(f"By {label} (NDCG@10, mean over all repos)", file=sys.stderr)
    for name in names:
        vals = [pick(r)[name] for r in results if name in pick(r)]
        print(f"  {name:<16}  {sum(vals) / len(vals):.3f}  (n={len(vals)} repos)", file=sys.stderr)


def _record_usage(into: dict[str, dict[str, object]], role: str, client: object | None) -> None:
    """Fold a provider client's own request and token tallies into the run's usage record.

    A client that counts nothing - the local Model2Vec embedder, or no reranker at all -
    contributes no entry, so an absent role means "nothing was bought over the wire",
    never "not measured". Counters live on the provider, not on the cache wrapper, so an
    entry reports what actually left the machine.

    :param into: The run's usage record, keyed by role.
    :param role: ``embedder`` or ``reranker``.
    :param client: The embedder or reranker in use, possibly a caching wrapper, or None.
    """
    if client is None:
        return
    inner = getattr(client, "inner", client)
    requests = getattr(inner, "request_count", None)
    tokens = getattr(inner, "total_tokens", None)
    if requests is None and tokens is None:
        return
    entry = into.setdefault(role, {"model_id": getattr(inner, "model_id", "?"), "requests": 0, "tokens": 0})
    entry["requests"] = int(entry["requests"]) + int(requests or 0)  # type: ignore[arg-type]
    entry["tokens"] = int(entry["tokens"]) + int(tokens or 0)  # type: ignore[arg-type]


def _print_hit_group(
    results: list[RepoResult], label: str, pick: Callable[[RepoResult], dict[str, dict[str, float]]]
) -> None:
    """Print the mean hit@1/5/10 per group value, skipping groups no repo reported."""
    names = sorted({name for r in results for name in pick(r)})
    if not names:
        return
    print(file=sys.stderr)
    print(f"By {label} (hit rates, mean over all repos)", file=sys.stderr)
    for name in names:
        rows = [pick(r)[name] for r in results if name in pick(r)]
        cells = "  ".join(
            f"{_HIT_LABELS[key]}={sum(row.get(key, 0.0) for row in rows) / len(rows):.3f}" for key in _HIT_KEYS
        )
        print(f"  {name:<16}  {cells}  (n={len(rows)} repos)", file=sys.stderr)


def _bench_quality(
    repo_tasks: dict[str, list[Task]],
    specs: dict[str, RepoSpec],
    *,
    verbose: bool = False,
    latency_runs: int = _LATENCY_RUNS,
    reranker: Reranker | None = None,
    usage: dict[str, dict[str, object]] | None = None,
) -> list[RepoResult]:
    """Run quality benchmarks (NDCG@5, NDCG@10, hit rates, latency) for each repo.

    :param usage: Optional record that per-repo provider request and token tallies are folded into.
    """
    print(
        f"{'Repo':<12} {'Language':<12} {'Chunks':>6} {'Tokens':>8} {'index':>9} {'NDCG@5':>8} {'NDCG@10':>8} "
        f"{'hit@1':>7} {'hit@5':>7} {'hit@10':>7} {'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8}",
        file=sys.stderr,
    )
    print(
        f"{'-' * 12} {'-' * 12} {'-' * 6} {'-' * 8} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}",
        file=sys.stderr,
    )
    results: list[RepoResult] = []
    for repo, tasks in sorted(repo_tasks.items()):
        spec = specs[repo]
        started = time.perf_counter()
        index = ZembleIndex.from_path(spec.benchmark_dir)
        index_ms = (time.perf_counter() - started) * 1000
        outcome = evaluate(index, tasks, verbose=verbose, latency_runs=latency_runs, reranker=reranker)
        if usage is not None:
            _record_usage(usage, "embedder", index.embedder)
        p50, p90, p95, p99 = np.percentile(outcome.latencies, [50, 90, 95, 99]).tolist()
        result = RepoResult(
            repo=repo,
            mode="auto",
            language=spec.language,
            chunks=len(index.chunks),
            tokens=outcome.tokens,
            ndcg5=outcome.ndcg5,
            ndcg10=outcome.ndcg10,
            p50_ms=p50,
            p90_ms=p90,
            p95_ms=p95,
            p99_ms=p99,
            index_ms=index_ms,
            by_category=outcome.by_category,
            by_kind=outcome.by_kind,
            hits=outcome.hits,
            hits_by_category=outcome.hits_by_category,
            hits_by_kind=outcome.hits_by_kind,
            queries=outcome.queries,
        )
        results.append(result)
        hit_cells = " ".join(f"{outcome.hits[key]:>7.3f}" for key in _HIT_KEYS)
        print(
            f"{repo:<12} {spec.language:<12} {len(index.chunks):>6} {outcome.tokens:>8}"
            f"{index_ms:>8.0f}ms {outcome.ndcg5:>8.3f} {outcome.ndcg10:>8.3f} {hit_cells}"
            f" {p50:>7.2f}ms {p90:>7.2f}ms {p95:>7.2f}ms {p99:>7.2f}ms",
            file=sys.stderr,
        )
    return results


def _group_means(results: list[RepoResult], pick: Callable[[RepoResult], dict[str, float]]) -> dict[str, float]:
    """Return the mean NDCG@10 per group value across the repos that reported it."""
    means: dict[str, float] = {}
    for name in sorted({name for r in results for name in pick(r)}):
        vals = [pick(r)[name] for r in results if name in pick(r)]
        means[name] = round(sum(vals) / len(vals), 4)
    return means


def _mean_hits(results: list[RepoResult], pick: Callable[[RepoResult], dict[str, float]]) -> dict[str, float]:
    """Return the mean of each hit-rate key across the repos that reported it."""
    return {key: round(sum(pick(r).get(key, 0.0) for r in results) / len(results), 4) for key in _HIT_KEYS}


def _grouped_hit_means(
    results: list[RepoResult], pick: Callable[[RepoResult], dict[str, dict[str, float]]]
) -> dict[str, dict[str, float]]:
    """Return the mean hit rates per group value across the repos that reported it."""
    means: dict[str, dict[str, float]] = {}
    for name in sorted({name for r in results for name in pick(r)}):
        rows = [pick(r)[name] for r in results if name in pick(r)]
        means[name] = {key: round(sum(row.get(key, 0.0) for row in rows) / len(rows), 4) for key in _HIT_KEYS}
    return means


def _save_results(results: list[RepoResult], label: str, usage: dict[str, dict[str, object]]) -> None:
    """Write results to benchmarks/results/<label>-<sha12>.json."""
    languages = sorted({r.language for r in results})
    by_language = {lang: [r for r in results if r.language == lang] for lang in languages}

    lang_means = {
        lang: {
            "ndcg10": sum(r.ndcg10 for r in grouped) / len(grouped),
            "tokens": sum(r.tokens for r in grouped) / len(grouped),
            "p50_ms": sum(r.p50_ms for r in grouped) / len(grouped),
            "p90_ms": sum(r.p90_ms for r in grouped) / len(grouped),
            "p95_ms": sum(r.p95_ms for r in grouped) / len(grouped),
            "p99_ms": sum(r.p99_ms for r in grouped) / len(grouped),
            "index_ms": sum(r.index_ms for r in grouped) / len(grouped),
        }
        for lang, grouped in by_language.items()
    }
    cat_means = _group_means(results, lambda r: r.by_category)
    kind_means = _group_means(results, lambda r: r.by_kind)

    n_repos = len(results)
    output = {
        "tool": label,
        "model": resolve_embedder_spec(),
        "summary": {
            "ndcg10": round(sum(r.ndcg10 for r in results) / n_repos, 4),
            "tokens": round(sum(r.tokens for r in results) / n_repos, 0),
            "p50_ms": round(sum(r.p50_ms for r in results) / n_repos, 3),
            "p90_ms": round(sum(r.p90_ms for r in results) / n_repos, 3),
            "p95_ms": round(sum(r.p95_ms for r in results) / n_repos, 3),
            "p99_ms": round(sum(r.p99_ms for r in results) / n_repos, 3),
            "index_ms": round(sum(r.index_ms for r in results) / n_repos, 1),
            "by_category": cat_means,
            "by_kind": kind_means,
            **_mean_hits(results, lambda r: r.hits),
            "hits_by_category": _grouped_hit_means(results, lambda r: r.hits_by_category),
            "hits_by_kind": _grouped_hit_means(results, lambda r: r.hits_by_kind),
        },
        "reranker": resolve_reranker_spec(),
        "rerank_settings": asdict(RerankSettings.from_env()) | {"passage": RerankSettings.from_env().passage.value},
        "api_usage": usage,
        "by_language": {
            lang: {
                "repos": len(by_language[lang]),
                "tokens": round(sum(r.tokens for r in by_language[lang]) / len(by_language[lang]), 0),
                "ndcg10": round(v["ndcg10"], 4),
                "p50_ms": round(v["p50_ms"], 3),
                "p90_ms": round(v["p90_ms"], 3),
                "p95_ms": round(v["p95_ms"], 3),
                "p99_ms": round(v["p99_ms"], 3),
                "index_ms": round(v["index_ms"], 1),
            }
            for lang, v in lang_means.items()
        },
        "repos": [asdict(r) for r in results],
    }

    out_path = save_results(label, output)
    print(f"\nResults saved to {out_path}", file=sys.stderr)


def main() -> None:
    """Parse arguments and run the zemble hybrid benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark hybrid zemble search across the pinned benchmark repos.")
    add_filter_args(parser, verbose=True)
    add_repo_source_args(parser)
    parser.add_argument(
        "--latency-runs",
        type=int,
        default=_LATENCY_RUNS,
        help=(
            "How many times each query is re-issued for its latency median. Quality metrics are "
            "unaffected; with a hosted reranker every repetition is a paid request."
        ),
    )
    parser.add_argument(
        "--label-suffix",
        default="",
        help="Appended to the result label, so runs that differ only by environment do not overwrite each other.",
    )
    args = parser.parse_args()
    repo_specs, tasks = load_filtered_tasks(
        args.repo or None, args.language or None, args.repos_file, args.annotations_dir
    )
    print("Loading model...", file=sys.stderr)
    started = time.perf_counter()
    print(f"Loaded in {(time.perf_counter() - started) * 1000:.0f} ms", file=sys.stderr)
    print(file=sys.stderr)
    repo_tasks = grouped_tasks(tasks)
    usage: dict[str, dict[str, object]] = {}
    reranker = load_reranker()
    results = _bench_quality(
        repo_tasks,
        repo_specs,
        verbose=args.verbose,
        latency_runs=args.latency_runs,
        reranker=reranker,
        usage=usage,
    )
    _record_usage(usage, "reranker", reranker)
    _print_summary(results)
    if usage:
        for role, entry in sorted(usage.items()):
            print(
                f"\n{role}: {entry['model_id']}  requests={entry['requests']}  tokens={entry['tokens']}",
                file=sys.stderr,
            )
    if not args.repo and not args.language:
        label = results_label("zemble-hybrid", args.repos_file, list(repo_tasks))
        _save_results(results, f"{label}-{args.label_suffix}" if args.label_suffix else label, usage)


if __name__ == "__main__":
    main()
