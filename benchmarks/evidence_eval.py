"""Measure evidence bundles against plain search on a local annotation set.

For every query the harness builds a bundle at several token budgets and records
whether an annotated file made it into the bundle as content, whether it made it
in at all, what the bundle cost, and what reading the annotated files outright
would have cost. Plain top-5 search is measured on the same queries as the
reference point.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmarks.data import (
    BENCHMARKS_DIR,
    Task,
    available_repo_specs,
    load_tasks,
    path_matches,
    save_results,
)
from zemble.evidence.bundle import Presentation, build_bundle
from zemble.evidence.tokens import estimate_tokens
from zemble.graph import SqliteGraphProvider, build_graph
from zemble.index import ZembleIndex

DEFAULT_BUDGETS = (1500, 3000, 6000)
DEFAULT_REPOS = BENCHMARKS_DIR / "local" / "repos.json"
DEFAULT_ANNOTATIONS = BENCHMARKS_DIR / "local" / "annotations"
SEARCH_TOP_K = 5

_CONTENT_FORMS = (Presentation.CONTENT, Presentation.TRUNCATED)


@dataclass
class QueryResult:
    """What one query produced at one budget."""

    query: str
    kind: str
    budget: int
    bundle_tokens: int
    content_hit: bool
    any_hit: bool
    named_hit: bool
    chunk_hit: bool
    expansion_hit: bool
    items: int
    omitted: int
    build_ms: float


@dataclass
class QueryFacts:
    """What one query costs and hits regardless of budget."""

    query: str
    kind: str
    full_file_tokens: int
    search_tokens: int
    search_hit: bool
    relevant: list[str] = field(default_factory=list)


def _kind_of(task: Task) -> str:
    """Return the query's declared kind, falling back to its category."""
    return task.kind or task.category


def _full_file_tokens(root: Path, task: Task) -> int:
    """Estimate what reading every annotated file outright would cost."""
    total = 0
    for target in task.relevant:
        try:
            total += estimate_tokens((root / target.path).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return total


def _hits(paths: Sequence[str], task: Task) -> bool:
    """Return True if any annotated file is among the paths."""
    return any(path_matches(path, target.path) for path in paths for target in task.relevant)


def _evaluate(
    index: ZembleIndex,
    graph: SqliteGraphProvider,
    tasks: Sequence[Task],
    root: Path,
    budgets: Sequence[int],
    top_k: int,
) -> tuple[list[QueryResult], list[QueryFacts]]:
    """Run every query at every budget and collect the raw records."""
    results: list[QueryResult] = []
    facts: list[QueryFacts] = []
    for number, task in enumerate(tasks, 1):
        found = index.search(task.query, top_k=SEARCH_TOP_K)
        search_paths = [result.chunk.file_path for result in found]
        facts.append(
            QueryFacts(
                query=task.query,
                kind=_kind_of(task),
                full_file_tokens=_full_file_tokens(root, task),
                search_tokens=sum(estimate_tokens(result.chunk.content) for result in found),
                search_hit=_hits(search_paths, task),
                relevant=[target.path for target in task.relevant],
            )
        )
        for budget in budgets:
            started = time.perf_counter()
            bundle = build_bundle(index, graph, task.query, budget, top_k=top_k)
            elapsed = (time.perf_counter() - started) * 1000
            content = [item.file_path for item in bundle.items if item.presentation in _CONTENT_FORMS]
            every = [item.file_path for item in bundle.items]
            searched = [item.file_path for item in bundle.items if item.tier == 0]
            expanded = [item.file_path for item in bundle.items if item.tier > 0]
            named = every + [entry.file_path for entry in bundle.omitted]
            results.append(
                QueryResult(
                    query=task.query,
                    kind=_kind_of(task),
                    budget=budget,
                    bundle_tokens=estimate_tokens(bundle.render()),
                    content_hit=_hits(content, task),
                    any_hit=_hits(every, task),
                    named_hit=_hits(named, task),
                    chunk_hit=_hits(searched, task),
                    expansion_hit=_hits(expanded, task),
                    items=len(bundle.items),
                    omitted=len(bundle.omitted),
                    build_ms=elapsed,
                )
            )
        print(f"  {number}/{len(tasks)} {task.query[:70]}", file=sys.stderr)
    return results, facts


def _rate(values: Sequence[bool]) -> float:
    """Return the share of True values, or 0 for an empty sequence."""
    return round(sum(1 for value in values if value) / len(values), 3) if values else 0.0


def _summarise(
    results: Sequence[QueryResult], facts: Sequence[QueryFacts], budgets: Sequence[int]
) -> dict[str, object]:
    """Aggregate the raw records overall and per query kind."""
    by_kind = sorted({fact.kind for fact in facts})
    search_by_kind = {kind: _rate([f.search_hit for f in facts if f.kind == kind]) for kind in by_kind}
    full_by_kind = {
        kind: round(statistics.mean([f.full_file_tokens for f in facts if f.kind == kind]), 1) for kind in by_kind
    }
    summary: dict[str, object] = {
        "queries": len(facts),
        "search_top5_hit_rate": _rate([fact.search_hit for fact in facts]),
        "search_top5_hit_rate_by_kind": search_by_kind,
        "mean_full_file_tokens": round(statistics.mean([fact.full_file_tokens for fact in facts]), 1),
        "mean_search_top5_tokens": round(statistics.mean([fact.search_tokens for fact in facts]), 1),
        "mean_full_file_tokens_by_kind": full_by_kind,
        "budgets": {},
    }
    per_budget: dict[str, object] = {}
    for budget in budgets:
        rows = [row for row in results if row.budget == budget]
        mean_tokens = statistics.mean([row.bundle_tokens for row in rows])
        per_budget[str(budget)] = {
            "content_hit_rate": _rate([row.content_hit for row in rows]),
            "any_hit_rate": _rate([row.any_hit for row in rows]),
            "named_hit_rate": _rate([row.named_hit for row in rows]),
            "chunk_hit_rate": _rate([row.chunk_hit for row in rows]),
            "expansion_hit_rate": _rate([row.expansion_hit for row in rows]),
            "expansion_only_hit_rate": _rate([row.expansion_hit and not row.chunk_hit for row in rows]),
            "mean_bundle_tokens": round(mean_tokens, 1),
            "mean_items": round(statistics.mean([row.items for row in rows]), 1),
            "mean_build_ms": round(statistics.mean([row.build_ms for row in rows]), 1),
            "compression_vs_full_files": round(summary["mean_full_file_tokens"] / mean_tokens, 2)  # type: ignore[operator]
            if mean_tokens
            else 0.0,
            "by_kind": {
                kind: {
                    "queries": len([row for row in rows if row.kind == kind]),
                    "content_hit_rate": _rate([row.content_hit for row in rows if row.kind == kind]),
                    "any_hit_rate": _rate([row.any_hit for row in rows if row.kind == kind]),
                    "mean_bundle_tokens": round(
                        statistics.mean([row.bundle_tokens for row in rows if row.kind == kind]), 1
                    ),
                    "mean_full_file_tokens": full_by_kind[kind],
                    "search_top5_hit_rate": search_by_kind[kind],
                }
                for kind in by_kind
            },
        }
    summary["budgets"] = per_budget
    return summary


def _print_summary(summary: dict[str, object]) -> None:
    """Print the tables that go into the documentation."""
    budgets: dict[str, dict[str, object]] = summary["budgets"]  # type: ignore[assignment]
    print()
    print(f"{summary['queries']} queries; plain search top-5 hit rate {summary['search_top5_hit_rate']}")
    print(f"mean full-file cost of the annotated answers: {summary['mean_full_file_tokens']} tokens")
    print(f"mean cost of the plain search top-5 chunks: {summary['mean_search_top5_tokens']} tokens")
    print()
    print(
        f"{'budget':>7} {'content':>8} {'any':>8} {'named':>8} {'chunk':>7} {'expand':>7} {'only':>6} "
        f"{'tokens':>8} {'items':>6} {'ratio':>6} {'ms':>7}"
    )
    for budget, row in budgets.items():
        print(
            f"{budget:>7} {row['content_hit_rate']:>8} {row['any_hit_rate']:>8} {row['named_hit_rate']:>8} "
            f"{row['chunk_hit_rate']:>7} {row['expansion_hit_rate']:>7} {row['expansion_only_hit_rate']:>6} "
            f"{row['mean_bundle_tokens']:>8} {row['mean_items']:>6} {row['compression_vs_full_files']:>6} "
            f"{row['mean_build_ms']:>7}"
        )
    for budget, row in budgets.items():
        print()
        print(f"budget {budget}, by kind:")
        print(f"{'kind':<14} {'n':>3} {'content':>8} {'any':>8} {'search@5':>9} {'tokens':>8} {'full':>8}")
        for kind, values in row["by_kind"].items():  # type: ignore[union-attr]
            print(
                f"{kind:<14} {values['queries']:>3} {values['content_hit_rate']:>8} {values['any_hit_rate']:>8} "
                f"{values['search_top5_hit_rate']:>9} {values['mean_bundle_tokens']:>8} "
                f"{values['mean_full_file_tokens']:>8}"
            )


def main() -> None:
    """Build the index and graph once, then measure every query at every budget."""
    parser = argparse.ArgumentParser(description="Measure evidence bundles against plain search.")
    parser.add_argument("--repos-file", type=Path, default=DEFAULT_REPOS, help="Repo descriptor file.")
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS, help="Annotation directory.")
    parser.add_argument("--repo", default="javaweb", help="Repo name inside the descriptor file.")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS), help="Token budgets to test.")
    parser.add_argument("--top-k", type=int, default=20, help="Search results each bundle expands.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N queries (0 = all).")
    parser.add_argument("--no-save", action="store_true", help="Do not write a result file.")
    args = parser.parse_args()

    specs = available_repo_specs(args.repos_file, args.annotations_dir)
    spec = specs.get(args.repo)
    if spec is None:
        parser.error(f"repo {args.repo!r} not available in {args.repos_file}")
    root = spec.benchmark_dir
    tasks = [task for task in load_tasks(specs, args.annotations_dir) if task.repo == args.repo]
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"Indexing {root} ...", file=sys.stderr)
    started = time.perf_counter()
    index = ZembleIndex.from_path(root)
    index_ms = (time.perf_counter() - started) * 1000
    print(f"  {len(index.chunks)} chunks in {index_ms / 1000:.1f}s", file=sys.stderr)

    print("Building the symbol graph ...", file=sys.stderr)
    started = time.perf_counter()
    stats = build_graph(str(root))
    graph_ms = (time.perf_counter() - started) * 1000
    print(f"  {stats.symbols} symbols, {stats.edges} edges in {graph_ms / 1000:.1f}s", file=sys.stderr)

    graph = SqliteGraphProvider(str(root))
    try:
        results, facts = _evaluate(index, graph, tasks, root, args.budgets, args.top_k)
    finally:
        graph.close()

    summary = _summarise(results, facts, args.budgets)
    _print_summary(summary)
    if args.no_save:
        return
    payload = {
        "tool": "evidence-bundles",
        "repo": args.repo,
        "root": str(root),
        "top_k": args.top_k,
        "chunks": len(index.chunks),
        "index_ms": round(index_ms, 1),
        "graph_ms": round(graph_ms, 1),
        "graph": {"symbols": stats.symbols, "edges": stats.edges},
        "summary": summary,
        "queries": [asdict(fact) for fact in facts],
        "records": [asdict(row) for row in results],
    }
    out_path = save_results(f"evidence-{args.repo}", payload)
    print(f"\nResults saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
