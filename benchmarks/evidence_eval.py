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
from collections import Counter
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
from zemble.evidence.intent import Intent, classify
from zemble.evidence.tokens import estimate_tokens
from zemble.graph import SqliteGraphProvider, build_graph
from zemble.index import ZembleIndex

DEFAULT_BUDGETS = (1500, 3000, 6000)
DEFAULT_REPOS = BENCHMARKS_DIR / "local" / "repos.json"
DEFAULT_ANNOTATIONS = BENCHMARKS_DIR / "local" / "annotations"
SEARCH_TOP_K = 5

_CONTENT_FORMS = (Presentation.CONTENT, Presentation.TRUNCATED)

#: The two orders under test: the shipped default, and the one the intent chooses.
BASE_VARIANT = "base"
INTENT_VARIANT = "intent"
VARIANTS = (BASE_VARIANT, INTENT_VARIANT)

# AIDEV-NOTE: the eval set's `kind` labels are ground truth for scoring the classifier
# and are never an input to it; this is the only place the two vocabularies meet.
_KIND_TO_INTENT = {
    "symbol": Intent.SYMBOL,
    "behavioural": Intent.BEHAVIOUR,
    "architecture": Intent.ARCHITECTURE,
    "bug-report": Intent.BUG,
    "consumer": Intent.CONSUMER,
}


@dataclass
class QueryResult:
    """What one query produced at one budget."""

    query: str
    kind: str
    variant: str
    budget: int
    bundle_tokens: int
    content_hit: bool
    any_hit: bool
    named_hit: bool
    chunk_hit: bool
    expansion_hit: bool
    items: int
    omitted: int
    seeded: int
    build_ms: float


@dataclass
class QueryFacts:
    """What one query costs and hits regardless of budget."""

    query: str
    kind: str
    full_file_tokens: int
    search_tokens: int
    search_hit: bool
    intent: str = Intent.UNKNOWN.value
    intent_rule: str = ""
    intent_correct: bool = False
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
    variants: Sequence[str],
) -> tuple[list[QueryResult], list[QueryFacts]]:
    """Run every query at every budget, in every variant, and collect the raw records."""
    results: list[QueryResult] = []
    facts: list[QueryFacts] = []
    for number, task in enumerate(tasks, 1):
        found = index.search(task.query, top_k=SEARCH_TOP_K)
        search_paths = [result.chunk.file_path for result in found]
        detected = classify(task.query)
        expected = _KIND_TO_INTENT.get(_kind_of(task))
        facts.append(
            QueryFacts(
                query=task.query,
                kind=_kind_of(task),
                full_file_tokens=_full_file_tokens(root, task),
                search_tokens=sum(estimate_tokens(result.chunk.content) for result in found),
                search_hit=_hits(search_paths, task),
                intent=detected.intent.value,
                intent_rule=detected.rule,
                intent_correct=expected is not None and detected.intent is expected,
                relevant=[target.path for target in task.relevant],
            )
        )
        for variant, budget in [(v, b) for v in variants for b in budgets]:
            # The base variant is the shipped default; the intent variant applies what
            # the classifier decided, which is what `explain --intent` does by hand.
            forced = None if variant == BASE_VARIANT else detected.intent
            started = time.perf_counter()
            bundle = build_bundle(index, graph, task.query, budget, top_k=top_k, intent=forced)
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
                    variant=variant,
                    budget=budget,
                    bundle_tokens=estimate_tokens(bundle.render()),
                    content_hit=_hits(content, task),
                    any_hit=_hits(every, task),
                    named_hit=_hits(named, task),
                    chunk_hit=_hits(searched, task),
                    expansion_hit=_hits(expanded, task),
                    items=len(bundle.items),
                    omitted=len(bundle.omitted),
                    seeded=bundle.seeded,
                    build_ms=elapsed,
                )
            )
        print(f"  {number}/{len(tasks)} {task.query[:70]}", file=sys.stderr)
    return results, facts


def _rate(values: Sequence[bool]) -> float:
    """Return the share of True values, or 0 for an empty sequence."""
    return round(sum(1 for value in values if value) / len(values), 3) if values else 0.0


def _intent_summary(facts: Sequence[QueryFacts], by_kind: Sequence[str]) -> dict[str, object]:
    """Score the classifier against the annotation kinds, as a diagnostic only."""
    labelled = [fact for fact in facts if _kind_to_intent_known(fact.kind)]
    return {
        "accuracy": _rate([fact.intent_correct for fact in labelled]),
        "labelled_queries": len(labelled),
        "accuracy_by_kind": {
            kind: _rate([fact.intent_correct for fact in labelled if fact.kind == kind]) for kind in by_kind
        },
        "detected_by_kind": {
            kind: dict(Counter(fact.intent for fact in facts if fact.kind == kind)) for kind in by_kind
        },
        "rules": dict(Counter(fact.intent_rule for fact in facts)),
    }


def _kind_to_intent_known(kind: str) -> bool:
    """Return True when the annotation kind has an intent to be scored against."""
    return kind in _KIND_TO_INTENT


def _budget_rows(
    rows: Sequence[QueryResult],
    by_kind: Sequence[str],
    search_by_kind: dict[str, float],
    full_by_kind: dict[str, float],
    full_tokens: float,
) -> dict[str, object]:
    """Aggregate one variant's rows at one budget, overall and per kind."""
    mean_tokens = statistics.mean([row.bundle_tokens for row in rows])
    return {
        "content_hit_rate": _rate([row.content_hit for row in rows]),
        "any_hit_rate": _rate([row.any_hit for row in rows]),
        "named_hit_rate": _rate([row.named_hit for row in rows]),
        "chunk_hit_rate": _rate([row.chunk_hit for row in rows]),
        "expansion_hit_rate": _rate([row.expansion_hit for row in rows]),
        "expansion_only_hit_rate": _rate([row.expansion_hit and not row.chunk_hit for row in rows]),
        "seeded_queries": len([row for row in rows if row.seeded]),
        "mean_seeded": round(statistics.mean([row.seeded for row in rows]), 2),
        "mean_bundle_tokens": round(mean_tokens, 1),
        "mean_items": round(statistics.mean([row.items for row in rows]), 1),
        "mean_build_ms": round(statistics.mean([row.build_ms for row in rows]), 1),
        "compression_vs_full_files": round(full_tokens / mean_tokens, 2) if mean_tokens else 0.0,
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


def _summarise(
    results: Sequence[QueryResult], facts: Sequence[QueryFacts], budgets: Sequence[int], variants: Sequence[str]
) -> dict[str, object]:
    """Aggregate the raw records per variant, overall and per query kind."""
    by_kind = sorted({fact.kind for fact in facts})
    search_by_kind = {kind: _rate([f.search_hit for f in facts if f.kind == kind]) for kind in by_kind}
    full_by_kind = {
        kind: round(statistics.mean([f.full_file_tokens for f in facts if f.kind == kind]), 1) for kind in by_kind
    }
    full_tokens = round(statistics.mean([fact.full_file_tokens for fact in facts]), 1)
    return {
        "queries": len(facts),
        "search_top5_hit_rate": _rate([fact.search_hit for fact in facts]),
        "search_top5_hit_rate_by_kind": search_by_kind,
        "mean_full_file_tokens": full_tokens,
        "mean_search_top5_tokens": round(statistics.mean([fact.search_tokens for fact in facts]), 1),
        "mean_full_file_tokens_by_kind": full_by_kind,
        "intent": _intent_summary(facts, by_kind),
        "variants": {
            variant: {
                "budgets": {
                    str(budget): _budget_rows(
                        [row for row in results if row.budget == budget and row.variant == variant],
                        by_kind,
                        search_by_kind,
                        full_by_kind,
                        full_tokens,
                    )
                    for budget in budgets
                }
            }
            for variant in variants
        },
    }


def _print_variant(name: str, budgets: dict[str, dict[str, object]]) -> None:
    """Print one variant's overall and per-kind tables."""
    print()
    print(f"=== variant: {name}")
    print(
        f"{'budget':>7} {'content':>8} {'any':>8} {'named':>8} {'chunk':>7} {'expand':>7} {'only':>6} "
        f"{'seeded':>7} {'tokens':>8} {'items':>6} {'ratio':>6} {'ms':>7}"
    )
    for budget, row in budgets.items():
        print(
            f"{budget:>7} {row['content_hit_rate']:>8} {row['any_hit_rate']:>8} {row['named_hit_rate']:>8} "
            f"{row['chunk_hit_rate']:>7} {row['expansion_hit_rate']:>7} {row['expansion_only_hit_rate']:>6} "
            f"{row['seeded_queries']:>7} {row['mean_bundle_tokens']:>8} {row['mean_items']:>6} "
            f"{row['compression_vs_full_files']:>6} {row['mean_build_ms']:>7}"
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


def _print_comparison(summary: dict[str, object]) -> None:
    """Print the base-versus-intent deltas the decision rule is read from."""
    variants: dict[str, dict[str, object]] = summary["variants"]  # type: ignore[assignment]
    if BASE_VARIANT not in variants or INTENT_VARIANT not in variants:
        return
    base: dict[str, object] = variants[BASE_VARIANT]["budgets"]  # type: ignore[assignment,index]
    intent: dict[str, object] = variants[INTENT_VARIANT]["budgets"]  # type: ignore[assignment,index]
    print()
    print("=== base -> intent")
    print(f"{'budget':>7} {'kind':<14} {'content':>16} {'any':>16}")
    for budget in base:
        for metric_kind in ("overall", *sorted(base[budget]["by_kind"])):  # type: ignore[index]
            if metric_kind == "overall":
                left, right = base[budget], intent[budget]  # type: ignore[index]
            else:
                left = base[budget]["by_kind"][metric_kind]  # type: ignore[index]
                right = intent[budget]["by_kind"][metric_kind]  # type: ignore[index]
            content = f"{left['content_hit_rate']} -> {right['content_hit_rate']}"  # type: ignore[index]
            every = f"{left['any_hit_rate']} -> {right['any_hit_rate']}"  # type: ignore[index]
            print(f"{budget:>7} {metric_kind:<14} {content:>16} {every:>16}")


def _print_summary(summary: dict[str, object]) -> None:
    """Print the tables that go into the documentation."""
    print()
    print(f"{summary['queries']} queries; plain search top-5 hit rate {summary['search_top5_hit_rate']}")
    print(f"mean full-file cost of the annotated answers: {summary['mean_full_file_tokens']} tokens")
    print(f"mean cost of the plain search top-5 chunks: {summary['mean_search_top5_tokens']} tokens")
    intent: dict[str, object] = summary["intent"]  # type: ignore[assignment]
    print()
    print(f"intent detection: {intent['accuracy']} over {intent['labelled_queries']} labelled queries")
    print(f"  by kind: {intent['accuracy_by_kind']}")
    print(f"  rules that fired: {intent['rules']}")
    for name, variant in summary["variants"].items():  # type: ignore[union-attr]
        _print_variant(name, variant["budgets"])
    _print_comparison(summary)


def main() -> None:
    """Build the index and graph once, then measure every query at every budget."""
    parser = argparse.ArgumentParser(description="Measure evidence bundles against plain search.")
    parser.add_argument("--repos-file", type=Path, default=DEFAULT_REPOS, help="Repo descriptor file.")
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS, help="Annotation directory.")
    parser.add_argument("--repo", default="javaweb", help="Repo name inside the descriptor file.")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS), help="Token budgets to test.")
    parser.add_argument("--top-k", type=int, default=20, help="Search results each bundle expands.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N queries (0 = all).")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS),
        choices=list(VARIANTS),
        help="Tier orders to measure: the fixed order, the intent-chosen one, or both.",
    )
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
        results, facts = _evaluate(index, graph, tasks, root, args.budgets, args.top_k, args.variants)
    finally:
        graph.close()

    summary = _summarise(results, facts, args.budgets, args.variants)
    _print_summary(summary)
    if args.no_save:
        return
    payload = {
        "tool": "evidence-bundles",
        "repo": args.repo,
        "root": str(root),
        "top_k": args.top_k,
        "variants": args.variants,
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
