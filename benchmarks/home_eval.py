"""Measure `zemble home` against the homes the javaweb workspace already declares.

Ground truth is the capability table in that workspace's CLAUDE.md: every row whose
"Mechanism home" cell names exactly one module. The queries are hand-written
paraphrases in `benchmarks/local/home_queries.json`, so a row's own wording is not
what finds it. Each query is answered twice: once with the declared-table lane
DISABLED, which measures what search, the symbol graph and the module order can do
on their own, and once with it enabled, which measures the whole thing.

Beside those positives the file carries NEGATIVE queries - capabilities the workspace
does not have, mechanisms two sibling modules would have to share, and questions written
in a declared row's vocabulary about something else. They have no declared home to rank,
and what they measure is the verdict: an `EXTEND_EXISTING` there is a confident wrong
answer, the failure mode this eval exists to catch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.data import BENCHMARKS_DIR, available_repo_specs, save_results
from zemble.graph import SqliteGraphProvider, build_graph
from zemble.home.answers import DEFAULT_TOP_K, build_answer
from zemble.home.config import HomeConfig
from zemble.home.decide import HomeAnswer, Verdict
from zemble.index import ZembleIndex
from zemble.types import ContentType

DEFAULT_QUERIES = BENCHMARKS_DIR / "local" / "home_queries.json"
DEFAULT_REPOS = BENCHMARKS_DIR / "local" / "repos.json"


@dataclass
class Record:
    """What one query produced in one lane."""

    title: str
    kind: str
    query: str
    declared_home: str | None
    tables: bool
    candidates: list[str]
    hit_at_1: bool
    hit_at_3: bool
    verdict: str
    expected_verdict: list[str]
    verdict_correct: bool
    named_home: str | None
    named_home_correct: bool
    suggested_home: str | None
    top_mechanism: str | None
    ms: float
    why: str

    @property
    def ranked(self) -> bool:
        """Whether this query has a declared home, so hit@1 and hit@3 mean anything."""
        return self.declared_home is not None

    @property
    def overconfident(self) -> bool:
        """Whether the answer claimed an existing mechanism where that was not acceptable."""
        return self.verdict == Verdict.EXTEND_EXISTING.value and not self.verdict_correct


def _why(answer: HomeAnswer, expected: str) -> str:
    """One line saying why an answer missed its declared home."""
    top = answer.candidates[0] if answer.candidates else None
    if top is None:
        return "no candidates: the search found nothing to place this beside"
    position = next(
        (index for index, candidate in enumerate(answer.candidates, 1) if candidate.module == expected), None
    )
    place = f"#{position}" if position else "not in the top 3"
    share = next((entry for entry in answer.module_hits if entry.module == top.module), None)
    mass = f"{share.hits} of {sum(entry.hits for entry in answer.module_hits)} hits" if share else "no hits"
    return f"{expected} is {place}; {top.module} led with {mass}"


def _judge(answer: HomeAnswer, entry: dict) -> tuple[bool, str]:
    """Say whether an answer met the query's expectations, and why not when it did not."""
    expected = entry.get("expected_verdict") or [Verdict.EXTEND_EXISTING.value]
    forbidden = entry.get("forbidden_homes") or []
    if answer.verdict.value not in expected:
        return False, f"verdict {answer.verdict.value}, expected {' or '.join(expected)}"
    if answer.home in forbidden:
        return False, f"named {answer.home}, which is one of the modules that must not own this"
    if entry.get("expect_suggested_home") and not answer.suggested_home:
        return False, "no shared home was suggested for two modules that cannot depend on each other"
    return True, ""


def _evaluate(
    index: ZembleIndex,
    graph: SqliteGraphProvider,
    config: HomeConfig,
    queries: Sequence[dict],
    top_k: int,
) -> list[Record]:
    """Answer every query in both lanes."""
    records: list[Record] = []
    for number, entry in enumerate(queries, 1):
        for tables in (False, True):
            started = time.perf_counter()
            answer = build_answer(index, graph, config, entry["query"], top_k=top_k, use_tables=tables)
            elapsed = (time.perf_counter() - started) * 1000
            expected = entry.get("home")
            modules = [candidate.module for candidate in answer.candidates]
            hit_1 = bool(modules) and expected is not None and modules[0] == expected
            correct, complaint = _judge(answer, entry)
            records.append(
                Record(
                    title=entry["title"],
                    kind=entry.get("kind", "declared"),
                    query=entry["query"],
                    declared_home=expected,
                    tables=tables,
                    candidates=modules,
                    hit_at_1=hit_1,
                    hit_at_3=expected is not None and expected in modules,
                    verdict=answer.verdict.value,
                    expected_verdict=list(entry.get("expected_verdict") or [Verdict.EXTEND_EXISTING.value]),
                    verdict_correct=correct,
                    named_home=answer.home,
                    named_home_correct=expected is not None and answer.home == expected,
                    suggested_home=answer.suggested_home,
                    top_mechanism=answer.mechanisms[0].label if answer.mechanisms else None,
                    ms=round(elapsed, 1),
                    why=complaint if expected is None else ("" if hit_1 else _why(answer, expected)),
                )
            )
        print(f"  {number}/{len(queries)} {entry['title'][:60]}", file=sys.stderr)
    return records


def _rate(values: Sequence[bool]) -> float:
    """Return the share of True values."""
    return round(sum(1 for value in values if value) / len(values), 3) if values else 0.0


def _summarise(records: Sequence[Record]) -> dict[str, object]:
    """Aggregate both lanes, ranking the declared rows and judging every query's verdict."""
    summary: dict[str, object] = {
        "queries": len({record.title for record in records}),
        "declared_queries": len({record.title for record in records if record.ranked}),
        "negative_queries": len({record.title for record in records if not record.ranked}),
    }
    for tables in (False, True):
        rows = [record for record in records if record.tables is tables]
        ranked = [row for row in rows if row.ranked]
        negative = [row for row in rows if not row.ranked]
        summary["with_table" if tables else "without_table"] = {
            "hit_at_1": _rate([row.hit_at_1 for row in ranked]),
            "hit_at_3": _rate([row.hit_at_3 for row in ranked]),
            "named_home_correct": _rate([row.named_home_correct for row in ranked]),
            "verdict_correct": _rate([row.verdict_correct for row in rows]),
            "verdict_correct_declared": _rate([row.verdict_correct for row in ranked]),
            "verdict_correct_negative": _rate([row.verdict_correct for row in negative]),
            "extend_existing": _rate([row.verdict == Verdict.EXTEND_EXISTING.value for row in rows]),
            "new_mechanism": _rate([row.verdict == Verdict.NEW_MECHANISM.value for row in rows]),
            "uncertain": _rate([row.verdict == Verdict.UNCERTAIN.value for row in rows]),
            "overconfident": sum(1 for row in rows if row.overconfident),
            "mean_ms": round(sum(row.ms for row in rows) / len(rows), 1) if rows else 0.0,
        }
    return summary


def _print_summary(summary: dict[str, object], records: Sequence[Record]) -> None:
    """Print the table and the failures that go into the documentation."""
    print()
    print(
        f"{summary['declared_queries']} declared-home rows, paraphrased,"
        f" plus {summary['negative_queries']} negative queries"
    )
    print()
    print(
        f"{'lane':<16} {'hit@1':>7} {'hit@3':>7} {'home ok':>8} {'verdict':>8} {'v-decl':>7} {'v-neg':>7}"
        f" {'EXTEND':>7} {'NEW':>7} {'UNSURE':>7} {'over':>5} {'ms':>7}"
    )
    for key, label in (("without_table", "search only"), ("with_table", "search + table")):
        row: dict[str, float] = summary[key]  # type: ignore[assignment]
        print(
            f"{label:<16} {row['hit_at_1']:>7} {row['hit_at_3']:>7} {row['named_home_correct']:>8}"
            f" {row['verdict_correct']:>8} {row['verdict_correct_declared']:>7} {row['verdict_correct_negative']:>7}"
            f" {row['extend_existing']:>7} {row['new_mechanism']:>7} {row['uncertain']:>7}"
            f" {row['overconfident']:>5} {row['mean_ms']:>7}"
        )
    for tables, label in ((False, "search only"), (True, "search + table")):
        lane = [record for record in records if record.tables is tables]
        misses = [record for record in lane if record.ranked and not record.hit_at_1]
        print()
        print(f"{label}: {len(misses)} miss(es) at rank 1")
        for record in misses:
            print(f"  {record.title[:44]:<44} want {str(record.declared_home):<16} {record.why}")
        wrong = [record for record in lane if not record.verdict_correct]
        print(
            f"{label}: {len(wrong)} wrong verdict(s),"
            f" {sum(1 for record in lane if record.overconfident)} of them over-confident"
        )
        for record in wrong:
            mark = "OVER-CONFIDENT" if record.overconfident else "wrong"
            print(f"  [{mark}] {record.title[:40]:<40} {record.why or record.verdict}")


def main() -> None:
    """Build the index and the graph once, then answer every query in both lanes."""
    parser = argparse.ArgumentParser(description="Measure `zemble home` against the declared homes.")
    parser.add_argument("--queries-file", type=Path, default=DEFAULT_QUERIES, help="Ground-truth query file.")
    parser.add_argument("--repos-file", type=Path, default=DEFAULT_REPOS, help="Repo descriptor file.")
    parser.add_argument("--repo", default="javaweb", help="Repo name inside the descriptor file.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Code results each answer weighs.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N queries (0 = all).")
    parser.add_argument("--no-save", action="store_true", help="Do not write a result file.")
    args = parser.parse_args()

    specs = available_repo_specs(args.repos_file, BENCHMARKS_DIR / "local" / "annotations")
    spec = specs.get(args.repo)
    if spec is None:
        parser.error(f"repo {args.repo!r} not available in {args.repos_file}")
    root = spec.benchmark_dir
    payload = json.loads(args.queries_file.read_text(encoding="utf-8"))
    queries = payload["queries"][: args.limit] if args.limit else payload["queries"]

    config = HomeConfig.load(root)
    if config.generic:
        parser.error(f"{root} declares no .zemble/home.toml, so there is nothing to measure against")

    print(f"Indexing {root} (code + docs) ...", file=sys.stderr)
    started = time.perf_counter()
    index = ZembleIndex.from_path(root, content=[ContentType.CODE, ContentType.DOCS])
    index_ms = (time.perf_counter() - started) * 1000
    print(f"  {len(index.chunks)} chunks in {index_ms / 1000:.1f}s", file=sys.stderr)

    print("Building the symbol graph ...", file=sys.stderr)
    started = time.perf_counter()
    stats = build_graph(str(root))
    graph_ms = (time.perf_counter() - started) * 1000
    print(f"  {stats.symbols} symbols, {stats.edges} edges in {graph_ms / 1000:.1f}s", file=sys.stderr)

    graph = SqliteGraphProvider(str(root))
    try:
        records = _evaluate(index, graph, config, queries, args.top_k)
    finally:
        graph.close()

    summary = _summarise(records)
    _print_summary(summary, records)
    if args.no_save:
        return
    out_path = save_results(
        f"home-{args.repo}",
        {
            "tool": "home",
            "repo": args.repo,
            "root": str(root),
            "top_k": args.top_k,
            "chunks": len(index.chunks),
            "index_ms": round(index_ms, 1),
            "graph_ms": round(graph_ms, 1),
            "graph": {"symbols": stats.symbols, "edges": stats.edges},
            "summary": summary,
            "records": [asdict(record) for record in records],
        },
    )
    print(f"\nResults saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
