"""Sweep reranker configurations over an evaluation set, one loaded index at a time.

Every configuration is scored against a repo while its index is in memory, so a whole grid
costs one indexing pass instead of one per configuration, and only one index is resident.
"""

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from benchmarks.data import Task, load_filtered_tasks
from benchmarks.metrics import ndcg_at_k, target_rank
from zemble import ZembleIndex
from zemble.rerank.base import Reranker
from zemble.rerank.registry import PassageMode, RerankSettings, parse_reranker_spec

_TOP_K = 10


@dataclass(frozen=True)
class Config:
    """One point of the sweep grid."""

    spec: str
    reranker: Reranker | None
    settings: RerankSettings

    @property
    def label(self) -> str:
        """The passage/alpha/k triple as shown in the table, blanked for the baseline."""
        if self.reranker is None:
            return "-"
        return f"{self.settings.passage.value}/{self.settings.alpha}/{self.settings.top_k}"


@dataclass
class Accumulator:
    """Per-configuration totals, held per repo.

    AIDEV-NOTE: every figure is a mean over repos of that repo's own mean, which is what
    ``benchmarks.run_benchmark`` reports. A micro-average over queries would weight the
    repos with the most annotations and would not compare with the recorded results.
    """

    ndcg: list[float] = field(default_factory=list)
    p50: list[float] = field(default_factory=list)
    p90: list[float] = field(default_factory=list)
    by_kind: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    by_category: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))


@dataclass(frozen=True)
class ConfigResult:
    """One configuration's quality and latency over the whole evaluation set."""

    reranker: str
    passage: str
    alpha: float
    rerank_k: int
    ndcg10: float
    p50_ms: float
    p90_ms: float
    by_kind: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, float] = field(default_factory=dict)


def score_repo(index: ZembleIndex, tasks: list[Task], config: Config, runs: int, into: Accumulator) -> None:
    """Score one configuration over one repo's tasks, folding the outcome into an accumulator.

    :param index: The repo's loaded index.
    :param tasks: The repo's tasks.
    :param config: The configuration to score.
    :param runs: Query repetitions used for the per-query latency median.
    :param into: The accumulator collecting this configuration's results.
    """
    values: list[float] = []
    latencies: list[float] = []
    kinds: dict[str, list[float]] = defaultdict(list)
    categories: dict[str, list[float]] = defaultdict(list)

    for task in tasks:
        timings: list[float] = []
        results = []
        for _ in range(runs):
            started = time.perf_counter()
            results = index.search(task.query, top_k=_TOP_K, reranker=config.reranker, rerank_settings=config.settings)
            timings.append((time.perf_counter() - started) * 1000)
        latencies.append(float(np.median(timings)))
        ranks = [rank for t in task.all_relevant if (rank := target_rank(results, t)) is not None]
        value = ndcg_at_k(ranks, len(task.all_relevant), _TOP_K)
        values.append(value)
        categories[task.category or "unknown"].append(value)
        if task.kind:
            kinds[task.kind].append(value)

    p50, p90 = np.percentile(latencies, [50, 90]).tolist()
    into.ndcg.append(sum(values) / len(values))
    into.p50.append(float(p50))
    into.p90.append(float(p90))
    for name, group in kinds.items():
        into.by_kind[name].append(sum(group) / len(group))
    for name, group in categories.items():
        into.by_category[name].append(sum(group) / len(group))


def _means(values: dict[str, list[float]]) -> dict[str, float]:
    """Mean per group, rounded, in name order."""
    return {name: round(sum(v) / len(v), 4) for name, v in sorted(values.items())}


def _result(config: Config, accumulated: Accumulator) -> ConfigResult:
    """Turn one configuration's accumulated values into a result row."""
    return ConfigResult(
        reranker=config.spec,
        passage=config.settings.passage.value if config.reranker is not None else "-",
        alpha=config.settings.alpha if config.reranker is not None else 0.0,
        rerank_k=config.settings.top_k if config.reranker is not None else 0,
        ndcg10=round(sum(accumulated.ndcg) / len(accumulated.ndcg), 4),
        p50_ms=round(sum(accumulated.p50) / len(accumulated.p50), 1),
        p90_ms=round(sum(accumulated.p90) / len(accumulated.p90), 1),
        by_kind=_means(accumulated.by_kind),
        by_category=_means(accumulated.by_category),
    )


def build_configs(specs: list[str], passages: list[str], alphas: list[float], rerank_ks: list[int]) -> list[Config]:
    """Expand the grid, keeping the baseline to a single row because it ignores every knob.

    :param specs: Reranker spec strings.
    :param passages: Passage modes.
    :param alphas: Blend weights.
    :param rerank_ks: Window sizes.
    :return: The configurations to run, in grid order.
    """
    built: dict[str, Reranker | None] = {spec: parse_reranker_spec(spec) for spec in specs}
    configs: list[Config] = []
    for spec, passage, alpha, rerank_k in itertools.product(specs, passages, alphas, rerank_ks):
        if built[spec] is None and (passage, alpha, rerank_k) != (passages[0], alphas[0], rerank_ks[0]):
            continue
        settings = RerankSettings(top_k=rerank_k, alpha=alpha, passage=PassageMode(passage))
        configs.append(Config(spec=spec, reranker=built[spec], settings=settings))
    return configs


def _print_table(results: list[ConfigResult]) -> None:
    """Print the results as a markdown table, one column per query kind."""
    kind_names = sorted({name for r in results for name in r.by_kind})
    kind_header = "".join(f" {name} |" for name in kind_names)
    print()
    print("| reranker | passage | alpha | k | NDCG@10 |" + kind_header + " p50 ms |")
    print("| --- " * (6 + len(kind_names)) + "|")
    for result in results:
        kinds = "".join(f" {result.by_kind.get(name, float('nan')):.3f} |" for name in kind_names)
        print(
            f"| {result.reranker} | {result.passage} | {result.alpha} | {result.rerank_k} "
            f"| {result.ndcg10:.4f} |" + kinds + f" {result.p50_ms:.0f} |"
        )


def main() -> None:
    """Run the configured grid and print a markdown table."""
    parser = argparse.ArgumentParser(description="Sweep reranker configurations over an evaluation set.")
    parser.add_argument("--repos-file", type=Path, default=Path("benchmarks/local/repos.json"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("benchmarks/local/annotations"))
    parser.add_argument("--repo", action="append", default=[], help="Restrict to these repos, repeatable.")
    parser.add_argument("--reranker", action="append", default=[], help="Spec, repeatable. Add 'none' for a baseline.")
    parser.add_argument("--passage", action="append", default=[], choices=[m.value for m in PassageMode])
    parser.add_argument("--alpha", action="append", type=float, default=[])
    parser.add_argument("--rerank-k", action="append", type=int, default=[])
    parser.add_argument("--runs", type=int, default=3, help="Query repetitions for the latency median.")
    parser.add_argument("--output", default=None, help="Write the raw results as JSON here.")
    args = parser.parse_args()

    specs, tasks = load_filtered_tasks(args.repo or None, None, args.repos_file, args.annotations_dir)
    repo_tasks: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        repo_tasks[task.repo].append(task)

    configs = build_configs(
        args.reranker or ["none"],
        args.passage or [PassageMode.CONTEXT.value],
        args.alpha or [1.0],
        args.rerank_k or [50],
    )
    accumulators = [Accumulator() for _ in configs]

    for repo in sorted(repo_tasks):
        started = time.perf_counter()
        index = ZembleIndex.from_path(specs[repo].benchmark_dir)
        print(f"indexed {repo} in {(time.perf_counter() - started) * 1000:.0f} ms", file=sys.stderr)
        for config, accumulated in zip(configs, accumulators):
            began = time.perf_counter()
            score_repo(index, repo_tasks[repo], config, args.runs, accumulated)
            print(
                f"  {repo:<20} {config.spec:<52} {config.label:<18} ({time.perf_counter() - began:.0f}s)",
                file=sys.stderr,
            )
        del index

    results = [_result(config, accumulated) for config, accumulated in zip(configs, accumulators)]
    for result in results:
        print(
            f"{result.reranker:<52} passage={result.passage:<8} alpha={result.alpha:<4} k={result.rerank_k:<3} "
            f"ndcg@10={result.ndcg10:.4f} p50={result.p50_ms:.0f}ms",
            file=sys.stderr,
        )
    _print_table(results)

    if args.output:
        with open(args.output, "w") as handle:
            json.dump([asdict(r) for r in results], handle, indent=2)


if __name__ == "__main__":
    main()
