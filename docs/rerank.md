# Pairwise reranking

Hybrid retrieval scores a query against a chunk without ever putting the two in the same
forward pass: BM25 counts shared tokens, the static embedder compares two vectors that were
computed independently. A **cross-encoder** does the thing neither can - it reads the query
and the passage together and answers one question, "does this passage answer this query" -
and it costs a transformer forward pass per candidate.

zemble runs one after the other: fusion and the ranking heuristics pick the head of the
list, and a reranker is allowed to reorder that head only. It is **off by default**; the
measurements below are why.

## The seam

`Reranker` is a two-member protocol - `model_id`, and `score(query, passages) -> list[float]`
with higher meaning more relevant. Scores are only ever compared inside one call, so an
implementation may return logits, probabilities or anything else monotonic in relevance.

Specs are resolved by `zemble.rerank.registry`, the same shape the embedder seam uses:

```
none                 # default: no rerank pass at all
cross:<hf-model>     # local cross-encoder through transformers
voyage:<model>       # the Voyage rerank API
```

The spec comes from `--reranker` on `zemble search`, else `ZEMBLE_RERANKER`. An unknown
scheme is a loud `RerankerSpecError`, never a silent fall back to no reranking. Three knobs
tune the pass, all read from the environment (`RerankSettings.from_env`):

| Knob | Env var | Default | Meaning |
| --- | --- | --- | --- |
| window | `ZEMBLE_RERANK_K` | 50 | How many head candidates are rescored. |
| blend | `ZEMBLE_RERANK_ALPHA` | 1.0 | Weight of the reranker score against the fused score. |
| passage | `ZEMBLE_RERANK_PASSAGE` | `context` | `context` = capsule + content, `content` = the code alone. |

`cross:` needs the optional extra: `pip install 'zemble[rerank]'` (torch + transformers).
Importing zemble never imports torch, and building a `cross:` reranker does not load the
model - the first `score()` call does. A test asserts this.

## Where it hooks in

In `zemble.search.search`, after RRF fusion, the multi-chunk boost, the identifier boost and
the path penalties have produced the ranked list. With a reranker configured the pipeline
ranks `max(top_k, rerank_k)` candidates instead of `top_k`, rescores that head, and truncates
to `top_k` afterwards; without one, nothing about the pipeline changes.

The blend is

```
final = alpha * normalized_rerank + (1 - alpha) * normalized_fused
```

with both sides min-max normalized over the window alone (an all-equal side normalizes to
zeros, never to a division by zero). `alpha = 1.0` hands the head to the reranker outright;
`alpha = 0.7` lets the fused score keep a third of the vote, which turns out to matter.

The returned score column is the window's own original scores handed out in the new order,
not the blend: the blend lives on a scale of its own and emitting it would leave the
reranked head incomparable with the untouched tail below it. Every pass logs its own timing
(`zemble.rerank.apply`, INFO).

## What was measured

Two sets, both through `benchmarks/rerank_sweep.py`, which scores every configuration
against a repo while its index is loaded so a grid costs one indexing pass:

- **javaweb** - 80 hand-verified queries over the local multi-repo Java workspace
  (`benchmarks/local/`), the hard set: NDCG@10 0.5669 before this step.
- **upstream** - the pinned 63-repo, ~1250-query benchmark.

Both figures are means over repos of that repo's own mean, which is what
`benchmarks.run_benchmark` reports; the sweep reproduces the recorded javaweb baseline
(0.5669 vs the recorded 0.567) exactly.

### javaweb: the full grid on the smallest model

`cross-encoder/ms-marco-MiniLM-L-6-v2` (22M), every combination of passage shape, blend
weight and window. The E0 row is the current tree with no reranker.

| passage | alpha | k | NDCG@10 | architecture | behavioural | bug-report | consumer | symbol | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _E0 (none)_ | - | - | 0.5669 | 0.416 | 0.570 | 0.383 | 0.147 | 0.978 | 130 |
| context | 1.0 | 20 | 0.5703 | 0.368 | 0.585 | 0.463 | 0.268 | 0.909 | 2115 |
| context | 1.0 | 50 | 0.5760 | 0.354 | 0.573 | 0.450 | 0.398 | 0.897 | 2558 |
| context | 0.7 | 20 | 0.5850 | 0.414 | 0.574 | 0.463 | 0.256 | 0.952 | 1083 |
| **context** | **0.7** | **50** | **0.5909** | 0.362 | 0.595 | 0.463 | 0.329 | 0.952 | 2588 |
| content | 1.0 | 20 | 0.5015 | 0.404 | 0.583 | 0.389 | 0.257 | 0.651 | 782 |
| content | 1.0 | 50 | 0.4749 | 0.383 | 0.545 | 0.356 | 0.276 | 0.615 | 1917 |
| content | 0.7 | 20 | 0.5684 | 0.453 | 0.610 | 0.405 | 0.202 | 0.869 | 790 |
| content | 0.7 | 50 | 0.5671 | 0.441 | 0.588 | 0.381 | 0.242 | 0.892 | 1884 |

Three things the grid settles:

1. **The capsule is what makes the reranker usable.** Content-only loses 0.07-0.10 NDCG
   against the same settings with the capsule, and the damage is concentrated in `symbol`
   (0.978 -> 0.615 at alpha 1.0): a cross-encoder trained on prose cannot tell that a bare
   method body is the definition of the identifier being asked for, and the capsule's path,
   package and signature is exactly the evidence it needs.
2. **Handing the head over outright is worse than a blend.** alpha 0.7 beats alpha 1.0 at
   every window and both passage shapes. The fused score already encodes zemble's
   identifier and path priors, and a general-purpose reranker overrides them badly.
3. **A wider window helps, and costs linearly.** k=50 beats k=20 with the capsule, and
   doubles the latency.

### javaweb: other models

The grid above was run in full only on the smallest model; larger ones were run at its
winning setting only (capsule passage, alpha 0.7).

| model | params | passage | alpha | k | NDCG@10 | p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| _E0 (none)_ | - | - | - | - | 0.5669 | 130 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 22M | context | 0.7 | 50 | 0.5909 | 2588 |
| NamanAgnih0tri/code-reranker-miniLM-staqc | 22M | context | 1.0 | 20 | 0.5255 | 1017 |
| mixedbread-ai/mxbai-rerank-xsmall-v1 | 71M | context | 0.7 | 50 | not measured | - |
| BAAI/bge-reranker-base | 278M | - | - | - | not measured | - |
| Alibaba-NLP/gte-reranker-modernbert-base | 149M | - | - | - | not measured | - |

`code-reranker-miniLM-staqc` is the one code-adapted cross-encoder on the Hub that is both
small enough and permissively licensed (Apache-2.0; it is `ms-marco-MiniLM-L-6-v2`
fine-tuned on StaQC's Stack Overflow question/code pairs). It scores **below the E0
baseline** at the setting where its unadapted parent scores above it: Stack Overflow
snippets are not repository code, and the fine-tune appears to have cost more than it
bought. The other code-labelled rerankers on the Hub are either 4B causal models
(`hq-bench/coreb-code-reranker` is a Qwen3-4B) or have single-digit download counts.

The three larger models were dropped mid-sweep on a wall-clock budget: at 1-3 s per query
for a 22M model, a 149M-278M model cannot come close to the latency bar even if it wins on
quality, so measuring it would change nothing about the decision below.

### Upstream 63-repo set: not run

The upstream regression check exists to justify flipping the default on. It was started
(E0 plus the best javaweb configuration in one pass) and stopped at 29 of 63 repos: the
javaweb latency alone already disqualifies every configuration from being the default, so
no upstream number could change the outcome. Nothing in this step alters the no-reranker
path, so the recorded upstream baseline still describes the shipped behaviour.

## Latency

Measured on this workstation, CPU only, median over queries of each query's own median of
three runs, on a machine that was not otherwise idle - the same configuration measured
1083 ms and 2588 ms at k=20 and k=50 while a Java build was running, and the baseline row
was 130 ms. Treat these as an order of magnitude, not a benchmark.

| k | passage | p50 ms |
| --- | --- | --- |
| - (no reranker) | - | 130 |
| 20 | content | ~790 |
| 20 | context | ~1100 |
| 50 | content | ~1900 |
| 50 | context | ~2600 |

The shape is what matters: cost is linear in the window, roughly 20-50 ms per candidate for
a 22M cross-encoder over 512-token pairs, against a **130 ms** whole-query baseline. The
bar for this step was a warm p50 under 300 ms. A 22M model over 50 candidates misses it by
an order of magnitude, and even k=20 with content-only passages - the cheapest useful
configuration - misses it by 2.6x while giving up most of the quality.

## Decision

**The default stays `none`.** The best configuration clears the quality bar
(0.5909 vs 0.5669, +0.024 >= +0.02) but misses the latency bar by roughly 9x, and the
upstream regression check was therefore not completed. The pass is kept, wired and tested,
and is one flag away:

```bash
pip install 'zemble[rerank]'
export ZEMBLE_RERANK_ALPHA=0.7        # the capsule passage and k=50 are already the defaults
zemble search "how are conditional GETs handled" . --reranker cross:cross-encoder/ms-marco-MiniLM-L-6-v2
```

Recommended setting when you do turn it on: **capsule passage, alpha 0.7, k=50**.

`--reranker` runs the search in the CLI process rather than through a warm daemon, because
the daemon protocol carries no reranker field; `ZEMBLE_RERANKER` in a daemon's own
environment is the way to rerank on the daemon path.

The realistic path to an on-by-default pass is a **hosted reranker** (`voyage:<model>`),
where the model runs on someone else's accelerator and the cost is one network round trip
rather than seconds of local CPU. That has now been measured - see **`docs/voyage.md`**.
Short version: `voyage:rerank-2.5` at capsule/0.7/50 is javaweb **+0.1135** (0.567 ->
0.680) with the upstream set unmoved (-0.0006), so it passes both quality bars by a wide
margin, at p50 570 ms. It still misses the 300 ms latency bar, and no hosted model can
pass it - one 50-passage round trip is 309 ms on its own - so the default stays `none`,
but a hosted reranker is now the configuration to reach for when quality matters.

### What it wins and what it costs, per query kind

Never read the headline alone - the win is not uniform:

| kind | E0 | best config | delta |
| --- | --- | --- | --- |
| consumer | 0.147 | 0.329 | **+0.182** |
| bug-report | 0.383 | 0.463 | **+0.080** |
| behavioural | 0.570 | 0.595 | +0.025 |
| symbol | 0.978 | 0.952 | **-0.026** |
| architecture | 0.416 | 0.362 | **-0.054** |

The reranker earns its keep exactly where fusion is weakest - symptom-only bug reports and
"which layer consumes this" questions, the two kinds where the query shares no vocabulary
with the target - and it damages the two kinds zemble's own heuristics are tuned for. A
`symbol` query is a solved problem at 0.978, and the cross-encoder can only lose there.
That asymmetry, not the average, is the argument for a per-query switch rather than a
global default.
