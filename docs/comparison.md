# One table: seven retrieval configurations on the same tree

Everything measured before this document was measured on a different tree, with a
different metric, or both: `docs/capsules.md` predates the `.hwk` template lane,
`docs/voyage.md` was run at `ebd5dd4` before templates were indexed, and all of it
reports NDCG@10 only. NDCG is the right number for tuning and the wrong number for a
human: it says how good the whole ranked list is, not whether the answer was on the
screen.

This document is one run of each configuration, all on the same working tree, all
over the same 90 queries, reporting hit rates beside NDCG.

**hit@k = the fraction of queries where at least one relevant file appeared at rank k
or better.** It ignores how many relevant files a query has and how they are ordered
inside the cutoff. hit@1 is "the first result was right"; hit@5 is "the answer was in
the visible list"; hit@10 is "the answer came back at all".

## The configurations

| run | what it is |
| --- | ---------- |
| A | `ZEMBLE_CAPSULE=off` - the pre-capsule pipeline, i.e. what semble does |
| B | zemble defaults: potion-code-16M plus full context capsules in dense and BM25 |
| C | `ZEMBLE_EMBEDDER=voyage:voyage-4-lite@1024` |
| D | `ZEMBLE_EMBEDDER=voyage:voyage-code-4@1024` |
| E | `ZEMBLE_RERANKER=voyage:rerank-2.5`, `ALPHA=0.7`, `K=50`, default embedder |
| F | C + E: the `voyage-4-lite` embedder *and* `rerank-2.5` |
| G | C + `voyage:rerank-2.5-lite`, same alpha and K |

All seven ran the javaweb local set (`benchmarks/local/`, 90 queries: 80 answered by a
Java file, 10 by a Hawkeye template), on the same 77,092-chunk index of the same
worktree, with the same `.zembleignore` and the same `.hwk` lane in effect.

## All 90 queries

| run | NDCG@10 | hit@1 | hit@5 | hit@10 | p50 | cost per query |
| --- | ------: | ----: | ----: | -----: | --: | -------------: |
| A Semble-equivalent (capsules off) | 0.5390 | 0.467 | 0.644 | 0.667 | 89 ms | $0 |
| B Zemble local (defaults) | 0.5569 | 0.500 | 0.622 | 0.689 | 95 ms | $0 |
| C voyage-4-lite@1024 | 0.6803 | 0.589 | 0.811 | 0.856 | 350 ms | $0.0000002 |
| D voyage-code-4@1024 | 0.6228 | 0.489 | 0.767 | 0.811 | 354 ms | $0.0000013 |
| E rerank-2.5 on the default embedder | 0.6912 | 0.622 | 0.767 | 0.822 | 578 ms | $0.000584 |
| F voyage-4-lite + rerank-2.5 | **0.7531** | **0.656** | 0.867 | **0.911** | 802 ms | $0.000574 |
| G voyage-4-lite + rerank-2.5-lite | 0.7380 | **0.656** | **0.878** | 0.900 | 796 ms | $0.000230 |

## What the fusion profile changed afterwards

The seven runs above were measured before embedders declared a fusion weight bonus
(`docs/embedders.md`). Hosted embedders now claim +0.15 of the RRF fusion by default, so
**C, D, F and G are no longer what those configurations produce today**. C was re-run on
the shipped default; the reranked rows were not.

| run | NDCG@10 | hit@1 | hit@5 | hit@10 |
| --- | ------: | ----: | ----: | -----: |
| C as measured above (bonus forced to 0) | 0.6803 | 0.589 | 0.811 | 0.856 |
| **C as shipped today (+0.15)** | **0.7102** | **0.611** | **0.833** | **0.867** |

The whole gain is one kind: `consumer` hit@5 goes 0.455 -> 0.636 and every other kind is
unchanged, which is the first time anything but a reranker has moved that column. On the
80 code queries C becomes 0.6896 / hit@5 0.812 (from 0.6568 / 0.787); the ten template
queries do not move (0.875 vs 0.868, hit@5 1.000 either way).

Re-running the two reranked rows was not worth another 5M rerank tokens for a row that
is already the recommendation; expect F and G to move less than C did, because the
reranker reorders the same window either way.

A and B are unaffected: the default embedder declares a bonus of 0.0, and B re-measured
at exactly 0.5569 / 0.500 / 0.622 / 0.689 on the current tree.

All the rows in this section were measured inside one 30-minute window, over a
77,094-to-77,098-chunk index, which matters: the javaweb corpus is a **live worktree**
that other work edits. Re-running B ninety minutes later, at 77,132 chunks, reads 0.5546
- the same hit rates, 0.002 of NDCG of pure corpus drift. Compare runs from the same
window; never a number here against one measured on another day.

## The 80 code queries only

The ten template queries are the ones whose annotated answer is a `.hwk` file. Pulling
them out is the fairest comparison against every number recorded before the template
lane existed.

| run | NDCG@10 | hit@1 | hit@5 | hit@10 |
| --- | ------: | ----: | ----: | -----: |
| A Semble-equivalent (capsules off) | 0.5188 | 0.463 | 0.600 | 0.625 |
| B Zemble local (defaults) | 0.5447 | 0.500 | 0.588 | 0.662 |
| C voyage-4-lite@1024 | 0.6568 | 0.562 | 0.787 | 0.838 |
| D voyage-code-4@1024 | 0.6425 | 0.512 | 0.787 | 0.812 |
| E rerank-2.5 on the default embedder | 0.6790 | 0.613 | 0.750 | 0.800 |
| F voyage-4-lite + rerank-2.5 | **0.7471** | **0.650** | 0.850 | **0.900** |
| G voyage-4-lite + rerank-2.5-lite | 0.7298 | 0.637 | **0.863** | 0.887 |

And the ten template queries on their own, which is a sample of ten and is reported
only because it is where the two embedders disagree most:

| run | A | B | C | D | E | F | G |
| --- | -: | -: | -: | -: | -: | -: | -: |
| NDCG@10 | 0.701 | 0.655 | 0.868 | 0.465 | 0.789 | 0.801 | 0.804 |
| hit@5 | 1.000 | 0.900 | 1.000 | 0.600 | 0.900 | 1.000 | 1.000 |

## hit@5 per query kind (all 90)

| run | symbol (24) | behavioural (29) | architecture (16) | bug-report (10) | consumer (11) |
| --- | ----------: | ---------------: | ----------------: | --------------: | ------------: |
| A | 1.000 | 0.655 | 0.438 | 0.400 | 0.364 |
| B | 1.000 | 0.621 | 0.438 | 0.400 | 0.273 |
| C | 1.000 | 0.862 | 0.688 | 0.800 | 0.455 |
| D | 0.958 | 0.793 | 0.688 | 0.800 | 0.364 |
| E | 1.000 | 0.828 | 0.562 | 0.500 | 0.636 |
| F | 1.000 | 0.897 | 0.750 | 0.700 | 0.818 |
| G | 1.000 | 0.897 | 0.750 | 0.800 | 0.818 |

`symbol` is saturated in every configuration but D: a bare identifier is a BM25
problem and BM25 already solves it. Every difference in the overall column is bought on
the other four kinds.

## What each row changed

**A - capsules off.** The pre-capsule pipeline: BM25 over the chunk body plus path
tokens, fused with a static embedding of the body alone. Note what is *not* turned off:
`.hwk` templates are still indexed, the workspace `.zembleignore` is still in effect,
and the javac fact files are still present, so this is "zemble minus capsules", not a
faithful reproduction of another tool. It answers 2 queries in 3 somewhere in the top
10 and gets the first result right slightly under half the time.

**B - defaults.** Adding the context capsule to both the dense document and the BM25
document buys +0.018 NDCG@10 and +0.033 hit@1, and it is the one row where a hit rate
moves the *wrong* way: hit@5 drops 0.644 -> 0.622. Capsules pull the right file to rank
1 more often and, on a handful of queries, push a mid-list right answer out of the top
five. hit@10 recovers (+0.022). On this set the capsule is a small, real, uneven gain -
the same shape `docs/capsules.md` measured, now visible as three separate numbers
instead of one.

**C - voyage-4-lite embedder.** The largest single move in the table: +0.123 NDCG@10 and
+0.189 hit@5 over B. Two queries in three that the default embedder could not put in the
top five are now there. `bug-report` - symptom-only queries with no identifier in them -
doubles, 0.400 -> 0.800 hit@5, which is exactly the kind of query a static order-free
embedder cannot serve. The price is latency: the query itself must be embedded over the
network, so p50 goes 95 ms -> 350 ms.

**D - voyage-code-4 embedder.** Worse than the lite model on everything except
`bug-report` parity, at six times the token price, reproducing the ordering
`docs/voyage.md` found at every width. New here: the ten template queries are where it
loses most (NDCG 0.465 vs C's 0.868, hit@5 0.600 vs 1.000). A code-specialised embedder
is worse at Hawkeye markup than a general one, and D is also the only run that drops a
`symbol` query out of the top five.

**E - rerank-2.5 on the default embedder.** A cross-encoder reading query and passage
together, over the top 50 fused candidates, blended 0.7 with the fused score. It is the
best hit@1 of the local-embedder rows (0.622) and it is the only thing in this table
that moves `consumer` queries - "which test proves this", "which layer calls this" -
from 0.273 to 0.636 hit@5. It costs the rerank round trip: p50 578 ms. It does not fix
`bug-report` (0.400 -> 0.500) the way a better embedder does, because the reranker can
only reorder what fusion already retrieved.

**F - both.** The combination never measured before, and the best row on NDCG@10 (0.753),
hit@1 (0.656) and hit@10 (0.911). The two mechanisms fix different queries: the embedder
gets the right file into the window, the reranker puts it at the top, and `consumer`
lands at 0.818 hit@5 where B had 0.273. Two API round trips per query, p50 802 ms.

**G - rerank-2.5-lite instead.** Statistically indistinguishable from F on this set -
it wins hit@5 (0.878 vs 0.867) and `bug-report` (0.800 vs 0.700), loses hit@10 and
NDCG@10 - for 40% of F's per-query cost. On 90 queries a 0.011 difference in hit@5 is
one query.

## Cost

Cost per query is the **steady-state** figure: what one query sends over the wire once
the index exists. It is the query embedding (10.8 tokens, measured) plus, where a
reranker is on, one rerank request of 50 passages (11,475 tokens, measured). List
prices: `voyage-4-lite` $0.02/M, `voyage-code-4` $0.12/M, `rerank-2.5` $0.05/M,
`rerank-2.5-lite` $0.02/M.

Building the index is separate and one-off. These runs paid for only 3,111 chunks - the
`.hwk` templates, absent from a cache built before the template lane - at 563,135 tokens:
$0.011 with `voyage-4-lite`, $0.068 with `voyage-code-4`. A cold full index of this
workspace is 15.5M tokens, i.e. $0.31 and $1.86 respectively (`docs/voyage.md`).

The whole seven-run programme in this document consumed 16.7M tokens for **$0.70** at
list price, inside Voyage's complimentary tier.

| run | embedder requests / tokens | reranker requests / tokens | run cost |
| --- | ---: | ---: | ---: |
| A | - | - | $0 |
| B | - | - | $0 |
| C | 1,069 / 563,135 | - | $0.0113 |
| D | 475 / 563,135 | - | $0.0676 |
| E | - | 450 / 5,253,965 | $0.2627 |
| F | 450 / 4,850 | 450 / 5,163,862 | $0.2583 |
| G | 450 / 4,850 | 450 / 5,163,276 | $0.1034 |

Every query is issued five times per run for the latency median, so the request counts
are 5x what one pass over the set costs; `--latency-runs` lowers that. C and D show the
same token total because they embedded the same 3,111 uncached chunks; the request
counts differ only in how the client batched them.

## Caveats

- **This is our own evaluation set.** 90 hand-written, hand-verified queries over one
  40-repo Java workspace, built by the same people who tuned the retrieval. It is not a
  public benchmark and it is not neutral.
- **n is small.** On 90 queries one query is 1.1 points of hit rate; on the 11 `consumer`
  queries one query is 9 points. Treat any per-kind difference under ~0.10 as noise, and
  read F vs G as a tie.
- **Absolute numbers do not match `docs/voyage.md`.** That document was measured at
  `ebd5dd4`, before `.hwk` templates were indexed, on the 80-query set. Templates both
  add 10 queries and compete with the annotated Java files for the other 80, which
  re-baselined the default configuration from 0.5669 to 0.5569. Only the deltas carry
  over between the two documents.
- **Latencies come from one loaded machine**, sequentially, over a residential
  connection, and every hosted row is dominated by network round trips. p50 here ranks
  the configurations; it does not predict anyone else's wall clock.
- **hit@k is a coarser instrument than NDCG by design.** It cannot tell "the answer was
  at rank 1" from "the answer and four distractors filled the top five", and a query
  with six relevant files scores exactly like a query with one. It is reported *beside*
  NDCG, never instead of it.
- **C, D, F and G predate the fusion profile.** Hosted embedders now weight the dense
  lane +0.15 by default; the section above re-measures C on the shipped configuration.
- **One run each.** No configuration was repeated, so nothing here has an error bar.
  Retrieval is deterministic given the index, so the quality numbers are stable; the
  latency numbers are not.
- **No recommendation is drawn here.** The defaults are unchanged; this is the
  measurement, not a decision.

Result JSONs, including every query's ranks and hit flags, are in `benchmarks/results/`
as `local-javaweb-zemble-hybrid-<A..G>-*-b46840360106.json`.
