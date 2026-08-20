# Voyage AI: hosted embeddings and hosted reranking, measured

Step 1 shipped a Voyage embedding client and step 7 a Voyage rerank client, both
tested against a fake server and both unmeasured: there was no API key on this
machine. There is one now. This document is what the two hosted models actually do
inside zemble's hybrid pipeline, on the same two evaluation sets every other step was
measured on, and what it costs.

Two questions were being asked:

1. **Does a stronger embedder matter inside a hybrid pipeline?** zemble fuses BM25
   with a static 16M-parameter Model2Vec embedder; the dense lane is deliberately the
   cheap one. Replacing it with a frontier code embedder is the direct test.
2. **Is a hosted reranker the path to reranking on by default?** A local
   cross-encoder cleared the quality bar and missed the latency bar by 9x
   (`docs/rerank.md`). A hosted one runs on someone else's accelerator.

The sets:

- **javaweb** - 80 hand-verified paraphrased queries over the 40-repo Java workspace
  (`benchmarks/local/`). The hard set.
- **upstream** - the pinned 63-repo, ~1250-query public benchmark.

Both figures are means over repos of that repo's own mean, which is what
`benchmarks.run_benchmark` reports. Every run below re-established its own E0 on the
tree it ran on rather than trusting a recorded one.

All of it was measured on the tree at `ebd5dd4`, before the `.hwk` template lane landed
(step 9). That step re-baselines the 80 original javaweb code queries at 0.545 because
templates now compete with the annotated Java files; the E0 rows here are 0.5669, the
pre-template baseline, and every Voyage row is against that same tree. The deltas are
what carry over, not the absolute numbers.

## Part A: embeddings

### javaweb

One paid pass built the `voyage-code-4` index at 2048 dimensions; 1024, 512 and 256
were then **derived from the sqlite cache by Matryoshka slicing** (`docs/embedders.md`),
which is a claim the run proves rather than assumes: rebuilding at 512 and at 256 made
**zero** API requests, and 1024 made exactly one (21 chunks of workspace drift between
the two runs). `voyage-4-lite` is a second paid pass.

| embedder | dims | NDCG@10 | architecture | behavioural | bug-report | consumer | symbol | q-p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _E0_ `model2vec:potion-code-16M-v2` | 256 | 0.5669 | 0.416 | 0.570 | 0.383 | 0.147 | 0.978 | 11 ms |
| `voyage:voyage-code-4@2048` | 2048 | 0.6668 | 0.524 | 0.694 | 0.617 | 0.196 | 1.000 | 488 ms |
| `voyage:voyage-code-4@1024` | 1024 | 0.6468 | 0.493 | 0.647 | 0.653 | 0.202 | 0.982 | 398 ms |
| `voyage:voyage-code-4@512` | 512 | 0.6318 | 0.471 | 0.615 | 0.657 | 0.183 | 0.985 | 349 ms |
| `voyage:voyage-code-4@256` | 256 | 0.6363 | 0.506 | 0.624 | 0.651 | 0.202 | 0.960 | 252 ms |
| **`voyage:voyage-4-lite@1024`** | 1024 | **0.6687** | 0.538 | 0.671 | 0.617 | 0.247 | 1.000 | 267 ms |

Four things this settles:

1. **A stronger embedder matters, and by a lot.** +0.10 NDCG@10 on a set where every
   step so far moved 0.01-0.035. It is by far the largest single jump measured on the
   javaweb set.
2. **The win is concentrated where the static model is blind.** `bug-report`
   (symptom-only queries, no shared vocabulary with the target) goes 0.383 -> 0.617,
   `behavioural` 0.570 -> 0.694, `architecture` 0.416 -> 0.538. `symbol` was already
   0.978 and reaches 1.000. Nothing regresses.
3. **`consumer` is still the floor** (0.147 -> 0.25 at best). "Which layer consumes
   this mechanism" is not an embedding problem - see Part B, where a reranker triples it.
4. **The cheap model wins.** `voyage-4-lite` scores at or above `voyage-code-4` at
   every width tried, at **1/6th the price** ($0.02 vs $0.12 per million tokens) and
   with a query round trip 130 ms shorter. The code-specialised model is not the one
   to reach for here.

Width buys little: 256 dimensions retains 0.636 of the 0.667 available at 2048, and
truncating is free once the wide vectors are bought. Latency falls with width because
the round trip carries fewer floats, not because the search got faster.

### The cost of an index

| pass | chunks | requests | tokens | wall | list price |
| --- | --- | --- | --- | --- | --- |
| `voyage-code-4` javaweb index | 73,957 | 578 | 15,526,808 | 1436 s | $1.86 |
| `voyage-4-lite` javaweb index | 73,978 | 578 | 15,531,476 | 1058 s | $0.31 |
| every derived width (1024/512/256) | 73,978 | 0-1 | 1,448 | 25 s each | $0.00 |

The whole programme in this document consumed well under the 200M-token complimentary
tier, so nothing was actually billed; the list-price column is what it would cost a
user who has spent theirs. `docs/embedders.md` previously estimated the javaweb index
at "roughly 31M tokens" - the measured figure is half that, and has been corrected.

Warm re-indexing is free: a second build over an unchanged tree makes zero API
requests, which the protoblast smoke run confirmed against the live API (7,565 chunks,
60 requests cold, 0 warm).

### Upstream 63 repos

`voyage-code-4@1024`, one paid pass, against an E0 re-run on the current tree
(0.8643, which reproduces the recorded capsule-era baseline exactly).

**59 of the 63 repos completed.** The four that did not - `zig`, `zig-clap`, `zls`,
`zod` - are not a result about those repos: the run degraded to under one embedded
chunk per second part-way through `zig` (26k chunks), while a fresh request from the
same machine to the same endpoint still returned 128 vectors in 2.4 s. Something in a
15-hour single process on this workstation, not the API, is the cause. The comparison
below is therefore restricted to the 59 repos both runs cover, with E0 recomputed over
exactly those repos, so it is like-for-like.

| | E0 (59 repos) | `voyage-code-4@1024` | delta |
| --- | --- | --- | --- |
| NDCG@10 | 0.8667 | **0.8852** | **+0.0186** |
| q-p50 | 7.2 ms | 257.2 ms | +250 ms |

| language | repos | E0 | voyage | delta | | language | repos | E0 | voyage | delta |
| --- | --- | --- | --- | --- |-| --- | --- | --- | --- | --- |
| bash | 3 | 0.867 | 0.880 | +0.013 | | lua | 3 | 0.870 | 0.883 | +0.013 |
| c | 3 | 0.774 | 0.853 | +0.078 | | php | 3 | 0.885 | 0.929 | +0.045 |
| cpp | 3 | 0.911 | 0.862 | -0.049 | | python | 9 | 0.875 | 0.887 | +0.012 |
| csharp | 3 | 0.885 | 0.892 | +0.007 | | ruby | 3 | 0.898 | 0.922 | +0.024 |
| elixir | 3 | 0.915 | 0.896 | -0.019 | | rust | 3 | 0.856 | 0.841 | -0.014 |
| go | 3 | 0.910 | 0.939 | +0.029 | | scala | 3 | 0.915 | 0.924 | +0.009 |
| haskell | 3 | 0.769 | 0.856 | +0.087 | | swift | 3 | 0.858 | 0.911 | +0.053 |
| java | 3 | 0.831 | 0.845 | +0.014 | | typescript | 2 | 0.790 | 0.825 | +0.035 |
| javascript | 3 | 0.915 | 0.932 | +0.017 | | kotlin | 3 | 0.836 | 0.835 | -0.001 |

Voyage is **above baseline on both sets**, so the rule that gates a default change is
satisfied on quality. The upstream gain is a tenth of the javaweb gain, which is the
expected shape: at 0.867 there is little headroom, the queries are short and lexical,
and BM25 already answers most of them. Four of eighteen languages regress, led by
`cpp` (-0.049, almost entirely `abseil-cpp` at -0.16); the biggest wins are `haskell`
(+0.087), `c` (+0.078) and `swift` (+0.053). Per repo the spread runs from -0.16
(`abseil-cpp`) to +0.11 (`xmonad`, `libuv`).


### Fusion weights

`zemble.ranking.weighting` gives the dense lane a fixed share of the RRF fusion: 0.30
for a query that looks like a symbol lookup, 0.50 otherwise. Those constants were tuned
around a static embedder. With a stronger dense lane the optimum should move, and it
does - but only for the stronger lane, which is what makes the comparison worth having.

| semantic weight (symbol / NL) | `voyage-code-4@1024` | default `potion-code-16M` |
| --- | --- | --- |
| 0.30 / 0.50 (shipped) | 0.6468 | **0.5669** |
| 0.45 / 0.65 (+0.15) | **0.6623** | 0.5591 |
| 0.60 / 0.80 (+0.30) | 0.6572 | 0.5480 |

Per kind at the winning `voyage-code-4@1024` setting (+0.15): architecture 0.502,
behavioural 0.653, bug-report 0.701, consumer 0.251, symbol 0.982 - the gain is in
`bug-report` (+0.048) and `consumer` (+0.049), and `symbol` does not move at all.

The shipped weights are **exactly right for the shipped embedder** - every increase
makes it worse, monotonically - and **too low for a hosted one**, worth about +0.016.
This is a per-embedder knob, not a global one, and no default is changed here: the
weight that wins with Voyage is the weight that loses with the default.

## Part B: the hosted reranker

Same pipeline position as `docs/rerank.md`: fusion and the ranking heuristics pick the
head, the reranker reorders that head only. The local cross-encoder's winning setting
(capsule passage, blend alpha 0.7, window k=50) is the starting point, and the local
model's other finding - that handing the head over outright is worse than a blend -
holds for the hosted models too (alpha 1.0 scores 0.6704 against alpha 0.7's 0.6804).

### javaweb, default local embedder

| reranker | alpha | k | NDCG@10 | architecture | behavioural | bug-report | consumer | symbol | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _none (E0)_ | - | - | 0.5669 | 0.416 | 0.570 | 0.383 | 0.147 | 0.978 | 92 ms |
| `cross:ms-marco-MiniLM-L-6-v2` (local, `docs/rerank.md`) | 0.7 | 50 | 0.5909 | 0.362 | 0.595 | 0.463 | 0.329 | 0.952 | 2588 ms |
| `voyage:rerank-2.5-lite` | 0.7 | 20 | 0.6331 | 0.442 | 0.658 | 0.426 | 0.400 | 0.966 | 425 ms |
| `voyage:rerank-2.5-lite` | 0.7 | 50 | 0.6672 | 0.481 | 0.702 | 0.426 | 0.492 | 0.971 | 542 ms |
| `voyage:rerank-2.5` | 0.7 | 20 | 0.6439 | 0.457 | 0.667 | 0.463 | 0.400 | 0.968 | 436 ms |
| **`voyage:rerank-2.5`** | **0.7** | **50** | **0.6804** | 0.484 | 0.728 | 0.463 | 0.500 | 0.968 | 570 ms |

**+0.1135 NDCG@10** over no reranker, five times the local cross-encoder's +0.024, and
without the local model's damage: architecture goes 0.416 -> 0.484 where the local
cross-encoder dropped it to 0.362, and `symbol` gives up 0.010 rather than 0.026.
`consumer` - the kind zemble has never been able to answer - goes **0.147 -> 0.500**.

A hosted reranker is a genuinely different quality tier from a 22M local one, and it
is the single largest improvement measured anywhere in this fork. The gap between the
two Voyage models is real but small (0.668 vs 0.680) and `rerank-2.5-lite` costs 2.5x
less.

### What a reranked query costs

At k=50 with the capsule passage, one javaweb query sends **11,841 tokens** in one
request (50 passages plus the query; the client batches at 100 documents, so k=50 is
always a single round trip).

| model | $/M tokens | per query | per 1,000 queries |
| --- | --- | --- | --- |
| `rerank-2.5` | $0.05 | $0.00059 | $0.59 |
| `rerank-2.5-lite` | $0.02 | $0.00024 | $0.24 |

Under a tenth of a cent a query, and the first 200M tokens are free - roughly 17,000
free javaweb queries.

### Upstream 63 repos: does it regress?

The on-by-default rule needs the upstream set not to move by more than 0.003. It does
not move at all.

| reranker | NDCG@10 | architecture | semantic | symbol | p50 |
| --- | --- | --- | --- | --- | --- |
| none | 0.8643 | 0.824 | 0.857 | 0.952 | 5 ms |
| `voyage:rerank-2.5` context/0.7/50 | 0.8637 | 0.794 | 0.866 | 0.951 | 406 ms |

**-0.0006 overall**, inside the tolerance. The composition is worth reading: the
reranker *gains* on `semantic` (+0.009) and gives it back on `architecture` (-0.030),
which is the same asymmetry the local cross-encoder showed, only much smaller. On a
set where retrieval is already at 0.864 there is very little headroom; the reranker
earns its keep on hard sets, not on easy ones.

### The three bars

| bar | threshold | best hosted config | verdict |
| --- | --- | --- | --- |
| javaweb quality | >= +0.02 | +0.1135 | **pass**, 5.7x over |
| upstream regression | <= 0.003 | -0.0006 | **pass** |
| warm p50 | < 300 ms | 570 ms | **fail** |

The latency bar is missed by 1.9x, and it is not an implementation problem: measured in
isolation, one 50-passage rerank call to Voyage from this machine takes **309 ms
(rerank-2.5) / 328 ms (rerank-2.5-lite)** median, so the round trip alone exceeds the
whole-query budget before any retrieval work is counted. A k=20 window brings the
query to 436 ms and gives up a third of the win. No hosted configuration can pass a
300 ms bar over a public API at this network distance; the bar and a hosted reranker
are incompatible, not merely unmet.

**Decision: the default stays `none`, and this is now a recommendation to *use* the
flag rather than a curiosity.** The local cross-encoder was hard to recommend at
2.6 s a query for +0.024; `voyage:rerank-2.5` is +0.11 for half a second and a
twentieth of a cent, which is a trade many callers will take.

## Decisions

### Embeddings: the default stays local, Voyage becomes the documented upgrade

`model2vec:minishlab/potion-code-16M-v2` stays the default. Not because it is better -
it is not, by 0.10 NDCG@10 on the hard set - but because the properties it is the
default *for* are the ones Voyage gives up:

- **No network.** A query goes from 11 ms to 267-488 ms, entirely in the round trip to
  embed the query text. That is a 25-40x latency regression on every search, and it
  makes zemble unusable offline.
- **No key, no account, no bill.** The default has to work on a fresh checkout.
- **The code never leaves the machine.** Every indexed chunk is sent to a third party.
  That is a decision a user makes, not one a search tool makes for them.

But the quality gap is too large to leave undocumented. `voyage:voyage-4-lite@1024` is
**the recommended upgrade**: it beat the code-specialised `voyage-code-4` at every
width, costs $0.02 per million tokens, and indexes the whole javaweb workspace for
about 31 cents. Reach for it when retrieval quality matters more than latency and the
code may leave the machine.

### The reranker: still off by default, but now worth turning on

`voyage:rerank-2.5` at the shipped defaults plus `ZEMBLE_RERANK_ALPHA=0.7` is
**+0.1135 NDCG@10** on javaweb, does not regress upstream, and costs six hundredths of
a cent a query - but adds roughly 450 ms. On by default would mean paying that on every
query, including the `symbol` lookups that are already at 0.978 and can only lose. Off
by default, prominently documented, is the right shape.

### Turning each on

```bash
export VOYAGE_API_KEY=pa-...

# Hosted embeddings. The first index pays for every chunk; later runs pay only for
# what changed, because the sqlite cache is content-addressed.
export ZEMBLE_EMBEDDER=voyage:voyage-4-lite@1024

# Hosted reranking. k=50 and the capsule passage are already the defaults.
export ZEMBLE_RERANKER=voyage:rerank-2.5      # or voyage:rerank-2.5-lite, 2.5x cheaper
export ZEMBLE_RERANK_ALPHA=0.7

# Both are per-invocation flags too:
zemble search "how are conditional GETs handled" . \
  --embedder voyage:voyage-4-lite@1024 --reranker voyage:rerank-2.5
```

Switching `ZEMBLE_EMBEDDER` rebuilds the index: an index records its embedder, and
vectors from two models are never mixed. `--reranker` on the command line runs the
search in the CLI process rather than through the warm daemon; `ZEMBLE_RERANKER` in the
daemon's own environment is the way to rerank on the daemon path.

### Not changed

- **The fusion weights.** +0.15 on the dense lane wins with Voyage (+0.016) and loses
  with the default embedder (-0.008). It would have to be a per-embedder default, and
  no mechanism for that exists; it is not worth inventing one for 0.016.
- **The reranker default.** Still `none`.
- **The embedder default.** Still Model2Vec.
