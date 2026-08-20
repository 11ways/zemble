# Voyage AI: hosted embeddings and hosted reranking, measured

Step 1 shipped a Voyage embedding client and step 7 a Voyage rerank client, both
tested against a fake server and both unmeasured: there was no API key on this
machine. There is one now. This document is what the two hosted models actually do
inside zemble's hybrid pipeline, on the same two evaluation sets every other step was
measured on, and what it costs.

AIDEV-NOTE: every figure here predates the `.hwk` template lane. For one table of the
same configurations re-measured on the current tree, with hit@1/5/10 beside NDCG, see
[docs/comparison.md](comparison.md).

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
`zod` - are not a result about those repos. Part-way through `zig` (26k chunks) the
process degraded from ~40 embedded chunks per second to under one, then stopped
progressing entirely, while a fresh request from the same machine to the same endpoint
still returned 128 vectors in 2.4 s throughout. Something in a 15-hour single process
on this workstation, not the API, is the cause, and the run was stopped. The comparison
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
makes it worse, monotonically - and **too low for a hosted one**.

That is where this document stopped in its first pass: the finding was real and there
was no mechanism to act on it. There is one now, and the section below is the sweep that
chose its number.

## Part A2: the fusion profile

An embedder now **declares** how much of the fusion its dense lane earns beyond the
shipped weights (`semantic_weight_bonus`, beside `is_remote` on the embedder seam;
`docs/embedders.md`). Model2Vec declares 0.0, every HTTP provider declares 0.15, an
embedder that declares nothing gets 0.0, and `ZEMBLE_SEMANTIC_WEIGHT_BONUS` overrides
all of them. Nothing else in ranking changed: at bonus 0 the fusion is bit-identical to
what it was, which the first row of every table below reproduces exactly.

All of the runs in this part are on the current tree and the **90-query** javaweb set
(80 code + 10 template), so they are comparable with `docs/comparison.md` and not with
Part A's 80-query numbers.

### javaweb: choosing the bonus

| bonus | embedder | NDCG@10 | hit@1 | hit@5 | hit@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.00 | `model2vec:potion-code-16M-v2` | **0.5569** | 0.500 | 0.622 | 0.689 |
| +0.15 | `model2vec:potion-code-16M-v2` | 0.5533 | 0.489 | 0.600 | 0.689 |
| 0.00 | `voyage:voyage-4-lite@1024` | 0.6803 | 0.589 | 0.811 | 0.856 |
| +0.10 | `voyage:voyage-4-lite@1024` | 0.6986 | 0.600 | 0.822 | 0.856 |
| **+0.15** | **`voyage:voyage-4-lite@1024`** | **0.7102** | **0.611** | **0.833** | 0.867 |
| +0.20 | `voyage:voyage-4-lite@1024` | 0.7039 | 0.578 | 0.822 | **0.878** |
| 0.00 | `voyage:voyage-code-4@1024` | 0.6208 | 0.489 | **0.767** | **0.811** |
| +0.15 | `voyage:voyage-code-4@1024` | 0.6237 | 0.511 | 0.711 | 0.778 |

`voyage-4-lite` peaks at +0.15 and turns back down at +0.20, so the choice is not a tie
that has to be broken by preferring the smaller number; +0.15 wins outright by 0.012
over +0.10. Per kind it is the same shape Part A found on `voyage-code-4`, only larger:

| kind | 4-lite at 0 | 4-lite at +0.15 |
| --- | ---: | ---: |
| architecture | 0.527 | 0.543 |
| behavioural | 0.702 | 0.737 |
| bug-report | 0.572 | 0.626 |
| consumer | 0.247 | 0.361 |
| symbol | 1.000 | 0.985 |

`consumer` - the kind nothing but a reranker had ever moved - gains **+0.114** from a
fusion weight, and the whole cost is 0.015 of an already saturated `symbol`.

`voyage-code-4` is the one to be careful about: NDCG@10 moves +0.003, but hit@5 drops
0.767 -> 0.711 and hit@10 0.811 -> 0.778, i.e. five queries lose their answer from the
visible list while the ones that remain are ordered better. On the previous tree and the
80-query set the same setting was worth +0.016 NDCG (Part A above). So the bonus is not
uniformly good for every hosted model; it is clearly good for the recommended one, and
roughly neutral-to-mixed for the code-specialised one that is already not recommended.

### Upstream 63 repos, and the number Part A never had

`voyage-code-4` was measured upstream on 59 of 63 repos because a 15-hour single process
degraded to nothing. `voyage-4-lite` had **no** upstream number at all. It has one now,
over **all 63 repos**, because the paid pass was run as one short-lived process per repo
instead of one long one: no run exceeded four minutes, nothing degraded, and the three
repos the earlier attempt lost came back.

| configuration | NDCG@10 | hit@1 | hit@5 | hit@10 | architecture | semantic | symbol | p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| _E0_ `model2vec:potion-code-16M-v2` | 0.8643 | 0.759 | 0.967 | 0.992 | 0.824 | 0.857 | 0.952 | 7 ms |
| `voyage:voyage-4-lite@1024`, bonus 0 | 0.8878 | 0.795 | 0.983 | 0.996 | 0.849 | 0.880 | 0.970 | 267 ms |
| **`voyage:voyage-4-lite@1024`, +0.15** | **0.8898** | **0.796** | **0.987** | 0.996 | 0.849 | **0.884** | **0.971** | 267 ms |

E0 reproduces the recorded 0.8643 exactly, which is what makes the two rows above it
trustworthy. `voyage-4-lite` is **+0.0235** over the default embedder here, against
`voyage-code-4`'s +0.0186 over 59 repos - the same ordering the javaweb set found.

The bonus **does not regress upstream at all**: +0.0020, in the same direction as
javaweb. Per language it is within noise nearly everywhere, gaining on `bash` (+0.010),
`lua` (+0.011), `rust` (+0.009) and `kotlin` (+0.013), giving back on `zig` (-0.012) and
`csharp` (-0.010).

### The decision

The gate was: ship the bonus as the default for remote contextual embedders only if it
improves javaweb **and** does not regress upstream by more than 0.003 for the same
embedder.

| bar | threshold | measured | verdict |
| --- | --- | --- | --- |
| javaweb improves | > 0 | +0.0299 | **pass** |
| upstream regression | <= 0.003 | +0.0020 (an improvement) | **pass** |
| default embedder untouched | bit-identical | 0.5569 and 0.8643 reproduced | **pass** |

**Shipped: 0.15 for every HTTP-backed embedder, 0.0 for Model2Vec.** The default
configuration does not move by a single digit; a Voyage user gets +0.030 NDCG@10 on the
hard set for free, without a second request or a second index.

### What this pass consumed

| pass | tokens | list price |
| --- | ---: | ---: |
| upstream 63-repo `voyage-4-lite` document index (the paid one) | 28,859,496 | $0.58 |
| upstream query embeddings, both bonus runs | 7,970 | $0.00 |
| javaweb query embeddings, six runs | 95,156 | $0.00 |

**~29.0M tokens, about $0.58 at list price**, inside the complimentary tier. The second
upstream run cost nothing but its queries: the documents were already in the sqlite
cache, which is exactly the property that made measuring two fusion weights on a paid
index affordable.

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

- **The fusion weights** were left alone in this document's first pass, for want of a
  per-embedder mechanism. Part A2 built one and shipped +0.15 for hosted embedders; the
  default embedder's weights are unchanged, and were re-measured to prove it.
- **The reranker default.** Still `none`.
- **The embedder default.** Still Model2Vec.

## What the whole programme consumed

Every figure below is Voyage's own `usage.total_tokens`, accumulated by the clients.

| pass | tokens | model | list price |
| --- | --- | --- | --- |
| protoblast smoke index | 1,475,301 | voyage-code-4 | $0.18 |
| javaweb index @2048 | 15,526,808 | voyage-code-4 | $1.86 |
| javaweb drift re-embed @1024 | 1,448 | voyage-code-4 | $0.00 |
| javaweb index @1024 | 15,531,476 | voyage-4-lite | $0.31 |
| upstream index @1024 (59 of 63 repos) | ~34,000,000 | voyage-code-4 | ~$4.08 |
| query embeddings (~10,000 across every benchmark pass) | ~150,000 | both | $0.02 |
| upstream index @1024 (63 of 63 repos, Part A2) | 28,859,496 | voyage-4-lite | $0.58 |
| javaweb rerank sweeps (~560 queries) | ~6,000,000 | rerank-2.5 / -lite | ~$0.20 |
| upstream rerank check (~1,250 queries) | ~15,000,000 | rerank-2.5 | ~$0.75 |

Roughly **87M tokens**, comfortably inside the 200M complimentary tier, so nothing was
billed. At list price the whole programme would have been about **$7.40**, and the
single most expensive thing in it - a full javaweb index with the recommended
`voyage-4-lite` - is 31 cents.

Derived-width indexes are the free lunch: 1024, 512 and 256 all came out of the sqlite
cache by slicing the 2048-dimension vectors, at zero requests and zero tokens.

## Surprises worth keeping

1. **The code-specialised model lost to the general cheap one.** `voyage-4-lite` beat
   `voyage-code-4` on javaweb at every width, at a sixth of the price. "code" in a
   model name is not evidence.
2. **The reranker is a bigger lever than the embedder** on the hard set: +0.11 against
   +0.10, while leaving the index and its cost completely alone. Both together were not
   measured - the reranker sweep ran on the default embedder deliberately, to isolate it.
3. **256 dimensions is nearly free quality.** Slicing to 256 keeps 0.636 of the 0.667
   at 2048 - still +0.069 over the default embedder - at a quarter of the vector
   storage and the shortest query round trip of the four widths.
4. **`consumer` queries are a reranker problem, not an embedder problem.** The best
   embedder moved them 0.147 -> 0.247; the reranker moved them 0.147 -> 0.500.
5. **Paying for a big index in one `embed_documents` call was a real hazard.** Before
   this step the whole workspace was embedded in one call that wrote to the cache once,
   at the end: a failure or an OOM anywhere in a 24-minute paid run threw away every
   vector bought. It also accumulated the JSON response rows in memory - 6.5 GB peak at
   2048 dimensions. Both are fixed by flushing every 512 texts (0.75 GB peak, and a
   retry only pays for the unflushed tail).
