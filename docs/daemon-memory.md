# Daemon memory defect (2026-10-02)

## Bounds and defaults

`ZEMBLE_DAEMON_MAX_RSS_MB` defaults to `min(15% of MemTotal, 4096)` MiB and must
be positive. Loads and rebuilds reserve headroom before allocating, evicting idle
indexes in LRU order. Requests pin their serving roots. A rebuild that cannot fit
leaves its old generation available, reports the deferral in status, and retains
one full-rescan bit for a retry after 30 seconds. Initial loads without room return
a memory refusal, not an in-process fallback.

The budget counts private memory (`VmData`: heap and private writable mappings), and
the Linux allocation backstop is `RLIMIT_DATA` at that ceiling, or a stricter inherited
limit; status reports the effective lower ceiling. Index stores are mapped files, page
cache the kernel can drop, so they count against neither; counting them (as address
space, as this used to) made one large mapped index look like gigabytes of allocation and
refused a second root. A missed estimate still cannot allocate beyond the ceiling. An
allocation refusal during rebuilding preserves the old generation. A ceiling below
startup private memory refuses startup with a diagnostic. Detached daemons use one BLAS/OMP
thread and at most two glibc arenas; graph extraction uses one worker, not a fleet
of child interpreters.

Only one construction job runs daemon-wide, including persistence and graph work.
Memory-intensive requests are serialized instead of multiplying scratch by arbitrary
client concurrency. Warm searches still answer from an immutable generation during
an admitted rebuild; ping and status do not wait for it. Graph-backed requests share
the construction slot because ensuring a graph can build it.

One root has one resident content selection. Code+docs replaces a narrower code
selection, and subsequent code requests share its chunks, vectors and postings
through a content selector. Docs cannot appear in a code answer, including when the
caller supplies `paths` or `exclude`. BM25 corpus statistics are the covering index's,
as with an ancestor/subtree view. The eval below measures this intentional difference.
On disk too, one root keeps one index: the widest one stored (`cache.covering_content`);
saving it removes the narrower ones it covers, and `zemble clear orphans` removes those
an older zemble left. A wider index saved before a narrower one covers nothing: the
narrower one keeps answering until the wider one is synced and saved again.

The daemon rereads its env file at startup using an explicit path. An inherited
`_ZEMBLE_USER_ENV_LOADED` from a long-running MCP client no longer suppresses newly
added settings. Explicit shell environment values still override the file, and an
explicit `--max-indexes` overrides both. A file edit does not reconfigure an already
running daemon: restart that daemon, not every MCP client. Status prints the effective
index count and memory budget; JSON also includes `private_mb` and `quiet_seconds`.

## Causes

Saving an index did not release its mutable BM25 dictionaries, Python chunk objects,
or anonymous embedding matrix. Rebuilds copied and normalized entire vector matrices,
and several roots could own scratch at once. Code and code+docs for the same root had
separate resident stores. Range splicing scanned every previous run for every reused
file, quadratic work that prolonged generation ownership.

There was also a singleton native-memory growth mechanism. `watchfiles` cannot drain
its native event set while its async generator is suspended in an awaited rebuild
callback. Ignored `.class` output still enters that set. After a blocked job finishes,
turning the accumulated set into Python tuples creates another peak. Resolving every
ignored artifact through the facts matcher then prevents prompt drainage.

The callback now returns immediately, retaining one bounded change set per resident
root. Rebuilds wait for two seconds of relevant-event quiet and share one construction
slot. Facts matching rejects nonmatching basenames before expensive realpath work.
Ignore specs, facts config generations and declared-path caches are bounded. Rebuilt
vectors use temporary mappings and bounded normalization scratch; saved generations
are reloaded as `memmap` / `ChunkList` with no heap BM25 delta. Range splicing bisects
directly to overlapping runs. Eviction removes root metadata and pending work.

## Replays

Both scripts use deterministic offline 1024-dimensional embeddings, a private
cache/socket/env file under `/tmp/opencode`, a PID ownership check, a 7 GiB emergency
stop on only the owned daemon, and a `free -g` / 10 GiB available-RAM precondition.
No measurement opens or writes the real `~/.cache/zemble` indexes.

```sh
.venv/bin/python -m tests.measure_daemon_churn --revision 4d96f86 --output tests/daemon_churn_before.json
.venv/bin/python -m tests.measure_daemon_churn --output tests/daemon_churn_after.json
.venv/bin/python -m tests.measure_daemon_singleton --revision 4d96f86 --output tests/daemon_singleton_before.json
.venv/bin/python -m tests.measure_daemon_singleton --output tests/daemon_singleton_after.json
```

### Multiple roots

The generated root has 128k code chunks and is requested again as code+docs (136k),
plus a second code+docs root (68k). Six rounds overlap six clients with two seconds
of writes and three seconds of quiet. RSS is sampled every 20 ms, including builds
and a bounded settling window. Budget: 4096 MiB.

| RSS, MiB | 4d96f86 | Budgeted daemon |
| --- | ---: | ---: |
| After first code load | 1134.7 | 98.8 |
| After code+docs request | 2256.8 | 113.5 |
| After second root load | 2813.7 | 122.5 |
| Peak including initial loads | 3481.8 | 2143.4 |
| Peak during churn/settling | 3481.8 | 1984.8 |
| Final RSS | 2735.9 | 138.0 |
| Final anonymous memory | 2709.2 | 108.9 |
| Final resident variants / rebuilding / pending roots | 3 / 3 / 3 | 2 / 0 / 0 |

The duplicate code matrix alone was 500 MiB; code+docs was 531.2 MiB and the second
root 265.6 MiB. Baseline resident stores were anonymous `ndarray` vectors and
`SplicedChunks`, with full mutable BM25 state still present. The fixed stores were
`memmap` / `ChunkList`, with zero BM25 delta documents after saving. Summed child RSS
in the raw report includes short-lived identity/git processes and can double-count
inherited pages; the table reports only the daemon PID's RSS.

### One root and stale config

The singleton has 25,383 chunks and 1,757 files. After one real index rebuild, the
script holds the graph phase and creates/deletes 500k unique ignored build artifacts.
Tracing starts after the graph hold, isolating event backlog from construction.
It simulates a slow or writer-blocked graph job without touching Archdev or buying
embeddings. Tracing stops before drainage to avoid distorting the full-batch cost.

| Singleton | 4d96f86 | Budgeted daemon |
| --- | ---: | ---: |
| RSS before ignored churn, MiB | 239.1 | 112.2 |
| RSS after 500k artifacts while graph held, MiB | 583.8 | 251.7 |
| Peak RSS including backlog drain, MiB | 1286.5 | 465.8 |
| Raw events drained while held | 0 | 851085 |
| File's requested max_indexes / effective | 1 / 4 | 1 / 1 |
| File's requested budget / reported, MiB | 2048 / absent | 2048 / 2048 |

Baseline RSS rose at every 50k artifacts while new traced Python memory stayed at
0.0 MiB and no batches drained. The index did not grow. The fixed daemon drained
ignored batches during the same hold and plateaued around 285-294 MiB instead of
retaining all history. This proves a singleton native-event growth mechanism; it
does not claim to reproduce Archdev's exact 7.0 GiB or 10.9 GiB live-tree peaks.
The allocation backstop also covers native allocations Python tracing does not see.

## Retrieval check

Local Model2Vec, no hosted reranker, one latency repetition. E0 uses exported
`4d96f86` code, and E0/after use the same frozen input files with private index caches.
The 63 upstream repos were fetched at their pinned revisions into a private directory.
The local annotations use a frozen copy of current `zenit-workspace` because aeor's
old `javaweb` lacks the annotated declarations. August's local number is not
reproducible on this changed corpus; no annotation was tuned or rewritten. Ordinary
code-only before/after have identical per-query records on both sets: 1251 upstream
and 90 local queries.

| Eval / selection | NDCG@10 | hit@1 | hit@5 | hit@10 |
| --- | ---: | ---: | ---: | ---: |
| Upstream E0 and code-only after | 0.8651 | 0.7634 | 0.9666 | 0.9912 |
| Upstream code selector over code+docs | 0.8652 | 0.7634 | 0.9666 | 0.9912 |
| Local E0 and code-only after | 0.4961 | 0.4333 | 0.5778 | 0.6111 |
| Local code selector over code+docs | 0.5007 | 0.4444 | 0.5778 | 0.6111 |

| Local kind | E0 NDCG@10 | Shared NDCG@10 | E0 hit@1/5/10 | Shared hit@1/5/10 |
| --- | ---: | ---: | --- | --- |
| architecture | 0.3636 | 0.3636 | 0.3750 / 0.3750 / 0.4375 | same |
| behavioural | 0.4302 | 0.4317 | 0.2414 / 0.5862 / 0.6552 | same |
| bug-report | 0.3244 | 0.3613 | 0.3000 / 0.4000 / 0.4000 | 0.4000 / 0.4000 / 0.4000 |
| consumer | 0.0452 | 0.0452 | 0.0000 / 0.1818 / 0.1818 | same |
| symbol | 0.9422 | 0.9422 | 0.9583 / 0.9583 / 0.9583 | same |

Upstream has categories instead of `kind`: architecture NDCG 0.8263 and hit@1/5/10
0.7215/0.9616/0.9937; semantic 0.8574 -> 0.8575 and 0.7402/0.9592/0.9889; symbol
0.9486 and 0.9040/0.9867/0.9933. Full query records are under
`benchmarks/results/*daemon-memory-{e0,after,shared}-4d96f8634dd9.json`.
Mapped normalization tests assert bit identity for unit, nonunit, zero and near-zero
rows across block boundaries. Hosted Voyage quality was not reevaluated.

## Verification

Full suite: 1027 passed, 42 existing sqlite ResourceWarnings; ruff check/format and
`git diff --check` passed. On aeor the command is:

```sh
ZEMBLE_TEST_ZENIT_ROOT=/home/skerit/projects/zenit-workspace/zenit ZEMBLE_DAEMON=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/pytest
```

The old real-workspace graph journey failed identically on E0 because its hardcoded
checkout lacks PageWindow. The fixture now accepts a current checkout, and the
"one shipped storage adapter" assertion excludes test-only implementations. New
regressions cover budget eviction, active-generation protection, deferred rebuilding,
shared content stores and filter composition, quiet-period coalescing, capped queues,
global single-flight work, eviction cleanup, mapped generations, model single-flight
loading, real allocation refusal, bounded caches, ignored-artifact rejection, and
stale inherited env flags with explicit-environment precedence.

## Streaming builds (2026-10-05)

A build materialized the whole index before saving it: the vector matrix copied to the
heap and normalized into a second copy, every BM25 posting as a Python dict, every chunk
blob joined in memory, and a save that folded the whole BM25 corpus again. One edited
file cost a copy of the index. Builds now stream their generation to disk (see
[the daemon doc](daemon.md#rebuilding-beside-the-index-that-is-being-served)). Peak
private memory on the 177,780-chunk zenit workspace, measured phase by phase:

| Build | Before | After |
| --- | --- | --- |
| Cold load, index current | 38 MiB | 32 MiB |
| Cold load, 1 file changed | 1,509 MiB | 180 MiB |
| Cold load, 3,000 files changed | 1,996 MiB | 222 MiB |
| Daemon rebuild, 1 file changed | 688 MiB | 169 MiB |
| Build from nothing | not measured (> 2 GiB) | 283 MiB |

Admission reserves 192 MiB plus three bytes per byte of new source, measured against the
stored manifest of the root that will answer, not the whole tree.

The symbol graph had the same shape: a build held every extraction and, past 400 target
files, the whole symbol table. It now extracts and resolves 100 files at a time through a
scratch database (see [the graph doc](graph.md#nothing-more-than-the-targets-need)). Peak
private memory of the graph worker on the same workspace:

| Graph build | Before | After |
| --- | --- | --- |
| 1 file changed | 41 MiB | 41 MiB |
| 1,000 files changed | 552 MiB | 162 MiB |
| From nothing (12,458 files, 2.06M edges) | 2,184 MiB | 189 MiB |
| Every facts file moved (163 files, 903k fact edges) | 3,388 MiB | 338 MiB |

The worker's ceilings dropped accordingly: 2048 MiB construction, 4096 MiB aggregate.

