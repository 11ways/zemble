# Query profile baseline (unchanged upstream retrieval)

Workspace: `/home/skerit/projects/javaweb` (the whole multi-repo tree).
Index: 15,958 files, 181,912 chunks, 474 MB under `~/.cache/zemble`.
Build: 156.3 s to index + 1.4 s to write (10-core CPU, `potion-code-16M-v2`).

Measured with `benchmarks/profile_query.py` (phase timers around the same calls
the CLI makes), plus `time zemble search ...` for the end-to-end number.

- **Cold** = a fresh process: interpreter start, imports, cache validation, index
  load from disk, then one query. The OS page cache was warm, so `load_*` is
  deserialization cost, not disk seek cost.
- **Warm** = the same process querying the already-loaded index (median of 5).

## Phase table (ms)

| Phase | Symbol query, cold | Symbol query, warm | NL query, cold | NL query, warm |
| --- | ---: | ---: | ---: | ---: |
| interpreter start (residual vs `time`) | ~120 | - | ~129 | - |
| imports (numpy, model2vec, zemble) | 201.0 | - | 214.3 | - |
| cache validation (walk 15,958 files, mtime check) | 3238.0 | - | 3246.5 | - |
| load metadata.json | 8.4 | - | 8.0 | - |
| load BM25 index | 4847.6 | - | 4864.6 | - |
| load dense vectors | 2120.5 | - | 1717.6 | - |
| load chunks.json | 930.2 | - | 947.6 | - |
| load embedding model | 202.1 | - | 197.9 | - |
| resolve alpha | 0.01 | 0.004 | 0.01 | 0.003 |
| embed query | 1.60 | 0.27 | 1.62 | 0.27 |
| dense search | 8.91 | 9.77 | 9.12 | 7.90 |
| BM25 search | 8.09 | 10.15 | 138.07 | 131.43 |
| fusion + ranking | 551.84 | 538.71 | 2.45 | 2.15 |
| format output | 0.08 | 0.07 | 0.09 | 0.07 |
| **total** | **12,238** (`time`: 12.24 s) | **559** | **11,477** (`time`: 11.48 s) | **142** |

Queries: `EventDelegationPlanner` (symbol) and `where does the compiler turn
per-element loop event handlers into a delegated listener` (natural language).
Both rank `hawkeye/hawkeye-core/.../EventDelegationPlanner.java` first.

## Where the milliseconds go

Startup dominates completely: 11.5 of the 12.2 seconds are spent before the
query is even looked at, and the actual retrieval is 0.15 s (NL) to 0.56 s
(symbol). Within startup, three costs matter - deserializing the BM25 index
(4.8 s), validating the cache by walking every file in the tree (3.2 s), and
loading the dense vectors plus chunk list (3.0 s). Every CLI invocation and
every MCP cold start pays all of it again, which is exactly what a warm daemon
(plan step 2) removes. The remaining per-query costs are workload-shaped rather
than fixed: a symbol query pays ~540 ms in fusion/ranking (the query-boost pass
scans all 181k chunks), while a natural-language query pays ~130 ms in BM25
because it has many more query terms. Both are large only because this index is
large; on a single repo they are single-digit milliseconds.
