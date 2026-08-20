# Load-time work: before and after

Same workspace, same index, same queries as `docs/profile-baseline.md`, but on the
smaller corpus the workspace's `.zembleignore` produces.

Workspace: `/home/skerit/javaweb-loadtime-snapshot`, a hardlink snapshot of
`/home/skerit/projects/javaweb` taken so that the live tree (which other agents keep
editing) cannot invalidate the cache mid-measurement.
Index: 6,431 files, 73,972 chunks, `potion-code-16M-v2`.
Measured with `benchmarks/profile_query.py`, which now also times the two phases the
baseline left untimed: building the embedder (`load_model`) and constructing the
`ZembleIndex` object (`build_index_object`, which used to read every indexed file).

Queries: `EventDelegationPlanner` (symbol) and `where does the compiler turn
per-element loop event handlers into a delegated listener` (natural language). Both
still rank `EventDelegationPlanner.java` first, before and after.

The machine was under heavy load (load average 16-23, other agents building) during
both the before and the after runs, so absolute numbers are pessimistic on both sides.

## Phase table (ms)

| Phase | Symbol before | Symbol after | NL before | NL after |
| --- | ---: | ---: | ---: | ---: |
| interpreter start | ~120 | ~120 | ~120 | ~120 |
| imports | 456.7 | 122.1 | 313.6 | 144.6 |
| build embedder | 0.0 | 0.2 | 0.0 | 0.2 |
| cache validation (walk 6,431 files) | 2207.4 | 265.0 | 1773.7 | 242.1 |
| load metadata.json | 3.7 | 5.5 | 3.7 | 4.0 |
| load BM25 index | 2606.6 | 1.8 | 2332.5 | 1.4 |
| load dense vectors | 82.5 | 0.3 | 55.4 | 0.3 |
| load chunks | 616.0 | 4.2 | 483.1 | 3.6 |
| load symbol definitions | - | 1.1 | - | 1.0 |
| build index object | 186.7 | 20.1 | 185.2 | 17.6 |
| resolve alpha | 0.01 | 0.02 | 0.01 | 0.01 |
| embed query (loads the model) | 372.2 | 443.3 | 266.2 | 381.5 |
| dense search | 13.0 | 10.5 | 12.4 | 41.3 |
| BM25 search | 14.3 | 0.9 | 99.3 | 8.4 |
| fusion + ranking | 355.8 | 21.7 | 2.0 | 9.0 |
| format output | 0.1 | 0.1 | 0.1 | 0.1 |
| **cold total** | **~7,035** | **~1,017** | **~5,647** | **~975** |
| **warm query (median of 5)** | **379.4** | **16.8** | **95.1** | **15.5** |

`time zemble search` on the same index after the change: 1.01 s (symbol), 1.20 s (NL).

Targets: cold under 1.5 s - met (1.0 s). Warm symbol under 60 ms - met (17 ms).
Warm NL under 60 ms - met (16 ms).

## What changed

- **BM25 (2.6 s to 1.8 ms).** The per-document JSON counters are gone. The index is
  a terms table, CSR postings (`int64` offsets, `int32` document indices, `uint16`
  term frequencies when they fit), document lengths and document ids, each an `.npy`
  loaded with `mmap_mode="r"`. Scoring slices one term's postings and accumulates with
  a fancy-index add, which is a true accumulate because a term names each document
  once. The mutable dictionaries still exist for building; a loaded index thaws into
  them lazily, only when a mutation is requested.
- **Chunks (616 ms to 4 ms).** One content blob plus offsets, a deduplicated path
  table, a language table and a line array. `index.chunks[i]` builds a `Chunk` on
  demand; the path and language columns answer whole-index questions (the file and
  language maps, the stats, `resolve_chunk`) without building any.
- **Dense vectors (82 ms to 0.3 ms).** The matrix is mapped rather than read, and
  vicinity's re-normalization pass at load is skipped because the constructor already
  normalized the rows before they were written. Incremental reindexing asks for a
  writable copy, since it writes new rows into the matrix.
- **Cache validation (2.2 s to 0.26 s).** The walk uses `os.scandir` and carries the
  root-relative path down the recursion instead of calling `Path.relative_to` per
  path per ignore spec (70% of the old time). Ignore specs are compiled once per
  directory and cached by modification time, per-pattern decisions are precomputed,
  and each spec gets one union regex that rejects a non-matching path in a single
  search instead of one per pattern.
- **Symbol rerank (356 ms to 22 ms cold, 350 ms to 9 ms warm).** The definition scan
  over every chunk is precomputed at save time into name -> chunk-index tables, using
  the same keyword vocabulary and boundary rules as `_definition_pattern`. A name the
  tables cannot express (anything that is not an identifier or namespace chain) falls
  back to scanning.
- **Index construction (187 ms to 20 ms).** File sizes for the token-savings stats are
  read on demand instead of reading every indexed file at load.
- **Imports (390 ms to 150 ms for `import zemble.cli`).** model2vec (and through it
  huggingface_hub), questionary and asyncio are imported where they are used. The
  model2vec import moved into the first embed call, which is why `embed query` looks
  slower after: it now carries what `imports` used to.

## What is left

`embed query` (380-440 ms) is the largest remaining cold phase and is almost entirely
`StaticModel.from_pretrained`: the huggingface_hub import plus reading the model. It is
unavoidable per process for any query and untouched by this work.

Building an index costs ~4 s more than before at this size, because the symbol
definitions are scanned at save time. Loading a previous index for an incremental
rebuild got faster (3.5 s to 0.8 s), but a mutation still thaws the columnar postings
back into dictionaries (3.4 s at 74k chunks), which is the same work the old load did
unconditionally.
