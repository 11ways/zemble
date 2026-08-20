# Warm daemon (plan step 2)

A CLI query used to pay the whole index load on every invocation: the profile in
`profile-baseline.md` measured 11.5 s of a 12.2 s query before the query itself was
looked at. The daemon holds the index in RAM for one user, watches the roots it
holds, and answers CLI and MCP requests over a unix socket.

## Design

| Piece | What it is |
| --- | --- |
| `zemble/daemon/protocol.py` | Wire format (newline-delimited JSON), socket/pidfile/lock/log locations, error types. Imports nothing heavy: every short-lived process loads it. |
| `zemble/daemon/client.py` | Connect, auto-start, one request, one response. Raises `DaemonError`; every caller falls back in-process. |
| `zemble/daemon/server.py` | The `Daemon` object (warm indexes, watchers, per-root locks) plus the `COMMANDS` dispatch table. |
| `zemble/daemon/watch.py` | `IgnoreRules` (the file walker's own gitignore machinery, reused) and `RootWatcher` (watchfiles). |
| `zemble/daemon/cli.py` | `zemble daemon run\|start\|stop\|restart\|status`. |
| `zemble/index_cache.py` | The index cache, moved out of `zemble/mcp.py` so the daemon and the in-process MCP path share one implementation. |

Requests are `{"id", "cmd", "args"}` and answers are `{"id", "ok", "result"|"error"}`,
one JSON object per line. A response can be a whole result set, so the client reads
with a buffered file object and never assumes a line length. A connection may carry
several requests in a row.

### Commands

`COMMANDS` is a plain `dict[str, Handler]`; a new command is one async function and
one entry in that table. Nothing else has to change: the client is generic.

| Command | Answers |
| --- | --- |
| `ping` | Liveness plus the daemon's pid. |
| `status` | pid, uptime, RSS, request count, idle time, and per-index root/content/embedder/chunks/files/last-used/watching/rebuilding/last-rebuild, plus builds in flight and pending reindexes. |
| `search` | The same payload `zemble search` and the MCP `search` tool print. |
| `find_related` | The same payload as `find-related`; a location that is not indexed answers `chunk_missing`. |
| `stats` | What one index holds. |
| `graph` | `command: "ensure"` guarantees a fresh symbol graph; any other name is a provider query answered through `zemble.graph.mcp.answer`. |
| `explain` | The evidence bundle `zemble explain` and the MCP `explain` tool render, built over the warm index and the daemon's graph; honours `budget`, `top_k` and `content`. |
| `outline` | The outline of a file or a type; an ambiguous or unknown target comes back as an error payload, not a failed command. |
| `signatures` | A symbol's signature and its exactly resolved call sites, with the same refusal shape. |
| `refresh` | Force a rebuild check for one root (loads it first if needed). |
| `evict` | Drop one root from memory and stop its watcher. |
| `shutdown` | Stop after answering. |

### Index cache

`IndexCache` keeps the semantics it had inside the MCP server: keyed by
(resolved source, content types), LRU eviction, one in-flight build shared by
concurrent callers, and a staleness re-check against the on-disk cache with a
cooldown scaled by build time. The daemon adds an eviction callback (so an evicted
root stops being watched), a resident-index limit, last-used timestamps, and
`replace()` for the atomic swap a rebuild ends with.

### Serving a sub-path from an ancestor index

A request naming a directory *inside* a root that is already indexed is answered from
that root, filtered to the sub-directory, instead of building a second index over the
same files. `IndexCache.get_with_key` resolves the request first (`resolve_index_root`
in `zemble/cache.py`), and the daemon's `index_for` returns the ANCESTOR's cache key
together with a restricted view of its index.

- Order of preference: an index of exactly the requested path (in memory or on disk)
  keeps serving it; else the nearest ancestor that is loaded, or whose on-disk index
  validates for the same content types; else the path is indexed on its own, as before.
- The view is a `ZembleIndex` sharing the ancestor's chunks, vectors and postings, with a
  path-prefix chunk selector (dense `selector`, BM25 `weight_mask`) applied to every
  query - the same mechanism `find_related` already used to stay inside one language.
  Ranking is the big index's ranking, restricted to the sub-tree.
- **Result paths stay relative to the ancestor root** (`zenit/src/Foo.java`, not
  `src/Foo.java`). That keeps one vocabulary across search, `find_related`, the symbol
  graph, `explain` and `outline`, all of which are answered from the same root.
- `stats` describes the sub-tree: its files, its chunks, its languages.
- One INFO line is logged per resolution:
  `serving /work/zenit from the /work index (subtree filter)`.
- If the ancestor holds nothing under the sub-directory, the sub-directory is indexed on
  its own instead of answering emptily.

This is what a sub-repo request costs now: no build, no embedding, no second resident
index. `zemble daemon status` keeps showing one index, the workspace.

### Watching

Each loaded local root is watched recursively with `watchfiles`. Events are filtered
through the *file walker's own* ignore rules: the same `.gitignore`/`.zembleignore`
loader, the same default ignored directories, the same extension set (plus `.java`,
so the symbol graph stays fresh). A watcher that disagreed with the indexer about
what a source file is would either rebuild on noise or miss edits.

Changes are coalesced with a 500 ms debounce and the resulting set of paths is what the
rebuild works from: `create_index_from_path(previous=..., changed_paths=...)` re-chunks
and re-embeds exactly those files and reuses every other file's chunks, vectors and
postings from the previous index, without walking the tree at all. The same set is
handed to `build_graph(changed_paths=...)`, where it replaces two walks: the source
walk and the `**/build/zemble/*.jsonl` discovery of the graph's facts files. The
watcher can stand in for the second because its ignore rules always admit a facts
file by name, whatever `.gitignore` says about the `build/` directory it lives in.
The full walk is still what a cold build, `zemble daemon refresh` and the CLI's cache
validation use: a walk discovers changes, a watcher reports them, and only a reported
set may skip the discovery.

A named path is still judged the way the walk judges one - extension, `.gitignore`,
readability - so a watcher that over-reports cannot get a file into the index that a
build would have skipped. The reverse is a real obligation on the caller: whatever the
change set does not name is assumed unchanged.

The rebuilt index is swapped into the cache, written back to the on-disk cache (at most
once every 10 s), and one line per rebuild is logged with the file counts and the
milliseconds.

The `graph_ms` on that line is the whole symbol-graph refresh. On the javaweb
workspace it is around 0.6 s for a template edit, 0.9 s for a Java one and 2.4 s when
that edit renames a method other files call; the numbers per edit shape, and what
they were before, are in [the symbol graph doc](graph.md).

### Rebuilding beside the index that is being served

A rebuild never writes into the index answering queries. It builds a new one next to it
and the swap is a single dict write on the event loop, so an in-flight search keeps
reading the object it started with and a search that arrives mid-rebuild is answered by
the index from before it. Only rebuilds of one root are serialised against each other
(`Daemon.rebuild_lock_for`); queries take no lock.

What that costs is bounded because of how the BM25 index is shaped:

| Piece | Shared or copied |
| --- | --- |
| BM25 columnar postings (`_Frozen`) | **Shared**, memory-mapped, never written to. |
| BM25 delta (added documents' postings, removed base rows, document order) | Copied - it holds only what moved since the base was built. |
| Vector matrix | Copied: the rebuild writes the changed files' rows into it. |
| Chunk list, manifest | Rebuilt, cheap. |

`BM25.for_update()` is that derivation. Scoring adds the base and the delta together:
corpus size, average document length and every term's document frequency count the live
documents (base minus removed, plus added), and a removed base row is masked out of the
term's posting list, so a query cannot tell where a document came from. Identity with a
from-scratch index is asserted in `tests/index/test_bm25.py`.

`BM25.fold()` turns base plus delta back into one base with a vectorized pass over the
postings, and no per-document dictionary anywhere. A save always writes the folded form,
so a cold load never pays for a warm process's updates, and `for_update()` folds instead
of deriving once the delta has grown past a tenth of the base. Every column is written to
a temporary file and moved onto its target: an older index generation may still have that
very file mapped, and truncating it under a live mapping is a SIGBUS, not a stale read -
which is exactly how the first measurement run of this work died, before the columns were
shared at all.

### On demand, and only on demand

The daemon is started by a zemble command that needs it, never at login, never by a
timer or a unit file. It exits by itself after `ZEMBLE_DAEMON_IDLE_MINUTES` without a
request (default 30; `0` never exits), and `zemble daemon stop` ends it immediately.

## Locations and settings

| Thing | Where |
| --- | --- |
| Socket | `$XDG_RUNTIME_DIR/zemble/daemon.sock` when that directory exists, else the zemble cache folder (`~/.cache/zemble/daemon.sock`). Overridden by `ZEMBLE_DAEMON_SOCKET`, or `ZEMBLE_DAEMON_DIR` for the whole directory. |
| Pidfile | `daemon.sock.pid`, beside the socket. |
| Lock | `daemon.sock.lock`, beside the socket: a `flock` held for the daemon's lifetime, so two daemons can never own one socket. |
| Log | `~/.cache/zemble/daemon.log` (the resolved cache folder), appended by detached daemons. |

| Variable | Default | Meaning |
| --- | --- | --- |
| `ZEMBLE_DAEMON=0` | unset | Never use or start a daemon in this process. |
| `ZEMBLE_DAEMON_MAX_INDEXES` | 4 | Resident index variants before the least recently used is evicted. |
| `ZEMBLE_DAEMON_IDLE_MINUTES` | 30 | Idle shutdown delay; `0` never exits. |
| `ZEMBLE_DAEMON_DIR` / `ZEMBLE_DAEMON_SOCKET` | unset | Move the runtime directory or the socket itself (tests use this). |

A socket with a dead owner is detected (connect fails, pidfile pid is gone) and
removed before a new daemon starts; a pidfile owned by a live process is left alone,
so a daemon that is still starting is never pulled out from under itself.

## Fallback rules

The daemon is an accelerator, never a requirement.

- `zemble search`, `find-related`, `stats`, `graph *`, `explain`, `outline` and `signatures`
  go through the client by default, as do their MCP tools.
- If the socket is absent or dead, the client spawns `python -m zemble.daemon run`
  detached (new session, stdio to the log) and waits up to 10 s for it to answer.
- Any failure (cannot start, connection lost, protocol error) raises `DaemonError`,
  and the caller answers in-process after one stderr line:
  `daemon unavailable (<reason>); running in-process`.
- `--no-daemon` and `ZEMBLE_DAEMON=0` skip the daemon silently: an opt-out is not a failure.
- `--embedder` skips it too, silently: the daemon holds one embedder (the environment
  default), and an override is answered in the calling process.
- The MCP server tries the daemon first and falls back to its own in-process cache, so
  several agent sessions share one RAM copy of a workspace index.
- A request blocks until its index is ready; the first request after a start may still
  be a cold build. `zemble daemon status` shows a build in progress.

## Measured

Workspace `/home/skerit/projects/javaweb`, 6,430 files, 73,957 chunks, `potion-code-16M-v2`,
query `EventDelegationPlanner`, `-k 3 --max-snippet-lines 0`. Wall time is process start
to output, five consecutive runs each.

| Run | `--no-daemon` | through the daemon |
| --- | ---: | ---: |
| 1 | 1.48 s | 0.58 s |
| 2 | 1.58 s | 0.56 s |
| 3 | 1.52 s | 0.57 s |
| 4 | 1.50 s | 0.58 s |
| 5 | 1.57 s | 0.59 s |
| median | **1.52 s** | **0.58 s** |

- `zemble daemon start`: 0.60 s. The first query after that (index load from disk into
  the daemon): 10.97 s, paid once instead of once per invocation.
- Daemon-side round trip, measured on the socket itself: 315-390 ms. The rest of the
  0.58 s is the client's own interpreter start and imports, which no daemon can remove.
- RSS with the workspace index resident: **293.8 MB**. After one incremental rebuild
  and the write-back: **700.3 MB** (the rebuild copies the vector matrix and the
  write-back builds the columns to write; Python does not return that to the OS). The
  write-back no longer materializes a dictionary per document, but the copy stands.

### One-file edit

`touch` on one Java file in the workspace (84,091 chunks, 7,251 files, 102,375 graph
symbols, `potion-code-16M-v2`), measured through `Daemon._on_change` with a query loop
running against the same root throughout. Two runs of each, both starting from a valid
on-disk cache.

| Phase | Before | After |
| --- | ---: | ---: |
| Index rebuild, first one after a cold load | 4,585 / 5,395 ms | 1,473 / 809 ms |
| Index rebuild, steady state | (same, every time) | **247 ms** |
| Longest query answered while that rebuild ran | 4,606 / 5,406 ms (blocked) | 797 / 707 ms (never blocked) |
| Median query while rebuilding | 22 / 39 ms | 21 / 15 ms |
| Symbol graph refresh | 47.4 / 55.4 s | 40.5 s (walk: 44.3 s) |
| Write-back of the whole index | 5,757 / 6,510 ms | 4,753 / 5,107 ms |
| ... of which the BM25 index | - | 547 ms |
| Warm query, no rebuild running | 9 / 21 ms | 11 / 10 ms |
| Cold load of the index into the daemon | 775 / 672 ms | 784 / 617 ms |

The first rebuild after a cold load is dearer than the ones after it because it pages the
memory-mapped vector matrix in; that is disk, not work. The steady-state number is three
consecutive rebuilds of the same root.

The graph numbers vary far too much run to run (the graph is a 1.9 GB sqlite database) for
the daemon-side pair to mean anything, so they come from six alternating isolated builds of
the same one-file edit: **44.3 s walking, 40.5 s from the change set**. The walk itself is
only 0.6 s of that. The other 40 s is the resolve pass - every symbol and every hierarchy
edge read back out of sqlite for one changed file - and it is what to attack next.

## Known limitations

- **A change set is trusted, not verified.** Only the paths the watcher reports are
  looked at, so an event the watcher never delivered leaves the index and the graph
  stale until something else touches that file. `zemble daemon refresh` and every cold
  build still walk the tree, which is the way back to a known-good state.
- **The symbol graph is still the slow half.** A one-file edit costs ~40 s of graph
  refresh against a workspace this size, essentially all of it the resolve pass; the
  index is ready in a quarter of a second. The graph refresh does not block queries
  either, but it does compete with them for CPU.
- **Symbol definitions are re-attached only when the rebuild is persisted.** The
  definition lookup is built at save time; between a rebuild and the next write-back
  (10 s throttle) a symbol query reranks with its own scan instead. Results are the
  same, the query is slower.
- **One embedder per daemon**, the environment default. `--embedder` is answered in-process.
- **Unix sockets only.** No Windows named-pipe transport; there, everything falls back
  in-process.
