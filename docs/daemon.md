# Warm daemon (plan step 2)

A CLI query used to pay the whole index load on every invocation: the profile in
`profile-baseline.md` measured 11.5 s of a 12.2 s query before the query itself was
looked at. The daemon holds the index in RAM for one user, watches the roots it
holds, and answers CLI and MCP requests over a unix socket.

## Design

| Piece | What it is |
| --- | --- |
| `zemble/daemon/protocol.py` | Wire format (newline-delimited JSON), socket/pidfile/lock/log locations, error types. Imports nothing heavy: every short-lived process loads it. |
| `zemble/daemon/client.py` | Connect, auto-start, one request, one response. Reports errors without authorizing local indexing. |
| `zemble/daemon/server.py` | The `Daemon` object (warm indexes, watchers, per-root locks) plus the `COMMANDS` dispatch table. |
| `zemble/daemon/memory.py` | RAM-derived memory budget, memory refusal, and kernel allocation backstop. |
| `zemble/daemon/admission.py` | Bounded concurrent reads, FIFO waiting room and request deadlines; leases survive client timeout until actual work finishes. |
| `zemble/daemon/graph_jobs.py`, `graph_worker.py` | Joined graph jobs, one bounded construction process, progress/status and complete worker teardown. |
| `zemble/daemon/watch.py` | `IgnoreRules` (the file walker's own gitignore machinery, reused) and `RootWatcher` (watchfiles). |
| `zemble/daemon/cli.py` | `zemble daemon run\|start\|stop\|restart\|status`. |
| `zemble/index_cache.py` | The index cache, moved out of `zemble/mcp.py` so the daemon and the in-process MCP path share one implementation. |

Requests are `{"id", "cmd", "args"}` and answers are `{"id", "ok", "result"|"error"}`,
one JSON object per line. A response can be a whole result set, so the client reads
with a buffered file object and never assumes a line length. A connection may carry
several requests in a row.

### Admission and graph construction

Immutable reads have four execution slots and a 32-request waiting room. A request
carries one optional `deadline_ms` covering preparation, queueing and execution
(default 25000, maximum 900000). A full queue or expired deadline returns retry
information (`retry_after_ms`) instead of starting a client-side index. Status
reports actual active reads, including work whose caller has stopped waiting.

Cold graph preparation owns no read slot. Callers join one construction job per
canonical root; their deadlines cannot cancel it or block unrelated warm searches.
The worker has its own bounded lifetime and private-memory limit, writes through
the existing graph publication protocol, and is reaped before the job completes.
Watcher changes arriving during construction coalesce into a bounded follow-up pass.
`graph_construction.jobs` reports queued/building/ready/failed/cancelled states,
worker PID, budget, elapsed/build time and peak worker RSS. Stop closes graph jobs
and drains running reads before exiting.

Repeated identical `home` and graph questions join one read computation and reuse
successful answers in a 64-entry LRU. Keys contain weak serving-index identities
and successful graph publication generations; edits therefore invalidate answers
without retaining old indexes. Errors are never cached. During a refresh, readers
continue using the last published generation instead of waiting for construction.

The aggregate default is 8192 MiB. Construction is capped at 6144 MiB and narrowed
by the serving process's current private memory plus concurrent-read scratch.
During construction the parent and worker private-memory limits (`RLIMIT_DATA`) sum
to at most the aggregate budget; both are reported in `memory_configuration`.

### Commands

`COMMANDS` is a plain `dict[str, Handler]`; a new command is one async function and
one entry in that table. Nothing else has to change: the client is generic.

| Command | Answers |
| --- | --- |
| `ping` | Liveness plus the daemon's pid, the zemble version it runs and the revision it was started from. |
| `status` | pid, uptime, RSS, request count, idle time, a `runtime` block (version, source root, revision, start time, `stale`, `note`), and per-index root/content/embedder/chunks/files/last-used/watching/rebuilding/last-rebuild, plus builds in flight and pending reindexes. |
| `search` | The same payload `zemble search` and the MCP `search` tool print; honours `paths` and `exclude`. |
| `find_related` | The same payload as `find-related`; a `file_path:line` that names no chunk answers `unresolved_location: true` with an error naming the nearest indexed path or the file's chunk spans. Honours `paths` and `exclude`. |
| `stats` | What one index holds. |
| `graph` | `command: "ensure"` guarantees a fresh symbol graph; any other name is a provider query answered through `zemble.graph.mcp.answer`. |
| `explain` | The evidence bundle `zemble explain` and the MCP `explain` tool render, built over the warm index and the daemon's graph; honours `budget`, `top_k`, `content`, `paths` and `exclude`. |
| `outline` | The outline of a file or a type; an ambiguous or unknown target comes back as an error payload, not a failed command. |
| `signatures` | A symbol's signature and its exactly resolved call sites, with the same refusal shape. |
| `refresh` | Force a rebuild check for one root (loads it first if needed). |
| `evict` | Drop one root from memory and stop its watcher. |
| `shutdown` | Stop after answering. |

### Which code is answering

zemble is installed as an editable checkout, so every process is a snapshot of the
source taken when it started: a daemon that came up before a pull keeps answering from
the code it loaded, for as long as its idle window allows. The daemon therefore names
its snapshot.

- `status` carries a `runtime` block: version, source root, revision, process start
  time, a `stale` flag (a *.py file under the loaded package changed, or HEAD moved,
  since the daemon started) and a human `note` when it is stale.
  `zemble daemon status` prints it, with `STALE` where it applies.
- Every response carries `zemble_version` and `zemble_rev`. Clients observe every
  daemon revision transition and continue through the current socket. A mismatch
  with a stale client is not a reason to restart a newer daemon or load a local index.
  MCP `status` includes the daemon's runtime and its last observed revision.
- `zemble status` prints this process's identity and the daemon's side by side.

`zemble daemon restart` is the fix; nothing reloads source in place.

### Index cache

`IndexCache` keeps the semantics it had inside the MCP server: keyed by
(resolved source, content types), LRU eviction, one in-flight build shared by
concurrent callers, and a staleness re-check against the on-disk cache with a
cooldown scaled by build time. A build that was told to `exclude` paths carries a
third key element, the digest of those patterns - and only then, so a plain build's
key, in memory and on disk, is byte-identical to what it always was. The daemon adds an eviction callback (so an evicted
root stops being watched), a resident-index limit, last-used timestamps, and
`replace()` for the atomic swap a rebuild ends with.

The daemon's `ResidentCache` reloads saved construction results as mapped columns;
keeping the original object would retain its mutable postings and anonymous vectors.
Watchers own freshness while enabled, so queries do not start independent builds of
a churning tree. `--no-watch` retains query-side staleness checks. Code and code+docs
requests for one root share one covering resident index, with content selectors for
narrower answers. On disk a root keeps only the widest stored index; a narrower request
is answered from it (`ZembleIndex.for_content`).

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
- **Paths are relative to the path the caller named** (`src/Foo.java`, not
  `zenit/src/Foo.java`), in and out: result paths have the routing prefix stripped, and a
  `file_path` given to `find_related`, or a `paths`/`exclude` filter, is taken relative to
  the requested path. That is the spelling the symbol graph, `outline`, `explain` and
  `dupes` already use for that path, so a location copied from any one tool resolves in
  every other, and joins with the path the caller holds. `explain` and `home` therefore
  open the REQUESTED path's graph and `home.toml`, not the ancestor's.
- A `file_path` that resolves to no chunk is reported as an argument problem, naming the
  nearest indexed path (`No indexed file matches 'zenit/src/Foo.java'. Did you mean
  'src/Foo.java'?`) or, for a known file, the line spans its chunks cover; the answer
  carries `unresolved_location: true`. Any line inside a chunk resolves, not only its first.
- `stats` describes the sub-tree: its files, its chunks, its languages.
- One INFO line is logged per resolution:
  `serving /work/zenit from the /work index (subtree filter)`.
- If the ancestor holds nothing under the sub-directory, the sub-directory is indexed on
  its own instead of answering emptily.

This is what a sub-repo request costs now: no build, no embedding, no second resident
index. `zemble daemon status` keeps showing one index, the workspace.

### Narrowing an answer: `paths` and `exclude`

Both are optional on `search`, `find_related` and `explain`, both are lists of strings
relative to the path the caller named, and both are refused as a command error if they
are not lists of strings - an unreadable filter is never silently ignored.

- On a root the daemon can already serve, they filter at **query time**: `index.filtered()`
  builds a restricted view sharing the same chunks, vectors and postings, and the view's
  files are dropped from the candidate set *before* the top-k is taken, so k results still
  come back full. Views are cached per filter, bounded, and cost one pass over the chunk
  list to build.
- On a root with **no index at all**, `exclude` is additionally handed to the walker that
  builds it, so an oversized tree can be indexed without its fat directories in the same
  call that was just refused. That build is a different index of a different file set, so
  it is keyed separately and never served as the plain index of the root.

The routing prefix of a sub-path request (see below) and a caller's filter compose: the
prefix is stripped before the filter is matched, so a caller's `paths`/`exclude` are always
relative to the repo it named, whichever root actually answers.

### Watching

Each loaded local root is watched recursively with `watchfiles`. Events are filtered
through the *file walker's own* ignore rules: the same `.gitignore`/`.zembleignore`
loader, the same default ignored directories, the same extension set (plus `.java`,
so the symbol graph stays fresh). A watcher that disagreed with the indexer about
what a source file is would either rebuild on noise or miss edits.

Events have a 500 ms debounce, but the callback only records paths and returns.
Rebuilding waits for two seconds of relevant-event quiet, checked again after waiting
for the daemon-wide construction slot. Each root owns one job and at most 4096
deduplicated paths; overflow becomes a full-rescan bit. Eviction discards pending
work and historical root metadata. The resulting set of paths is what the
rebuild works from: `write_index(previous=..., changed_paths=...)` re-chunks
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

Every successful rebuild is published and reloaded as mapped columns before the
swap. One line per rebuild logs the file counts and milliseconds, including persistence.

The `graph_ms` on that line is the whole symbol-graph refresh. On the javaweb
workspace it is around 0.6 s for a template edit, 0.9 s for a Java one and 2.4 s when
that edit renames a method other files call; the numbers per edit shape, and what
they were before, are in [the symbol graph doc](graph.md).

### Rebuilding beside the index that is being served

A rebuild never writes into the index answering queries. It builds a new one next to it
and the swap is a single dict write on the event loop, so an in-flight search keeps
reading the object it started with and a search that arrives mid-rebuild is answered by
the index from before it. All roots share one construction slot; memory-intensive
requests use bounded concurrent read slots, and warm search does not take the construction lock.

Every build lane - a cold build, a stale cache brought up to date, a clone, a watcher
rebuild - is `ZembleIndex.build`: `write_index` streams the next generation into a
staging folder beside the variant, `generation.publish` renames it into place file by
file (metadata last, under the variant's `index.lock`), and the result is loaded mapped.
A process that has the old generation mapped keeps reading its inodes; a load opening
the variant waits on the lock rather than mixing two generations.

Nothing is held whole while writing:

| Store | How the next generation gets it |
| --- | --- |
| Chunks | Each new file's chunks are appended as they are chunked; a reused file's rows are copied byte for byte from the mapped store. |
| BM25 | New documents' postings are kept as three flat integers each; reused documents are carried by row and merged term by term in blocks of 2^20 postings. |
| Symbols | New chunks are scanned; reused chunks keep the names their previous generation found. |
| Vectors | A mapped matrix is preallocated; reused rows are copied in blocks, fresh rows embedded in batches of 2048 after the bill guard has judged all of them at once. |

What stays in memory is what is new plus a few integers per chunk. On the 177k-chunk
workspace a rebuild for one edited file peaks at ~170 MiB of private memory and a
build from nothing at ~280 MiB. Carried-over postings come first within a term, which no
score can see because each document appears once; identity with a from-scratch build is
asserted in `tests/index/test_bm25.py`.

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

A detached daemon is spawned with `--log-file`, which points its logging at that path
through a `RotatingFileHandler`: `LOG_MAX_BYTES` (4 MB) per file, `LOG_BACKUP_COUNT` (3)
backups kept as `daemon.log.1` .. `daemon.log.3`, so the log costs at most 16 MB no matter
how long the daemon lives. Both constants are declared in `daemon/protocol.py` beside the
other daemon defaults. A foreground `zemble daemon run` still logs to stderr. The child's
raw stdout/stderr are pointed at the same file so a crash before logging is configured is
still recorded; that inherited descriptor follows the inode across a rotation, so such
output can end up in a backup.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ZEMBLE_DAEMON=0` | unset | Never use or start a daemon in this process. |
| `ZEMBLE_DAEMON_MAX_INDEXES` | 4 | Resident roots before LRU eviction; content variants of one root share a resident index. |
| `ZEMBLE_DAEMON_MAX_RSS_MB` | min(15% of MemTotal, 4096) MiB | Serving-process private memory (`VmData`) budget and `RLIMIT_DATA` backstop; mapped index files are not counted. |
| `ZEMBLE_DAEMON_TOTAL_MEMORY_MB` | 8192 MiB | Aggregate parent/graph-worker reservation ceiling. |
| `ZEMBLE_GRAPH_BUILD_MEMORY_MB` | 6144 MiB | Construction cap, narrowed by aggregate headroom. |
| `ZEMBLE_DAEMON_READ_SLOTS` | 4 | Concurrent immutable read tasks. |
| `ZEMBLE_DAEMON_QUEUE_LIMIT` | 32 | Waiting-room capacity; excess requests receive retry information. |
| `ZEMBLE_DAEMON_IDLE_MINUTES` | 30 | Idle shutdown delay; `0` never exits. |
| `ZEMBLE_DAEMON_DIR` / `ZEMBLE_DAEMON_SOCKET` | unset | Move the runtime directory or the socket itself (tests use this). |

A socket with a dead owner is detected (connect fails, pidfile pid is gone) and
removed before a new daemon starts; a pidfile owned by a live process is left alone,
so a daemon that is still starting is never pulled out from under itself.

## Fallback rules

Daemon-enabled clients never fall back to a local index. The explicit `--no-daemon`
switch is the only client opt-out. The daemon itself may use shared local library
code, but that is backend execution, not a client fallback.

- `zemble search`, `find-related`, `stats`, `graph *`, `explain`, `outline` and `signatures`
  go through the client by default, as do their MCP tools.
- If the socket is absent or dead, the client spawns `python -m zemble.daemon run`
  detached (new session, stdio to the log) and waits up to 10 s for it to answer.
- Startup failures, disconnects and response timeouts report `Daemon unavailable,
  retry`. A full waiting room or expired request deadline reports `Daemon busy,
  retry` with retry information. Clients keep a default 30-second response timeout;
  the server's default 25-second deadline leaves time to deliver its answer. Shared
  cold construction may continue so a later retry can use its result.
- **A refusal is not an outage.** A deliberate, deterministic "no" - any `Refused`, which
  includes daemon memory admission, a root too broad or holding more source than one build may chunk
  (`ScopeRefused`) and a bill over the budget (`EmbeddingBudgetExceeded`) - is
  answered as `{"ok": false, "kind": "refused", "error": ...}` and raises `CommandRefused`
  (a `CommandFailed`) in the client. Callers surface it instead of falling back: refused
  work is not an outage. A fallback could allocate unbounded client memory or repeat a
  full paid build only to reproduce a work/bill refusal. `ErrorKind` in `protocol.py` is the
  one home for that vocabulary, and an unknown kind from a newer daemon is read as
  `failed` and reported without fallback; unknown members fail closed.
  The daemon logs the refusal's own text, and each refusal carries the environment variable
  that would raise the ceiling it hit (`knob` in the root's `last_error`). `knob` is declared
  once, on the `Refused` base in `zemble/refusal.py`, and `REFUSAL_TYPES` is that base rather
  than a hand-maintained union, so a new refusal type is reported here with no second edit and
  can never reach this handler without the field it reads.
- `--no-daemon` deliberately permits local indexes. MCP accepts it in either
  `zemble --no-daemon` or `zemble mcp --no-daemon` form; local model loading is lazy.
- `ZEMBLE_DAEMON=0` alone refuses access and tells the caller to unset it or pass the
  explicit opt-out. `--embedder`, `--reranker`, `--intent` and confirmation similarly
  cannot implicitly select local execution: unsupported overrides require the opt-out.
- MCP keeps an empty local cache and no model while using the daemon. Availability,
  command, budget and malformed/empty-reply errors do not populate it.
- New callers send `accepts_busy: true`. Legacy callers receive the known `refused`
  error kind for busy replies, with retry text, because old index clients would
  otherwise treat an unknown `busy` kind as permission to fall back.
- A request blocks until its index is ready; the first request after a start may still
  be a cold build. `zemble daemon status` shows a build in progress.

## Measured

Current memory measurements and effective defaults are documented in
[the memory report](daemon-memory.md). The latency measurements below are historical.

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

- **A failed graph refresh is reported, never repaired mid-flight.** The daemon logs it
  at ERROR saying the graph is now STALE and keeps serving the graph it has; the next
  build is what fixes it, and a store sqlite calls malformed is rebuilt from source
  there ([storage](graph.md#durability)).
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
- **One embedder per daemon**, the environment default. `--embedder` requires explicit `--no-daemon`.
- **Staleness is reported, never enforced.** A daemon that finds itself stale keeps
  serving; only a restart changes the code it runs. Staleness is judged from mtimes of
  the *.py files present at startup plus the HEAD revision, so a file added after the
  daemon started is only noticed through a moved revision.
- **Unix sockets only.** No Windows named-pipe transport; local client execution there
  requires explicit `--no-daemon`, not an automatic transport fallback.
