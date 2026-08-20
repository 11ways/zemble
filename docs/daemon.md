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

### Watching

Each loaded local root is watched recursively with `watchfiles`. Events are filtered
through the *file walker's own* ignore rules: the same `.gitignore`/`.zembleignore`
loader, the same default ignored directories, the same extension set (plus `.java`,
so the symbol graph stays fresh). A watcher that disagreed with the indexer about
what a source file is would either rebuild on noise or miss edits.

Changes are coalesced with a 500 ms debounce, then one incremental reindex runs in a
worker thread (`create_index_from_path(previous=...)`), the result is swapped into the
cache atomically, the index is written back to the on-disk cache (at most once every
10 s), and the symbol graph is refreshed if a `.java` file moved. One line per rebuild
is logged with the file counts and the milliseconds.

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
  write-back serializes the BM25 index; Python does not return that to the OS).

Rebuild after `touch` on one Java file in the workspace, straight from the log:

```
INFO watchfiles.main: 1 change detected
INFO zemble.daemon.server: rebuilt /home/skerit/projects/javaweb: 0 added, 1 changed, 0 removed, 73957 chunks in 6213 ms
```

The graph refresh that followed took 4,602 ms. Most of the 6.2 s is the walk over the
whole tree that any incremental reindex does, not the one file that changed.

## Known limitations

- **Queries for the root being rebuilt block for the rebuild.** The incremental path
  consumes its `PreviousIndex`: it mutates the BM25 index in place and writes into the
  vector matrix. Copying a 74k-document BM25 index costs more time and memory than the
  block it would avoid, so the rebuild and the swap are done under that root's lock.
  Other roots are unaffected, and a query never sees a half-updated index.
- **Symbol definitions are re-attached only when the rebuild is persisted.** The
  definition lookup is built at save time; between a rebuild and the next write-back
  (10 s throttle) a symbol query reranks with its own scan instead. Results are the
  same, the query is slower.
- **One embedder per daemon**, the environment default. `--embedder` is answered in-process.
- **Unix sockets only.** No Windows named-pipe transport; there, everything falls back
  in-process.
