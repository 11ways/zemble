# Embedding server

One machine holds the paid vector cache and the provider key; the others ask it for vectors
and reranker scores, so a chunk is bought once across all of them. Indexes, the symbol graph
and the daemon stay on each machine.

## Running the server

```bash
zemble embed-server add-key --label aeor      # once per client machine; prints the key
zemble embed-server run --host 0.0.0.0 --port 8765
```

- **What it serves:** the family of `ZEMBLE_EMBEDDER` and the hosted `ZEMBLE_RERANKER` from the
  server's own environment (`~/.config/zemble/env`, which also holds `VOYAGE_API_KEY`), or
  whatever `--embedder SPEC` / `--reranker SPEC` name (repeatable). A family is scheme plus
  model, so `voyage:voyage-4-lite` serves every width of that model. Anything else is refused
  with 403, naming what is served: an `openai:` spec carries its base URL, so a client can
  never point the server's key at a URL of its own choosing.
- **Keys:** `~/.config/zemble/server-keys` (mode 600, `--keys-file` overrides), one
  `<label> <key>` per line. The file is re-read when it changes, so `add-key` needs no restart;
  deleting a line revokes that key. The server refuses to start without a key. The label
  appears in the access log.
- **Data:** `~/.local/share/zemble/embed-server/<family>.sqlite` (`--data-dir` overrides): the
  same file format as the local cache, deliberately outside `~/.cache/zemble`, so
  `zemble clear orphans` on the host can never mistake it for an unused client cache.
- **TLS:** `--certfile` and `--keyfile` serve HTTPS directly. Without them the key crosses the
  network in the clear; use them, a reverse proxy, or a private network for anything beyond a LAN.
- `GET /v1/health` answers without a key; everything else needs `Authorization: Bearer <key>`.

## Using it from a machine

Add two lines to `~/.config/zemble/env`:

```bash
ZEMBLE_EMBED_SERVER=http://embed-host:8765
ZEMBLE_EMBED_SERVER_KEY=<the key add-key printed>
```

Then `zemble daemon restart` (a running daemon read its environment when it started) and
restart MCP processes that run without the daemon. `zemble embed-server check` shows what the
server holds, what it did since it started, and what it bought from its providers over the last
31 days per day and client key label (texts and estimated tokens per model and kind, kept in
`spend.db` beside the cache files, so a restart does not reset it).

With a server configured, a paid embedder spec builds a `ServerEmbedder` and a `voyage:`
reranker builds a `ServerReranker`. The client needs no provider key and keeps no vector
cache. `ZEMBLE_EMBED_CACHE=0` does not bypass the server: the server always caches. A local
embedder (`model2vec:`) never touches the server.

- **Indexes stay valid.** The client reports the provider's own model id
  (`voyage:voyage-4-lite@1024`), width and fusion bonus, so an index built before the switch is
  served after it without a rebuild.
- **The budget guard still applies.** It asks the server which texts are not paid for yet,
  exactly as it asked the local cache, so a build that is cached on the server never prompts.
  `zemble embed-status` reads the server's coverage too and names the server as its cache.
- **No fallback.** An unreachable server is an error, retried like any provider (429/5xx with
  backoff). A provider failure on the server comes back as 424 with the provider's message
  and is not retried again by the client: the server already retried.

### Moving a machine's vectors to the server

```bash
zemble embed-server push                  # this machine's cache files, for the families served
zemble embed-server push --remove-local   # ... then delete each file the server now holds in full
zemble embed-server push seed.sqlite --family voyage:voyage-4-lite   # any cache file, any name
```

`push` uploads stored vectors, never re-buys them, and never replaces a vector the server
already has. Run it once on every machine before relying on the server, so what each machine
already paid for is not paid for again. With `--remove-local` a file is deleted only after
every one of its rows was accepted, and only when no process holds it open (restart the
daemon first). That deletion is the disk this saves per machine.

## How one text is bought once

Requests are answered on a thread each. A documents request looks every text up first; the
misses are claimed per (family, text hash, width) before the provider is called. A second
request that needs a claimed text waits for that claim and then reads the stored vector
instead of buying it again; if the first purchase failed, the claim is released and the
waiting request buys it itself. Purchases are flushed in slices of 512 (`FLUSH_EVERY`), which
is also how many texts one client request carries, so a failure loses at most one slice.

Queries are never stored: for an asymmetric model a query vector must never be served where a
document vector was asked for.

## Counting what the server holds

`GET /v1/status` reads each family's vector count from the file itself, never by scanning it:
two triggers in the file move a counter in `cache_meta` on every row added or removed, whoever
writes (`put_many`, `put_missing`/`push`, `gc`, an older zemble, a hand-run statement), inside
that writer's own transaction, so a daemon and a CLI on one client file cannot tear it. A file
from before the counter gets the triggers when it is first opened and is counted once, on a
background thread with a connection of its own; until that scan ends its status says
`"vectors": null, "counting": true` (`check` prints "vectors not counted yet"), then the exact
number. On a 9.4 GB synthetic file of 2,000,000 vectors (2026-11-21, page cache evicted) status
took 3.3 s before (0.09 s warm), and 0.05 s after with the scan running behind it for 3.4 s, then
0.002 s.

## Keeping used vectors alive

The server cannot see its clients' indexes, and an index that is only loaded never asks for its
vectors again, so the `used` stamp (the last day a vector was used) is the server's only
evidence of what is still wanted. Three things write it: storing a vector (`documents`, `store`),
serving one (`documents`), and a `covered` hit, since a pre-flight only asks about the chunks of
a live tree. Every stamping path shares one refresh window (`STAMP_REFRESH_DAYS`, 7, in
`zemble.embedding.cache`): a new hash is stamped today, but an existing stamp is rewritten only
once it is 7 days old, so a vector in daily use costs one stamp write a week and a batch whose
stamps are all fresh takes no writer lock at all. A vector in use can therefore show a stamp up
to 6 days old, and both collections refuse a grace period under 8 days.

Stamping never runs on the connection that serves. A store stamps inside its own insert; served
and covered hashes go, after the answer, to a side stamper with its own sqlite connection per
family (a bounded queue of 64 batches; when it is full their stamps are dropped and counted in
`stamps_dropped`, the daily report being what keeps an index's vectors alive). Stale stamps are
written 250 per write transaction (`STAMPS_PER_TRANSACTION`), so a buyer's insert gets the writer
between two of them.

Each client also reports what its saved indexes hold: the daemon, a random time after it starts
(so hosts restarted together do not report together; 1 minute to half its idle limit, 15 minutes
for the default 30, or 10 to 60 minutes when it never exits idle; until that first report has run,
and while any report runs, the daemon does not exit idle; a failed report lets it exit and the
next start reports again) and then every
`ZEMBLE_DAEMON_REFERENCED_HOURS` (6; `0` disables), sends the server the text hash of every chunk
of every saved index and lane overlay of a paid family (`POST /v1/referenced`, 2,000 hashes per
request, each distinct hash once). The server stamps the ones it holds on the side connection
and never creates a stamp for a hash it does not. It stamps one report batch at a time, server
wide; a batch arriving meanwhile gets `503` with `Retry-After: 2`, and the client waits with a
growing pause (up to 60 s, up to 30 minutes in all) and resumes with the same batch. On one host
the daemon and `zemble embed-server report [--force]` share an exclusive lock
(`~/.cache/zemble/embed-server-referenced.lock`); the second one skips and says so. Each family
goes once a day per server, and its day is written to
`~/.cache/zemble/embed-server-referenced.json` as soon as that family is done, so a restart does
not resend. A server older than the route answers 404: the client logs it once and tries again
the next day.

Measured 2026-11-21 on aeor: the client's 38 saved indexes and overlays of `voyage-4-lite` hold
3.35M chunks, 0.96M distinct hashes, read and hashed in 27 s with 57 MB of private memory. On a
synthetic 4M-vector server file with the page cache evicted, a 2,000-hash batch of stale stamps
takes 0.9 s, of which the eight write transactions hold the writer at most 78 ms each (500 per
transaction: 106 ms, 1,000: 144 ms). Three reporters of 60 such batches each, run at once against
a server on that file, finished in 93, 117 and 198 s (180 batches in 200 s, 15 answered 503);
meanwhile a `documents` request of 32 cached texts answered in 30 ms at the median, 393 ms at
worst (34 and 85 ms with no report running), and a 2,000-hash `covered` request in 51 ms and
203 ms (213 and 693 ms, colder, before).

## Garbage collection

```bash
zemble embed-server gc --dry-run          # on the server host
zemble embed-server gc                    # stop the server first; VACUUM needs the file alone
```

It keeps every vector used within `--grace-days` (180 by default, against 14 on a client, at
least 8 because of the refresh window, else it refuses with exit code 1) and
sweeps the rest. Then it VACUUMs, but only when the disk can hold it: in WAL mode VACUUM copies
the live pages into a temp database (in `SQLITE_TMPDIR`, else `TMPDIR`, else `/var/tmp`) and
then writes the whole rebuilt file through the WAL before the file shrinks, so it needs about
the live size twice. When it is short the deletes are still committed and the WAL checkpointed,
and both `gc` and `zemble clear embeddings` say "VACUUM skipped: needs ~X on <folder>, Y free;
freed pages are reused by new vectors"; a dry run predicts the same and reports the free pages
the file already holds.

## Wire format

JSON over HTTP; vectors travel as base64 of little-endian float32 (`zemble.embedding.wire`, the
one home of every route and limit both sides share).

| Route | Body | Answer |
| --- | --- | --- |
| `POST /v1/documents` | `spec`, `texts` | `rows`, `dimensions`, `vectors` |
| `POST /v1/queries` | `spec`, `texts` | `rows`, `dimensions`, `vectors` |
| `POST /v1/covered` | `spec`, `digests`, `probe` | `covered`, `dimensions` (hits are stamped) |
| `POST /v1/referenced` | `spec`, `digests` | `held` (the held ones are stamped) |
| `POST /v1/describe` | `spec` | `model_id`, `dimensions`, `family`, `semantic_weight_bonus` |
| `POST /v1/store` | `spec`, `rows` of `[text sha256, dims, base64 vector]` | `added` |
| `POST /v1/rerank` | `spec`, `query`, `passages` | `scores` |
| `GET /v1/status` | - | families (`vectors`, `counting`, `bytes`), counters since start, `spend` (per day, label, model, kind) |
| `GET /v1/health` | - | `ok`, `version` (no key needed) |

## Measured

On aeor, with the server on localhost and a scratch data directory, 2026-10-05:

| Step | Result |
| --- | --- |
| `push` of aeor's real cache file (1.4 GB) | 302,984 vectors in 18.8 s |
| `embed-status` of `hawkeye` (17,310 chunks) through the server | 17,274 cached, 36 uncached, 0.2 s lookup |
| Cold index of `hawkeye` plus a reranked search, fresh client cache | 12.8 s; 36 texts bought (8,237 provider tokens) |
| The same from a second fresh client cache | 6.4 s; nothing bought |

## Not done

- No spending ceiling on the server itself: each client's budget guard still refuses a build
  over its own budget, but the server pays whatever its key holders ask for.
- The client's provider retry policy applies to an unreachable server too, so an outage takes
  about 15 s of backoff to report.
