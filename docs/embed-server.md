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
server holds and what it did since it started.

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

## Garbage collection

```bash
zemble embed-server gc --dry-run          # on the server host
zemble embed-server gc                    # stop the server first; VACUUM needs the file alone
```

The server cannot see its clients' indexes, and an index that is only loaded never asks for
its vectors again, so a use stamp is its only evidence. It keeps every vector stored or served
within `--grace-days` (180 by default, against 14 on a client) and sweeps the rest.

## Wire format

JSON over HTTP; vectors travel as base64 of little-endian float32 (`zemble.embedding.wire`, the
one home of every route and limit both sides share).

| Route | Body | Answer |
| --- | --- | --- |
| `POST /v1/documents` | `spec`, `texts` | `rows`, `dimensions`, `vectors` |
| `POST /v1/queries` | `spec`, `texts` | `rows`, `dimensions`, `vectors` |
| `POST /v1/covered` | `spec`, `digests`, `probe` | `covered`, `dimensions` |
| `POST /v1/describe` | `spec` | `model_id`, `dimensions`, `family`, `semantic_weight_bonus` |
| `POST /v1/store` | `spec`, `rows` of `[text sha256, dims, base64 vector]` | `added` |
| `POST /v1/rerank` | `spec`, `query`, `passages` | `scores` |
| `GET /v1/status` | - | families, counters since start |
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
