# Client index ownership

## Finding and routing

At `f424be3`, `_via_daemon` (CLI) and `_daemon_call` (MCP) caught general daemon
errors and selected local indexing. Once an MCP cache loaded an index it retained
that generation, independently of the daemon's RSS budget. Startup also eagerly
initialized the client's embedder even when every request used the daemon.

| Condition at f424be3 | Previous behavior | Current behavior |
| --- | --- | --- |
| Healthy daemon | Remote answer | Remote answer, no client model/index |
| Index budget/scope refusal | Explicit refusal for primary index tools | Explicit refusal for all daemon-backed MCP tools |
| Graph MCP refusal | General catch answered locally | Explicit refusal, no local graph build |
| Occupied execution/construction slot | Waited; an eventual timeout/outage could fall back | Immediate busy/retry reply |
| Startup/connect failure, disconnect, timeout | Loaded/answered locally | Unavailable/retry, no local index |
| Malformed, missing/empty or failed reply | Generic failure could fall back | Explicit failure, no local index |
| Unknown error kind | Generic failure, local fallback | Generic failure, no local fallback |
| Revision mismatch | One warning; successful replies remained remote | Observe every revision transition and continue remote |
| ZEMBLE_DAEMON=0 | Local execution | Refuse access unless explicit --no-daemon |
| --embedder / --reranker / --intent / confirmation | Some implicitly selected local execution | Require explicit --no-daemon for unsupported overrides |
| --no-daemon | Local execution | Deliberate local execution, lazy model loading |

`--no-daemon` is available for both bare `zemble` stdio mode and `zemble mcp`.
Successful retrieval/scoring is unchanged. Graph CLI reads a provider after a
daemon-gated ensure; a failed ensure no longer builds locally. Explicit local graph
maintenance also requires the opt-out. `dupes` remains a separate deliberate scanner,
not an index-loading fallback; it does not load a `ZembleIndex`.

Response deadlines default to 30 seconds. Availability errors do not cause automatic
replay of a command that may already be running in the daemon. A retry uses a fresh
connection; a still-running construction can reply busy instead of buying the same
index in another process. Null results cannot act as a local-routing sentinel.

## Revision changes and rollout

The client remembers the revision observed on each valid response envelope, logs
each transition, and reconnects to the current daemon socket for every call. A client
on older source keeps using the newer daemon rather than asking it to downgrade or
loading an index locally. MCP `status` distinguishes the MCP process's own identity
from the daemon runtime and last observed daemon revision. This is protocol routing,
not Python module hot reload or dynamic replacement of an old tool schema.

New callers declare `accepts_busy: true`. A busy reply to an unmarked legacy caller
uses its existing `refused` vocabulary with retry text, so pre-fix primary index tools
do not classify a new unknown kind as a fallback opportunity. This cannot repair an
old client's own outage fallback implementation or release its resident objects.
Restart old pre-fix MCP processes once to install the new routing and release their
already-loaded indexes. After that, daemon revision changes do not require an MCP
restart for existing compatible tools. No existing user process was killed during
this work.

## Isolated stdio measurement

```sh
.venv/bin/python -m tests.measure_client_memory --revision f424be3 --output tests/client_memory_before.json
.venv/bin/python -m tests.measure_client_memory --output tests/client_memory_after.json
```

The script generates identical 32k-code / 34k-all chunks for one root and 17k-all
chunks for a second root. It uses the real MCP stdio transport, an owned daemon,
offline deterministic 1024-dimensional vectors, and private cache/socket/runtime/env
directories under `/tmp/opencode`. A `free -g` / 10 GiB available-RAM check precedes
each run; sampling stops owned processes at 7 GiB combined RSS. Real caches, sockets
and hosted embedding APIs are not used.

It makes a healthy request, injects availability, busy, generic failure and budget
refusal answers, then restarts the daemon while keeping the MCP process alive. Busy
and command error replies are injected at the wire boundary to reproduce old client
classification; unit tests separately hold real execution locks and verify immediate
busy replies and legacy-safe refusal encoding. Daemon revision labels are controlled
in the measurement envelope so restart detection can be asserted while testing a
working tree.

| MCP client | f424be3 | Current |
| --- | ---: | ---: |
| Peak RSS, MiB | 768.2 | 84.6 |
| Local indexes after errors | 3 | 0 |
| Local build/load calls | 3 | 0 |
| Local model loaded | yes | no |
| RSS before daemon restart, MiB | 649.7 | 84.4 |
| RSS after daemon restart, MiB | 649.7 | 84.4 |
| Same MCP PID across restart | yes | yes |
| Observed revision after restart | not tracked | daemon-two |

The before process loaded one mapped code index and constructed two additional
content/root variants after failures. A healthy daemon response later did not evict
those client-owned stores. The after process answered errors explicitly, left its
cache empty and its model uninitialized, and successfully queried the replacement
daemon from the same MCP PID. The daemon peaks were 525.5 / 532.3 MiB; this change
removes client duplication rather than shifting the same retained heap into an
unbounded daemon. The load demonstrates the mechanism, not the exact Archdev 5.1 GiB
live-process peak.

## Verification

Tests cover all daemon-backed MCP index/graph surfaces across refusals, unavailable,
busy, failed and malformed responses; CLI index surfaces across the same failures;
environment disablement versus explicit local opt-out; missing results; startup
without model loading; lazy local execution; socket timeouts; multiple revision
transitions in one MCP instance; current daemon identity in status; real admission
locks; and legacy caller error vocabulary.

The successful query payload SHA-256 is identical before/after:
`1a0872d89c7782049c1dd9d264511d7e1c78750bd7a14e199b3b7dbe0ba54733`.
No ranking,
chunking, embedding text or scoring changes were made, so retrieval-quality eval sets
were not rerun. The previous memory report's full retrieval measurements remain in
`daemon-memory.md`. Full test execution uses the current workspace fixture as before:

Full suite: 1092 passed, with 42 existing sqlite ResourceWarnings. Ruff check/format
and `git diff --check` passed; pydoclint reports no findings in the changed client,
protocol, main MCP and main CLI modules.

```sh
ZEMBLE_TEST_ZENIT_ROOT=/home/skerit/projects/zenit-workspace/zenit ZEMBLE_DAEMON=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/pytest
```
