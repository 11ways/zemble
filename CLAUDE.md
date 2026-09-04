# zemble

Fork of Semble (MinishLab) with a workspace code-intelligence layer on top: symbol
graph + compiler facts, context capsules, evidence bundles, duplication detection,
`home`, pluggable embedders/rerankers, warm daemon, columnar index. Python 3.11+,
one small Java sub-project (`javac-facts/`). `docs/plan.md` is the step-by-step
status table with every measured number; read it first.

## Layout

- `src/zemble/` -- `index/` (chunk/BM25/dense stores, create, file walker, symbols),
  `chunking/` (tree-sitter chunking, `capsule.py`), `ranking/`, `search.py`,
  `embedding/` (Embedder seam, providers, sqlite cache), `rerank/`, `languages/`
  (one `LanguageSpec` per bundled grammar: declaration/call node kinds plus path
  expressions; the home of every grammar-specific fact outside the Java and hwk lanes),
  `graph/` (Java + hwk extractors, `generic.py` spec-driven extractor for every other
  grammar, resolver, sqlite store, `facts.py` overlay, provider),
  `evidence/` (explain/outline/signatures), `dedup/`, `home/`, `daemon/`
  (server, client, watcher), `index_cache.py` (shared with the MCP server), `cli.py`,
  `mcp.py`, `installer/` + `agents/` (agent config templates).
- Surfaces stay OUT of the shared files: each package has its own `cli.py`/`mcp.py`
  and appends one entry to `_SUBCOMMAND_RUNNERS` / `register_*_tools` / the daemon
  `COMMANDS` table. Follow that pattern for anything new.
- `benchmarks/` -- upstream harness (`run_benchmark.py`, 63 repos in
  `~/.cache/zemble-bench`) plus `local/` (the javaweb eval set: 80 code + 10 template
  queries, `home_queries.json`), `rerank_sweep.py`, `evidence_eval.py`,
  `home_eval.py`, `profile_query.py`; results JSON under `benchmarks/results/`.
- `docs/` -- one doc per capability; `docs/graph-facts.md` is a CONTRACT other tools
  implement (do not change field names casually); `docs/comparison.md` is the
  seven-configuration table with hit rates.
- `javac-facts/` -- standalone javac plugin emitting graph facts; built with plain
  `javac`/`jar` via its `Makefile` (`make build`, `make test`). No Gradle here.

## Build, test, lint

```
uv venv && uv pip install -e ".[mcp,dev]"      # .venv for development
ZEMBLE_DAEMON=0 .venv/bin/pytest                # full suite (~30 s); conftest forces the guard anyway
.venv/bin/pytest -m "not slow"                  # skip the real-workspace journeys
ruff check src tests && ruff format --check src tests
```

- Never let a test spawn a real daemon; `tests/conftest.py` sets `ZEMBLE_DAEMON=0`.
- `mypy` is broken by a numpy-stub/py3.14 mismatch; not a gate.
- The machine install is a uv tool editable install (`~/.local/bin/zemble`) pointing at
  this checkout; after adding a dependency run
  `uv tool install --editable --reinstall ".[mcp]"` or the daemon lane misses it.

## Measuring (non-negotiable)

Every retrieval-affecting change is measured on BOTH eval sets before it ships, and
E0 must reproduce the recorded baseline first:

```
.venv/bin/python -m benchmarks.run_benchmark \
  --repos-file benchmarks/local/repos.json --annotations-dir benchmarks/local/annotations   # javaweb, ~2 min
.venv/bin/python -m benchmarks.run_benchmark                                                # upstream 63 repos, ~10 min
```

Report NDCG@10 AND hit@1/5/10, per kind. A change that loses on either set ships only
with a stated reason. Variants are env-switchable (`ZEMBLE_CAPSULE`, `ZEMBLE_EMBEDDER`,
`ZEMBLE_RERANKER`, `ZEMBLE_RERANK_ALPHA/K`); use `--label-suffix` so runs do not
overwrite each other. Never tune on individual eval queries.

## Conventions

- ASCII only in files. Docblocks: one-sentence summary, gotchas only. `AIDEV-NOTE:`
  for surprising logic; never delete one without instruction.
- Linear git history: no merge commits. Work on a branch/worktree, `git rebase main`,
  fast-forward. Commit subject starts with a real Unicode gitmoji, max 3 lines.
- Vocabularies have one home (enum/sealed type, exhaustive dispatch); unknown members
  fail closed. A language is one spec in `languages/catalog.py`; the graph and dedup both
  read it, and `tests/test_languages.py` fails the build on a node kind the grammar lacks
  or a bundled code grammar without a spec. Never guess node kinds: parse a fixture and
  print the tree. Ranking must stay bit-identical across refactors that are not meant
  to change it (prove it with the benchmark).
- MCP tools return their payload as an object (or as plain text), never as a
  `json.dumps` string from a `-> str` signature: that makes the client parse JSON
  out of JSON. Every tool is registered `@server.tool(structured_output=False)`: FastMCP
  otherwise wraps a non-object return type as `{"result": ...}` in `structuredContent`,
  and Claude Code renders that twin instead of the text. `tests/test_mcp.py` fails the
  build when a tool advertises an output schema or answers with structured content.
- Defaults are local/offline/no key. Hosted providers (Voyage) are opt-in via env;
  `docs/comparison.md` carries the recommendation.
- Secrets: `VOYAGE_API_KEY` lives in `~/.config/zemble/env` (mode 600) on the dev
  machine; load with `set -a; . ~/.config/zemble/env; set +a`. Never print it, never
  put it on a command line, never write it into the repo, logs, or docs.
- Two guards, two units: runaway WORK is refused from the walk in bytes (`ZEMBLE_INDEX_WORK_LIMIT_MB`, 180 MB, every embedder) and a runaway BILL is refused from the uncached set in money (`ZEMBLE_EMBED_BUDGET_USD`, $5.00; `ZEMBLE_EMBED_BUDGET_TOKENS` caps a model with no documented price, and `MAX_BUDGET_TOKENS` is the absolute volume backstop under the money ceiling, DERIVED as the budget at the price table's own dearest documented rate = 38.5M tokens, because `$5.00 / price` is only as honest as the price table). It binds by construction: the capsule overhead scales with FILE SIZE (measured 1.05x on 20 KB files up to 2.60x on 20 B ones), so 180 MB of source carries ~52M estimated tokens at the low end and ~130M at the high, 38.5M sits under even the low end, and a rate mistyped low narrows the ceiling instead of deleting it - the worst real bill a build at the work ceiling can carry is ~$16.87 at the dearest documented rate. It bounds the ESTIMATE, so a density that under-counts slips past it - only the byte work ceiling bounds that. Never price bytes: the embedding cache is invisible before chunking. The bill guard lives at the seams that BUY vectors, never inside `CachingEmbedder`, which `ZEMBLE_EMBED_CACHE=0` removes, and what a repeated text costs is the buyer's own answer (`pending_purchase`), never a rule the report owns. `zemble embed-status <path> [--exclude ...]` reports both ceilings and the verdict, embedding nothing.
- Caches: `~/.cache/zemble/` (indexes, `embeddings/*.sqlite`, `javac-facts/`,
  `daemon.log`); daemon socket `$XDG_RUNTIME_DIR/zemble/daemon.sock`. The embedding
  cache is keyed by chunk text + dims: changing capsule text means a full re-embed.
- Every MCP tool's `repo` is optional: `zemble.mcp_repo` fixes the default to the server
  process's start directory once at import and announces the resolved path in each tool's
  parameter description. Graph tools cap answers at `limit` (default 50) and always carry
  `total`, plus a `truncated` note when the cap bit.
- A path inside an already indexed root is served from that root's index, filtered to the
  sub-tree. Every tool speaks paths relative to the path the caller passed (the view rebases
  chunk paths in and out; graph, `dupes`, `explain`, `home` use that path's own graph/config),
  and the capsule's path segment is repo-relative (`<git-root name>/<inner path>`) so both
  roots embed the same text.
- Upstream remote is `upstream` (MinishLab/semble); origin is `11ways/zemble`. Keep
  Semble's attribution in README, CITATION.cff and LICENSE.
