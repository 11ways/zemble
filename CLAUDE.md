# zemble

Fork of Semble (MinishLab) with a workspace code-intelligence layer on top: symbol
graph + compiler facts, context capsules, evidence bundles, duplication detection,
`home`, pluggable embedders/rerankers, warm daemon, columnar index. Python 3.11+,
one small Java sub-project (`javac-facts/`). `docs/plan.md` is the step-by-step
status table with every measured number; read it first.

## Layout

- `src/zemble/` -- `index/` (chunk/BM25/dense stores, create, file walker, symbols),
  `chunking/` (tree-sitter chunking, `capsule.py`), `ranking/`, `search.py`,
  `embedding/` (Embedder seam, providers, sqlite cache), `rerank/`, `graph/`
  (Java + hwk extractors, resolver, sqlite store, `facts.py` overlay, provider),
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
  fail closed. Ranking must stay bit-identical across refactors that are not meant
  to change it (prove it with the benchmark).
- Defaults are local/offline/no key. Hosted providers (Voyage) are opt-in via env;
  `docs/comparison.md` carries the recommendation.
- Secrets: `VOYAGE_API_KEY` lives in `~/.config/zemble/env` (mode 600) on the dev
  machine; load with `set -a; . ~/.config/zemble/env; set +a`. Never print it, never
  put it on a command line, never write it into the repo, logs, or docs.
- Paid embedders are budget-guarded (`ZEMBLE_EMBED_BUDGET_TOKENS`, refusal before any request); `zemble embed-status <path>` reports what a build would embed and cost, embedding nothing.
- Caches: `~/.cache/zemble/` (indexes, `embeddings/*.sqlite`, `javac-facts/`,
  `daemon.log`); daemon socket `$XDG_RUNTIME_DIR/zemble/daemon.sock`. The embedding
  cache is keyed by chunk text + dims: changing capsule text means a full re-embed.
- Upstream remote is `upstream` (MinishLab/semble); origin is `11ways/zemble`. Keep
  Semble's attribution in README, CITATION.cff and LICENSE.
