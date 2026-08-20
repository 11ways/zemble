# zemble plan

## Step 0: Fork hygiene

Fork, upstream remote, upstream benchmark baseline recorded, javaweb eval set
(50-100 real queries with expected files), cold/warm query profile to see where
the milliseconds go.

## Step 1: Embedder seam + external providers

An `Embedder` interface (`embed_documents`, `embed_queries`, `dimensions`,
`model_id`). Implementations: local Model2Vec (default), Voyage API
(voyage-code-4, voyage-4-lite), OpenAI-compatible HTTP (Ollama and friends).

A content-hash embedding cache keyed on (model_id, dims, chunk_hash). Index
metadata records the embedder; mixing embedders is refused loudly.

Benchmark Potion against voyage-code-4 at 256 and 2048 dimensions (one API pass,
Matryoshka slice).

## Step 2: Warm daemon + incremental watch

A long-lived process per workspace holding the index in RAM, with the CLI and
MCP server as thin clients. A filesystem watch reindexes changed files through
the embedding cache.

## Step 3: Context capsules

Prefix the embedding text with path, package, enclosing class, signature,
annotations and interfaces; return the original code only. Measure on both eval
sets.

## Step 4: Symbol graph for Java

tree-sitter-java extraction of defs, refs, extends, implements, overrides,
annotations, imports, and test-to-subject edges. Stored beside the index and
incrementally updated. Queries: callers, implementations, tests-of, definition,
related-by-graph. Kept behind a provider interface so zenit-dev (javac-grade)
can replace it later.

## Step 5: Evidence bundles

`explain <query> --budget N` = search, then one-hop graph expansion, dedupe, and
pack under a token budget with a one-line reason per item. Plus signatures-only
and file-outline modes. Measure tokens-to-answer against raw Read.

## Step 6: Duplication mode

Normalized-AST hashing for exact and alpha-renamed clone classes over Java
methods. Logic duplication = embedding candidates plus a structural check (call
set, control-flow shape, literals) with a stated reason. Output shape mirrors
`zenit-dev duplication`.

## Step 7: Pairwise reranker behind a flag

A cross-encoder over the top 50 (local small model, optionally the Voyage rerank
API). Kept only if it beats the heuristics on the hard subset.

## Status (2026-08-20)

Every step 0-10 shipped on `main`; step 7 (reranker) shipped behind a flag, default `none`. Linear history,
one agent branch per step, rebased in. Numbers below are from the docs each
step wrote; the eval sets are the upstream 63-repo benchmark and the javaweb
local set (benchmarks/local, 80 queries).

| Step | Shipped | Measured |
| --- | --- | --- |
| 0 | fork, `benchmarks/local` javaweb set, profile | upstream NDCG@10 0.8517 (bit-identical to unmodified upstream); javaweb 0.532; cold query 12.2 s of which 11.5 s index load |
| 1 | `Embedder` seam (`model2vec:` / `voyage:` / `openai:` specs), sqlite content-hash cache with Matryoshka slicing, `--embedder`, `zemble stats` | measured with a real key (`docs/voyage.md`): javaweb 0.567 -> **0.669** with `voyage-4-lite@1024` (0.667 for `voyage-code-4@2048`, 0.636 even at 256 sliced from cache), upstream +0.0186 over 59 of 63 repos; 15.5M tokens and $0.31-$1.86 for a full javaweb index, zero requests on a warm rebuild, but 11 ms -> 267 ms per query. DEFAULT STAYS local Model2Vec (offline, no key, code stays home); `voyage-4-lite` is the documented upgrade. Fusion weight +0.15 wins with Voyage and loses with the default, so it is DECLARED per embedder (`semantic_weight_bonus`, 0.15 for HTTP providers, 0.0 for Model2Vec): javaweb `voyage-4-lite@1024` 0.680 -> **0.710**, upstream 0.8878 -> 0.8898 over all 63 repos, default embedder bit-identical |
| 2 | on-demand daemon (unix socket, idle exit 30 min, LRU), watchfiles incremental reindex + graph refresh, CLI/MCP daemon-first with in-process fallback | CLI query 1.52 s -> 0.58 s (the rest is the client interpreter); RSS 294 MB with the workspace loaded; one-file rebuild 6.2 s |
| 3 | context capsules (path/package/type chain/signature/imports) in the dense text AND the BM25 document, `Chunk.context` | javaweb 0.532 -> 0.567 (+0.035), upstream 0.8517 -> 0.8643 (+0.013); BM25 carries the win |
| 4 | Java symbol graph (tree-sitter, two-pass resolver with EXACT/UNIQUE_NAME/AMBIGUOUS/UNRESOLVED, sqlite, `GraphProvider` seam), CLI `zemble graph *`, 5 MCP tools | whole workspace 33 s, 101k symbols / 923k edges; queries < 12 ms; six CLAUDE.md facts asserted |
| 5 | `explain` (tiered budgeted bundles, degrade-before-drop), `outline`, `signatures`; CLI + MCP | bundles draw level with search hit rate only at 3000 tokens (tier 0 IS the search set; the graph hop found something search missed in 1/80); `outline` is the token saver (~150-300 tokens per class) |
| 6 | `zemble dupes`: exact / alpha-renamed / logic clone classes over Java bodies + statement windows, zenit-dev-shaped report, MCP tool | workspace exact+renamed 45 s; first real finding: `trimToNull`-family copied 9x across 5 repos; detector false positive (`this.f = f` ctors) found and fixed |
| 8 | `zemble home` (config-driven `.zemble/home.toml`: module order, forbidden deps, declared-home table, skills, rules; existing mechanisms + candidate homes + verdict + checklist; CLI/MCP/daemon) | 61 paraphrased capability rows: declared home ranked #1 in 90% (77% with the table lane off), top-3 93%; ~0.3 s per answer |
| 7 | `Reranker` seam (`none` / `cross:<hf>` / `voyage:<model>`), post-fusion window blend, optional `zemble[rerank]` extra, `--reranker` / `ZEMBLE_RERANKER` | local cross-encoder: javaweb +0.024 at p50 ~2.6 s. Hosted, measured with a real key (`docs/voyage.md`): `voyage:rerank-2.5` context/0.7/50 is javaweb **+0.1135** (0.567 -> 0.680; consumer 0.147 -> 0.500) with upstream at -0.0006, i.e. quality and no-regression bars both passed - but p50 570 ms against a 300 ms bar, and one 50-passage round trip alone is 309 ms, so no hosted config can pass it. DEFAULT STAYS NONE; the flag is now a recommendation rather than a curiosity, at $0.0006 a query |
| 9 | `.hwk` templates in the code lane (HTML-grammar chunking + lexical capsule), template graph extractor (TEMPLATE/BLOCK symbols; REFERENCES to custom elements via Hawkeye's tag rule, CALLS to `@HawkeyeFunction`s by namespace+name, EXTENDS/render includes), 10 template queries in the eval set | 619 templates / 3,114 chunks; 7,806 template edges (element refs 98.7% EXACT, includes 100%); `graph callers StringFunctions.presence` now lists templates; original 80 code queries 0.567 -> 0.545 (templates that use a mechanism outrank the annotated Java file on 5 queries; NOT tuned away), new 10 at 0.655 |
| 10 | Graph facts overlay (generic JSONL contract `docs/graph-facts.md`: sha256 freshness, per-language `RefMapper`, fresh facts REPLACE tree-sitter CALLS/OVERRIDES/EXTENDS/IMPLEMENTS per file, `zemble graph facts status`) + `zemble-javac-facts` (standalone javac plugin in `javac-facts/`, any Java project) + javaweb wiring (`ZembleFactsInstaller` in protoblast-gradle-plugin, opt-in by jar presence, one file per JavaCompile task) | workspace: 1,483 files covered, all fresh; in covered files calls = 32,505 EXACT + 20,570 external, ZERO by-name/ambiguous (uncovered files keep the ladder: 138k ambiguous); +2.6 s per full repo build; next lever = map generated `Tpl_*` facts back to `.hwk` via source maps |
| B1 | columnar BM25 (CSR + vectorized scoring), mmap vectors, columnar lazy chunks, precomputed symbol-definition table, scandir walker, lazy imports | cold query 7.0 s -> 1.0 s, warm symbol 379 -> 17 ms, NL 95 -> 16 ms; ranking bit-identical on both sets |

Open: query-kind-aware tier order in `explain` (the `consumer` kind is where bundles should win and
do not); facts coverage for repos on the publish-only plugin variant and the apps;
mapping generated-template facts back to `.hwk`; revisit the 5 code-query annotations
that now have template answers.

## Later

Git-history retrieval, repo-adapted rerankers, hierarchical module vectors.

## Rules

Every step is measured against the upstream benchmark AND the javaweb eval set.
Nothing merges that regresses either without a stated reason.
