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

## Status (2026-08-21)

Every step 0-11 shipped on `main`; step 7 (reranker) shipped behind a flag, default `none`; Jelle's machine runs voyage-4-lite + rerank-2.5-lite via ~/.config/zemble/env. Linear history,
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
| 11 | Engineering wave 2026-08-21: daemon change-set rebuild + delta BM25 + non-blocking swap + atomic column writes; generated `Tpl_*` facts mapped to `.hwk` via Hawkeye `// @hwk:` markers; incremental graph refresh (`SqliteLookup`, planned facts overlay, `decl_keys`, two pre-existing correctness fixes); per-embedder fusion bonus (+0.15 hosted); `explain` intent classifier + intent tier orders behind `--intent` (default order kept: measured no win); dupes lanes/keys/ignore/baseline/brief + MCP double-encoding audit (9 tools fixed); sub-path served from an ancestor index + repo-relative capsule paths + refusal hint; pre-flight `embed-status` + budget guard; `~/.config/zemble/env` loader; `zemble.parallel` (never fork a threaded host) | one-file edit: index 247 ms steady, graph 0.6-2.8 s (was ~40 s), queries never blocked; ignored generated facts 65,857 -> 36 (51k mapped onto 147 templates); javaweb voyage-4-lite 0.680 -> 0.710 (hit@5 83%), upstream 4-lite 0.890 (all 63 repos); dupes over MCP 8.6 s instead of hanging; sub-repo search served from the workspace index with zero embedding |
| 12 | Dupes feedback round 2026-08-21 (from the javaweb dedup campaign): content-only class keys (no file paths; logic keyed by the alpha-renamed stream) so keys survive file moves and are scan-root independent; every nested `.zemble/dupes.ignore` honoured by an ancestor-root scan, violations named per file; baseline diff gains a CHANGED bucket (gone entry paired with the re-keyed class on shared member files, score delta printed; a re-keyed suppressed class silently claims its entry) and `baseline`/`save_baseline` booleans over MCP against `<repo>/.zemble/dupes.baseline.json` (BASELINE_VERSION 2, v1 refused); logic reasons aggregated at 3+ copies (consensus line + outliers only); cross-module home verdicts driven by `.zemble/home.toml` (candidate-home / forbidden-dep / no-shared-ancestor per class spanning declared modules, on text + JSON) | key stability, ignore merge, changed pairing, aggregation and all three verdicts covered by journeys (15 dedup tests); full suite 682 green; verdict machinery reuses `HomeConfig` unchanged |
| 13 | `existing-home`, a fourth cross-module dupes verdict: a clone class whose most-core member module is the candidate home AND whose copy in that module is a whole body named by a declared capability-table row (`[[tables]]` of `home.toml`) is reported as an EXISTING mechanism, with `symbol`, `location` and `declared-row` `evidence` on text and JSON. Evidence is declared rows only - no graph, no callers, no embeddings, no search - so classification never triggers indexing; symbol matching fails closed (a bare `Type` row covers its members, `Other.weave` never claims `OneShared.weave`); `forbidden-dep` still outranks it; a missing table file degrades to `candidate-home` plus a note | 11 dedup journeys added (existing-home from a row and from a bare class row, candidate-only, near-miss, forbidden precedence, no-shared-ancestor, missing and unparseable table, key stability across clone/baseline/ignore, and a no-retrieval guard); clone, baseline and ignore keys proven unmoved; javaweb production renamed scan, warm: 23.7 s before, 23.4 s after (noise - the tables are read once per run, and only once a class spans two declared modules). Live outcome on javaweb: 1 existing-home class (`McpServerConnection.findMeaningfulMessage` in zenit-ai, named by CLAUDE.md's AI-layer row through the bare-class rule, copy in quirkyquarters); `Texts.trimmedOrNull` stays candidate-home because no row names `Texts` |
| 14 | Bug-report wave 2026-08-21 (two assistant reports: `dupes` on a Zig repo, `existing-home` on javaweb). Findings: the Java gate, `order`-as-dependency, the money-only embed guard and the single `existing-home` verdict were implementation shortcuts retro-documented as deliberate; none was an approved scope. Shipped, one branch per stream, rebased in: (A) `dupes` is profile-driven (`dedup/languages/`: `LanguageProfile` registry keyed by extension, Java + Zig, drift tests against the real grammars' node kinds; `units.py` holds no language vocabulary), a zero-files scan refuses instead of printing `No duplication found.`, per-file extraction failures are counted and named, relative `--paths`/`--exclude` resolve against the scan root, `Unit.modifiers` + `visibility`; (B) a pre-parse size guard for EVERY embedder lane (`index/scope.py`: walk + byte estimate in ~0.2 s, top-8 directory breakdown, remedies name `.zembleignore` and sub-path narrowing; local-lane default 50M tokens, priced-lane 2M, one env knob), daemon refusals carry `kind: refused` and are never retried in-process, `paths`/`exclude` on `search`/`find_related`/`explain` over MCP, CLI and daemon (query-time filter when the index exists, build-time exclusion otherwise, default cache key unchanged); (C) `runtime/` identity (`zemble status`, MCP `status` tool, daemon `ping`/`status` carry version/revision/start time/`stale`), pool children take `PDEATHSIG` + a parent watchdog, MCP server exit hygiene; (D) `home.toml` gains a dependency DAG (`depends_on` + Gradle discovery, declarations override; `order` is ranking only; `Reachability` with UNKNOWN distinct from UNREACHABLE; source sets common/server/browser/test with one compatibility table; `RowMatchKind` so a bare `Type` row never counts as a declaration; sibling/misplaced lanes with `suggested_home` = deepest shared dependency), and the dupes verdict split into `existing-reusable-api` / `existing-implementation-not-api` / `candidate-home` / `siblings-need-common-home` / `review-required` / `forbidden-dep` / `no-shared-ancestor` with kind-tagged evidence; `call or extend it` appears only on `existing-reusable-api`. | Java equivalence: javaweb 6309 files / 365 classes, 0 key or member differences before vs after the profile refactor; sketerm (453 `.zig` files) analyzed 453/0 failed, 594 exact+renamed classes (82.9 s, windows dominate; 2.8 s with `--no-windows`); sketerm refusal 94 s -> 0.24 s, local lane now guarded; javaweb Gradle discovery 154 edges / 27 modules in 0.14 s, zenit-flow <-> zenit-widget unreachable both ways; home eval 61 positives hit@1 0.869 -> 0.852, home-ok 0.803 -> 0.787 (two rules named in docs/home.md), 12 negatives added, 5 still over-confident (3 retrieval, 2 need an absolute floor not tuned here); javaweb dupes: all 14 former `existing-home` verdicts gone, 0 `call or extend` lines remain (exact/renamed 26 judged: 18 candidate, 6 siblings, 2 forbidden; logic 44 judged: 43 review-required); retrieval bit-identical (javaweb local embedder NDCG@10 0.5576, hit@1/5/10 0.5111/0.6222/0.6778 before and after); suite 698 -> 800+ tests. |
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

A feature that emits a VERDICT (home, existing-home, any "call it / extend it" instruction)
ships with negative eval cases (does not exist; exists but not reusable; siblings) and a
per-query expected verdict; a rate of confident answers is not a metric.
A scan that analyzed nothing never renders as a clean result; an unknown language,
module pair, dependency or visibility fails closed and says "unknown".
Scope restrictions ("Java only", "order is permission") are recorded in this plan as
decisions with a rationale BEFORE they ship, or they do not ship.
