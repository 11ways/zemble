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

## Later

Git-history retrieval, repo-adapted rerankers, hierarchical module vectors.

## Rules

Every step is measured against the upstream benchmark AND the javaweb eval set.
Nothing merges that regresses either without a stated reason.
