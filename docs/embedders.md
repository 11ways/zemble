# Embedders

Zemble embeds text through a pluggable `Embedder`. The default is a local
Model2Vec static model; API providers are opt-in and selected with one string.

## Spec grammar

An embedder spec is a scheme plus a model, with an optional output width:

```
model2vec:<hf-model-or-local-path>
voyage:<model>[@<dims>]
openai:<base_url>#<model>[@<dims>]
```

Examples:

```
model2vec:minishlab/potion-code-16M-v2      # the default
voyage:voyage-code-4                        # Voyage's own default width (1024)
voyage:voyage-code-4@256                    # Matryoshka-truncated to 256
voyage:voyage-4-lite@512
openai:http://localhost:11434/v1#nomic-embed-text
openai:https://api.openai.com/v1#text-embedding-3-small@1536
```

A spec without a known scheme is an error, not a guess. `model2vec:` rejects
`@<dims>`, because a static model's width is fixed by the model itself.

The normalized form of a spec, with the resolved width made explicit, is the
embedder's `model_id`. It is what an index records, and two indexes with the
same `model_id` are comparable.

## Choosing one

In order of precedence:

1. `--embedder <spec>` on `zemble search`, `zemble find-related` and `zemble stats`.
2. `ZEMBLE_EMBEDDER=<spec>`.
3. `ZEMBLE_MODEL_NAME=<hf-model>`, the legacy variable, which maps to
   `model2vec:<hf-model>`.
4. The default, `model2vec:minishlab/potion-code-16M-v2`.

`zemble stats <path>` prints the embedder and width of an index alongside its
file and chunk counts.

## Providers

### Model2Vec (local, default)

A static model, no forward pass, no network. Documents and queries are embedded
identically. The model is loaded from the Hugging Face cache or a local path.

### Voyage

`POST https://api.voyageai.com/v1/embeddings`, authenticated with a bearer token
read from `VOYAGE_API_KEY`. An unset key is refused by name before any request is
made. Documents and queries are asymmetric: they are sent with `input_type` set
to `document` and `query` respectively, which is what the code models are trained
for. `@<dims>` is passed as `output_dimension`.

Requests are batched to 128 texts and roughly 100K estimated tokens, both well
under the documented ceilings (1,000 texts; 320K tokens for `voyage-code-4` and
`voyage-4`, 1M for `voyage-4-lite`). `truncation` is on, so an oversized chunk is
shortened by the API rather than failing the batch.

### OpenAI-compatible

Any server exposing `POST <base_url>/embeddings` in the OpenAI shape: Ollama,
LM Studio, vLLM, OpenAI itself. The API key is read from `OPENAI_API_KEY`, or
from the variable named by `ZEMBLE_EMBEDDER_KEY_ENV`; when neither is set, no
`Authorization` header is sent, which is what local servers want.

The OpenAI schema has no `input_type`, so documents and queries are embedded
identically here even if the model behind it is asymmetric. `@<dims>` is passed
as `dimensions`.

### Retries

429 and 5xx are retried up to 5 attempts with exponential backoff and jitter,
honouring `Retry-After`. Every other 4xx is raised immediately, carrying the
provider's own message.

## The embedding cache

Remote providers are wrapped in a content-hash cache, so a chunk that has not
changed is never paid for twice.

- Location: `~/.cache/zemble/embeddings/<family>.sqlite` (or
  `$ZEMBLE_CACHE_LOCATION/embeddings/`, or the platform cache dir).
- One file per embedder **family** (scheme plus model, without dimensions).
- Table: `(text_sha256, dims, vec)`, primary key `(text_sha256, dims)`.
- Matryoshka fallback: a request at 256 dimensions is served by slicing a stored
  1024-dimension vector of the same text and renormalizing. The slice is not
  stored, because it is derivable.
- Misses are handed to the provider 512 texts at a time and every slice is written
  before the next is asked for. A cold workspace index is one `embed_documents` call
  of tens of thousands of texts and half an hour of paid requests; without that
  boundary a single failure at the end throws away every vector already bought, and
  the response rows pile up in memory (6.5 GB peak on javaweb at 2048 dimensions,
  0.75 GB with the boundary).
- Only documents are cached. Queries always reach the provider: for an
  asymmetric model a query vector must never be served where a document vector
  was asked for.
- `ZEMBLE_EMBED_CACHE=0` disables the wrapper entirely.

Model2Vec is deliberately *not* wrapped: embedding locally is faster than a
sqlite round trip.

## Mixing embedders is refused

An index records its `embedder` and `dimensions`. If a cached index was built
with a different embedder than the one requested, zemble logs one line naming
both and rebuilds. Vectors from two models are never mixed.

## Running Voyage

```bash
export VOYAGE_API_KEY=pa-...
zemble search "where is the retry policy" ./my-project --embedder voyage:voyage-code-4@256
zemble stats ./my-project --embedder voyage:voyage-code-4@256
```

The first run embeds every chunk over the API; later runs pay only for chunks
whose content changed, because everything else comes out of the sqlite cache.

### Cost

`voyage-code-4` is $0.12 per million tokens, and the first 200M tokens are free.
A full index of the javaweb workspace measured 15.5M tokens over 73,957 chunks and
578 requests, i.e. $1.86 of paid-equivalent inside the free tier, and a re-index
after the cache is warm costs only the changed files. See `docs/voyage.md` for what
that buys in retrieval quality.

## Cost visibility and the budget guard

A hosted embedder bills per token, and the only moment that matters is *before* a
build starts. Two things make that visible.

### `zemble embed-status`

```bash
zemble embed-status /path/to/workspace --embedder voyage:voyage-4-lite@1024
zemble embed-status . --content all --json
```

It chunks the tree exactly as a build would - same walker, same capsules, same
mtime-based reuse of a previous index - then asks the sqlite cache which of the
would-be-embedded texts are already paid for. It **never embeds anything**, never
contacts a provider (not even to learn a model's vector width) and needs no API key.

It reports chunks total / reusable from the previous index / cached / uncached, the
estimated tokens and cost for the uncached ones, the cache file, the embedder, and
whether the budget below would refuse the build.

On the javaweb workspace (77,092 chunks) a cold pass costs about 11 s of chunking plus
0.3 s of cache lookup; when the previous index covers every file the walk alone answers
in well under a second.

### The budget

Before a **remote** embedder embeds a batch of uncached documents, the whole pending
set is estimated and compared with `ZEMBLE_EMBED_BUDGET_TOKENS` (default 2,000,000,
roughly one full javaweb index). Over budget is a loud `EmbeddingBudgetExceeded` naming
the estimate, the price, the budget and the two ways out, and **nothing is sent**:

```
Refusing to embed 61,000 uncached chunk(s) with voyage:voyage-4-lite@1024: ~15,500,000
estimated tokens (~$0.31) exceeds the budget of 2,000,000 tokens. Pass --yes (or set
ZEMBLE_EMBED_CONFIRM=1) to embed anyway, or raise ZEMBLE_EMBED_BUDGET_TOKENS.
```

- The check sits in the caching embedder, at the one point where the uncached set for a
  whole build is known, so the CLI, the MCP server and the daemon all inherit it. It is
  never per 512-text slice.
- Local embedders are never gated, with or without a confirmation.
- `zemble search`, `zemble stats` and `zemble find-related` take `-y/--yes`; in-process
  they print the refusal and exit non-zero.
- The daemon catches a refusal, logs one line, keeps the previous index serving (nothing
  is embedded and nothing is swapped) and shows it in `zemble daemon status` as the
  root's `last_error`.
- Every paid embed logs one INFO line first:
  `embedding 812 uncached chunk(s), ~171000 tokens, ~$0.02 with voyage:voyage-4-lite@1024`.

### Prices

Per million tokens, from each provider's price list. A model that is not in the table is
reported as "unknown price" rather than guessed, and an unknown price never becomes free.

| model | $/M tokens | | model | $/M tokens |
| --- | --- | - | --- | --- |
| `voyage-code-4` | 0.12 | | `voyage-3.5` | 0.06 |
| `voyage-4` | 0.06 | | `voyage-3.5-lite` | 0.02 |
| `voyage-4-lite` | 0.02 | | `text-embedding-3-small` | 0.02 |
| `model2vec:*` | free (local) | | `text-embedding-3-large` | 0.13 |

**The estimate is an estimate.** Tokens are counted as characters / 3.6, the density
measured on javaweb (15,526,808 provider-reported tokens for 73,957 chunks); a tree of
minified or non-Latin text will differ. The provider's own `usage.total_tokens` is what
is billed.

## The user env file

All `ZEMBLE_*` settings and provider keys can live in `~/.config/zemble/env`
(`KEY=VALUE` lines, `#` comments, optional `export`; `ZEMBLE_ENV_FILE` overrides the
location). `zemble.userenv.load_user_env` applies it at every entry point (CLI, MCP
server, `python -m zemble.daemon`) for keys not already set, so the process environment
always wins. Tests pin `ZEMBLE_ENV_FILE` to a missing path so a developer's real file is
never read by the suite.
