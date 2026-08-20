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
The javaweb workspace measures roughly 31M tokens across 63 repositories, so a
full index of it is comfortably inside the free tier, and a re-index after the
cache is warm costs only the changed files.
