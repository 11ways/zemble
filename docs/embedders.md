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

## The fusion profile

Search fuses a dense lane with BM25, and `zemble.ranking.weighting` decides how much of
that fusion the dense lane gets: 0.30 for a query that looks like a symbol lookup, 0.50
otherwise. Those two constants were tuned around the static default embedder, and they
are measurably too low for a contextual one - the same weight that wins with Voyage
loses with Model2Vec (`docs/voyage.md`).

So the number is not global. Every embedder declares its own share:

```python
class Model2VecEmbedder:
    semantic_weight_bonus = 0.0     # the weights were tuned on exactly this

class HttpEmbedder:
    semantic_weight_bonus = 0.15    # hosted contextual models, measured
```

- The bonus is **added to both** auto-detected weights (0.30 -> 0.45, 0.50 -> 0.65) and
  the result is clamped to [0, 1].
- An **explicit** `alpha` - `--alpha`, or the parameter on `search()` - is the caller's
  number and is never adjusted.
- An embedder that declares nothing gets **0.0**, which is exactly the shipped weights.
  An unrecognised embedder never silently retunes ranking.
- `ZEMBLE_SEMANTIC_WEIGHT_BONUS` overrides every declaration, which is how the sweep in
  `docs/voyage.md` was run. A value that is not a number is ignored with one warning.
- The caching wrapper forwards the wrapped provider's bonus: caching changes who pays,
  not how good the vectors are.

Measured on the javaweb set: `voyage-4-lite@1024` goes 0.6803 -> 0.7102 NDCG@10 with the
bonus, `potion-code-16M-v2` goes 0.5569 -> 0.5533 without one, and upstream moves
+0.0020. The sweep, the per-kind breakdown and the decision are in `docs/voyage.md`.

## The embedding cache

Remote providers are wrapped in a content-hash cache, so a chunk that has not
changed is never paid for twice.

- Location: `~/.cache/zemble/embeddings/<family>.sqlite` (or
  `$ZEMBLE_CACHE_LOCATION/embeddings/`, or the platform cache dir).
- One file per embedder **family** (scheme plus model, without dimensions).
- WAL readers leave no open transaction between preflight and provider work.
  Vector batches and their use stamps commit atomically under `BEGIN IMMEDIATE`;
  contending writers use SQLite's bounded 30-second busy timeout, without retrying
  provider calls. A per-family `.sqlite.init.lock` serializes first-open journal
  transitions and schema setup. Garbage collection selects and deletes under one
  writer transaction.
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
- `ZEMBLE_EMBED_CACHE=0` disables the wrapper entirely. It does NOT disable the budget
  guard, which sits at the seams that buy rather than inside the wrapper; an uncached build
  simply has nothing already paid for, so every text counts against the ceiling.

Model2Vec is deliberately *not* wrapped: embedding locally is faster than a
sqlite round trip.

## The capsule path is repo-relative

The cache is keyed by the text a chunk is embedded as, and that text starts with the
chunk's context capsule, whose first segment is a path. That path is therefore **not**
index-root-relative: for a file inside a git working tree it is
`<git-root directory name>/<path inside that repo>`, resolved from the nearest ancestor
holding a `.git` entry and cached per directory. Outside any git repository it falls back
to the index-root-relative path.

Without this, the same file embedded as part of a workspace
(`zenit/src/.../PageWindow.java`) and as its own repo (`src/.../PageWindow.java`) is two
different texts, and a sub-repo index misses the cache for every chunk it holds - which
is exactly how a refusal to embed ~15,000 already-paid-for chunks happened.

- `Chunk.file_path` is unchanged: it stays relative to the index root.
- For a workspace whose root is not itself a repo (each sub-directory is), the text is
  byte-identical to what it was before this rule existed: 88,592 of 88,592 javaweb
  capsules unchanged, so no index rebuilds and `CACHE_FORMAT_VERSION` is not bumped.
- An index built *for a sub-repo root* before this rule keeps serving with its old
  capsules until a file changes (only re-chunked files pick up the new text). Clear it
  with `zemble clear index` to move it over in one go; the embeddings then come from the
  cache and cost nothing.

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
zemble embed-status . --exclude 'vendored/' 'third_party/'
```

It chunks the tree exactly as a build would - same walker, same capsules, same
mtime-based reuse of a previous index - then asks the sqlite cache which of the
would-be-embedded texts are already paid for. It **never embeds anything**, never
contacts a provider (not even to learn a model's vector width) and needs no API key.

It reports chunks total / reusable from the previous index / cached / uncached, the
estimated tokens and cost for the uncached ones, the cache file, the embedder, the source
volume a build would chunk against the work ceiling, the spending ceilings, and whether a
build would be refused. The verdict covers **both** guards below, and reads each one's
verdict from the guard's own home (`work_refusal`, `bill_refusal`) over the inputs the build
would use, because a report that knew only about the bill once called a build affordable that
the work guard refused - and then, one layer down, computed the work verdict here from
different inputs. `--exclude` takes the same patterns a build does, so the report can model
the recovery a refusal advertises rather than answering for a build nobody is running.

```
work       1.3 MB of source to chunk, against a 180.0 MB ceiling
budget     $5.00 and 38,461,538 tokens
verdict    a build would be allowed
```

What a repeated chunk costs is asked of the BUYER, through the same `pending_purchase` seam
the guard reads: the caching embedder gives every copy of one text a single provider slot, so
it is billed once, while the bare remote embedder `ZEMBLE_EMBED_CACHE=0` hands a build really
does buy every copy. Deduplicating unconditionally reported a twentieth of the bill that
build would be refused over.

The report asks that question UNPROBED (`pending_purchase(..., may_probe=False)`), because
deciding what is already bought reads a vector width and reading one off a model that declares
none is a provider request this report never makes. The width then comes from what the family's
own cache file holds, which leaves two cases where the lanes differ, both confined to a model
that declares no width. A file holding two widths over-reports: the report cannot tell which one
a build would read, so it counts nothing as stored. A file holding ONE width that is not the
width the build resolves under-reports: the report matches every digest at the stored width and
can answer "nothing to buy" for a build that then buys the lot. Neither is a spending hole - the
build's own guard measures the real set through `pending_documents` - but `would_refuse` is
advisory for such a model, in both directions.

On the javaweb workspace (77,092 chunks) a cold pass costs about 11 s of chunking plus
0.3 s of cache lookup; when the previous index covers every file the walk alone answers
in well under a second.

### The two guards

Two different harms, two guards, two units: **refuse work by work, refuse money by money.**
Only the cache-aware number gates money.

| Harm | Guard | Where | Unit | Applies to |
| --- | --- | --- | --- | --- |
| runaway WORK (minutes of chunking) | `require_affordable_scope` | `index/scope.py`, pre-parse | BYTES of source a build would chunk | every embedder, local included |
| runaway BILL | `require_affordable_bill` | `embedding/pricing.py`, post-chunk | USD of the UNCACHED texts, or their VOLUME where no bill can be computed | every remote embedder |

**Query-side spend sits outside both guards, deliberately.** They exist because a BUILD is
unbounded in the size of the tree handed to it; a query is not. `zemble.search` embeds the
query itself (`embed_queries`, one short text - a few dozen tokens, and under a millionth of
a dollar at any rate in the price table), and a hosted reranker (`rerank/voyage.py`) scores
one window of candidates per query. What bounds that is the caller's own two constants, not
anything walked off disk: `ZEMBLE_RERANK_K` passages (default 50), split into requests of at
most 100 documents and ~100,000 estimated tokens each, with `truncation` on so a long passage
cannot lift a request past the provider's ceiling. `VoyageReranker` keeps its own running
`total_tokens` and `request_count` from the provider's own usage figures. Putting a per-build
ceiling in front of a per-query cost would refuse a search for the size of a repository it
never reads.

The pre-parse guard cannot know a bill. The content-addressed embedding cache is invisible
to it by construction - there is no chunk text to hash yet - so a tree whose chunks were all
paid for last week looks exactly like a tree nobody has ever indexed. Pricing those bytes
anyway is what refused `home` on a workspace whose real bill was $0.000016.

#### 1. The pre-parse work guard (every embedder, local included)

Before a local index is chunked, the tree is walked - the same walker a build uses, so
the same `.gitignore`, `.zembleignore`, default-ignored directories and 1 MB file cap
apply. Files a previous index already covers unchanged are left out, because a build would
reuse them without embedding anything, so an incremental rebuild is never refused for the
size of the tree it already indexed. Over the ceiling is an `OversizedRootRefused`, and
**nothing is parsed**:

```
Refusing to index /home/me/sketerm with model2vec:minishlab/potion-code-16M-v2: 7,372
files, 253.4 MB of source exceeds the 180.0 MB this build may chunk. Nothing was parsed
or embedded.
  zig-pkg/                    6821 files  234.0 MB  (~92%)
  src/                         461 files  17.9 MB  (~7%)
  vendor/                       79 files  1.3 MB  (~1%)
Exclude paths with /home/me/sketerm/.zembleignore (gitignore syntax), or point repo at a
sub-path such as /home/me/sketerm/src, or raise ZEMBLE_INDEX_WORK_LIMIT_MB / set
ZEMBLE_EMBED_CONFIRM=1 (--yes on the CLI) in the environment of the process that builds
(a running daemon does not see a client's environment; restart it).
```

The ceiling is `ZEMBLE_INDEX_WORK_LIMIT_MB`, named in MEGABYTES because that is the unit a
human reads a tree in, and it defaults to **180 MB of source** - where runaway work begins.
A real multi-repo workspace is ordinary work and is not refused: javaweb measures 9,039
files and 64.0 MB of code, 78.7 MB with docs and config. The tree above, which carries
thirteen copies of its own source in a package directory, is refused. `0` or less disables
the guard. It names no price and no token count: it cannot know either.

This guard exists because the expensive half of an oversized build is the parse, not the
embed: on a 414 MB tree, chunking took 94 s before the old post-chunk guard could refuse,
and a local embedder was never gated at all and would have run the whole thing to
completion. The walk itself takes 0.2 s.

The byte estimate is a *lower* bound on what is embedded - a context capsule adds a
header to every chunk, measured at +21% on this repository and +52% on the small test
fixture tree. That is honest for work, and void for money.

Both scope guards sit at `create_index_from_path`, the one construction seam every index
build passes, so the daemon's watcher rebuild is judged by exactly the same rule as the
CLI. Before that they sat above `ZembleIndex.from_path` only, and the daemon re-chunked
and re-embedded, unguarded, the very tree the CLI had just been refused.

A rebuild the watcher drives from a KNOWN change set is measured on the paths it names, not
by walking the tree: `plan_changed_files` exists to avoid that walk, and walking anyway cost
0.26 s and 9,766 stats on every coalesced file event. It also measured the wrong set - every
file whose modification time had drifted, rather than the ones the build would chunk - so
after a large `git checkout` a one-file rebuild could be refused for drift it would have
REUSED, and since a refused rebuild swaps nothing the manifest never advanced and every later
event refused again, wedging that root until the daemon restarted.

#### 2. The post-chunk bill guard (remote embedders)

Before a **remote** embedder embeds a batch of uncached documents, the whole pending set -
and only that set, the cached texts are already excluded - is priced and compared with
`ZEMBLE_EMBED_BUDGET_USD` (default **$5.00**). That is 16x a full javaweb index at
`voyage-4-lite` ($0.31) and 2.7x the same index at the dearest code model ($1.86), so a
legitimate workspace index never prompts and a runaway still does. Over budget is a loud
`EmbeddingBudgetExceeded` naming the estimate, the cost, the ceiling and the ways out, and
**nothing is sent**:

```
Refusing to embed 640000 uncached chunk(s) with voyage:voyage-code-4@1024: ~250,000,000
estimated tokens (~$30.00) exceeds the budget of $5.00 (ZEMBLE_EMBED_BUDGET_USD). Exclude
paths with <root>/.zembleignore (gitignore syntax), or point repo at a sub-path such as
<root>/src, or raise ZEMBLE_EMBED_BUDGET_USD / set ZEMBLE_EMBED_CONFIRM=1 (--yes on the
CLI) in the environment of the process that builds (a running daemon does not see a
client's environment; restart it).
```

A model with **no documented price** cannot be billed, so its VOLUME is capped instead:
`ZEMBLE_EMBED_BUDGET_TOKENS`, default 2,000,000 tokens. That cap bounds VOLUME, not spend -
an undocumented model may charge more per token than anything in the table, and 2M tokens at
$2 per million is $4.00 - but it is fail-closed, because an unknown price is never treated as
free and never as unlimited, and the refusal says so:

```
Refusing to embed 9000 uncached chunk(s) with openai:http://localhost:11434/v1#nomic-embed-text:
~2,500,000 estimated tokens exceeds the ceiling of 2,000,000 tokens for a model with no
documented price (ZEMBLE_EMBED_BUDGET_TOKENS). ...
```

**The volume backstop.** Money is judged first, and a token ceiling sits under it as an
absolute limit that applies to a PRICED model too: `MAX_BUDGET_TOKENS`, **derived** as
`DEFAULT_BUDGET_USD / the dearest rate the table documents` - $5.00 at $0.13 per million, so
38,461,538 tokens. It is derived rather than typed because the money ceiling is `$5.00 /
price`, which makes the price table load-bearing: a rate that is 10x too low admits 10x the
tokens while the guard keeps printing "$5.00". Denominating the backstop in the DEAREST
documented rate is what bounds that - at the backstop even the dearest model in the table
bills exactly $5.00 - so a rate mistyped low for any other model NARROWS the ceiling instead
of deleting it. A mistyped *dearest* entry does lift it, which is what `PRICES_CHECKED_ON`
and the unit-sanity test are for.

It has to BIND, and it does. What 180 MB of source carries in estimated tokens depends on the
capsule overhead, and that scales with FILE SIZE rather than with the tree: measured over real
builds, 1.05x on 20 KB files, 1.10x at 400 B, 1.50x at 80 B and 2.60x at 20 B. The ends of that
range are declared once, as `CAPSULE_OVERHEAD_LOW` and `CAPSULE_OVERHEAD_HIGH` in
`embedding/pricing.py`, and a test fails when this prose stops quoting them. So the work
ceiling admits roughly 52M estimated tokens for a tree of large files and up to ~130M for one
of tiny ones, and 38.5M sits under even the low end - it binds whatever the tree is made of -
while the measured full javaweb code-and-docs index (~21.6M estimated tokens) passes with room
over it. The binding argument needs the LOW end; the reader who wants the worst case wants the
high one - a build at the work ceiling can really bill up to ~$16.90 at the dearest documented
rate, which is what the work ceiling, not this backstop, bounds.

The backstop has not always bound. At its first value, 100,000,000 tokens, it sat *above* the
work ceiling and could therefore refuse nothing at all: with `voyage-code-4` mistyped one order
of magnitude low, a build at the work ceiling billed $7.26 for real while the guard computed
$0.73 and allowed it. Raising `ZEMBLE_EMBED_BUDGET_USD` does not raise the backstop; a build that is genuinely
bigger names `ZEMBLE_EMBED_BUDGET_TOKENS` deliberately, which the refusal says. The price
table carries the date it was last read (`PRICES_CHECKED_ON`) and a test fails once that is
more than 180 days old.

`ZEMBLE_EMBED_BUDGET_TOKENS`, when a caller sets it deliberately, replaces whichever token
ceiling would otherwise apply - the unpriced cap or the backstop - on any model that costs
anything: a caller who names a token ceiling means it. Setting it to `0` or less disables the
volume half, and for a model with **no documented price** the volume half is the whole bill
guard, so that is a way to disable the money ceiling entirely for that model.
`ZEMBLE_EMBED_BUDGET_USD=0` disables the money half. A free (local) model is capped by
neither: it is the work guard's business.

Both refusals name the same three remedies, in the same order, from one helper: exclude
paths, narrow the root, or raise/confirm the ceiling that refused. The first two work from
inside a tool call; the third needs the environment of whichever process builds.

- The check sits at the seams that BUY document vectors - `index/dense.py::embed_chunks` for
  an index build, `dedup/detect.py` for logic-mode duplication - and asks the embedder what
  it would actually have to buy, so a caching embedder answers with its uncached set and a
  bare one with everything. It is per build, never per 512-text slice, and it deliberately
  does NOT live inside `CachingEmbedder`: that wrapper is optional (`ZEMBLE_EMBED_CACHE=0`,
  or a library caller handing in a bare remote embedder), and one environment variable must
  not be able to delete every spending ceiling. A drift test names those seams, so a third
  place that buys vectors fails the build instead of shipping unguarded.
- Local embedders are never billed, with or without a confirmation.
- Every subcommand that can be REFUSED takes `-y/--yes`: `search`, `stats`, `find-related`,
  `explain`, `home` and `dupes`. In-process they print the refusal and exit non-zero.
  `dupes` is there for its `--kind logic` lane, which buys a vector per candidate body
  without building an index at all - "never builds an index" was never the same claim as
  "cannot be refused". `outline`, `signatures` and `graph` answer from the symbol graph, which
  a drift test proves by running each with BOTH refusable seams trip-wired: the index build
  and the seam that buys vectors.
- The daemon catches a refusal, logs the refusal's own text (not just "refused"), keeps the
  previous index serving (nothing is chunked, embedded or swapped) and shows it in
  `zemble daemon status` as the root's `last_error`.
- A refusal travels the daemon protocol as `{"ok": false, "kind": "refused", ...}` and
  raises `CommandRefused` in the client. Callers report it instead of falling back
  in-process: the same request refuses identically in every process, and rebuilding it
  locally only pays for the refusal twice. See "refusals versus outages" in
  `docs/daemon.md`.
- An agent can recover from a refusal without leaving the tool call, by passing
  `exclude` to `search` / `find_related` / `explain`: on a root with no index yet those
  patterns prune the walk that builds it, and the pruned build gets its own cache entry
  so it can never be mistaken for the plain index of the same root.
- A refusal for a path that sits inside an already-indexed tree names that tree and the
  way out, because indexing a sub-repo of an indexed workspace is almost never what was
  wanted: `/work/zenit is inside /work, which is already indexed: search /work instead,
  or pass the workspace root; to index /work/zenit on its own anyway set
  ZEMBLE_EMBED_CONFIRM=1.` (Normally the request never gets that far - see
  "serving a sub-path from an ancestor index" in `docs/daemon.md`.)
- Every paid embed logs one INFO line first:
  `embedding 812 uncached chunk(s), ~171000 tokens, ~$0.02 with voyage:voyage-4-lite@1024`.

### Broad-root guard

Before a new local semantic index is chunked, Zemble checks whether its root appears to
be a container of unrelated workspaces. A Git root is an explicit project boundary, and
a multi-repository workspace declares itself with `.zemble/home.toml`. A smaller ad-hoc
directory remains valid, but a non-declared root containing at least eight nested Git
repositories is refused before anything reaches the embedder. This catches accidental
requests for paths such as `~/projects` without blocking a declared workspace such as
`javaweb`.

The refusal names the root and all ways out: search a narrower project, declare the
workspace, or deliberately override this check, the work ceiling and the budget at once
with `--yes` / `ZEMBLE_EMBED_CONFIRM=1`. It shares a base class, `ScopeRefused`, with the
work guard above, which is what lets every surface treat both as deliberate answers. A
confirmed CLI request requires explicit `--no-daemon` because an already-running
daemon cannot inherit an environment decision made by its client. Confirmation alone
does not authorize a local index.

### Prices

Per million tokens, from each provider's price list. A model that is not in the table is
reported as "unknown price" rather than guessed, and an unknown price never becomes free.

| model | $/M tokens | | model | $/M tokens |
| --- | --- | - | --- | --- |
| `voyage-code-4` | 0.12 | | `voyage-3.5` | 0.06 |
| `voyage-4` | 0.06 | | `voyage-3.5-lite` | 0.02 |
| `voyage-4-lite` | 0.02 | | `text-embedding-3-small` | 0.02 |
| `model2vec:*` | free (local) | | `text-embedding-3-large` | 0.13 |

Every rate above was last read off its provider's price list on the date in
`PRICES_CHECKED_ON` (`embedding/pricing.py`), and `test_the_price_table_is_dated_and_sane`
fails once that is 180 days old or a rate is outside the plausible band for USD per million
tokens. That is not tidiness: the token ceiling a priced build is judged by is derived from
these numbers.

**The estimate is an estimate.** Tokens are counted as characters / 3.6, the density
measured on javaweb (15,526,808 provider-reported tokens for 73,957 chunks); a tree of
minified or non-Latin text will differ, under-counting several-fold in the worst case. The
provider's own `usage.total_tokens` is what is billed. Since the ceiling became money, that
density risk converts directly into dollars: a build estimated at $4.99 on CJK or minified
source could really bill several times that. The volume backstop does NOT bound that: it
compares the same estimate against a token ceiling, and an under-counted density makes the
estimate smaller, not larger, so it slips past both halves together. The only true bound on
what such a build can really cost is the byte work ceiling above - which is measured in
bytes, of which no density estimate can talk it out.

## The user env file

All `ZEMBLE_*` settings and provider keys can live in `~/.config/zemble/env`
(`KEY=VALUE` lines, `#` comments, optional `export`; `ZEMBLE_ENV_FILE` overrides the
location). `zemble.userenv.load_user_env` applies it at every entry point (CLI, MCP
server, `python -m zemble.daemon`) for keys not already set, so the process environment
always wins. Tests pin `ZEMBLE_ENV_FILE` to a missing path so a developer's real file is
never read by the suite.
