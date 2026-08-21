# zemble

zemble is a fork of [Semble](https://github.com/MinishLab/semble) by MinishLab. Semble is
fast local hybrid code search for coding agents: tree-sitter chunking, BM25 over
identifiers and API names, static Model2Vec embeddings, fused with reciprocal rank fusion
and reranked with code-aware heuristics, all on CPU with no API key and no external
service. zemble keeps that core, and adds a workspace code-intelligence layer on top of
it: a symbol graph carrying compiler-resolved facts, context capsules in the retrieval
text, evidence bundles (`explain`, `outline`, `signatures`), duplication detection,
`home` ("does this already exist, and which module should own it"), pluggable embedders
and rerankers, a warm daemon, and a columnar index that loads in milliseconds.

The retrieval core is Semble's work and the credit for it belongs upstream. The
improvement that touches that core, the context capsule, also held on Semble's own
63-repo benchmark (NDCG@10 0.852 to 0.864), so it is not only a private win on this
fork's workspace.

[What you get](#what-you-get) - [Benchmarks](#benchmarks) - [Suggested configuration](#suggested-configuration) - [Install](#install) - [CLI](#cli) - [MCP server](#mcp-server) - [How it works](#how-it-works)

## What you get

**Search** (`zemble search`, MCP `search`). Natural-language or code queries against a
local path or a git URL, answered from chunks rather than whole files. `--content` picks
the `code` (default), `docs`, `config` or `all` lane. Every chunk is embedded and indexed
together with a [context capsule](docs/capsules.md): its path, package, enclosing type
chain, signature and the imports it actually uses. The capsule is retrieval text only,
never output.

**Find related** (`zemble find-related`, MCP `find_related`). Give it a file and a line
and it returns the chunks nearest to the code living there.

**Symbol graph** (`zemble graph *`, MCP `graph_definition`, `graph_callers`,
`graph_implementations`, `graph_tests_of`, `graph_neighbors`). Callers, callees,
references, implementations, supertypes, overrides, tests-of and neighbour walks over a
Java and Hawkeye workspace. Every answer carries the rung it landed on (exact, unique
name, ambiguous, unresolved) and a one-line reason, because the extractor is tree-sitter
plus a name resolver and not a compiler. Where a build emits
[compiler facts](docs/graph-facts.md), those replace the guessed edges for the files they
cover. See [docs/graph.md](docs/graph.md).

**Evidence bundles** (`zemble explain`, `outline`, `signatures`). `explain` searches,
follows the graph one hop out of what it found, and packs the result under a token budget
with a reason per item and an explicit list of what did not fit. `outline` is a
signature-only view of a file or a type (150 to 300 tokens for a whole class) and
`signatures` prints a declaration plus the call sites the graph resolved exactly.
[docs/evidence.md](docs/evidence.md) also carries the honest measurement: bundles do not
beat plain search on hit rate, they buy structure.

**Duplication** (`zemble dupes`, MCP `dupes`). Exact, alpha-renamed and logic clone
classes over declaration bodies and statement windows, ranked by weight. Java and Zig
today; a language is one profile module in `src/zemble/dedup/languages/`. Logic clones are
never reported on embedding similarity alone: a structural check has to agree, and the
reason is printed. It is a report, never a gate. See [docs/dedup.md](docs/dedup.md).

**Home** (`zemble home`, MCP `home`). "Does this feature already exist, and which module
should it live in?" Answered from the existing mechanisms search and the graph find,
plus the module map, forbidden dependencies and declared-home tables the workspace states
about itself in `.zemble/home.toml`. See [docs/home.md](docs/home.md).

**Warm daemon** (`zemble daemon`). One process per user holding the indexes and the graph
in RAM, watching the roots it holds and reindexing incrementally. It starts on demand,
never at login, and exits after 30 idle minutes. Every surface is daemon-first with an
in-process fallback, so it is an accelerator and never a requirement. See
[docs/daemon.md](docs/daemon.md).

**Pluggable embedders and rerankers**. One spec string selects the embedder
(`model2vec:`, `voyage:`, `openai:` for anything speaking the OpenAI embeddings shape)
and one selects the reranker (`none`, `cross:<hf-model>`, `voyage:<model>`). Remote
embeddings are cached by content hash, and each embedder declares how much of the hybrid
fusion its dense lane earns, so a stronger embedder is weighted for without a flag. See
[docs/embedders.md](docs/embedders.md) and [docs/rerank.md](docs/rerank.md).

## Benchmarks

Two sets are involved. The public one is Semble's: 63 repositories, 19 languages, about
1,250 queries, run with the harness in [`benchmarks/`](benchmarks/README.md). The other
one is ours and is local: 90 hand-written, hand-verified queries (80 answered by a Java
file, 10 by a Hawkeye template) over one private 40-repo Java workspace.

Seven configurations, one working tree, one run each, all 90 queries
([docs/comparison.md](docs/comparison.md)):

| configuration | NDCG@10 | hit@1 | hit@5 | hit@10 | p50 | cost per query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A Semble-equivalent (capsules off) | 0.539 | 0.467 | 0.644 | 0.667 | 89 ms | $0 |
| B zemble local defaults | 0.557 | 0.500 | 0.622 | 0.689 | 95 ms | $0 |
| C `voyage-4-lite@1024` embedder | 0.680 | 0.589 | 0.811 | 0.856 | 350 ms | $0.0000002 |
| D `voyage-code-4@1024` embedder | 0.623 | 0.489 | 0.767 | 0.811 | 354 ms | $0.0000013 |
| E `rerank-2.5` on the default embedder | 0.691 | 0.622 | 0.767 | 0.822 | 578 ms | $0.00058 |
| F `voyage-4-lite` + `rerank-2.5` | 0.753 | 0.656 | 0.867 | 0.911 | 802 ms | $0.00057 |
| G `voyage-4-lite` + `rerank-2.5-lite` | 0.738 | 0.656 | 0.878 | 0.900 | 796 ms | $0.00023 |

Rows C to G were measured before hosted embedders started claiming a larger share of the
fusion. On the shipped configuration C is now 0.710 NDCG@10 and 0.833 hit@5
([docs/voyage.md](docs/voyage.md)); the reranked rows move less and were not re-measured.

hit@5 is the fraction of queries where a correct file is in the top 5. NDCG@10 scores the
whole ranked list of 10: bigger is better, and 1.0 means the right file came first on
every query.

In plain words, on this set: with the local defaults about 62% of queries have the answer
in the top 5; with a hosted embedder that is about 83%; with a hosted embedder and a
hosted reranker together it is about 87 to 88%. The two mechanisms fix different queries.
The embedder gets the right file into the window (symptom-only bug reports go from 0.400
to 0.800 hit@5), the reranker puts it at the top ("which layer consumes this" queries go
from 0.273 to 0.818 hit@5).

On Semble's own 63-repo benchmark the capsule moved NDCG@10 from 0.852 to 0.864, and a
hosted embedder adds about +0.02 on top of that (measured over the 59 repos both runs
covered). The headroom there is small because that set is largely short lexical queries
BM25 already answers.

Speed, same workspace: a cold CLI query went from 12.2 s to about 1.0 s and a warm symbol
query from 379 ms to 17 ms once the index became columnar and mmap-backed
([docs/profile-loadtime.md](docs/profile-loadtime.md)). Through the warm daemon a CLI
invocation over a 74k-chunk workspace index is 0.58 s against 1.52 s in-process, and most
of what is left is the client interpreter start that no daemon can remove.

Read the numbers with these caveats:

- The 90-query set is **our own**, hand-labelled by the same people who tuned the
  retrieval, over one private Java workspace. It is not a public benchmark and it is not
  neutral. The public benchmark is Semble's, and it is reproducible with the harness.
- **n is small.** One query is 1.1 points of hit rate; per query-kind figures move 9
  points per query. F and G are a tie.
- **One run each.** Nothing here has an error bar. Retrieval is deterministic given the
  index, so the quality numbers are stable; the latencies are not.
- **Latencies come from one loaded machine**, sequentially, over a residential
  connection, and every hosted row is dominated by network round trips. They rank the
  configurations; they do not predict anyone else's wall clock.
- **hit@k is coarse by design.** It cannot tell "the answer was at rank 1" from "the
  answer and four distractors filled the top five", which is why NDCG is printed beside
  it.

## Suggested configuration

**The defaults are local, offline and free**, and they need no key, no GPU and no
account. That is the configuration everyone gets, and nothing below changes it.

**The measured best value for money**, if you have a Voyage key and are willing to send
code to it, is a hosted embedder plus the small hosted reranker:

```bash
export VOYAGE_API_KEY=pa-...
export ZEMBLE_EMBEDDER=voyage:voyage-4-lite@1024
export ZEMBLE_RERANKER=voyage:rerank-2.5-lite
export ZEMBLE_RERANK_ALPHA=0.7
export ZEMBLE_RERANK_K=50
```

**Before a first paid index, run `zemble embed-status <path>`.** It chunks the tree the
way a build would and reports how many chunks are already cached, how many would be
embedded, and what those cost at the model's list price - without embedding anything or
needing a key. A build that would spend more than `ZEMBLE_EMBED_BUDGET_TOKENS`
(default 2,000,000) is refused before a single request, naming the estimate and how to
proceed; see [docs/embedders.md](docs/embedders.md#cost-visibility-and-the-budget-guard).

Put those lines in `~/.config/zemble/env` (mode 600; or point `ZEMBLE_ENV_FILE` at another
file) and zemble loads them itself at startup for the CLI, the MCP server and the daemon;
an explicit environment variable always wins over the file, so nothing secret has to live in
an agent's MCP config.

That is row G above: about 88% hit@5, roughly $0.0002 and 0.8 s per query. Dropping the
reranker (the embedder alone, row C) is about 83% hit@5 at roughly 0.35 s per query and a
fifth of a millionth of a dollar. The reranker is worth its round trip on the queries
fusion is worst at, and it slightly hurts bare-identifier lookups, which BM25 already
answers perfectly.

Be clear about what this turns on: with a hosted embedder, **the text of every chunk is
sent to Voyage** when the index is built, and every query is sent on every search; with a
hosted reranker, the top 50 candidate passages are sent too. That is why it is opt-in and
why the default stays local.

Embedding an index is a one-off. Remote embeddings are cached by content hash in
`~/.cache/zemble/embeddings/`, so re-indexing pays only for chunks whose content changed,
and a narrower width is sliced out of a wider cached vector rather than bought again.
A full index of a 74k-chunk workspace measured 15.5M tokens, which is $0.31 with
`voyage-4-lite`. See [docs/embedders.md](docs/embedders.md) and
[docs/voyage.md](docs/voyage.md).

## Install

zemble is **not published on PyPI**. Install it from a checkout with
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone <this repository> /path/to/zemble
uv tool install --editable "/path/to/zemble[mcp]"
zemble --version
```

The `[mcp]` extra pulls in the MCP server dependencies. Use
`"/path/to/zemble[mcp,rerank]"` if you also want to run a local cross-encoder
(`cross:<hf-model>`); that extra drags in torch and transformers, and nothing imports
them until a `cross:` reranker actually scores something.

On first use zemble downloads the default embedding model from Hugging Face into the
standard Hugging Face cache (`~/.cache/huggingface/`, or `$HF_HOME`). That is the only
network access the defaults need, and it happens once.

`zemble install` still works and configures whichever coding agents it detects (MCP
server, AGENTS.md / CLAUDE.md instructions, a `zemble-search` sub-agent). It notices an
editable install and points the generated configs at your checkout path rather than at a
PyPI release, so they launch the code you have. `zemble uninstall` removes them again.
For per-agent manual setup, see [the installation docs](docs/installation.md).

## Quickstart

```bash
zemble search "how is authentication handled" ./my-project
```

The index is built on first use and cached; later runs walk the tree, reindex only what
changed, and answer from the cache. Point it at a local path or an https git URL.

## CLI

```bash
# Search a local repo (index built and cached automatically)
zemble search "authentication flow" ./my-project

# Search a remote repo (cloned on demand)
zemble search "save model to disk" https://github.com/MinishLab/model2vec

# More results, shorter snippets (0 = path and line range only)
zemble search "authentication flow" ./my-project --top-k 10 --max-snippet-lines 10

# Search docs or config instead of code
zemble search "deployment guide" ./my-project --content docs   # or: config, all

# Find code similar to a known location
zemble find-related src/auth.py 42 ./my-project

# What does this index hold?
zemble stats ./my-project

# Which zemble code is running here, and is the daemon on the same revision?
zemble status

# What would indexing this cost with a paid embedder? (embeds nothing)
zemble embed-status ./my-project --embedder voyage:voyage-4-lite@1024
```

Graph queries take a simple name (`PageWindow`), a qualified name or `Type.member`:

```bash
zemble graph callers ./my-project PageWindow.of
zemble graph implementations ./my-project StorageAdapter
zemble graph tests-of ./my-project PageWindow
zemble graph neighbors ./my-project PageWindow --hops 2
zemble graph build ./my-project --stats
zemble graph facts status ./my-project
```

Evidence, duplication and placement:

```bash
zemble explain ./my-project "how is pagination computed" --budget 3000
zemble outline ./my-project PageWindow --members page
zemble signatures ./my-project PageWindow.of
zemble dupes ./my-project --kind exact,renamed --limit 20
zemble home ./my-project "cache a computed thumbnail on disk"
```

The daemon manages itself, but it can be driven by hand:

```bash
zemble daemon status
zemble daemon start
zemble daemon stop
```

`path` defaults to the current directory. The graph and evidence commands exit `0` when
they answered, `1` when nothing matched and `2` when the name was ambiguous, with the
candidates on stderr; `zemble home` exits `1` when nothing matched at all.
`--no-daemon` (or `ZEMBLE_DAEMON=0`) answers in the calling process instead.
`zemble savings` reports how many tokens searches saved against reading the matched
files outright, and `zemble clear index|savings|orphans|all` empties the
caches. If `zemble` is not on `$PATH`, use `uvx --from "/path/to/zemble[mcp]" zemble`.

<details>
<summary>Controlling which files are indexed</summary>

zemble reads `.gitignore` and `.zembleignore` to decide what to index. Both use gitignore
syntax and their patterns are merged; a `.zembleignore` in a subdirectory applies to that
subtree. It exists so zemble-specific rules do not have to go into `.gitignore`.

```
# .zembleignore
generated/     # exclude a generated directory
*.pb.go        # exclude Go protobuf files
!*.proto       # include a non-default extension
!*.cob
```

A leading `!` on an extension pattern force-includes files zemble would not index by
default. Well-known non-source directories (`node_modules/`, `.venv/`, `dist/`, `build/`,
`__pycache__/` and friends) are always skipped regardless of ignore files.

</details>

<details>
<summary>Storage</summary>

Indexes and usage stats live in the OS cache folder (`~/.cache/zemble/` on Linux,
`~/Library/Caches/zemble/` on macOS, `%LOCALAPPDATA%\zemble\Cache\` on Windows), with
`ZEMBLE_CACHE_LOCATION` overriding the whole location. The symbol graph is a sqlite file
beside the index, and remote embeddings are cached under `embeddings/` in the same place.

</details>

<details>
<summary>Python API</summary>

```python
from zemble import ContentType, ZembleIndex

# Index a local directory (code only, the default)
index = ZembleIndex.from_path("./my-project")

# Index docs and prose, or several lanes at once
index = ZembleIndex.from_path("./my-project", content=ContentType.DOCS)
index = ZembleIndex.from_path("./my-project", content=[ContentType.CODE, ContentType.DOCS])

# Index a remote git repository
index = ZembleIndex.from_git("https://github.com/MinishLab/model2vec")

# Search, then walk outwards from a result
results = index.search("save model to disk", top_k=3)
related = index.find_related(results[0], top_k=3)

result = results[0]
result.chunk.file_path   # "model2vec/model.py"
result.chunk.start_line  # 127
result.chunk.end_line    # 150
result.chunk.content     # "def save_pretrained(self, path: PathLike, ..."
result.chunk.context     # the capsule: path, package, enclosing chain, signature
```

</details>

## MCP server

Running `zemble` with no subcommand **is** the MCP server, so an agent config points at
the `zemble` binary (or `uvx --from "/path/to/zemble[mcp]" zemble`) with no arguments.
Repos are indexed on demand and cached, local paths are refreshed as files change, and
the server asks the warm daemon first so several agent sessions share one copy of an
index in RAM.

| Tool | Answers |
| --- | --- |
| `search` | Natural-language or code query over a repo, in the `code`, `docs`, `config` or `all` lane. |
| `find_related` | Chunks similar to a given file and line. |
| `graph_definition` | Where a symbol is declared. |
| `graph_callers` | Who calls it, with the resolution grade and a reason per hit. |
| `graph_implementations` | Implementations and subclasses of a type. |
| `graph_tests_of` | Tests naming or exercising a symbol. |
| `graph_neighbors` | An n-hop walk around a symbol, filterable by edge kind. |
| `explain` | A budgeted evidence bundle as markdown. |
| `outline` | Signature-only view of a file or a type. |
| `signatures` | A declaration plus its exactly resolved call sites. |
| `dupes` | Clone classes over the workspace's code (Java, Zig). |
| `home` | Existing mechanisms, candidate homes, verdict and checklist. |
| `status` | Which zemble code this server is running (version, source root, revision, start time) and whether the checkout moved under it. |

An ambiguous symbol comes back as an `error` payload with a `candidates` list rather than
as a failure. Per-agent setup is in the [installation docs](docs/installation.md#mcp-server).

### Restart after pulling

An editable install (`uv tool install --editable`, `pip install -e`) has every process
serve the snapshot of the source it started with. An MCP server or warm daemon started
before a pull keeps answering from the old code, silently. So:

- `status` (MCP tool) and `zemble status` (CLI) report the version, revision and start
  time of the process answering, and set `stale` once the checkout has moved under it.
  `zemble status` also prints the daemon's identity when one is reachable.
- A stale MCP server logs one WARNING to stderr on the next tool call, and the daemon
  client logs one WARNING per process when the daemon runs another revision. Neither
  refuses to answer.
- The fix is always a restart: restart the MCP server in your agent, and
  `zemble daemon restart` for the daemon.

## How it works

Files are split into code-aware chunks with
[tree-sitter](https://github.com/tree-sitter/py-tree-sitter). Every query is scored
against those chunks by two retrievers: static [Model2Vec](https://github.com/MinishLab/model2vec)
embeddings from the code-specialized
[potion-code-16M-v2](https://huggingface.co/minishlab/potion-code-16M-v2) model for
semantic similarity, and BM25 for lexical matches on identifiers and API names. The two
ranked lists are fused with reciprocal rank fusion. Because the embedding model is static
there is no transformer forward pass at query time, so all of it runs in milliseconds on
CPU.

After fusion the list is reordered with code-aware signals:

<details>
<summary><b>Ranking signals</b></summary>

- **Adaptive weighting.** Symbol-like queries (`Foo::bar`, `_private`, `getUserById`) get
  more lexical weight; natural-language queries stay balanced.
- **Definition boosts.** A chunk that defines the queried symbol outranks chunks that
  merely reference it.
- **Identifier stems.** Query tokens are stemmed against identifier stems, so `parse
  config` boosts `parseConfig`, `ConfigParser` and `config_parser`.
- **File coherence.** Several matching chunks in one file boost that file, so the top
  result reflects file-level relevance rather than one out-of-context chunk.
- **Noise penalties.** Test files, `compat/` and `legacy/` shims, example code and `.d.ts`
  stubs are down-ranked so canonical implementations surface first.

</details>

Three things this fork adds to that pipeline. The **capsule** puts each chunk's path,
package, enclosing type chain, signature and used imports into both the dense text and
the BM25 document, which is what lets a mid-body chunk be found by the name of the class
it lives in (and, on the reranker path, what a cross-encoder needs to tell a method body
from prose). The **graph overlay** is a second index beside the search index, built by a
tree-sitter extractor and a name resolver, whose edges are replaced by real compiler
output for any file a build emitted fresh facts for. The **daemon** holds both in RAM for
one user and reindexes incrementally on file changes, so the index load is paid once
instead of once per invocation.

Indexes are cached to disk on the first search. Later runs compare modification times and
reindex only added, removed or changed files; a full rebuild happens only when the
indexing settings change (a different model, chunker or cache format). The on-disk index
is columnar: BM25 postings in CSR form, the vector matrix mmap-mapped, chunks as one blob
plus offsets, and a precomputed symbol-definition table.

### Using a custom model

Set `ZEMBLE_MODEL_NAME` to a local path or a Hugging Face repository holding a
[Model2Vec](https://github.com/MinishLab/model2vec)-compatible model. The value is read
verbatim, which is also how you run without Hugging Face access at query time.

### Using an API embedder

One spec string on `--embedder` or in `ZEMBLE_EMBEDDER` picks the provider:

```bash
zemble search "auth flow" ./my-project --embedder voyage:voyage-4-lite@1024
zemble search "auth flow" ./my-project --embedder openai:http://localhost:11434/v1#nomic-embed-text
```

Mixing embedders in one index is refused loudly, never silently blended. See
[docs/embedders.md](docs/embedders.md) for the grammar, the environment variables, the
cache and the cost notes.

### Java compiler facts

[`javac-facts/`](javac-facts/README.md) holds `zemble-javac-facts`, a standalone javac
plugin that writes down what the compiler itself resolved: declared symbols, calls with
the exact overload javac selected, overrides, supertype edges and constant annotation
arguments. It depends only on the documented `com.sun.source` API and plugs into Gradle,
Maven or plain javac.

How the facts flow: the build emits one JSONL file per compile task, by convention at
`build/zemble/facts-*.jsonl`. zemble finds those files under the indexed root, and every
`file` line in them names the sha256 of the source it was derived from. When that hash
still matches the file on disk, the facts are **fresh** and the `CALLS`, `OVERRIDES`,
`EXTENDS` and `IMPLEMENTS` edges for that file come from the compiler instead of from
tree-sitter; when it does not, the facts for that one file are ignored and the extracted
edges stay. Freshness is per source file, so one edit does not invalidate the other 400
facts in the same file. `zemble graph facts status` reports coverage and staleness, and
every answer names the tool that produced the edge it is showing. The format is
documented in [docs/graph-facts.md](docs/graph-facts.md) and any analyzer can write it.

## Acknowledgements

- [Semble](https://github.com/MinishLab/semble) by MinishLab, which this is a fork of and
  which is the whole retrieval core.
- [model2vec](https://github.com/MinishLab/model2vec) for the static embedding models and
  [vicinity](https://github.com/MinishLab/vicinity) for the vector store.
- [tree-sitter](https://github.com/tree-sitter/py-tree-sitter) for the grammars every
  chunk boundary and every symbol comes from.
- [Voyage AI](https://www.voyageai.com/) for the hosted embedding and reranking models
  the optional configurations use.

## License

MIT, the same as upstream. See [LICENSE](LICENSE).

## Citing

zemble is derived from Semble, and work using it should cite the upstream software:

```bibtex
@software{minishlab2026semble,
  author       = {{van Dongen}, Thomas and Stephan Tulkens},
  title        = {Semble: Fast and Accurate Code Search for Agents},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19785932},
  url          = {https://github.com/MinishLab/semble},
  license      = {MIT}
}
```
