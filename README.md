<h2 align="center">
  zemble<br/>
  Fast and Accurate Code Search for Agents<br/>
  <sub>Uses ~99% fewer tokens than grep+read</sub>
</h2>

<div align="center">

[Quickstart](#quickstart) •
[CLI](#cli) •
[MCP Server](#mcp-server) •
[Installation](docs/installation.md) •
[Benchmarks](#benchmarks)


</div>

zemble is a fork of [Semble](https://github.com/MinishLab/semble) by MinishLab, focused on workspace code intelligence: a symbol graph, evidence bundles packed to a token budget, duplication detection, and pluggable embedders. The upstream README follows below; see [docs/plan.md](docs/plan.md) for what this fork is building.

On top of upstream's `search` and `find-related`, the fork adds `graph` (callers,
implementations, tests-of and friends over a Java symbol graph), `explain`,
`outline` and `signatures` ([evidence bundles](docs/evidence.md)), `dupes`
(exact, alpha-renamed and logic clone classes), `daemon` (one warm process per
user holding the indexes in RAM), and [`home`](docs/home.md) - "does this feature
already exist, and which module should it live in?", answered from the module
map, forbidden dependencies and declared-home tables a workspace states in
`.zemble/home.toml`. Every one of them is a CLI subcommand and an MCP tool.

Zemble is a code search library built for agents. It returns the exact code snippets they need instantly, using ~99% fewer tokens than grep+read. Indexing and searching a full codebase end-to-end takes under a second, matching the retrieval quality of a code-specialized transformer while indexing ~220x faster and querying ~17x faster (see [benchmarks](#benchmarks)). Everything runs on CPU with no API keys, GPU, or external services. Use it as an MCP server, a CLI tool via AGENTS.md, or a dedicated sub-agent, and any coding agent (Claude Code, Cursor, Codex, OpenCode, etc.) gets instant access to any repo.

## Quickstart

Your agent queries Zemble in natural language (e.g. `"How is authentication handled?"`) and gets back only the relevant code snippets, without grepping or reading full files.

The fastest way to get started is the interactive installer. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv tool install zemble
zemble install
```

`zemble install` detects installed coding agents such as Claude Code, Codex, and OpenCode, and then lets you choose which integrations to enable:

- **MCP server**: lets the agent call Zemble directly as a tool.
- **Instructions**: adds CLI usage guidance to AGENTS.md / CLAUDE.md.
- **Sub-agent**: installs a dedicated `zemble-search` sub-agent.

To undo the setup, run `zemble uninstall`.

For manual setup instructions (MCP config per agent, AGENTS.md snippet, sub-agent files), see the [installation docs](docs/installation.md).

<details>
<summary>Updating Zemble</summary>

```bash
uv tool upgrade zemble   # upgrade
uv cache clean zemble    # for MCP users (restart your MCP client after)
```

</details>

<details>
<summary>Unattended install</summary>

For sandboxed or scripted environments, skip the prompts with `--agent` and, optionally, `--type`:

```bash
zemble install --agent claude --type mcp subagent --yes
```

`--agent` accepts one or more agent ids (e.g. `claude`, `codex`, `pi`); `--type` accepts `mcp`, `instructions`, `subagent`, or `all` (default: all); `--yes` skips the confirmation prompt (requires `--agent` for a fully non-interactive run).

</details>

## Main Features

- **Fast**: indexes an average repo in ~500 ms and answers queries in ~1 ms, all on CPU.
- **Accurate**: NDCG@10 of 0.854 on our [benchmarks](#benchmarks), on par with code-specialized transformer models, at a fraction of the size and cost.
- **Token-efficient**: returns only the relevant chunks, using [~99% fewer tokens than grep+read](#benchmarks).
- **Zero setup**: runs on CPU with no API keys, GPU, or external services required.
- **MCP server**: works with Claude Code, Cursor, Codex, OpenCode, VS Code, and any other MCP-compatible agent.
- **Local and remote**: pass a local path or a git URL.

## CLI

Zemble also ships as a standalone CLI. This is useful in scripts or anywhere you want search results without an MCP session. Indexes are built and cached on first run, and invalidated automatically when files change.

```bash
# Search a local repo (index is built and cached automatically)
zemble search "authentication flow" ./my-project

# Search a remote repo (cloned on demand)
zemble search "save model to disk" https://github.com/MinishLab/model2vec

# Limit results
zemble search "save model to disk" ./my-project --top-k 10

# Search docs/config/everything instead of just code
zemble search "deployment guide" ./my-project --content docs   # or: config, all

# Find code similar to a known location
zemble find-related src/auth.py 42 ./my-project

# Show only the first N lines of each result's snippet (0 = path/line range only)
zemble search "authentication flow" ./my-project --max-snippet-lines 10
```

`--content` accepts `code` (default), `docs`, `config`, or `all`. `path` defaults to the current directory when omitted; git URLs are accepted. If `zemble` is not on `$PATH`, use `uvx --from "zemble[mcp]" zemble` in its place. `zemble --version` (or `-V`) prints the installed version.

<details>
<summary>Controlling which files are indexed</summary>

Zemble reads `.gitignore` and `.zembleignore` files to determine which files to index. Both files use standard gitignore syntax and their patterns are merged. `.zembleignore` lets you add zemble-specific rules without touching `.gitignore`. Rules are applied recursively, so a `.zembleignore` in a subdirectory applies to that subtree.

**Excluding files:** add patterns the same way you would in `.gitignore`:

```
# .zembleignore
generated/     # exclude generated dir
*.pb.go.       # exclude Go protobuf files
```

**Including non-default extensions:** prefix the extension pattern with `!` to force-include files that zemble wouldn't index by default:

```
# .zembleignore
!*.proto       # include Protobuf files
!*.cob         # include COBOL files
```

Zemble also always skips a set of well-known non-source directories regardless of ignore files (e.g. `node_modules/`, `.venv/`, `dist/`, `build/`, `__pycache__/`, and similar).

</details>

<details>
<summary>Savings</summary>

`zemble savings` shows how many tokens zemble has saved across all your searches:

```bash
zemble savings
```

```
  Zemble Token Savings
  ════════════════════════════════════════════════════════════════════════

  Total saved:  ~714.2M tokens  (94%)
  Total calls:  14.3k
  Efficiency:  ███████████████████████░  94%

  By Period
  ────────────────────────────────────────────────────────────────────────
  Period             Calls           Saved  Ratio
  ────────────────────────────────────────────────────────────────────────
  Today                198    ~1.4M tokens  ███████████████████████░  95%
  Last 7 days        13.1k  ~707.2M tokens  ███████████████████████░  94%
  All time           14.3k  ~714.2M tokens  ███████████████████████░  94%

  By Call Type
  ────────────────────────────────────────────────────────────────────────
  #     Call type            Calls  Share
  ────────────────────────────────────────────────────────────────────────
  1.    search               14.1k  ████████████████    99%
  2.    find_related           205  █░░░░░░░░░░░░░░░     1%
  ════════════════════════════════════════════════════════════════════════
```


Savings are calculated as follows: for each call, zemble records the total character count of the unique files containing returned chunks and the character count of the snippets returned. Estimated tokens saved is `(file chars − snippet chars) / 4` (4 chars per token). This is a conservative estimate: the baseline is reading matched files in full, which is how coding agents often explore unfamiliar code.

</details>

<details>
<summary>Storage</summary>

By default, your Zemble savings statistics and any saved indexes are stored in the OS cache folder (`~/Library/Caches/zemble/` on macOS, `~/.cache/zemble/` on Linux, `%LOCALAPPDATA%\zemble\Cache\` on Windows). To override this location you can supply an environment variable `ZEMBLE_CACHE_LOCATION` which should be the full path to the target cache location e.g. `~/my-folder/my-caches/zemble`.

On first use, Zemble also downloads the embedding model from Hugging Face and caches it in the standard Hugging Face cache (`~/.cache/huggingface/` by default, or `$HF_HOME` if set); this only happens once and requires network access.

Use `zemble clear` to remove cached data: `zemble clear index` (saved indexes), `zemble clear savings` (usage stats), `zemble clear orphans` (indexes for repos no longer present on disk), or `zemble clear all` (everything).

</details>

<details>
<summary>Library usage</summary>

Zemble can also be used as a Python library for programmatic access, useful when building custom tooling or integrating search directly into your own code.

```python
from zemble import ContentType, ZembleIndex

# Index a local directory (code only, the default)
index = ZembleIndex.from_path("./my-project")

# Index docs and prose (markdown, rst, etc.)
index = ZembleIndex.from_path("./my-project", content=ContentType.DOCS)

# Index everything (code, docs, and config)
index = ZembleIndex.from_path("./my-project", content=[ContentType.CODE, ContentType.DOCS, ContentType.CONFIG])

# Index code and docs together
index = ZembleIndex.from_path("./my-project", content=[ContentType.CODE, ContentType.DOCS])

# Index a remote git repository
index = ZembleIndex.from_git("https://github.com/MinishLab/model2vec")

# Search the index with a natural-language or code query
results = index.search("save model to disk", top_k=3)

# Find code similar to a specific result
related = index.find_related(results[0], top_k=3)

# Each result exposes the matched chunk
result = results[0]
result.chunk.file_path   # "model2vec/model.py"
result.chunk.start_line  # 127
result.chunk.end_line    # 150
result.chunk.content     # "def save_pretrained(self, path: PathLike, ..."
```

</details>

## MCP Server

Zemble runs as an MCP server so agents can search any codebase directly as a native tool call. Repos are indexed on demand and cached; local paths are re-indexed automatically on file changes.

| Tool | Description |
|------|-------------|
| `search` | Search a codebase with a natural-language or code query. Pass `repo` as a local path or an https:// git URL and `content` as `code`, `docs`, `config`, or `all` (default: `code`). |
| `find_related` | Given a file path and line number, return chunks semantically similar to the code at that location. |

For per-agent setup instructions, see the [installation docs](docs/installation.md#mcp-server).


## Benchmarks

We benchmark quality and speed across ~1,250 queries over 63 repositories in 19 languages (left), and token efficiency against grep+read at equivalent recall levels (right).

<table>
<tr>
<td><img src="https://raw.githubusercontent.com/MinishLab/semble/main/assets/images/speed_vs_ndcg_cold.png" alt="Speed vs quality"></td>
<td><img src="https://raw.githubusercontent.com/MinishLab/semble/main/assets/images/token_efficiency.png" alt="Token efficiency: recall vs. retrieved tokens"></td>
</tr>
</table>

The quality benchmark (left) scores retrieval quality (NDCG@10) against total latency; zemble matches the quality of the 137M-parameter [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed) while indexing 220x faster. The token efficiency benchmark (right) measures how many tokens each method needs to reach a given recall level; zemble uses 99% fewer tokens on average and hits 97% recall at only 2k tokens, while grep+read needs a full 100k context window to reach 85%. See [benchmarks](benchmarks/README.md) for per-language results, ablations, and full methodology.

## How it works

Zemble splits each file into code-aware chunks using [tree-sitter](https://github.com/tree-sitter/py-tree-sitter), then scores every query against the chunks with two complementary retrievers: static [Model2Vec](https://github.com/MinishLab/model2vec) embeddings using the code-specialized [potion-code-16M-v2](https://huggingface.co/minishlab/potion-code-16M-v2) model for semantic similarity, and BM25 for lexical matches on identifiers and API names. The two score lists are fused with Reciprocal Rank Fusion (RRF).

After fusing, results are reranked with a set of code-aware signals:

<details>
<summary><b>Ranking signals</b></summary>

- **Adaptive weighting.** Symbol-like queries (`Foo::bar`, `_private`, `getUserById`) get more lexical weight, while natural-language queries stay balanced between semantic and lexical retrievers.
- **Definition boosts.** A chunk that defines the queried symbol (a `class`, `def`, `func`, etc.) is ranked above chunks that merely reference it.
- **Identifier stems.** Query tokens are stemmed and matched against identifier stems in a chunk, giving an additional weight to chunks that contain them. For example, querying `parse config` boosts chunks containing `parseConfig`, `ConfigParser`, or `config_parser`.
- **File coherence.** When multiple chunks from the same file match the query, the file is boosted so the top result reflects broad file-level relevance rather than a single out-of-context chunk.
- **Noise penalties.** Test files, `compat/`/`legacy/` shims, example code, and `.d.ts` declaration stubs are down-ranked so canonical implementations surface first.

</details>

Because the embedding model is static with no transformer forward pass at query time, all of this runs in milliseconds on CPU.

Indexes are cached to disk automatically on the first search. On subsequent runs, Zemble walks the file tree and compares modification times; added, removed, or changed files are reindexed incrementally, without rebuilding the rest of the index. A full rebuild only happens if the indexing settings change (e.g., after a zemble upgrade that changes the model, chunking, or cache format). In MCP mode, the index is checked and refreshed automatically as files change, so results stay current across the session.

### Using a custom model

If you would like to use another model, you can set your `ZEMBLE_MODEL_NAME` environment variable to a local path or Hugging Face repository. This path is read verbatim, and should contain a [`Model2Vec`](https://github.com/MinishLab/model2vec) compatible model. This is particularly useful if you can't access Hugging Face at runtime.

### Using an API embedder

Zemble can also embed through the Voyage API or any OpenAI-compatible
`/v1/embeddings` endpoint (Ollama, LM Studio, vLLM, OpenAI), selected with a
single spec string on `--embedder` or in `ZEMBLE_EMBEDDER`:

```bash
zemble search "auth flow" ./my-project --embedder voyage:voyage-code-4@256
zemble search "auth flow" ./my-project --embedder openai:http://localhost:11434/v1#nomic-embed-text
```

Embeddings from API providers are cached by content hash, so unchanged code is
never paid for twice. See [docs/embedders.md](docs/embedders.md) for the full
spec grammar, environment variables, cache location and cost notes.

### Java compiler facts

For Java codebases, [`javac-facts/`](javac-facts/README.md) holds `zemble-javac-facts`, a small
standalone javac plugin that emits what the compiler itself resolved -- declared symbols, calls with
the exact overload javac selected, overrides, supertype edges and constant annotation arguments --
as JSONL in zemble's graph-facts format. It plugs into any Gradle or Maven build (or plain `javac`),
depends only on the documented `com.sun.source` API, and gives the graph precise Java edges without
re-implementing name and overload resolution on top of tree-sitter.

## Acknowledgements

Thanks to [Greptile](https://greptile.com) for providing free access to their AI code review platform.

## License

MIT

## Citing

zemble is derived from Semble. If you use it in your research, please cite the upstream work:

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
