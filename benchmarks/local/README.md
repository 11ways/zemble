# Local workspace evaluation sets

Evaluation sets over trees that live on this machine instead of a pinned
public clone. They use the same annotation format as
`benchmarks/annotations/*.json` and are selected with two flags:

```bash
uv run python -m benchmarks.run_benchmark \
  --repos-file benchmarks/local/repos.json \
  --annotations-dir benchmarks/local/annotations
```

Results are written to `benchmarks/results/local-<repos>-zemble-hybrid-<sha12>.json`.

## javaweb

A hand-built, hand-verified retrieval evaluation set over the `javaweb`
workspace (`/home/skerit/projects/javaweb`), a multi-repo Java web framework
workspace.

- `annotations/javaweb.json` - 80 queries with expected file locations.
- `repos.json` - the repo descriptor for the local workspace.

## Schema

Copied verbatim from `benchmarks/data.py`. An annotation file is a JSON
**list**; each item is:

```json
{
  "query": "text of the query",
  "relevant": ["path/relative/to/repo/root.java"],
  "secondary": ["another/acceptable/path.java"],
  "category": "symbol | semantic | architecture",
  "kind": "symbol | behavioural | architecture | bug-report | consumer"
}
```

- `relevant` / `secondary`: a target is either a plain string path or a mapping
  `{"path": ..., "start_line": N, "end_line": M}`. This set uses plain string
  paths only (whole-file relevance), which is what most of the harness's own
  annotation files do. NDCG is computed over `relevant + secondary`
  (`Task.all_relevant`), so a secondary entry counts as a hit, not as a
  distractor.
- `repo`: optional per-item override; omitted here, so it defaults to the
  annotation file stem (`javaweb`), which must match a `repos.json` name.
- `category`: consumed by the harness. Only the three values above are used by
  the harness's reporting (`semantic`, `architecture`, `symbol`); `data.infer_category`
  fills it in when absent, so an explicit value always wins.
- `kind`: an addition of this set, and now a first-class optional field on
  `data.Task`. It carries a finer query-mix label than `category`; when any
  annotation in a run declares one, `run_benchmark` reports a `By kind` table
  beside the stock `By category` one and stores `summary.by_kind` in the result
  JSON. Annotations without a `kind` are simply absent from that table.

Path matching is suffix-based (`data.path_matches`), so paths relative to the
repo root - as used here - match regardless of where the checkout lives.

## Query mix (80 total)

| kind          | count | mapped `category` | what it tests |
| ------------- | ----: | ----------------- | ------------- |
| `symbol`      |    20 | `symbol`          | bare identifier or `Type.member` lookup |
| `behavioural` |    25 | `semantic`        | natural language describing behaviour, never reusing the identifier |
| `architecture`|    15 | `architecture`    | mechanism / design questions ("how is X done") |
| `bug-report`  |    10 | `semantic`        | symptoms only, no identifiers, as a user would report them |
| `consumer`    |    10 | `semantic`        | tests that prove a behaviour, or the layer that consumes a mechanism |

Repos covered: protoblast (incl. protoblast-source-guard), hawkeye, zenit,
zenit-cms, zenit-auth, zenit-forms-adjacent surfaces, plumage, zenit-widget,
zenit-comms, zenit-media, zenit-microcopy, and the orcono app.

Queries are paraphrases: they deliberately avoid the identifier and avoid
sentences that appear verbatim in the target file's comments, so a hit cannot
come from a single literal string match with the docblock.

## How each ground truth was verified

For every entry:

1. Candidate located with the `semble` MCP tool / `find` over the workspace.
2. The candidate file **opened at the relevant region** and the implementation
   confirmed - the declaration of the class, or the specific method/field/branch
   the query describes (e.g. `EntityTags.matchesIfNoneMatch` at line 31,
   `SessionIds.forToken` at line 28, `PageWindow.OutOfRange` at line 22,
   `RecordSchedules` per-step `hasCapability` re-check at line 651,
   `AuthorizationMiddleware.owesPasswordRotation` at line 110).
3. A file that merely *calls* the mechanism was demoted to `secondary` or used
   only for a `consumer`-kind query where the caller IS the answer
   (`ConditionalRequest` delegating to `EntityTags`, `Paging` calling
   `PageWindow.offsetFor`).
4. A programmatic check confirms every one of the 100 referenced paths exists
   under `/home/skerit/projects/javaweb`, that no query text repeats, and that
   the file is pure ASCII.

The workspace's `CLAUDE.md` capability-to-home map was used as the seed for the
architectural and mechanism queries; each named home was still opened and
confirmed rather than trusted.

### Queries considered and dropped

- **`PlumageChart.layout`** (named in CLAUDE.md): no such class exists; the
  layout entry point is `ChartFunctions.layout` returning `ChartLayout`. Kept,
  but reworded to the real symbol with `ChartLayout` as secondary.
- **Byte-range / partial-content serving in zenit-media**: `MediaStreams` does
  no range handling, so the query would have had no honest home. Dropped and
  replaced with the gated-serve query the file actually answers.
- **`RunRecordSchedulesTask` as the per-step re-authorization home**: the
  sweeper does not do the authority check; `RecordSchedules` does. Query points
  at `RecordSchedules`, with the sweeper as secondary.
- **A `PageWindow` unit-test query**: there is no dedicated test class for it,
  so no `consumer`-kind query was written for it.
- **`Themes` as a primary target**: two classes named `Themes` exist (zenit and
  emberglyph). Kept only as a `secondary` under the `PreferenceCookie` query to
  avoid an ambiguous primary.

## Running against a LOCAL repo

The stock harness assumes every repo is a pinned remote clone. Three changes
make a local tree a first-class benchmark repo, and they are in the harness:

- `data.RepoSpec` has a `local_path: str | None = None` field, and
  `checkout_dir` returns it when set, so `available_repo_specs()` sees the
  workspace itself instead of `~/.cache/zemble-bench/javaweb`.
- `sync_repos.py` skips a spec with a `local_path` (one log line) and `--check`
  treats it as always current. `javaweb` is a *multi-repo workspace* - the root
  directory is not a git repo at all, and each subproject (`zenit`, `hawkeye`,
  ...) is its own repo, so there is no single SHA to pin. `revision` is the
  sentinel string `"local-worktree"`.
- `run_benchmark.py` takes `--repos-file` and `--annotations-dir`; both default
  to the upstream set, so nothing changes for the 63-repo benchmark.

Consequence: this set is pinned to the working tree, not to a SHA - re-verify
paths after large refactors.

`benchmark_root` is `null`: the whole workspace is indexed, which is
intentional (cross-repo retrieval is the thing being measured).

### Ignore list

The workspace root carries a `.zembleignore` (gitignore syntax, read by
`zemble.index.file_walker` next to any `.gitignore` in the same directory) that
drops `alchemy/` (legacy JavaScript predecessor), `testbeds/` and `references/`
and `resources/` (vendored third-party checkouts), plus `build/` and `.gradle/`.
No annotated target lives under any of them. It cuts the index from 181,912
chunks to 73,901 and cold index time from 155s to 38s.

### Baseline

Default embedder, `.zembleignore` in place, 80 queries:

| Metric  | Before ignore list | Baseline (after) |
| ------- | -----------------: | ---------------: |
| NDCG@5  |              0.500 |            0.521 |
| NDCG@10 |              0.510 |            0.532 |
| chunks  |            181,912 |           73,901 |
| index   |               155s |              38s |
| q-p50   |             186ms  |             94ms |

By kind (NDCG@10):

| kind          | Before | Baseline |
| ------------- | -----: | -------: |
| symbol        |  0.924 |    0.940 |
| behavioural   |  0.504 |    0.499 |
| architecture  |  0.350 |    0.416 |
| bug-report    |  0.383 |    0.383 |
| consumer      |  0.065 |    0.121 |

For reference, the 63-repo upstream benchmark scores NDCG@10 0.8517 overall and
0.849 on Java. This set is deliberately harder: paraphrased queries over a
40-repo workspace where many files legitimately discuss the same mechanism.

## Smoke test

Not a benchmark - a sanity check that the ground truth is findable at all. Ten
queries spread across all five kinds were run through the installed CLI
(`semble search "<query>" . --top-k 5`, the same retrieval stack) from the workspace root, scoring a
hit when any `relevant` path appears in the top 5 file paths.

**Result: 7/10 hit@5.**

Misses:

1. `checking that a submitted redirect destination stays inside this application`
   -> wanted `zenit/src/server/java/.../http/ReturnTarget.java`; got
   `RedirectResult`, `RedirectStatus` and legacy JS.
2. `how is a value owned by the server pushed into a reactive holder inside the browser`
   -> wanted `zenit/src/common/java/.../channel/SyncedRefs.java`; got hawkeye
   compiler/reactivity files.
3. `how is a listing narrowed to the rows the signed-in identity is allowed to see`
   -> wanted `zenit/src/common/java/.../data/RecordSource.java`; got zenit-auth
   identity sinks and legacy JS.

All three misses were re-checked against the source: the annotated file is the
correct home in each case, so these are retrieval failures, not label errors -
exactly the kind of query this set exists to measure.
