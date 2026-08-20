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

- `annotations/javaweb.json` - 90 queries with expected file locations.
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

## Query mix (90 total)

| kind          | count | mapped `category` | what it tests |
| ------------- | ----: | ----------------- | ------------- |
| `symbol`      |    24 | `symbol`          | bare identifier or `Type.member` lookup |
| `behavioural` |    29 | `semantic`        | natural language describing behaviour, never reusing the identifier |
| `architecture`|    16 | `architecture`    | mechanism / design questions ("how is X done") |
| `bug-report`  |    10 | `semantic`        | symptoms only, no identifiers, as a user would report them |
| `consumer`    |    11 | `semantic`        | tests that prove a behaviour, or the layer that consumes a mechanism |

Ten of those (added when `.hwk` indexing landed) answer with a **Hawkeye
template** rather than a Java file: four element-tag / tag-class symbol lookups
(`zc-inbox`, `zfm-media-field`, `PlTabsTrigger`, `ZfQueryBuilder`), four
behavioural queries describing what a piece of markup renders, one architecture
query about how a generated admin listing is laid out, and one consumer query
asking which templates drive a picker from a data provider. Every one of them
scored exactly 0 before templates were indexed, because no `.hwk` file could be
returned at all.

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
4. A programmatic check confirms every one of the 121 referenced paths exists
   under `/home/skerit/projects/javaweb`, that no query text repeats, and that
   the file is pure ASCII.

For the ten template queries the same rule applied, read off the `.hwk` source:
the `{% tag %}` declaration for each element-tag lookup (`{% tag ZcInbox %}`,
`{% tag ZfmMediaField %}`, `{% tag PlTabsTrigger extends PlTabsMember %}`,
`{% tag ZfQueryBuilder implements formAssociated %}`), the `markAllTarget` form
in `zc-inbox.hwk`, the `CmsShell.brandName`/`navSections`/`documentTheme` calls in
`shell.hwk`, the `.pl-date-picker__popup` rule in `date-picker.hwk`, the
`tag TimeTimerControl` block in orcono's `time/components.hwk`, and the toolbar,
filter, row and pagination sections of `pages/resource-list.hwk`. The consumer
query's eleven targets are every non-ignored `.hwk` file that writes both
`pl-select` and a `provider={% ... %}` attribute; the three `zenit-forms` field
templates are `relevant` and the eight application-side users are `secondary`.

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

Default embedder, `.zembleignore` in place, 80 queries, before `.hwk` files were
indexed:

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

### Indexing `.hwk` templates

Templates were invisible to the index until they were registered as a code-lane
language. Adding them grows the workspace by 3,123 chunks. The original 80
queries, same code, same machine, the extension registration the only difference:

| Metric               | Without `.hwk` | With `.hwk` |
| -------------------- | -------------: | ----------: |
| NDCG@5 (original 80) |          0.542 |       0.521 |
| NDCG@10 (original 80)|          0.567 |       0.545 |
| NDCG@10 (new 10)     |          0.000 |       0.655 |
| NDCG@10 (all 90)     |              - |       0.557 |
| chunks               |         73,957 |      77,080 |
| cold index           |            45s |         50s |
| q-p50                |           98ms |       139ms |

**The -0.022 on the original 80 is five queries, and one of them is most of it.**
Per-query, 74 of the 80 are unchanged, one improves, and these five lose:

| delta | kind | query |
| ----: | ---- | ----- |
| -1.000 | architecture | "how do an accordion, a tab strip and a multi-select table share one active-item engine" |
| -0.369 | bug-report | "asking for a page beyond the last one gives back an empty list instead of the final page" |
| -0.369 | behavioural | "the single place a credential is assigned to somebody who did not pick it themselves" |
| -0.043 | symbol | `ChartFunctions.layout` |
| -0.023 | behavioural | "per-row permission entries where a deny entry beats an allow entry" |

The first one is the whole story and it is a **label** problem more than a
retrieval one: the query names an accordion, a tab strip and a table, and the new
top three are `accordion.hwk`, `table.hwk` and `tabs.hwk` - the three components
it names, every one of them a real consumer of `Selection.*`. The annotated
answer, `SelectionFunctions.java`, was written when no template could be
retrieved, and it falls from rank 1 to just outside the top 10. The two -0.369
queries are genuine dilution: markup that mentions pagination or a date field is
not the answer to a question about clamping logic or password assignment.

Those annotations were deliberately **not** rewritten to include the templates.
Widening the ground truth of the queries a change happens to have moved is how a
benchmark stops measuring anything, and no ranking prior against `.hwk` was added
for the same reason: a language penalty tuned on five queries is not a finding.

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
