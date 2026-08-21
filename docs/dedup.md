# Duplication detection

`zemble dupes` reports duplicated Java code as **clone classes** ranked by
weight, in the shape `zenit-dev duplication` prints for `.hwk` templates, so a
reader who knows one report family recognises the other.

It is a **report, never a gate**: the exit code is 0 however much duplication it
finds. The only non-zero exits are a bad flag and a missing path.

It is **Java only, deliberately**. `.hwk` templates are indexed and are in the
symbol graph, but duplicated markup is `zenit-dev duplication`'s job: it matches
alpha-renamed `.hwk` subtrees off the Hawkeye compiler's own AST, which is a
better answer than anything a token stream could give here.

```
zemble dupes /home/skerit/projects/javaweb --kind exact,renamed --limit 20
zemble dupes . --kind logic --min-files 2 --json
zemble dupes . --lane production --brief
zemble dupes . --baseline dupes-baseline.json
```

## The three kinds

**`exact`** hashes the token stream with comments and whitespace removed.
Literals stay verbatim. Two bodies match when they are the same code, formatted
differently and commented differently.

**`renamed`** (alpha-renamed) additionally normalizes every identifier the unit
**declares** -- locals, parameters, lambda parameters, catch/resource/for-each
and pattern bindings, local types -- to positional placeholders in first-seen
order. Field names, method names, type names and every literal stay as they are:
a differing local name always matches, a differing literal or field name never
does. An identifier straight after a dot is a member name and is never renamed,
whatever it is spelled like (see the false positive below).

A group whose members all share one exact stream is pure `exact` duplication and
is reported there alone; a `renamed` class must span at least two distinct exact
streams.

**`logic`** takes embedding neighbours and then **refuses to report any of them
on similarity alone**. A candidate pair (cosine >= `--logic-threshold`, from the
`--logic-top-k` nearest neighbours of each body) is only reported when

1. the control-flow skeletons -- the sequence of `if`/`else`/`for`/`while`/`do`/
   `switch`/`try`/`catch`/`finally`/`return`/`throw`/`break`/`continue` -- are
   identical or within 2 edits, and
2. the called-name sets overlap by Jaccard >= 0.6, and
3. the pair is not already an exact or renamed match.

Each reported pair carries its reason, e.g. `control flow identical; calls
{addColumn, createTable, ...} shared, {addIndex} differs; 14 literals differ
("created_at", "email", ...)`.

At **3 or more copies** the per-pair reasons are aggregated instead of listed:
one consensus line (`7 copies; control flow identical across all copies; all
call {applyAttribute, setStringAttribute}; literals differ per copy`) followed
only by the members that deviate from it (`outlier DominoButtonElement.apply
also calls {toBooleanValue}`). A pair keeps the pair format: there the
aggregate would say nothing more.

## Lanes: production, mixed, test

Every unit is production or test code, decided by the symbol graph's own rule
(`zemble.graph.model.is_test_path`: a `test`, `tests`, `browserTest`,
`integrationTest` or `testFixtures` directory segment). A clone class is
**production** when every member is production code, **test** when every member
is test code, and **mixed** when it spans both.

The report prints one section per lane -- production, then mixed, then test --
each with the kind sections inside it, each ranked on its own. Scores are
untouched: a 25-copy browser-test fixture constructor still scores 25000, it just
sits in the test section where it cannot bury a two-copy production finding.
`--lane production|mixed|test` restricts the report to one of them.

That is the fix for the failure mode this report had on a whole repo: on
zenit-cms the top exact and renamed classes were both the browser-test panel
constructor, 125x the score of the best production class, and neither
`--min-files` nor `--min-tokens` could separate them. `--exclude <glob>`
(repeatable, gitignore-style, relative to the root, applied before anything is
parsed) drops files entirely when a lane is not enough -- generated sources, a
vendored tree.

## Class keys

Every clone class carries a stable key, printed as `key: exact:4c4c163666b7`:
its kind plus the first 12 hex characters of a sha256 over the sorted list of
its members' normalized stream hashes -- the exact stream for `exact`, the
alpha-renamed stream for `renamed` and `logic`.

File paths and line numbers are deliberately not in it, so editing around a
clone, **moving or renaming a file**, and **scanning from a different ancestor
root** all keep the key -- which is what lets a repo's own ignore entries hold
under a workspace-wide scan. Adding or removing a copy *does* change it, which
is what makes a stale suppression visible instead of silently covering a class
that grew; the baseline diff pairs such re-keyings up (see CHANGED below).

## Suppression: `.zemble/dupes.ignore`

Some duplication is deliberate (enum constructors, two bodies a driver's API
forces apart). Commit `.zemble/dupes.ignore`, one entry per line. A scan
honours the scanned root's file **plus every `.zemble/dupes.ignore` under a
directory that holds scanned files**, so each repo commits its own entries and
they hold whether that repo is scanned alone or as part of the workspace.
(A class that gains cross-repo copies under the wider scan re-keys, so the
repo-local entry is then reported stale -- deliberately: the justification was
written about a smaller class.)

```
# deliberate duplication, reviewed
exact:4c4c163666b7  the Couchbase driver hands N1qlRows back per query type
renamed:309a664361af  enum constructors, one per constant by design
```

Source-guard convention applies: an entry is `<key>` followed by whitespace and
a **justification**, and an entry **without** one is itself reported as a
violation and suppresses nothing. An entry that matches no class this run is
reported as **stale** -- except for kinds the run did not scan, so `--kind exact`
never declares a `renamed:` entry dead.

Suppressed classes leave the report and are counted in a trailing
`suppressed: N` line; `--show-suppressed` prints them.

## Baselines

```
zemble dupes . --kind exact,renamed --save-baseline dupes-baseline.json
# ... refactor ...
zemble dupes . --kind exact,renamed --baseline dupes-baseline.json
```

`--save-baseline` writes every class key of the run (with its kind, lane, copies,
score and member locations) as JSON (document version 2; version 1 files, whose
keys included file paths, are refused loudly). `--baseline` prints four
sections: **resolved** (in the baseline, gone now), **changed**, **remaining**
and **new**. The exit code stays 0 -- this is still a report, not a gate.
`--json --baseline` prints the diff as a structured object.

**CHANGED** is the honest middle: content-derived keys churn on edits, so a
class that shrank or grew shows up as one gone entry plus one new class. The
diff pairs each new class with the gone entry of its kind sharing the most
member files and reports `was 625 -> now 435 (score delta)` with both keys,
instead of pretending a partial resolution is a resolution plus a regression.

A suppressed class is neither new nor changed, and is not called resolved
either -- including when it re-keyed but still spans the entry's files: it is
still there, on purpose. A run narrowed with `--kind` or `--lane` only judges
the entries it actually looked for.

Over MCP the baseline lives at the fixed `<repo>/.zemble/dupes.baseline.json`:
`save_baseline=true` writes it, `baseline=true` diffs against it, and one call
may do both (the diff loads the old file before it is overwritten, so a
refactor loop is `baseline=true, save_baseline=true` each round).

## Cross-module verdicts

A workspace scan finds clone classes spanning repos, but a flat list cannot say
what to do about them: that depends on dependency direction. When the scanned
root has a `.zemble/home.toml` (the same file `zemble home` reads: module
`order`, module globs, `[[forbidden]]` rules), every class spanning two or more
declared modules carries one of four verdicts, printed as a `home:` line under
its members and as a `home` object in the JSON:

- **existing-home** -- the most core member module already declares one of the
  copies as a capability's home, so the mechanism EXISTS and the other copies
  are the drift:

  ```
  home: existing home zenit: Texts.trimmedOrNull
        declared by ARCH.md: Blank-safe string trimming
        downstream copies should call or extend it
  ```

  The evidence line carries the row's TITLE (its capability up to the first
  parenthesis, truncated at 100 characters): a capability cell is prose and
  routinely runs for hundreds of characters. The JSON adds `symbol`, `location`
  (`<file path>:<start line>` of that copy, the declaration's first line, not its
  line range) and `evidence` (one
  `{"kind": "declared-row", "capability", "file", "line"}` per row, with the
  whole capability cell) to the usual `verdict` / `modules` / `home` / `detail`.
- **candidate-home** -- the most core member module, when every other member may
  depend on it: `home: candidate home zenit (spans orcono, zenit; zenit is the
  most core member module and every other member may depend on it)`. This is a
  PLACE for a new mechanism, never a claim that a reusable one is already there;
  only `existing-home` says that.
- **forbidden-dep** -- a `[[forbidden]]` rule blocks some member from depending
  on the would-be home; the rule and its `why` are quoted, plus `a shared home
  must sit deeper than <module>`. This is the case where a naive
  "extract a shared class" is architecturally wrong.
- **no-shared-ancestor** -- no member module is in the declared order (sibling
  apps): the weakest finding, labelled as such.

`existing-home` reads the same `home_modules` the `home` tool does, so the home
cell must name its module IN BACKTICKS: a row whose home cell is prose declares
no module, matches no copy, and is silently no evidence -- no verdict change and
no note. `forbidden-dep` outranks `existing-home`: when a member may not depend on the
module that declares the mechanism, calling it is not the fix, whatever the
table says.

No `home.toml` means no verdicts and no noise; a malformed one is reported as a
note and skipped, because this is a report, never a gate.

### What counts as evidence for `existing-home`

Declared capability-table rows (the `[[tables]]` of `home.toml`) and nothing
else. A copy is promoted when it is a whole body -- not a statement window --
living in the candidate home module -- and not a synthetic member such as
`Type.<initializer>`, which nothing can call -- and a row whose `home` cell
backticks that module names that body. Three name shapes match and no more: the exact
`Type.member`, a `Type.member` row whose qualified tail the unit carries
(`Outer.Texts.trimmedOrNull` is named by `Texts.trimmedOrNull`), and a bare
`Type` row, which covers the members declared directly in that type. A row that
names a different type's member of the same name (`Other.trimmedOrNull` against
`Texts.trimmedOrNull`) does NOT match: symbol matching fails closed, because a
wrong `existing-home` sends a reader to code that does something else.

Callers, the symbol graph, embeddings and search are deliberately not consulted.
`dupes` is a cache-free scan that must run on any checkout without an index, and
usage is not intent: a helper twenty callers reach for is not thereby the
declared home of anything, while a mechanism a human wrote into the table is one
however few callers it has. Classifying a class therefore never triggers
indexing, embedding or retrieval. The tables are read once per run, and only
once a class actually spans two declared modules: a scan with nothing to judge
never opens them. A declared table file that is missing is then a note in the
report and no evidence; one that parses to no rows is simply no evidence.

## `--brief`

Header plus one line per class -- rank, kind, lane, copies x tokens, score, root
symbol, file count and key -- and nothing else. No member paths, no reasons, so
the output survives a pipe without losing the exit code the way `| grep` does.

## Units

Every method, constructor, annotation element and initializer **body**, plus
every window of `--min-statements` (default 6) consecutive statements inside any
block in one, each with its file and line span. Bodies and windows shorter than
`--min-tokens` (default 30) are dropped, which is what keeps getters, setters
and one-line delegates out of the report. Windows are capped at 24 statements
and never repeat a body's own full statement list.

Anonymous and local classes are part of their enclosing body's token stream
rather than units of their own.

## Ranking

`score = tokens x copies x files`, the weighting `zenit-dev duplication` uses
(the plan's `tokens x (members - 1)` was dropped for it, so both reports rank the
same way: a small shape spread over many files beats a big local one).

A class is dropped when a higher-ranked class with at least as many members
already contains every one of its members. That is what collapses the dozens of
overlapping window lengths of one copied run into the single widest one.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--kind` | `exact,renamed` | `exact`, `renamed`, `logic`, `all`, or a comma-separated list |
| `--limit` | 25 | Clone classes printed per section |
| `--min-files` | 1 | Only report classes spanning at least N files |
| `--min-tokens` | 30 | Smallest unit that may form a class |
| `--min-statements` | 6 | Smallest statement window compared inside a body |
| `--no-windows` | off | Compare whole bodies only |
| `--logic-threshold` | 0.92 | Cosine a logic candidate needs |
| `--logic-top-k` | 10 | Embedding neighbours considered per body |
| `--paths` | whole root | Restrict the scan to these paths |
| `--exclude` | none | Gitignore-style pattern dropped before parsing (repeatable) |
| `--lane` | `all` | Report one lane: `production`, `mixed`, `test` |
| `--brief` | off | Header plus one line per class |
| `--show-suppressed` | off | Also print the classes the ignore file took out |
| `--save-baseline` | off | Write this run's class keys to a file |
| `--baseline` | off | Report resolved / remaining / new against a saved baseline |
| `--embedder` | env default | Embedder spec used by `--kind logic` |
| `--jobs` | up to 8 | Extraction worker processes |
| `--json` | off | Machine-readable output |

## The MCP tool

`dupes(repo, kind, paths, exclude, lane, limit, min_files, format, brief,
baseline, save_baseline)`.

`format="text"` (the default) returns the report exactly as the CLI prints it,
as a plain string; `brief=true` trims it to the class lines. `format="json"`
returns the structured object itself -- FastMCP encodes it once, so a client
never has to parse JSON out of a JSON string. `baseline`/`save_baseline` are
booleans against the fixed `<repo>/.zemble/dupes.baseline.json` (see
Baselines). Neither surface needs a daemon or an index.

The `--kind all` text report is roughly a third of the tokens the JSON form
costs. Reasons are collapsed on both surfaces: one reason per class when every
pair agreed, and the consensus-plus-outliers aggregate at 3+ copies.

## Cost, measured on one repo

zenit-cms (270 Java files, 52 950 units of which 1670 are bodies):
`--kind exact,renamed` 6.5 s, `--kind logic --no-windows` 1.9 s of which 0.9 s
embedding with the local potion-code-16M-v2 model. Lanes: 10 production classes,
0 mixed, 54 test.

## Cost, measured on the javaweb workspace

6228 Java files, 556 321 units of which 29 374 are whole bodies, on 8 worker
processes:

| Run | Wall clock |
| --- | --- |
| `--kind exact,renamed` | 45.5 s |
| `--kind logic --no-windows` (potion-code-16M-v2) | 43.9 s, of which 15.8 s embedding |
| `--kind all` | 73.6 s |

Class counts for `--kind all`: 554 exact, 125 renamed, 1645 logic (4719 pairs
survived the structural check). The full JSON is
`benchmarks/results/dupes-javaweb-fd24a6849e1f.json`.

## Reading the top of that report

Every class below was opened in the source and classified.

| # | Kind | Class | Verdict |
| --- | --- | --- | --- |
(Measured before lanes existed: every "test scaffolding" verdict below is now
sectioned under TEST, and the production classes are what the report leads with.)

| 1 | exact | 19 copies of the `ActivityPanel(String slug, PanelPeer peer)` fixture constructor across zenit-cms browser tests | Genuine. Test scaffolding copied per test class; one shared fixture panel would remove all 19. |
| 2 | exact | 12 copies of a 7-statement window inside `createAdministrator` in zenit-auth tests | Genuine, and the same code as #3: a shared test helper is missing. |
| 3 | exact | 11 copies of `createUser(String email)` in zenit-auth tests (`Row` + 5 `set` + `save`) | Genuine. A `AuthTestUsers.create(...)` helper is the fix. |
| 4 | exact | 18 copies of `PrincipalConduit.setAttribute` (the stub `Conduit` used in tests) | Genuine. One test-fixture Conduit, copied into every repo that needed one. |
| 5 | exact | 6 copies of the user + password row creation window (orcono, zenit-auth) | Genuine, and crosses repos: this one belongs in a published test fixture. |
| 1 | renamed | 25 copies of the browser-test fixture panel constructor (exact class #1 plus 6 that only differ in locals) | Genuine. Same finding as exact #1, with the near copies folded in -- which is what `renamed` is for. |
| 2 | renamed | 9 copies of `trimToNull` / `trimmedOrNull` / `blankToNull` across proteus, QQ, zenit-ai, zenit-widget | Genuine, and the most actionable finding in the report: one text helper, five repos. |
| 3 | renamed | 7 copies of `countOccurrences(String, String)` in hawkeye tests | Genuine. A test-support helper. |
| 4 | renamed | 8 copies of `blankToNull` inside orcono alone | Genuine. Same helper, copied within one repo. |
| 5 | renamed | 5 copies of a `writeDirectGrant` window (zenit-auth production + its tests) | Genuine, and the interesting shape: the tests re-implement what `GrantService` already does. |
| 1 | logic | 35 migration `up()` bodies across arcana, orcono, zenit-auth, zenit-microcopy | Legitimate parallel. Every migration calls the same schema DSL in the same order; the 46 differing literals in the reason ARE the migration. This is the weakness of logic mode: builder-DSL bodies all look alike. |
| 2 | logic | 20 copies of `createAdministrator` in zenit-auth tests | Genuine, superset of exact #2/#3. |
| 3 | logic | 32 copies of the browser-test fixture panel constructor | Genuine, superset of exact #1 / renamed #1. |
| 4 | logic | 14 copies of `applyMigration` in zenit ORM tests (build a migration, run it, assert completed) | Genuine. Test scaffolding that should be one helper taking a table spec. |
| 5 | logic | 19 copies of the stub `setAttribute` | Genuine, superset of exact #4. |

Two honest observations from that table. First, `logic` re-states families that
`exact` and `renamed` already found, with the near misses folded in; the pair
exclusion only stops the *pair*, so a class can grow around an exact core. That
is useful (it shows the whole family) but it is not new information. Second, the
one class that is not worth surfacing -- the migrations -- is not a detector bug:
the bodies really do have identical control flow and an almost identical call
set. `--min-tokens` will not separate them; only a human reading the reason will.

## The false positive that was a real bug

The first workspace run ranked, as the top renamed class, 130 copies of a
constructor field-assignment run across 38 unrelated files (`TransportMode`,
`TextEdit`, `EventModifier`, `ServerDominoKeyboardEvent`, ...). The cause: in
`this.key = key;` the parameter `key` is a declared local, so the normalizer
replaced **both** occurrences, and every `this.field = field;` constructor in the
workspace collapsed to one stream. The fix is the dot rule above: an identifier
straight after `.` is a member name and is never renamed. Renamed classes fell
from 163 to 125 and the entire family disappeared. Fixture `CtorA`/`CtorB` in
`tests/fixtures/dedup` holds the case.

## Limits

- Java only. The unit extractor is tree-sitter Java; nothing else is scanned.
- No index and no incremental state: every run re-parses the tree. That is the
  45 s above, and it is why there is no cache to invalidate.
- Two call-free bodies trivially satisfy the call-set check (an empty set
  overlaps an empty set completely); their reason says
  `neither body calls anything` so a reader can see it happened.
- A window class and the body class containing it can both be reported when the
  window spans more files than the body does; that is deliberate (the window is
  the wider finding) but it does read as two entries for one family.
- `logic` embeds every body in the workspace on every run. It is 16 s with the
  local Potion model, but a paid embedder would make it a paid operation.
