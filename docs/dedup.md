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
| `--embedder` | env default | Embedder spec used by `--kind logic` |
| `--jobs` | up to 8 | Extraction worker processes |
| `--json` | off | Machine-readable output |

The MCP tool is `dupes(repo, kind, paths, limit, min_files)` and returns the same
JSON. Neither surface needs a daemon or an index.

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
