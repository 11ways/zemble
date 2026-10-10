# Where does this belong?

`zemble home` answers the question a new capability starts with: **does this
already exist, and if not, which module should own it?**

```bash
zemble home /home/skerit/projects/javaweb "let a user schedule a recurring action on one record"
```

It searches, asks the symbol graph who consumes what it found, reads the module
map the workspace declares about itself, and returns four sections: the existing
mechanisms that look like the description, the candidate homes ranked with
reasons, a verdict, and the rules, forbidden dependencies and skills that apply
to the candidates.

Everything it knows about a workspace comes from `<root>/.zemble/home.toml`.
Nothing about any particular workspace lives in zemble.

A `home` call from a nested directory searches upward for the nearest declaration,
including across nested Git repositories. That workspace supplies the index, graph,
module globs, rules and skills together, so a subrepo call can find mechanisms in
sibling modules. A local declaration wins over a parent declaration; a malformed
local declaration fails instead of being skipped. With no declaration, the query
stays at the requested directory and always answers `UNCERTAIN` with low confidence.
Search hits remain visible as leads, never a recommendation to create a mechanism.

Expanded answers name the requested path, resolved workspace and config file at the
top. JSON carries these in `scope`; all evidence paths are workspace-relative.
This expansion is specific to `home`: search, graph and dedup retain their existing
path-scoping contracts. `HomeConfig.load` still loads exactly the root given to it;
the home surfaces resolve the root first through `resolve_home_root`.

This changes query scope, not scoring or candidate ranking. Regression journeys
compare nested and root requests against the same index and real graph across
CLI, MCP and daemon, requiring identical mechanisms, candidates and verdicts.
Existing root queries with a declaration retain their decision algorithm.

## The answer

### Existing mechanisms

The distinct symbols behind the best code hits, at most eight. For each: its
signature, its module, and its **consumer spread** - the distinct modules that
call, import, subclass or otherwise point at it, its own module excluded. A type
is consumed by every incoming edge (`references`), a method by its callers.

A mechanism is marked a **strong match** when it is within 15% of the best hit's
score AND carries the shape of a mechanism rather than of one caller's private
helper: consumers in two or more modules, a position closer to the core than every
one of its consumers, or - the case consumer spread cannot see - **being the symbol
a matched declared row names**.

That last lane exists because a mechanism's wrappers often all live inside its own
module by design. `PreferenceCookie` is the reference case: `Themes`, `Timezones`
and `Disclosures` wrap it without leaving `zenit`, so its cross-module consumer
spread is zero and the first two rules read the declared mechanism as a private
helper. A row that writes a name in backticks - `PreferenceCookie.named(...)`,
`Brand`, `RecordSource.accessCriteria(AccessContext -> Criteria)` - is naming that
symbol as the mechanism, and a hit on it is "this already exists".

Only `Class` and `Class.member` are read as names: argument lists and generic
parameters are dropped first, the `Class.a/b/c` shorthand expands to one name per
member, and a backticked path, package or setting key (`common/holder`,
`comms.channels.*`) names nothing. The owner of a `Class.member` name counts as named
too, but as the weaker `BARE_TYPE` kind: the row is about that class rather than about
that exact symbol, and the answer prints which of the two it had.
And only rows within 15% of the best-matching row may name anything - the same
near-the-top rule the hits themselves are held to, because every row of such a table
names classes, and letting a faintly matching row speak turns its neighbour's
mechanism into "this already exists" (measured below).

A row that shares only VOCABULARY with the description declares nothing. The two facts a
row match carries are kept apart on purpose: `lexical_score` is the weighted word overlap,
and naming is whether the row writes one of the symbols the search actually found. Only
naming can make a mechanism strong, and the verdict says which of the two it had -
`declared (CLAUDE.md row names \`Texts.trimmedOrNull\`)`, `declared type (... the type this
is declared in)`, `graph evidence (consumer spread and module position)` - plus one line
per row that matched on words alone:
`lexically related row: 'Record comments' (word overlap 47% on model, nullable, record; it
names no symbol found here, so it declares nothing about this)`.

Three more lanes hang off this section: the declared-home rows the description
resembles, `find_related` on the single best hit ("also similar"), and the
documentation chunks the same description matched (only when the index covers the
docs lane, which the `home` surfaces always request).

### Candidate homes

Every module the code hits touched is scored:

| Term | Value | When |
| --- | --- | --- |
| relevance share | `module mass / total mass` | always; the module's share of the summed hit scores |
| declared bonus | up to `+0.8` | a matched declared-home row names this module, scaled by how well the row matched (word overlap weighted by how many rows use each word, so "record" or "user" cannot carry a match) |
| lexical bonus | up to `+0.3` | the row names symbols, none of which turned up here: it says which capability family this is, not where this belongs |
| core proximity | `+0.35` | the module already holds this family AND at least one other DECLARED hit module sits after it in `order` |
| forbidden | `-1.0` each | another module the description touched may not depend on this one |
| no dependency path | `-0.05` each | another module the description touched is a SIBLING of this one: neither can reach the other in the dependency graph |

A row that names no symbol at all is not discounted: there was nothing it could have
named, and a human still wrote the module into the home column. Only a row that names
symbols none of which the search found drops to the lexical bonus.

The top three are reported, each with its reasons as sentences ("3 of 5 hits live
here", "declared home for 'Pagination arithmetic' (row match 43%)", "closer to
the core than its 2 consumer module(s): zenit-cms, quirkyquarters", "would make
zenit-widget depend on zenit-cms: widget never depends on cms").

The forbidden check deliberately looks at **every** other module in the hits, not
only the ones further from the core: the module that may not depend on a
candidate is usually the one closer in, and scoping the check by order would make
the rule unreachable.

Reachability is checked at the same site and never replaces the rule. It only fires for a
SIBLING pair - two modules with no dependency path in either direction - because a module
closer to the core not depending on a candidate is the normal case, not a fault: charging
for that hands every answer to whatever sits nearest the core (measured: hit@1 0.869 ->
0.607).

### Verdict

| Verdict | When |
| --- | --- |
| `EXTEND_EXISTING` | a strong match was found - by consumer spread, by core position or by declaration; it is named, with "wire or extend it; do not duplicate it" |
| `NEW_MECHANISM` | no strong match, and one candidate leads clearly; its module is the home |
| `NEW_MECHANISM` (misplaced) | strong matches exist, the demand module can use NONE of them, and the best one's module can depend on the demand's module: what was found is a consumer's copy, and the home is the demand's module |
| `NEW_MECHANISM` (sibling) | a strong match exists, but one of the two modules that want it cannot depend on it and it cannot depend on that module either; the home is their `nearest_common_dependency` and `suggested_home` carries it |
| `UNCERTAIN` | the top two candidates are within 15% of each other, or nothing matched, or two siblings share nothing to put the mechanism in |

Both lanes ask the same question - can the module that WANTS this reach the module that
HAS it? - and answer it from the dependency graph rather than from `order`.

"The module that wants it" is read from the hits, never from the candidate ranking
(`demand_of`): the modules ordered by the relevance mass of their PRODUCTION hits. The
demand is the heaviest of them, and the sibling lane checks the two heaviest. Three rules
keep the lanes honest, each measured below:

- **demand is mass, not the top candidate.** A candidate's score answers "where would a
  new mechanism go" and carries the core-proximity bonus, so the top candidate is usually
  the most core module that got any hit at all. Read as demand, it made every mechanism
  outside that module "a consumer's copy" (`ContentStore` in `zenit` called a copy because
  `protoblast` led the candidates on one hit and the bonus).
- **a test hit is not demand.** A hit in a test source set (`config.source_set_of` is
  `TEST`) exercises a family without needing it in production: `hawkeye`'s browser-test
  templates made `hawkeye` the demand for typed form posting and `FormEndpoint` in `zenit`
  a copy.
- **a usable match before a copy.** When the best-scored strong match is out of the
  demand's reach but another strong match is reachable, that one is extended
  (`_usable_match`); only when none is reachable is the best one read as misplaced.
`AiRecordSources` in `zenit-ai` is the misplaced case: `zenit` leads the candidates,
`zenit` cannot reach into `zenit-ai`, and `zenit-ai` already depends on `zenit`, so the
shared registration belongs in `zenit` and what `zenit-ai` holds is its own copy.

The sibling lane exists because `EXTEND_EXISTING` naming one of two siblings is a
confidently wrong answer: `zenit-flow` cannot use a mechanism that lives in
`zenit-widget`, so "extend `WidgetMigrations`" is advice that does not compile. The
answer says so in one line - "zenit-widget and zenit-flow are siblings (no dependency
path); the shared mechanism belongs in plumage" - and names the shared home: the deepest
module both of them already depend on, not the root of the workspace.

It fires only from a KNOWN dependency graph. A workspace that declares no dependencies
and has no build files is told exactly that instead: "no dependency information for this
workspace: whether X may depend on Y is unknown, not confirmed".

Confidence is `high` / `medium` / `low`, and a lead of less than 40% of the
leader's score is never called high. A workspace without a config always returns
`UNCERTAIN` with low confidence, even when a search hit has multiple consumers.

### Checklist

The workspace's own rules whose scope covers the candidate modules, the forbidden
dependencies that touch them, and the skills to read before designing there.

## Configuration

`<root>/.zemble/home.toml`. A missing file is **generic mode**: modules are the
first path segment, there are no declared homes, rules or skills, and the answer
says so in a note. A file that is present but malformed is a loud error, never a
silent downgrade.

```toml
# Closer to the core first. This is a RANKING and nothing else: it says which module is
# the better home for a shared mechanism, never that anyone may depend on anyone. Who may
# depend on whom is the dependency graph below, and the two are never mixed.
order = ["protoblast", "hawkeye", "zenit", "plumage", "zenit-cms"]

# module name -> glob or list of globs. A path matching none of them falls back
# to its first path segment.
[modules]
protoblast = "protoblast/**"
hawkeye = ["hawkeye/**", "hawkeye-core/**"]

# A module that declares its own dependencies is written as a table instead. Doing so
# REPLACES Gradle discovery for that module entirely, an empty list included.
[modules.zenit]
globs = ["zenit/**"]
depends_on = ["protoblast", "hawkeye"]

# Where dependency edges may come from: "both" (default), "gradle" or "declared".
# `gradle_roots` limits the build-file scan to some directories; empty scans the tree.
[dependencies]
source = "both"
gradle_roots = []

# A version catalog a SETTINGS PLUGIN imports from compiled code, which no build file
# names. It applies to exactly the builds whose settings file applies `plugin`; `name`
# is the accessor (default "libs"), `path` is workspace-relative and must exist.
[[dependencies.catalogs]]
plugin = "be.elevenways.protoblast.settings"
name = "libs"
path = "protoblast/zenit-versions/libs.versions.toml"

# Which fold of a module a path is compiled into. Declaring a fold REPLACES its defaults.
# Keys are the source sets zemble knows: common, server, browser, test. Anything else is
# a loud error rather than a silently ignored line.
[source_sets]
common = ["src/common/**"]
server = ["src/server/**"]
browser = ["src/browser/**", "src/client/**"]

# Dependencies the workspace refuses. A candidate home that would create one loses.
# Either side may be a list: every pair is refused, never a module to itself.
[[forbidden]]
from = "zenit-widget"
to = "zenit-cms"
why = "zenit-widget never depends on zenit-cms"

[[forbidden]]
from = ["zenit-forms", "zenit-cms"]
to = "zenit-editor"
why = "forms and CMS never depend on rich text"

# Markdown tables that already DECLARE homes. Backticked names in the home column
# that resolve to a declared module are read as the home; the rest stays prose.
[[tables]]
file = "CLAUDE.md"
capability = "Capability"
home = "Mechanism home"
consumers = "Consumers (thin wiring)"

# Read before designing in a module.
[skills]
zenit-forms = ["zenit-forms-editing"]

# Echoed in every answer whose candidates the rule covers. `modules` is optional;
# without it the rule is unscoped and always applies.
[[rules]]
text = "Nothing lands without at least one wired consumer and a test"

[[rules]]
text = "zenit core owns no visual components beyond Icon"
modules = ["zenit"]
```

`order` may also be written inside `[modules]`, which is how the design note
spelled it; a workspace that does that cannot also have a module named `order`.
Reading the file needs Python 3.11 (`tomllib`).

## Module dependencies

`order` ranks; the dependency graph permits. They are different questions and the answer
keeps them apart: no part of `home` reads `order` as permission to depend on anything.

Edges come from two lanes, merged per module:

| Lane | What it reads |
| --- | --- |
| declared | `depends_on` on a `[modules.<name>]` table |
| discovered | `settings.gradle(.kts)` for the project layout and its version catalogs, then each `build.gradle(.kts)` for `project(':name')` references, `'group:artifact:version'` coordinates and version-catalog accessors (`libs.zenit.cms`, comma lists continued across lines), in any configuration ending in `implementation`, `api`, `compileOnly`, `compileOnlyApi` or `runtimeOnly` (so `commonCompileOnly`, `serverImplementation` and `browserTestImplementation` all count, while `annotationProcessor` does not) |

Discovery is a **heuristic text scan**, never a Gradle evaluation: a dependency built from
a variable or a loop is invisible to it. It is evidence, so a declaration always wins - a
module that writes `depends_on` ignores everything found in its build file. A coordinate
resolves to a module when the artifact name is that module, or is that module plus one
published fold suffix (`-common`, `-client`, `-browser`, `-server`, `-test-support`,
`-test`); anything else is an external library and contributes no edge.

### Version catalogs

A build sees the catalogs its settings file gives it (`gradle.settings_catalogs`), the
first source to name a catalog winning:

1. the settings file's own `versionCatalogs { libs { from(files('...')) } }` (or
   `create("libs") { ... }`); a `from('group:artifact:version')` import names a published
   catalog the scan cannot read, and is reported as such;
2. a `[[dependencies.catalogs]]` declaration whose `plugin` the settings file applies
   (`id '...'`). This is the fallback for a catalog a settings plugin imports from compiled
   code: zemble never evaluates Gradle or reads plugin sources, so the workspace names the
   catalog's source file once, and it applies only where that plugin is applied;
3. Gradle's convention file `gradle/libs.versions.toml` beside the settings file, as `libs`.

Aliases are normalised the way Gradle builds accessors (`-`, `_` and `.` all separate
segments). An accessor resolves to a library, to a bundle's libraries (`libs.bundles.x`),
or, when it names a GROUP of libraries (`libs.hawkeye` over `hawkeye-common`, `-client`,
`-server`), to the group's direct leaves only - never everything below it, so `libs.zenit`
is not every `zenit-*` artifact.

An accessor in a code configuration that resolves to nothing is REPORTED, never silently
dropped: the graph keeps it as `unresolved` with a reason (`no-catalog`,
`unreadable-catalog`, `unknown-alias`), `home` counts it in a note naming the first three,
and the graph summary (`dependencies` in the JSON answer, the `Dependency graph:` line in
the markdown) lists them all. Only `libs` and catalog names a build actually has are read as
catalog accessors; `sourceSets.main` is not one.

Discovery is lazy - nothing walks the workspace until something asks a dependency
question - and it is cached on the config. On the zenit workspace (41 declared modules, 7 of
them apps under `apps/`, each repository its own Gradle build, every
internal dependency a `libs.*` accessor of the catalog the settings plugin applies) it
takes 0.04 s and finds 234 edges out of 37 modules with no unresolved accessor; the four
without outgoing edges are `protoblast` (the root library), `emberglyph` (only external
dependencies), `zenit-auth-test-support` (declared, no directory of its own) and `alchemy`
(not checked out). Before the catalog was readable the same workspace had 0 edges, so
every reach was UNKNOWN and only `order` ranked anything. `zenit-flow -> zenit-widget` is
correctly absent: those two are siblings, which is exactly the case the verdict has to get right.

`config.reachable(a, b)` answers with one of five values:

| Value | Meaning |
| --- | --- |
| `DIRECT` | `a` declares or builds its production code against `b` (and every module reaches itself) |
| `TRANSITIVE` | `b` is reachable through other modules' production edges |
| `FORBIDDEN` | a `[[forbidden]]` rule refuses the pair; it overrides every edge |
| `UNREACHABLE` | the graph knows what `a` depends on, and `b` is not in it; "knows" includes a declared empty `depends_on` and a build file that was read and named no workspace module (the root library) |
| `UNKNOWN` | nothing at all is known about what `a` depends on: no declaration, no build file, or a build file with an unresolved catalog accessor (the missing edge may be the one asked about) |

Reachability is a question about the source module's PRODUCTION code. An edge written in a
test source set's configuration (`testImplementation`, `browserTestImplementation`,
`testFixturesApi`: the prefix is one of the graph's own test segment names,
`is_test_configuration`) stays in the graph with `test_only` set and grants nothing:
`textum` testing against `zenit` does not let `textum` use `zenit`. A pair written in both
a test and a production configuration keeps the production edge. A forbidden edge is never
walked through either, so a refusal removes its edge from every path, not only from the
pair it names. On the zenit workspace 28 of the 234 edges are test-only: 24 point at
`zenit-microcopy` or `zenit-auth-test-support`, and the other four are `textum -> zenit`,
`zenit-3d -> zenit`, `plumage -> protoblast` (still reached through `hawkeye`) and the
`zenit-cms -> zenit-pages` edge the workspace forbids.

`UNKNOWN` is never permission (`Reachability.usable` is False for it), and it is never
silently turned into "no" either: once a graph is known, an unknown pair fails closed for
scoring, and a workspace with NO edges at all makes the answer say the dependency
information is missing rather than pretend it decided something.

`config.nearest_common_dependency([a, b])` returns the **deepest** module every given
module can reach, or `None`. Nearest means nearest to THEM, not to the core: of the
modules they all depend on, the ones that are not themselves a dependency of another
shared module. Two `zenit-*` siblings depend on `protoblast`, `hawkeye`, `zenit` and
`plumage`; `plumage` reaches the other three, so `plumage` is the answer and the root
library is not. `config.common_dependencies([a, b])` returns that whole maximal set, and
when it holds more than one module `order` breaks the tie - most core first - with the
answer saying the tie happened.

Two consequences worth knowing. A `[[forbidden]]` rule removes an edge from reachability,
so a module sitting behind a refusal can become maximal: `zenit-comms` and `zenit-ai` both
reach `zenit-cms` and `zenit-pages`, and because `zenit-cms must not depend on
zenit-pages`, `zenit-pages` is the deepest thing they share. And a dependency CYCLE would
leave nothing maximal, so the whole shared set is used instead of reporting that the
modules share nothing.

## Source sets

A javaweb module compiles the same package into several folds, and a mechanism in the
server fold cannot be reached from the browser fold however close the modules are.
`config.source_set_of(path)` returns one of `COMMON`, `SERVER`, `BROWSER`, `TEST`,
`UNKNOWN`, from `[source_sets]` or from the defaults (`src/common/**`, `src/server/**`,
`src/browser/**` and `src/client/**`, matched against the path and against the same path
inside its module). A test source set is decided by the graph's own `is_test_path`, so
`src/browserTest/java` is TEST rather than the browser fold.

`config.source_set_compatible(consumer_path, provider_path)` reads one table:

| Consumer | May use |
| --- | --- |
| `COMMON` | `COMMON` |
| `SERVER` | `COMMON`, `SERVER` |
| `BROWSER` | `COMMON`, `BROWSER` |
| `TEST` | anything |
| `UNKNOWN` | `UNKNOWN` |

`UNKNOWN` pairs with itself and nothing else: a workspace that does not split its sources
answers every question about itself, and an unclassified path never claims reuse of a
classified one.

## Surfaces

| Surface | Call |
| --- | --- |
| CLI | `zemble home <path> "<description>" [-k 40] [--json] [--no-daemon]` |
| MCP | tool `home(description, repo, top_k)` |
| daemon | command `home` with `{path, description, top_k, content}` |

All three use the daemon without automatic local fallback (local CLI/MCP execution
requires explicit `--no-daemon`), and all three ask for the
`code` and `docs` content lanes: a design note naming a module is evidence too.
The CLI exits 1 when nothing matched at all.

## Measurement

`benchmarks/home_eval.py`, against the javaweb workspace. Two sets of queries, both in
`benchmarks/local/home_queries.json`:

- **61 declared rows** - one per row of the `CLAUDE.md` capability table whose "Mechanism
  home" cell names exactly one module, asked as a hand-written paraphrase ("I need one
  abstraction whose implementation differs on the JVM and in the browser" for the row
  that declares `protoblast` the home of platform seams), so a row's own words are not
  what finds it. `EXTEND_EXISTING` is the right verdict for all 61.
- **12 negative queries** - six capabilities javaweb does not have (`absent`), three
  mechanisms two sibling modules would share (`sibling`), one a consumer module already
  keeps its own copy of (`substrate`), and two written in a declared row's vocabulary
  about something else (`lexical-trap`). None of them may be answered with
  `EXTEND_EXISTING` naming a module that cannot own the mechanism; each carries
  `expected_verdict`, and where it matters `forbidden_homes` and `expect_suggested_home`.

Every query is answered twice: once with the declared-table lane **disabled**, which
measures what search, the graph and the module order can do alone, and once with it
enabled.

### With a real dependency graph (2026-10-10, zenit workspace)

Until the settings-plugin catalog was readable the zenit workspace had 0 dependency edges,
so every graph-gated lane above was silent and the numbers below this section were tuned
without it. With 234 edges the misplaced lane fired on eight declared rows and verdict
accuracy fell. Same eval set (73 queries, model2vec, `--repos-file` pointing at the zenit
workspace); "graph off" is the same tree with an empty `DependencyGraph`:

| Run | hit@1 | hit@3 | home ok | verdict | over-confident (of 12) |
| --- | --- | --- | --- | --- | --- |
| graph off (7b0ab5f) | 0.639 | 0.885 | 0.656 | 0.644 | 7 |
| graph on, 7b0ab5f | 0.656 | 0.852 | 0.689 | 0.575 | 4 |
| graph on, demand and reachability fixed | 0.656 | 0.852 | 0.705 | 0.671 | 4 |

Both lanes print the same numbers on this workspace: its capability table writes the
home column without backticks, so no row names a home module and the declared lane is
inert (an open item, not addressed here).

The causes, all general and each with a positive and a near-miss fixture test:

- the misplaced lane read the TOP CANDIDATE as the demand, and the core-proximity bonus
  makes that the most core module with any hit (`protoblast` on one hit for object
  storage, the A2UI adapter and test-restorable registries);
- it claimed a copy even when another strong match WAS reachable from the demand
  (`RecordGrantModel` in zenit-auth beside `RecordCapabilityDecision` in zenit);
- browser-test templates counted as demand (`hawkeye` for typed form posting);
- the sibling lane read its co-demand from the top two candidates, so seeing a sibling
  depended on bonuses: `emberglyph`, second by mass for the GPU pipeline, dropped to third
  candidate once test-only edges stopped counting and the answer became `EXTEND_EXISTING`;
- reachability walked test-only edges (`textum -> zenit` is `testImplementation`) and
  through forbidden edges.

Verdicts that changed against 7b0ab5f: test-restorable registries, record-scoped grants,
return navigation, typed form actions, the A2UI wire adapter, object storage and the
microcopy catalogs went from a misplaced `NEW_MECHANISM` to `EXTEND_EXISTING` (right
verdict; object storage and microcopy name `protoblast`, not the declared module); element
measurement went `UNCERTAIN` -> `NEW_MECHANISM` in hawkeye and the record-source substrate
`NEW_MECHANISM` -> `EXTEND_EXISTING` `RecordCapabilities` in zenit, both unchanged in
correctness. Against graph off, the one verdict still lost is raw-value coercion: the only
strong match is `DateFunctions` in plumage, zenit cannot reach it, and zenit's own
`SubmittedValueCoercion`/`PrimitiveCoercion` score just outside the 15% window, so the
answer names zenit as a new mechanism. Gained against graph off: the GPU pipeline and
speech transcription (siblings, no longer `EXTEND_EXISTING`) and the substrate case.

### Earlier measurements (javaweb tree)

73 queries, 81k chunks, 101k symbols / 923k edges, 0.4 s per answer. Before is the same
eval set run against the previous release, so the two tables are one measurement apart:

| Lane | hit@1 | hit@3 | verdict names the home | over-confident (of 12) |
| --- | --- | --- | --- | --- |
| search + graph, before | 0.738 | 0.902 | 0.754 | not measured |
| search + graph, after | 0.738 | 0.885 | 0.721 | 4 |
| plus the declared table, before | 0.869 | 0.918 | 0.803 | not measured |
| plus the declared table, after | 0.852 | 0.918 | 0.787 | 5 |

"hit@1" is the declared home ranked first among the candidates, over the 61 rows only;
"verdict names the home" is the module the verdict itself named, `null` whenever the
answer is `UNCERTAIN`; "over-confident" counts the negative queries answered
`EXTEND_EXISTING` when that was not an acceptable verdict.

The 61 positives cost one query at rank 1 (0.869 -> 0.852) and one at "verdict names the
home". That is the price of the two rules this release adds, and it is stated rather than
hidden:

- a declared row that names symbols, none of which the search found, now earns the
  lexical bonus (`+0.5` at most) instead of the declared one (`+0.8`);
- a candidate loses `0.05` per sibling module that was also hit.

Both were measured against wider settings and both were tightened afterwards: charging
every co-hit module that cannot reach a candidate (rather than only siblings) cost hit@1
0.869 -> 0.607, and a lexical bonus of `0.3` cost 0.869 -> 0.820. The value of the two
rules is on the negative set, which the previous release did not measure at all: the
`substrate` case - "register which module owns a record source" answered with "extend
`AiRecordSources` in `zenit-ai`" - is now `NEW_MECHANISM` in `zenit`, because `zenit`
leads the candidates and cannot depend on `zenit-ai` while `zenit-ai` already depends on
`zenit`. The "shared Flow/Widget migration state" case answers `zenit` too.

Five negatives are still answered `EXTEND_EXISTING`, and the reason is retrieval rather
than judgement: for "a shared way to render a field's validation errors, used both by
form fields and by the widgets that embed them" the search never surfaces `zenit-widget`
at all - the candidates are `zenit-forms`, `zenit`, `hawkeye` - so nothing in the answer
knows there are two siblings to serve. The sibling and misplacement rules can only fire on
modules the search actually found. The remaining two are an absent capability
(`bluetooth pairing`) and a lexical trap (`browser-side object storage`) whose top hit is
strong by consumer spread; an absolute "nothing matched well enough" floor is the obvious
next lever and is not in this release.

The six rank-1 misses that predate this work are unchanged in shape: the core-proximity
bonus pulls the answer one step towards the core.

| Row | Declared home | Answer |
| --- | --- | --- |
| Record-scoped grants | zenit-auth | zenit led (12 of 33 hits); the grant tables read as ORM |
| Field-level access decisions | zenit | zenit-forms led; zenit is #2 |
| Widget trees | zenit-widget | hawkeye led (14 of 29 hits); templates dominate, zenit-widget is #2 |
| Surface actions | zenit-widget | hawkeye led, same reason |
| Pages + public page routing | zenit-pages | zenit-cms led on routing |
| Microcopy/i18n catalogs | zenit-microcopy | zenit led; `zenit-microcopy` depends on `zenit`, so the bonus points the wrong way |

That bias is deliberate - "the mechanism lives as close to core as its mechanics allow" is
the rule it encodes - and it is why the answer is a ranked list with reasons rather than a
single module. hit@3 is 0.918: what the answer misses at rank 1 it almost always still
shows.

Three older measured non-improvements, kept so they are not tried again:

- widening the strong-match window from 15% to 30% raised `EXTEND_EXISTING` to 0.80 but
  dropped "verdict names the home" from 0.787 to 0.721: the extra matches are mechanisms
  in a *different* module, and an `EXTEND_EXISTING` pointing at the wrong module is a
  worse answer than a `NEW_MECHANISM` pointing at the right one;
- letting *every* matched row name symbols, rather than only the rows near the best
  match, reached `EXTEND_EXISTING` for 0.656 of the rows but dropped "verdict names the
  home" from 0.787 to 0.770;
- discounting a declared row by how little of ITSELF the overlap covers - the obvious fix
  for a row whose capability text is a paragraph - dropped hit@1 from 0.902 to 0.869,
  because the genuine rows in that table are long paragraphs too.

Four limits worth stating: a module the search does not touch at all is never a candidate,
however core it is, and the sibling rules cannot fire for it either; the declared-table
match is a token overlap, so a row phrased in vocabulary the description does not share is
not found; the answer only knows the modules that are checked out - the javaweb
`CLAUDE.md` names consumers such as `hohenheim` that live in no directory here; and
`nearest_common_dependency` returns the highest-ranked shared dependency, which in a
workspace where everything depends on one root library is that root
(`protoblast` for two `zenit-*` siblings) unless a nearer module is the only one both can
legally use.

## API for dedup verdicts

`zemble dupes` asks the same workspace the same questions, so the facts live here and are
called from there rather than re-derived. Everything below is on `HomeConfig`, except the
two vocabularies, which live in their own modules.

| Call | Returns | Semantics |
| --- | --- | --- |
| `config.reachable(consumer, home)` | `Reachability` | `DIRECT` / `TRANSITIVE` / `FORBIDDEN` / `UNREACHABLE` / `UNKNOWN` over production edges (test-only and forbidden edges are never walked); `Reachability.usable` is True for the first two only |
| `config.nearest_common_dependency([a, b, ...])` | `str` or `None` | the DEEPEST module every given module can reach - shared, and not itself reachable from another shared module - excluding the given modules themselves; ties are broken by `order`, most core first; `None` when they share nothing or the graph is empty |
| `config.common_dependencies([a, b, ...])` | `tuple[str, ...]` | the whole maximal shared set the line above picks from, so a caller can report the tie |
| `config.source_set_of(path)` | `SourceSet` | `COMMON` / `SERVER` / `BROWSER` / `TEST` / `UNKNOWN` for a workspace-relative path |
| `config.source_set_compatible(consumer_path, provider_path)` | `bool` | the one compatibility table; `UNKNOWN` pairs only with `UNKNOWN` |
| `config.dependencies` | `DependencyGraph` | the merged graph itself: `nodes`, `edges` (each with `origin` and the Gradle `configuration`), `known`, `targets_of`, `has_edges_from` |
| `zemble.home.tables.row_match_kind(declared, unit_name)` | `RowMatchKind` | `EXACT_MEMBER` (the row names this exact symbol), `BARE_TYPE` (it names only the type the symbol is declared in), `NONE` |
| `zemble.home.tables.row_names_symbol(declared, unit_name)` | `bool` | the same fact collapsed to "is this row about this symbol at all"; anything WEIGHING an answer must read the kind instead |

Two rules that a caller must not soften:

- **`UNKNOWN` is not `UNREACHABLE`.** `UNKNOWN` means nothing is known about what that
  module depends on - a module with no `depends_on` and no readable build file. It is
  never permission (`usable` is False). Inside a graph that HAS edges it fails closed, so
  an unknown pair is treated as unreachable and may be reported as such; when the whole
  workspace has no edges (`config.dependencies.known` is False) no boundary may be claimed
  at all, and the answer has to say the dependency information is missing instead. Check
  `known` before you conclude anything from a single `UNKNOWN`.
- **`BARE_TYPE` is not a declaration.** A bare `Type` row names a class, not the member
  found inside it. It is fine as "this row is about that class"; it is not "the workspace
  declared this member's home".
