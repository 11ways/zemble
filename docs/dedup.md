# Duplication detection

`zemble dupes` reports duplicated code as **clone classes** ranked by weight,
in the shape `zenit-dev duplication` prints for `.hwk` templates, so a reader who
knows one report family recognises the other.

**Every language with a bundled grammar** is compared: Java and Zig through hand-written
profiles, every other language through a profile derived from its grammar spec in
`src/zemble/languages/catalog.py` (the same spec the symbol graph reads), see
[Adding a language](#adding-a-language). The report header names the extensions the run
walked, so an empty result can never be mistaken for a clean workspace.

It is a **report, never a gate**: the exit code is 0 however much duplication it
finds. The only non-zero exits are a bad flag and a missing path.

It was Java-only for its first year, and that was an implementation shortcut,
not a principle: every node type the extractor knew was hard-coded tree-sitter
Java. Everything downstream of unit extraction -- hashing, grouping, ranking,
lanes, ignore files, baselines, home verdicts -- was already language-neutral, so
the fix was to lift the Java vocabulary into a profile, register a second one (Zig),
and then derive one from every grammar spec so the remaining languages arrived at once.

`.hwk` templates stay out on purpose. They are indexed and are in the symbol
graph, but duplicated markup is `zenit-dev duplication`'s job: it matches
alpha-renamed `.hwk` subtrees off the Hawkeye compiler's own AST, which is a
better answer than anything a token stream could give here.

## Architectural candidate channel

`zemble dupes . --kind architectural --json` (MCP: `kind="architectural"`)
uses the daemon's published symbol graph to find documented migrations and thin
callable delegation. It reports `candidates`, distinct from literal `classes`.
Each candidate includes its source declarations, evidence kind and reason, and
`equivalence_proven: false`: discovery still needs behavioral review. Small bodies
and changed control flow are allowed here; literal clone thresholds remain intact.
The current whole-workspace scan covers files with explicit deprecation evidence.
Unreadable or oversized source files are named in `unreadable_files`.

`--paths`, `--exclude`, `--min-files` and `--limit` bound this channel. Literal
baselines and lane filtering are refused for architectural candidates. `all`
continues to select the literal clone channels. Graph construction shares the
same isolated, bounded worker as `home` and related reads.

```
zemble dupes /home/skerit/projects/javaweb --kind exact,renamed --limit 20
zemble dupes . --kind logic --min-files 2 --json
zemble dupes . --lane production --brief
zemble dupes . --baseline dupes-baseline.json
```

## Sub-body and vocabulary channels: holed, idiom, reimplements, vocabulary

A deduplication audit of Hohenheim and the Zenit framework (2026-10) found about 1,800 lines of real
duplication that `exact`/`renamed`/`logic` could not see: the same call chain with different literals
983 times, 3-line twins below the 30-token floor, private copies of an existing framework method, and
one status vocabulary declared in eleven places. Four kinds cover those classes.

### Design: one more set of kinds, not a parallel tool

What existed: `exact`/`renamed`/`logic` compare whole bodies and statement windows, keep literals verbatim
by design and drop anything under 30 tokens; `architectural` needs the daemon's graph and only looks at
files that mention a deprecation; `home` judges a described capability, not code; `find_related` answers
one chunk at a time. None of them could carry a site list, a typed hole or a declared value.

Decision: the four channels are new `CloneKind` members, so they reuse everything downstream of a clone
class unchanged: lanes, content keys, `.zemble/dupes.ignore` (justified entries suppress them, stale and
unjustified entries are reported), baselines, `--brief`, home verdicts, `--limit` with totals, and the
`--kind` vocabulary on both surfaces (`--kind holed,idiom`, MCP `kind="vocabulary"`; `all` selects every
kind). One home per vocabulary:

- `CloneKind.facts` (`KindFacts`) is the only place a kind's key attribute, ranking (`Ranking.MASS` or
  `Ranking.SPREAD`), lead text, member cap, focus lane and embedding need are declared; a kind missing
  from the table raises, and `tests/test_dedup_shapes.py` fails on one without facts.
- `SiteKind` (`chain`, `wrapper`, `pair`, `constant`, `literal`, `set`, `switch`, `regex`) is the unit
  kind of a site. Sites live in their own lists beside the clone units, so exact and renamed never see one.
- `ShapeHooks` on the language profile is the home of every language fact the channels read (call
  receivers, the constant naming convention, forwarding bodies, `@Override`, the vocabulary walk). Java
  has them; a profile without them takes part in `holed` and `reimplements` through its literal kinds and
  call names, and has no idiom or vocabulary sites. A run of `idiom` or `vocabulary` names the languages
  it covered (`idiom and vocabulary sites cover: java`), so an empty Python result never reads as clean.
- `dupes` stays cache-free: no graph, no index. The re-implementation lane reads the same body vectors
  and local mirror `logic` reads (`find_related`'s embedding space), never the daemon.

The holed stream is the common normalisation: the renamed stream with every literal a typed hole
(`<str>`, `<num>`, `<chr>`, `<bool>`, `<null>`, `<lit>`, typed by the literal node kind's own name) and
every qualified constant reference (`Egress.NONE`, `MAX_SIZE`, the profile's `is_constant_name`) one
`<const>` hole.

### holed: bodies equal up to literal values

Whole bodies of at least 8 tokens grouped by holed stream. Floors replace the 30-token one: a class needs
`tokens x copies >= 40` (two copies of 20 tokens, three of 14, five of 8). A group that is one renamed
stream at 30+ tokens is exact or renamed duplication and stays there. Below 30 tokens a body that calls
nothing (`this.x = x;`) or only hands literals to one call (`return Icon.of("x");`) declares data and is
dropped. Ranked by mass like the clone kinds. Notes: the shape with its holes, copies and literal variants.

### idiom: one call shape at many sites

Sites, not bodies: every call chain of at least two calls and 8 tokens, the receiver-holed tails of a
chain whose head varies (`<anything>.resolve($0.getLocales(), $0.getMessageResolver())`, and every longer
suffix of an outermost chain, `<anything>.offset(n).limit(1000).all()`), and every
pair of different calls on one receiver with the same first argument (`getAttribute(K)` ...
`setAttribute(K, v)`). The key erases every typed hole AND every placeholder the site uses once, so a
wrapper `copy(String key) { return Microcopy.of(key).withFilter("scope", "x"); }` and an inline
`Microcopy.of("title").withFilter("scope", "x")` are one idiom; a placeholder used twice stays, because it
is the site's data flow. A chain that is its method's whole body is a `wrapper` site, and the class names
its wrappers.

A shape needs 3 sites in 2 files AND evidence that it is duplication rather than an API used as designed
(`.icon(Icon.of("x"))` at 95 sites is not a finding): some sites already wrap it while others inline it;
nearly every site (90%) repeats one literal; one value is wired into several calls; four or more different
calls are chained; or a read/write pair of one noun (`getAttribute`/`setAttribute`). A class whose only
evidence is a repeated literal that a better-ranked idiom inside it already repeats is dropped
(`_.label(Microcopy.of(...)...)` says nothing beyond the Microcopy idiom). Ranked by spread: copies x
files, copies counted up to 3 per file, so 255 sites packed into 7 migration files rank as the 21 they
spread like. Each class lists its first 12 sites (`... and N more site(s)`; the JSON keeps `copies`).
A shape and its prefix-extension cut at the same sites are one family: when they share 75% of the larger
site set (`SAME_SITES_SHARE`), the variant with more sites stays (the longer shape on a tie) and names the
other (`one family: 29 of these sites as _.tabs(...).build()`). An extension found at fewer of the prefix's
sites is a narrower finding and stays its own class. On zenit-cms this folds 535 idiom classes into 481,
on Hohenheim b9b8f221 397 into 350; no other class changes.

### reimplements: code that redoes an existing method

Each class is one API and the bodies that redo it, copies first and the API last:
`X, Z (2 bodies) re-implement Y; call Y (file:line)`, then one evidence line per copy.

- **Forwarding facade**: a type whose bodies (75% of them, at least 2) only pass their parameters, in
  order, to one other type that the scan holds (`Slugs` over `SlugText`). Read off the syntax.
- **Re-implementation**: a public method of a public type, outside the tests, not an `@Override` and not
  a builder (under 40% of its calls chained on a call result), at most 20 different calls, that a
  non-test body in another file and type repeats without calling it. Candidates come from three lanes,
  each with its own bar: the same code with locals renamed (a private copy of a public helper); a close
  embedding neighbour (cosine >= 0.85 among the 10 nearest) repeating 75% of the API's calls, two of them
  uncommon, control flow within 3 edits; or a pair sharing four uncommon calls (cosine >= 0.75, 60% of the
  API's calls, flow within 4 edits). A call made by more than 0.5% of the scanned bodies is common. When
  the API has two or more literals the copy must repeat half of them. Two same-named members only count
  when the copy is private or package-private: a public or protected namesake is an override or a
  parallel implementation. Embedding similarity alone never reports anything.
- **Same intent** (a fourth lane, for the copies the three code lanes cannot see because they do the same
  thing with different calls): a helper outside the tests whose intent text sits among the 10 nearest of a
  public method's (cosine >= 0.6). A helper is a static method or an instance method the Java profile proves
  reads no state of its own (`Signature.reads_instance`: no `this`/`super`, no instance field, no call an
  enclosing type does not declare static; it fails closed, so a shadowing local or an inherited call counts
  as a read). Both sides are at most 400 tokens (`INTENT_MAX_TOKENS`), and a pair with a side over 160
  (`HELPER_MAX_TOKENS`) must also share an uncommon call: a mechanism that redoes another with none of its
  calls is a parallel one. The intent text is what a
  member says it does: its name in words, its signature (`(Object) -> Integer`), its Javadoc and the names
  it calls, embedded beside the bodies in one purchase. The pair is then weighed by four signals, each
  declared once with its weight in `reimplements.Signal`: intent cosine 0.45, body cosine 0.25, name 0.20
  (how much of each name the other spells, rare words weighing most, a word matching its prefix so `int`
  meets `integer`), signature 0.10 (whether the copy could hand its own inputs to the API and use its
  result; below 0.5 the pair is refused). The weighted score must reach 0.80, and the copy must hold every
  literal the API holds (`copy(key)` with its own `"scope"` is a parallel helper). A body that is an
  accepted copy again (body cosine >= 0.9, control flow within 2 edits, 60% shared calls) joins it.
  These classes carry `inferred: true` in the JSON and rank after every class code evidences, and their
  copies never join a code-evidenced class: the code lanes' classes are the same with or without them.
- **The original, by architecture**: of every API a copy matches, the original is the one in the most core
  module the copy's module may depend on (`reimplements.Architecture`, read from the scanned root's
  `.zemble/home.toml` through `HomeConfig`: `order` ranks, the dependency graph and `[[forbidden]]` decide
  reach). Preference: reachable (direct or transitive), then reach unknown, then the most core module, then
  the best evidence. Of two public bodies the one in the more core module is the home and never a copy;
  inside one module, a `common` source set and then the shallower path decide as before. When every API a
  copy matches lives in a module its module may not depend on (forbidden or unreachable), the class says
  so (`..., but zenit may not depend on hohenheim (unreachable): no original to call`) instead of advising
  the call. Without a `home.toml` every body ranks alike and every reach is unknown, which reproduces the
  path order exactly. An original that is itself a reported copy hands its copies on to its own original
  (`B.helper is itself a copy of C.core`); a code-evidenced copy only follows code-evidenced links, and no
  link is followed to an original the copy reaches worse.
- **Signature facts**: the intent lane reads `Signature` (parameter and result types by simple name, open
  types such as `Object` and type variables, staticness, whether the body reads its instance, Javadoc)
  from the profile's `ShapeHooks.signature`; Java reads it, a profile without it takes no part in the lane.

To find copies of framework methods in an app, scan a root holding both and focus on the app:
`zemble dupes <workspace> --kind reimplements --focus apps/hohenheim`.

### vocabulary: one value declared in several places

Read from the language's vocabulary walk: constants with a plain string value, literal uses, value sets
(an enum's lower-cased members plus the strings its constants pass, a run of constants sharing a name
prefix such as `STATUS_*`, or a type's unprefixed constants), the string labels of a switch with what it
dispatches on, and regex literals (the first string argument of `compile`, `matches`, `replaceAll`,
`replaceFirst`, `split`). Four flavours, each ranked on its own and taking turns in the report:

- **values**: a value declared by 3+ constants in 2+ files under one meaning (names that spell the value,
  `FAILED`/`STATUS_FAILED`, or one repeated name, `STATE_COLUMN`; six constants holding "hohenheim" for six
  purposes are not one vocabulary), or a compound value (`instance-devices`, `host_admission`) written
  as a literal where a reachable constant holds it. A common word's literal uses are counted, never listed.
- **value sets**: sets sharing 3 values and 60% of the smaller one, unioned into families; a near twin
  (`success` beside `succeeded`, 5 shared leading characters) counts toward the overlap and is reported
  as drift.
- **dispatch**: switches on one call (`switch (column.name())`) in 3+ files, each restating its labels.
- **regex**: a regex of 8+ characters (an anchor, a class, an escape) written in 2+ places, as a call
  argument or a constant.

Each class carries a suggested home: the existing public declaration in a shared (`common`) source set,
shallowest first, or the members' deepest common directory.

### Measured (2026-10-10)

On the Hohenheim tree before its cleanup (b9b8f221, 912 production files; `exact`/`renamed` find 0
production classes there and `logic` 61): holed 85 classes, idiom 397, vocabulary 87 in 2-7 s each. On a
scratch root holding that tree plus zenit, zenit-cms and protoblast sources (4 512 files, 31 645 compared
bodies), reimplements 26 classes in 15-25 s warm. Every case the audit listed was looked up in these
outputs; the misses are below under Limits.

Same-intent lane (2026-10-10, same scratch root, 5 139 Java files, 31 679 bodies, 31 420 intents): 26 -> 62
classes, the 26 code-evidenced ones unchanged in membership, score and order (`AppDirectory.fixOf` still #6,
the `Slugs` facade still #19), 36 inferred ones after them. Recall on the audit's misses:

| Miss | Found | Rank, score | API it points at |
| --- | --- | --- | --- |
| `effective*` (7) | 4 of 7: SiteModel, InstanceVariableModel, InstanceTemplateModel, GitProviders | #27, 2100 (score 0.92) | `SiteDomainModel.effective`, the public copy; not `Row.afterWrite` |
| `ResourceLimits.asInteger`, `GitRepositoryResolver.providerIdOf` | both (providerIdOf as its twin) | #30, 558 | `IndexedScopes.intOf` (zenit), not `CmsSupport.parsedInt` |
| `StackDeploymentsPage.durationLabel` | no (score 0.69) | - | - |
| `trimmedOrNull`, `blankToNull` | both | #51 (0.85), #58 (0.82) | `Texts.trimmedOrNull`, `Texts.blankAsNull` |
| DatabaseParts regex vs `DatabaseModel.isValidName` | no (not attempted) | - | - |

What moved recall, measured by ranking each target pair among every gated pair: the intent embedding is
the signal that ranks the targets first or second (body cosine ranked them 11th to 784th); adding the called
names to the intent text took `asInteger` from 9th to 1st; the twin lane added `providerIdOf`. A
family-agreement signal (copies whose twins point at the same API) and an exact-signature bonus were
tried and dropped: both admitted more noise than targets. Precision, top 30 of the scratch root read and
judged: 25 code classes as before (7 noise by a strict reading: two panel declarations, two parallel
parsers, a registration, `fingerprint`/`backup`, `listOf`/`shown`, two HTTP header getters across
libraries) plus 5 inferred (`stringOf` copies, mostly real, its `stringOrEmpty` members borderline;
`effective`; `HohenheimFormCopy.label`; `CmsSupport.textOf`; `intOf`): 8 of 30. Whole javaweb workspace (11 987
files, 79 285 bodies, 76 578 intents): reimplements 717 s the first time (685 s of it buying ~3.9M intent
tokens, ~$0.08, with voyage-4-lite), 23 s warm, 226 classes.

Originals by architecture and the wider lane (2026-10-10; scratch root rebuilt from Hohenheim b9b8f221 plus
the framework sources at their current heads, 5 532 Java files, 35 022 bodies, 33 318 intents, with a
`home.toml` declaring protoblast -> zenit -> zenit-cms -> hohenheim). At 3b8aef0 the root gave 69 classes
(27 code-evidenced, 41 inferred, one facade lane); now 70. Without a `home.toml` the 27 code classes are
byte-identical; with it they are too, and the inferred ones change only where the architecture or the chase
says so: four zenit or zenit-cms helpers matching a Hohenheim helper now read `... but zenit may not depend on
hohenheim (unreachable): no original to call` instead of advising the call, a public zenit-cms method is no
longer reported as a copy of Hohenheim code, `CreatorGrantHook.install` points at zenit's `ActivityLog.install`
instead of Hohenheim's `ProxyReloadHooks.install`, and the two `bucketKeyOf` classes merged into one through
`InstanceQuota.memoryBucketOf`, itself a copy of `InstanceDeviceQuota.diskBucketOf`. The wider lane admitted
one new class, `HohenheimActivityAction.icon`/`ZenitCmsActivityAction.icon` re-implementing
`ActivityActions.unknownIcon` (both `return Icon.of("circle-info")`). Precision, top 30 read and judged by the
strict reading above: positions 1-29 are identical to 3b8aef0's on this root (27 code classes, then the
`stringOf` and `effective` inferred ones; 9 noise: a panel declaration, a parallel parser, a serializer
registration, a batch-insert pair, `fingerprint`/`backup`, two `close`/`listOf` parallels, two header
getters), and #30 changed from the `HohenheimFormCopy.label` class (borderline, now split by architecture)
to the `intOf` target class: 9 of 30 against 10 of 30. Recall on the audit's 14 bodies:

| Target | Found | Rank, score | Original it points at |
| --- | --- | --- | --- |
| `effective` in SiteModel, InstanceVariableModel, InstanceTemplateModel, GitProviders | 4 of 4 | #29 (0.92) | `SiteDomainModel.effective` (not `Row.afterWrite`) |
| `SiteDomainModel.effective`, `TenantWrites.effectiveSiteType`, `DnsZoneCascades.effectiveZoneId` | 0 of 3 | - | - |
| `ResourceLimits.asInteger`, `GitRepositoryResolver.providerIdOf` | 2 of 2 | #30 (0.92) | `IndexedScopes.intOf` (zenit, the most core) |
| `StackDeploymentsPage.durationLabel` | no | 0.69 | - |
| `InstanceConsoles.trimmedOrNull`, `ApiProviderClient.blankToNull` | 2 of 2 | #56 (0.85), #62 (0.82) | `Texts.trimmedOrNull`, `Texts.blankAsNull` |
| DatabaseParts regex, `DatabaseModel.isValidName` | no (not attempted) | - | - |
| `AppDirectory.fixOf` (guard) | yes | #7 | `RecordHealthReads.fixCell` |
| `Slugs` facade (guard) | yes | #20 | `SlugText` |

Why `Row.afterWrite` is not reached: every pair from the `effective` family to it scores 0.44-0.62
(`SiteDomainModel.effective`: intent 0.80, body 0.61, name 0, signature 1.0), and among the 4 196 gated pairs
whose names share no word it ranks 219th and 265th, behind `Builder.storedIn -> Nested.of` and its kind. No
pair whose names share nothing can reach 0.80 (the other three signals sum to at most 0.80 only when all are
perfect), and reweighting to let one through would admit those 218 first. The chase is in place: the day
`SiteDomainModel.effective` is linked to `Row.afterWrite`, its four copies follow. Why `durationLabel`
scores 0.69 against `RelativeTime.duration`: the copy has no Javadoc and makes one call (`longValue`), so its
intent text is its name and signature alone (intent 0.69); the API's name is one of its two words (name
0.62); and the API holds literals the copy does not (`1000.0` where the copy has `1000`, three `false`
flags, `null`), so the literal rule refuses it whatever the score. It also is not a drop-in copy: it prints
`5s` where the API prints `5 seconds`. Two principled changes were weighed on the measured signals and not
shipped: a name signal reading only the API's direction lifts it to 0.77, still under the bar, and number
normalisation (`1000.0 == 1000`) leaves the `false` flags failing the literal rule.

The wider lane, measured on the scratch root (same gates, signals and bar): judging every size took the
intent matrix from 7 854 APIs x 8 511 copies (1.0 s) to 8 749 x 22 552 (2.7 s); instance methods admitted 25
copies, every one but two reading its own state (`this.running`, an enum constant, `model()`), so only
stateless ones are judged; past 160 tokens six pairs reached 0.75 and the one over the bar
(`PanelRegistry.register -> PanelPlacements.register`, two registries) shared no uncommon call, so a side
over 160 tokens needs one; past 400 tokens nothing reached the bar. On the whole workspace the lane takes
5.2 s (3.7 s at the old gates, after its neighbour search was vectorised per block instead of per row) and
buys nothing new: the bodies and intents it compares were already embedded (`embed-status --dupes
reimplements`: 155 873 texts, 28 uncached by workspace drift, ~$0.0001). Whole workspace, load average
10-15 from other agents: 36 s for the first run (buying those 28), 26-44 s after (3b8aef0: 26 s in the
same window), 222 classes. The workspace's `home.toml` gives no dependency graph (its build files take the
version catalog from a settings plugin the Gradle scan cannot read, and `apps/hohenheim` is not declared),
so there `order` alone ranks originals and nothing is reported as unreachable; that already turns framework
helpers away from app code (`IndexedScopes.intOf` is no longer a copy of zenit-auth's
`CapabilityMatrixTransport.integer`, hawkeye's `NumberFunctions.parseInt` no longer one of Hohenheim's
`RawValues.parsedInt`).

Precision, top 30 per channel read and judged (real = a helper, constant or call would remove it):

| Channel | Hohenheim b9b8f221 | zenit-cms |
| --- | --- | --- |
| holed | 2 borderline of 30 (a DI constructor, a migration `down`) | 9 classes; 1 borderline |
| idiom | 5 of 30 borderline (column-builder DSL chains) | 4-5 of 30 (a log call, a test table builder, migrations) |
| reimplements | 5-6 of 26 (panel declarations, parallel registrations; on the scratch root) | not run alone (no API side) |
| vocabulary | 0-2 of 30 | 12 classes, all real |

Whole javaweb workspace (`/home/skerit/projects/zenit-workspace`, 11 986 files, load average ~30 from other
agents): holed 16 s, idiom 24 s, vocabulary 16 s, reimplements 59 s warm (490 s the first time, 227 s of it
embedding 79 315 bodies with voyage-4-lite), all four together 95 s / 1.6 GB warm; `--kind exact,renamed` on
the same machine and tree took 179 s.

### Focus

`--focus` works for all four kinds: they keep no per-root index, so a focused run computes the whole
run's classes for them and keeps the ones with a member under the focus. The clone kinds keep their
indexed focus lane; a run asking for both does both.

## The three clone kinds

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
never declares a `renamed:` entry dead, and except in a run that saw only part of
the root (`--paths`, `--focus`): there an entry whose class lies elsewhere matched
nothing, which says nothing about whether it is stale.

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
what to do about them: that depends on dependency direction, visibility and
source sets. When the scanned root has a `.zemble/home.toml` (the same file
`zemble home` reads: module `order`, module globs, `[[forbidden]]` rules,
`depends_on` or discovered Gradle edges, `[source_sets]`), every class spanning
two or more declared modules carries one verdict, printed as a `home:` line
under its members and as a `home` object in the JSON.

### The verdicts

- **existing-reusable-api** -- a declared capability row names one copy, that
  copy is public, every other copy's source set may use its source set, and
  every other copy's module provably reaches its module. This is the ONLY
  verdict that says the other copies should call it:

  ```
  home: existing reusable API zenit: Texts.trimmedOrNull
        declared by CLAUDE.md: Blank-safe string trimming (row names Texts.trimmedOrNull)
        public static
        every copy's module reaches zenit (quirkyquarters: transitive)
        downstream copies should call or extend it
  ```

- **existing-implementation-not-api** -- the mechanism is declared and in the
  right place, but it cannot be called from where the copies are: it is
  private, package-private or protected, its source set is unreachable, or the
  workspace declares no dependency graph at all. Every failing check is listed:

  ```
  home: existing implementation zenit-ai: AiRecordSources.declare (not a reusable API)
        declared by CLAUDE.md: Record sources (row names AiRecordSources.declare)
        private
        thoth common cannot use zenit-ai common
        expose it or extract the generic mechanism; do not call it as is
  ```

- **candidate-home** -- a PLACE for a new mechanism, never a claim that a
  reusable one is already there. A row that names only the copy's TYPE lands
  here too, saying exactly what it proved:

  ```
  home: candidate home zenit-cms: ResponseCache.appendPart
        lexically related row: Response caching + sitemap (names the type, not this member)
        every copy's module reaches zenit-cms (quirkyquarters: direct)
        no declared member; review before consolidating
  ```

- **siblings-need-common-home** -- no member module may depend on any other, so
  none of them is the home; the suggested home is the module they all reach
  (`config.nearest_common_dependency`), or nothing when the workspace declares
  no dependencies:

  ```
  home: siblings zenit-widget, zenit-flow: renameType
        no dependency path either way
        shared mechanism belongs in plumage
  ```

- **forbidden-dep** -- a `[[forbidden]]` rule blocks one member module from
  depending on another; the rule and its `why` are quoted, plus `a shared home
  must sit deeper than <module>`. This is the case where a naive "extract a
  shared class" is architecturally wrong.
- **no-shared-ancestor** -- a member module is not declared in `home.toml` at
  all (sibling apps, a checkout the workspace does not describe): the
  architecture cannot judge code it does not know about, and the verdict names
  the modules it could not place.
- **review-required** -- the class is a LOGIC clone (or a holed, idiom or re-implementation lead). Structural similarity is
  not equivalence, so the answer is a lead and never an instruction: the
  best-evidenced copy (a copy a declared row names, else the most core one) is
  named as a *possible* existing mechanism.

  ```
  home: possible existing mechanism zenit-ai: McpServerConnection.findMeaningfulMessage (logic clone)
        logic clone: similar control flow and call set, not the same code
        semantic review required; structural similarity is not equivalence
  ```

### The decision order

One ordered decision function, first answer wins, every step failing closed:

1. A `[[forbidden]]` rule between ANY two member modules -> `forbidden-dep`. It
   outranks every declaration: when a member may not depend on the module that
   declares the mechanism, calling it is not the fix, whatever the table says.
2. A member module that `home.toml` does not declare -> `no-shared-ancestor`.
3. `kind` is a lead kind (`logic`, `holed`, `idiom`, `reimplements`: its `KindFacts.lead` is set) ->
   `review-required`, never anything stronger; the head names the lead (`(logic clone)`, `(holed clone)`).
   `vocabulary` classes hold the same value, so they take the ordinary steps.
4. Dependency topology, when the workspace has a dependency graph at all: the
   home candidates are the member modules every other member module can reach
   (`Reachability.DIRECT` or `TRANSITIVE`), most core first. When there is no
   such module -> `siblings-need-common-home`. When the workspace declares and
   builds NO dependency graph, the most core member module is used as a place,
   and the verdict is capped below `existing-reusable-api`: `order` ranks
   modules, it never granted anyone permission to depend on anyone.
5. Declared rows for that home module: only an exact `Type.member` row on a
   whole-body member counts as a declaration. A bare `Type` row is recorded as
   `declared-type` evidence and yields `candidate-home`; no row at all yields
   `candidate-home` without a symbol.
6. With a declaration, three checks, all of which must pass for
   `existing-reusable-api`: visibility, source set compatibility of every other
   copy's file against the declared copy's file, and reachability of the
   declared copy's module from every other member module. Any failure ->
   `existing-implementation-not-api`, listing all of them.

Visibility is two facts, both of which must be `public`: the member's own level
and its declaring type's, the latter already folded through every enclosing type
(a public method on a package-private class is not reusable, and the evidence
line names the type and its level). The language profile computes both from the
parse, so it knows the implicit rules: a Java interface or annotation member is
public without saying so unless it is an explicit Java 9 `private` one, a nested
type of an interface is implicitly public, and a Zig declaration is public when
it is `pub` (or `export`) and file-private otherwise, its container included. The
levels are one vocabulary, `zemble.dedup.languages.Visibility` (`public`,
`protected`, `package-private`, `private`, `unknown`), they ship on every JSON
member as `visibility` and `container_visibility`, and they are kept out of every
hash so a visibility edit can never move a clone key. A statement window is
`unknown`: nothing can call one. `unknown` is never reusable, which is how a new
language fails closed until its profile places its members.

### Evidence

Every verdict carries an `evidence` list of `{kind, text}` items, and the text
report prints one line per item between the head line and the action line. The
kinds are `declared-member`, `declared-type`, `visibility`, `source-set`,
`dependency` and `clone-kind`; declared-row evidence adds `capability`, `file`
and `line`, with the whole capability cell (the printed line carries the row's
TITLE, its capability up to the first parenthesis, truncated at 100 characters,
because a capability cell is prose and routinely runs for hundreds of
characters).

The JSON verdict object carries `verdict` and `kind` (the same value; `verdict`
is the name the first readers were written against), `modules`, `home`,
`detail`, `symbol`, `location` (`<file path>:<start line>` of the named copy),
`suggested_home`, `evidence` and `lines` (exactly what the text report prints).

### What is proven, and what is not

Proven: the declaration (a human wrote the row), the visibility (from the
parse), the source-set fold (from the path), and the dependency direction (from
`depends_on` or the Gradle build files). Not proven, and deliberately not
claimed: that the two copies have compatible SIGNATURES, that they are
semantically equivalent, or that the caller's context allows the call. The
visibility proof covers the member and its declaring types and nothing beyond
them: a Java `public` member of a public type may still be unreachable because
its module is not exported (JPMS) or its package is shaded, and neither is read.

`home_modules` is read the way the `home` tool reads it, so the home cell must
name its module IN BACKTICKS: a row whose home cell is prose declares no
module, matches no copy, and is silently no evidence.

No `home.toml` means no verdicts and no noise; a malformed one is reported as a
note and skipped, because this is a report, never a gate.

### What counts as a declaration

Declared capability-table rows (the `[[tables]]` of `home.toml`) and nothing
else. A copy is promoted when it is a whole body -- not a statement window --
living in the home module -- and not a synthetic member such as
`Type.<initializer>`, which nothing can call -- and a row whose `home` cell
backticks that module names that body exactly. Two name shapes are a
declaration: the exact `Type.member`, and a `Type.member` row whose qualified
tail the unit carries (`Outer.Texts.trimmedOrNull` is named by
`Texts.trimmedOrNull`). A bare `Type` row is NOT a declaration of its members:
it says the row is about that class, which is a lexical relation and not a
statement that this member is the capability's API. A row that names a
different type's member of the same name (`Other.trimmedOrNull` against
`Texts.trimmedOrNull`) does not match at all: symbol matching fails closed,
because a wrong "call this" sends a reader to code that does something else.

Callers, the symbol graph, embeddings and search are deliberately not consulted.
`dupes` is a cache-free scan that must run on any checkout without an index, and
usage is not intent: a helper twenty callers reach for is not thereby the
declared home of anything, while a mechanism a human wrote into the table is one
however few callers it has. Visibility and modifiers come from the same
tree-sitter parse the clone detection already did, so reading them costs
nothing and reaches no index. Classifying a class therefore never triggers
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
| `--kind` | `exact,renamed` | `exact`, `renamed`, `logic`, `holed`, `idiom`, `reimplements`, `vocabulary`, `all`, or a comma-separated list |
| `--limit` | 25 | Clone classes printed per section |
| `--min-files` | 1 | Only report classes spanning at least N files |
| `--min-tokens` | 30 | Smallest unit that may form a class |
| `--min-statements` | 6 | Smallest statement window compared inside a body |
| `--no-windows` | off | Compare whole bodies only |
| `--logic-threshold` | 0.92 | Cosine a logic candidate needs |
| `--logic-top-k` | 10 | Embedding neighbours considered per body |
| `--paths` | whole root | Restrict the scan to these paths, resolved against the scan root (absolute paths taken as given) |
| `--focus` | none | Report only the classes with a member under these paths, compared against everything scanned (see Focused runs) |
| `--exclude` | none | Gitignore-style pattern dropped before parsing (repeatable) |
| `--lane` | `all` | Report one lane: `production`, `mixed`, `test` |
| `--brief` | off | Header plus one line per class |
| `--show-suppressed` | off | Also print the classes the ignore file took out |
| `--save-baseline` | off | Write this run's class keys to a file |
| `--baseline` | off | Report resolved / remaining / new against a saved baseline |
| `--embedder` | env default | Embedder spec used by `--kind logic` |
| `--jobs` | up to 8 | Extraction worker processes |
| `--json` | off | Machine-readable output |
| `-y`/`--yes` | off | Embed whatever `--kind logic` costs, past the spending budget |

## Focused runs

`--focus PATH...` (MCP: `focus`) answers "does THIS code duplicate anything?" at a
cost that follows the focus, not the workspace. It reports exactly the classes a
whole run reports that have a member under a focus path -- same kinds, floors,
keys, members and reasons -- and leaves every other class out. The JSON gains
`"focus": [...]` (empty on a whole run); nothing else in the shape changes.
`--baseline`/`--save-baseline` are refused with it: a baseline would read every
class it leaves out as resolved.

How it stays cheap:

- **Unit index.** Every scanned file's row -- both stream hashes of every unit,
  and its whole-body units -- is kept in `dupes-units.sqlite` in the root's cache
  folder, keyed by path, content digest and an extraction signature (options,
  the source of the dedup and language modules, grammar package versions). A
  warm run parses nothing but the files holding a hash that a focus unit holds
  and some other unit holds too; a clean focus parses nothing at all. Rows of a
  file a whole-root walk no longer finds go, and so do extractions nobody ran
  for a week. A `--kind logic` whole run refreshes the index too.
- **Vector mirror.** Logic mode needs every body's vector; an embedding server
  keeps no local copy, and fetching 57k vectors back from one took 22 s. They are
  mirrored in `dupes-vectors/` beside the index (the embedding cache's own store)
  and swept once it holds more than twice what a run reads.
- **Walk, not all-pairs.** A pair is a logic candidate when either body has the
  other among its `--logic-top-k` neighbours, and a class is the union of its
  accepted pairs. The walk starts at the focus bodies and, for every body it
  reaches, takes its own neighbour row (forward) and every body that holds it in
  ITS row (reverse: a similarity scan, then that body's own row), until no new
  body joins. Both run modes read their rows from one cosine space, so they can
  only differ where two matrix products of one row round apart.

Measured on the javaweb workspace (11 982 files, 1 757 515 units, 56 874 bodies,
voyage-4-lite through an embedding server), `--json --kind exact,renamed,logic
--logic-threshold 0.6 --lane all`, focus = 23 Java files across zenit, hawkeye,
protoblast and plumage:

| Run | Wall clock | Peak RSS |
| --- | --- | --- |
| whole workspace, before this mode | 326 s | 7.1 GB |
| focused, cold (index and mirror empty) | 134-175 s | 0.9 GB |
| focused, warm | 8 s | 0.8 GB |

## The MCP tool

`dupes(repo, kind, paths, focus, exclude, lane, limit, min_files, format, brief,
baseline, save_baseline)`.

`kind` takes one channel name (`exact`, `renamed`, `logic`, `holed`, `idiom`, `reimplements`,
`vocabulary`), `all`, or `architectural`; the parameters are the CLI's.

`format="text"` (the default) returns the report exactly as the CLI prints it,
as a plain string; `brief=true` trims it to the class lines. `format="json"`
returns the structured object itself -- FastMCP encodes it once, so a client
never has to parse JSON out of a JSON string. `baseline`/`save_baseline` are
booleans against the fixed `<repo>/.zemble/dupes.baseline.json` (see
Baselines). Neither surface needs a daemon or a search index.

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

## Adding a language

A language with a grammar spec gets its profile for free: `zemble.dedup.languages.generic`
derives one from the spec (its callable rules become the member kinds, its type,
namespace and extension rules the containers, its call rules the called names, its
binding paths the declared names), and the token-level vocabulary is a shared superset
(`CONTROL_KEYWORDS`, `LITERAL_KINDS`, `DECLARING_KINDS`) narrowed to what the grammar
actually has. Only a spec with a callable rule that has a body is comparable; a
stylesheet or a Dockerfile is not. So adding a language is adding a spec to
`src/zemble/languages/catalog.py` and a fixture to `_VISIBILITY_FIXTURES` in
`tests/test_dedup.py`.

A hand-written profile (Java, Zig) is one module under `src/zemble/dedup/languages/`
exporting a `LanguageProfile`, plus one entry in that package's `_HAND_WRITTEN` tuple.
`units.py` holds no node type of any language and must stay that way. The fields:

| Field | What it answers |
| --- | --- |
| `name`, `extensions` | The grammar name and the file extensions it claims. |
| `parser` | Returns the tree-sitter parser, or None when the platform lacks it. |
| `member_kinds` | Node kind -> unit kind, for every declaration with a comparable body. |
| `member_body`, `member_name` | Where that declaration's body and name segment are. |
| `container` | The nested namespace a node opens (`class Foo`, `const Foo = struct`), and how far it can be reached. |
| `flatten_kinds` | Wrapper nodes whose children are the real members. |
| `block_kinds` | Nodes whose statement children form a statement window. |
| `control_keywords` | Leaf kinds whose order is the control-flow skeleton. |
| `literal_kinds` | Nodes that are a literal; the walk never descends into one. |
| `declared_name_fields` | Kinds whose `name` field is an identifier the unit declares. |
| `declared_names_extra` | Declared names no `name` field can express (patterns, captures). |
| `call_names` | The names one node calls. |
| `member_separators` | Leaf texts after which an identifier is a member and is never renamed. |
| `modifiers` | The declaration's modifiers; reported, never hashed. |
| `visibility` | How far one member can be called from, its declaring body's kind included. |
| `hook_node_kinds` | The node kinds only the hooks above name, for the drift test. |
| `classify` | Optional: decides a node's unit kind when its node kind alone cannot (an Elixir `call` that is a `def`). |
| `descend` | Optional: whether a node that is neither member nor container is looked through for members. |

`tests/test_dedup_languages.py` fails the build when a profile names a node kind
its grammar does not have, or claims an extension zemble does not index as code.
Do not guess node kinds: parse a fixture and print the tree.

The unit kinds a profile declares are the home of the "is this a whole body"
vocabulary (`zemble.dedup.model.BODY_KINDS` is derived from them), so a new
language widens logic mode by itself. A file whose extension no profile claims is
never walked at all.

## Limits

- The sub-body channels are leads, not proofs: `holed`, `idiom` and `reimplements` classes carry
  `review-required` home verdicts. Their floors were tuned on Hohenheim and zenit-cms (see Measured).
- `idiom` and `vocabulary` read Java only (`ShapeHooks`); another language reports none and the run says
  so. `holed` folds constants into `<const>` only where the profile names a constant convention.
- `idiom` folds a prefix-extension into its prefix only when both are cut at the same sites (see idiom);
  a nested variant at a quarter or more fewer sites is still its own class. Fluent builder DSLs with four
  or more calls still pass the chain evidence.
- `holed` drops call-free and single-call bodies under 30 tokens and small twins inside one file, so a
  copied `return a != null ? a : b;` coalesce or a family of one-file conveniences is not reported.
- `reimplements`' intent lane cannot link two members whose names share no word: `Row.afterWrite` is the
  original of the `effective*` helpers, but each of those pairs scores 0.44-0.62 (name 0, see Measured), so
  they still point at Hohenheim's public `SiteDomainModel.effective`, and `TenantWrites.effectiveSiteType`
  and `DnsZoneCascades.effectiveZoneId` (a fixed field, signature 0.5) stay below the bar. `durationLabel`
  is not a drop-in copy of `RelativeTime.duration` (score 0.69, see Measured). Instance methods that read
  their own state are never intent copies. The architecture only knows the modules `home.toml` declares;
  a workspace whose build files the Gradle scan cannot read (a version catalog served by a settings plugin)
  has no dependency graph, so every reach there is unknown and `order` alone ranks. A regex literal is not
  linked to a predicate method over the same characters. A facade over a type outside the scan (a JDK type)
  is not reported, so scan the root that holds both sides.
- `vocabulary` reads string constants and literals; numeric vocabularies, values built by concatenation
  and regexes passed to a non-JDK API are not seen.

- A derived profile is only as precise as its spec: a grammar whose call node the spec
  does not describe compares bodies with an empty call list, so its logic clones lean on
  the skeleton and literals alone. `zemble dupes` names the extensions it walked.
- Statement windows dominate the cost on repositories with long function bodies.
  sketerm (453 Zig files) is 2.8 s with `--no-windows` and 83 s with them, for
  619 303 units against 13 284 bodies; javaweb's 6309 Java files are 46-53 s.
- No index and no incremental state: every run re-parses the tree. That is the
  45 s above, and it is why there is no cache to invalidate.
- Two call-free bodies trivially satisfy the call-set check (an empty set
  overlaps an empty set completely); their reason says
  `neither body calls anything` so a reader can see it happened.
- A window class and the body class containing it can both be reported when the
  window spans more files than the body does; that is deliberate (the window is
  the wider finding) but it does read as two entries for one family.
- `logic` embeds every body in the workspace on every run. It is 16 s with the
  local Potion model, but a paid embedder makes it a paid operation: it is one of
  the two seams the bill guard stands at, so on a remote embedder the run can be
  refused in money before a single body is sent, and `--yes` (or
  `ZEMBLE_EMBED_CONFIRM=1`) is what buys it anyway. The refusal is the answer on
  both surfaces - the CLI exits with it, the MCP tool returns it as its text.
- A file that fails to parse is counted in `failed_files` and named in a note,
  never silently dropped: 15 files on javaweb, `gradle-wrapper.jar` among them,
  reach the scan through a negated gitignore rule and show up there.
