# zemble health

`zemble health [path]` (MCP `health`) is a ranked, reference-free report of code-shape
findings plus a score. It reads the symbol graph (`docs/graph.md`), for layering the
workspace's own `.zemble/home.toml` (`docs/home.md`), and for the duplication tier the
`zemble dupes` channels (`docs/dedup.md`) through their per-root unit index; it never builds a
search index and never embeds anything. It is a REPORT, never a gate: it exits 0 whenever it
could answer, however many findings it has.

```
zemble health ./my-project                       # top 25 findings, the score and its axes
zemble health ./my-project --limit 100 --json    # the same payload as JSON
zemble health ./my-project --signals long_callable,complex_callable
zemble health ./my-project --paths src/core src/api
zemble health ./my-project/src/core              # a sub-directory: its own paths, its findings only
zemble health ./my-project --copying             # whole tree with the duplication tier (slow; see below)
zemble health ./my-project --since main          # the branch delta: findings touching changed lines
zemble health ./my-project --save-baseline base.json
zemble health ./my-project --since main --baseline base.json   # delta findings marked new or remaining
```

Every score line says `not a target: address or explain the findings`. The number is a
summary of the findings, not something to optimise: a finding is addressed by changing the
code or by explaining why the shape is deliberate.

## What it measures

Each signal is one member of `Signal` in `src/zemble/health/model.py`; its axis, tier, unit,
threshold, comparison, resolution floor, source and advice are facts on the member
(`SignalFacts`), and `src/zemble/health/signals.py` refuses to import without a producer for
every member. Test code is excluded from every signal: a unit in a test source set (the
graph's own `is_test`) is never measured.

| Signal | Axis | Tier | Finding when | Threshold source |
| --- | --- | --- | --- | --- |
| `long_callable` | size | syntactic | own body lines > 60 | PMD ExcessiveMethodLength reports at 100 by default; 60 flags the tail before it gets there |
| `many_parameters` | size | syntactic | declared parameters > 5 | PMD ExcessiveParameterList reports at 10 by default; 5 is the common review convention |
| `deep_nesting` | size | syntactic | control-flow nesting > 3 | Sonar S134 default maximum 3 |
| `complex_callable` | size | syntactic | cognitive complexity > 15 | Sonar S3776 default threshold 15 |
| `large_type` | size | syntactic | type declaration span > 500 lines | PMD ExcessiveClassLength reports at 1000 by default; 500 flags the tail before it gets there |
| `large_file` | size | syntactic | file lines > 1000 | Sonar S104 default maximum 1000 |
| `many_types_per_file` | size | syntactic | top-level types in one file > 5 | declared here; no common tool default |
| `single_implementation_interface` | machinery | graph | an interface, trait or protocol with exactly 1 production implementor | speculative generality (Fowler, Refactoring) |
| `single_subclass_abstract` | machinery | graph | an `abstract` type with exactly 1 production subtype | speculative generality (Fowler, Refactoring) |
| `lone_public_callable` | machinery | graph | a public method or function with at most 1 production caller | declared here |
| `unreferenced_symbol` | machinery | graph | a private or package-private method, function or type nothing references | declared here |
| `forbidden_crossing` | layering | graph | a file with edges into a module its own module may not depend on | the workspace's `[[forbidden]]` rules |
| `clone_mass` | copying | duplication | file lines inside exact or renamed clone classes > 50 per mille (5%) | declared here: PMD CPD reports every copy past its token minimum but has no per-file share default |
| `cloned_body` | copying | duplication | a callable body in an exact or renamed clone class with a copy in another file | membership (Fowler, Refactoring: duplicated code) |
| `pass_through` | machinery | duplication | a public callable whose whole body hands its parameters, in order, to one other callable | existence: middle man (Fowler, Refactoring) |
| `vocabulary_drift` | vocabulary | duplication | a value or value set declared in more than one place (one per `vocabulary` class) | existence: a `zemble dupes --kind vocabulary` class |

The tool defaults quoted above are as the step's design states them; they were not re-verified
against the tools' current releases.

The measurements come from the graph (`docs/graph.md`): `CallableMetrics` per callable
(lines, cognitive complexity, nesting), `FileMetrics` per file, symbol spans and parameter
lists. `long_callable` uses a callable's OWN lines: the span of every named callable declared
directly inside it (a decorated tool inside its `register_*`, a closure) is subtracted, since
that one is measured on its own. Cognitive complexity follows Sonar and keeps nested
lambdas and functions in the enclosing callable.

CONSISTENCY and HISTORY are later axes: they join `Axis` when a signal measures them, so no
axis exists that could only ever read "not measured".

## The duplication tier

The four `duplication` signals detect nothing themselves: `zemble.health.copying` runs
`find_duplication` for the `exact`, `renamed` and `vocabulary` kinds only (never `logic`, which
buys embeddings), production lane only, `.zemble/dupes.ignore` honoured, and turns its clone
classes and units into samples:

- `clone_mass`: per measured production file of a language dedup parses, the union of the line
  spans of every exact or renamed class member in it (bodies and statement windows, intra-file
  copies included), in thousandths of the file's lines.
- `cloned_body`: per production body of clone size (`--min-tokens`, 30), the number of OTHER
  files holding a member of an exact or renamed class it belongs to; the reason names them.
- `pass_through`: per public production body with at least one parameter, 1 when dedup's
  `forwards_to` is set (the whole body is `return Type.member(p1, ..., pn);` of its own
  parameters in order). Not measured: a body with no parameter (it passes nothing through; it
  names a value, `return Set.of();`), anything part of a contract (`@Override` or a
  framework annotation, a member of an interface, a callable that overrides or that something
  overrides), a constructor handing over to another (`this(...)`), and every language whose dedup
  profile has no shape hooks (today Java is the only one that reads forwarding).
- `vocabulary_drift`: one sample per vocabulary class, valued by its places and anchored at its
  first member in scope, the other places carried as `elsewhere`; every other declaration site
  (a constant, value set, switch or regex no class holds) is a clean sample of value 1.

Bodies are matched to graph callables by file, simple name and the widest line overlap, so their
keys are the same path-free identities as every other callable signal; a body no callable
matches is not measured. The run is rooted at the graph's root, so a sub-directory's copies are
compared against the whole tree.

### The cost rule

Over a whole tree the tier costs a parse of every file. Measured on the javaweb workspace
(7144 files, this machine): the dedup run alone 64-88 s and 2.4 GB peak; the whole health run
with the tier 103-114 s and 2.7 GB, against 12.3 s and 397 MB without it. That is over the
60 s budget, so:

- a whole-tree run measures the tier only with `--copying` (MCP `copying=true`) and otherwise
  says so in a note;
- a bounded run (`--since`, `--paths`, or a sub-directory) measures it by default, as a
  `--focus` dedup run: clone classes come from the per-root unit index (only files sharing a hash
  with the focus are parsed again), shaped bodies are parsed for the focus files only, and the
  vocabulary channel still reads every file's sites, since a vocabulary is compared over the tree;
- `--no-copying` (MCP `copying=false`) turns it off anywhere.

## The resolution rule

A graph signal names a floor, the weakest `Resolution` an edge may have to count:
`unique_name` for the machinery signals, `exact` for `forbidden_crossing`. AMBIGUOUS and
UNRESOLVED edges never support a finding. They also withhold one:

- A presence finding (a crossing, a count of exactly one) counts only edges at the floor.
- An absence finding (nothing calls it, nothing references it, exactly one implementor) is
  only made when no weaker edge might mean the symbol: an inbound edge resolved below the
  floor, or ANY destination-less edge in the whole graph that wrote its simple name, takes the
  unit out of the measured population entirely. It is neither a finding nor a clean unit.

Each graph finding reports the weakest edge supporting it (`exact` for a pure absence).

## Exclusions (what reaches code without a written call)

`lone_public_callable` and `unreferenced_symbol` skip a callable that:

- is a constructor, or is named `main`;
- carries any annotation or decorator (frameworks register code through them: `@Test`,
  `@Inject`, `@server.tool`, `@property`);
- overrides something: an OVERRIDES edge at any resolution; a supertype in its type's chain
  declaring a callable of the same name (Python writes no OVERRIDES edges); any supertype the
  graph cannot see into (unresolved, ambiguous, outside a sub-directory view); or an owner the
  graph holds no symbol for (the methods of a Rust `impl Trait for ForeignType`, and Go methods,
  whose container is the file);
- matches its language's `runtime_hook_patterns` (`LanguageSpec`): Python `__*__`, PHP `__*`,
  Go `init`;
- belongs to an anonymous class, or is declared inside another callable (a nested helper is
  reached through its scope).

`unreferenced_symbol` also skips a type that is annotated or declares an annotated member.

A remaining candidate with no inbound edge is finally checked against the source text,
because the graph has no edge for a callable used as a VALUE (a dispatch table, `key=f`,
`Foo::bar`): its simple name must not be written outside its own declaration in its own file
(private), its directory (package-private) or any file of the graph (anything wider; test
files count for `unreferenced_symbol` only). A mention in a comment or a string also withholds
the finding.

Visibility comes from the written modifiers first (`public`, `pub`, `export`, ...); a member of
an interface or trait without one is as visible as the interface; Java defaults to
package-private; other languages follow their `LanguageSpec` (`underscore_is_private`,
`capitalized_is_public`, `default_visibility`). A level no rule can place is UNKNOWN and is
never reported.

## Percentiles and ranking

A finding's percentile is the share of every measured value of the same signal and the same
language in the scope that is no more suspect than it, 0-100 to one decimal. "More suspect"
follows the signal's comparison: larger for `above`, smaller for `at_most`, closer to the
threshold for `equals`. Findings are ranked by percentile, then by how far past the threshold,
then by signal, file and line: one list, no grouping, capped by `limit` (default 25) with
`total` and `truncated`.

Each finding has a stable `key`: `signal:language:identity`, where identity is the symbol id
without its file (`com.example.Util.tangled(int,int)`), for a file-level unit the path, and for
a vocabulary class its content key (`vocabulary:<digest>`). Never a line number, so a baseline
diff survives edits around a finding.

## Baselines

The same document and vocabulary as `zemble dupes` (`zemble.baseline` writes, reads and refuses
the versioned JSON for both): `--save-baseline FILE` writes every finding of the scope (all of
them, not only the listed ones), `--baseline FILE` diffs the run against one, read before a
save in the same run writes. The MCP tool keeps it at `<repo>/.zemble/health.baseline.json`
(`baseline`, `save_baseline`), and a missing or unreadable one there is a note, not a failure.

```
Baseline: 24 finding(s); 1 resolved, 23 remaining, 1 new

== RESOLVED (in the baseline, gone now) ==
#1  unreferenced_symbol  was: src/.../Util.java:26-28  Util.deadHelper  key: ...
== REMAINING (in the baseline, still here) ==
== NEW ==
```

Health keys are identities, not content, so an edit never re-keys a finding: unlike dupes there
is no CHANGED bucket. An entry is called resolved only when the run measured its signal over its
file (`--signals`, `--paths` and a focused duplication tier narrow that). JSON adds a `baseline`
object (counts, resolved entries, remaining and new keys) and a `baseline` status on each listed
finding.

## The branch delta: `--since <ref>`

`zemble health --since main` lists only the findings whose span intersects a line changed or
added by `git diff --unified=0 main` of the working tree (staged and unstaged edits) or lying in
an untracked file git would add (`ls-files --others --exclude-standard`; a nested checkout is a
gitlink, never its files). A hunk that only deletes lines marks the line it follows. File-level
findings span their whole file, so any change in the file lists them; a vocabulary finding is
listed when any of its places is touched. Git is read by `zemble.health.git` with plain
subprocess calls; a ref that names no commit, or a tree outside a checkout, is refused.

The score section is the whole scope's, and the delta's own count is printed beside it
(`Since main: 6 finding(s) touch 2 changed file(s); the score is the whole scope's (23
finding(s))`). The composite is never recomputed over the delta: a score over a subset says
nothing about the code. For the same reason the duplication tier, which a `--since` run measures
over the changed files only, is listed but left out of the score (`unscored`, and "not scored"
on its axis lines). With `--baseline`, each delta finding is marked new or remaining.
`--since` takes no `--save-baseline`: a delta is not a scope.

## The score

```
mass(unit)        = value for a lines signal (long_callable, large_type, large_file), else 1
fraction(signal)  = sum(mass of units past the threshold) / sum(mass of every measured unit)
axis score        = 100 * (1 - mean(fraction of the axis's scored signals))
composite         = sum(weight(axis) * axis score) / sum(weight(axis))   over measured axes
```

Weights (`AXIS_WEIGHTS` in `src/zemble/health/score.py`, the one table): size 1, machinery 1,
layering 2, copying 1, vocabulary 1 (a forbidden edge breaks a rule the workspace wrote down;
the rest are general conventions). An axis with no measured unit (no `home.toml` with
`[[forbidden]]` rules, the duplication tier not run, or `--signals` leaving it out) is "not
measured" and takes no part in the composite; it is never 100. The computation is
deterministic: the same tree gives the same report (no wall time is printed in it).

## Surfaces

- CLI `zemble health [path] [--limit N] [--signals a,b] [--paths P ...] [--since REF]
  [--copying | --no-copying] [--baseline FILE] [--save-baseline FILE] [--json] [--no-daemon]`,
  through the warm daemon first, in process otherwise (graph built or refreshed as `outline`
  does). A path with no graph gets one built.
- MCP `health` (`repo`, `limit`, `signals`, `paths`, `since`, `copying`, `baseline`,
  `save_baseline`), returning the markdown; under `zemble nib-provider` it answers with one text
  part. Every surface builds one `HealthRequest` (`zemble.health.answers`), whose field names
  are the daemon's argument names.
- Daemon command `health`, reading its warm graph; no search index is loaded for it, the
  duplication tier included (it reads the unit index).
- A sub-directory of a tree with a graph is answered from that graph: paths relative to the
  sub-directory, only its units measured, callers anywhere counted. The config is the nearest
  ancestor's `home.toml`, as `home` reads it.

## Measured (2026-10-17)

In process over an existing graph (graph build excluded; `ZEMBLE_DAEMON=0`), this machine:

| Repo | Language | Wall | Score | size | machinery | layering |
| --- | --- | --- | --- | --- | --- | --- |
| zemble (this repo) | Python, Java | 0.40 s | 84.5 | 91.9 | 77.1 | not measured |
| javaweb workspace | Java, JS, hwk | 11.3 s (384 MB peak) | 91.7 | 89.0 | 77.7 | 100.0 |
| gin | Go | 0.14 s | 84.8 | 92.9 | 76.6 | not measured |
| express | JavaScript | 0.05 s | 57.1 | 96.0 | 18.2 | not measured |
| monolog | PHP | 0.10 s | 89.2 | 97.5 | 80.9 | not measured |
| serde | Rust | 0.17 s | 81.6 | 78.3 | 84.9 | not measured |
| gson | Java | 0.25 s | 80.2 | 88.4 | 72.0 | not measured |

Findings per signal:

| Repo | long | params | nesting | complex | large type | large file | types/file | single impl | single sub | lone public | unreferenced | forbidden |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zemble | 22 | 83 | 20 | 63 | 4 | 6 | 15 | 0 | 0 | 247 | 5 | 0 |
| javaweb | 674 | 669 | 490 | 1058 | 243 | 87 | 0 | 135 | 12 | 3025 | 11 | 0 |
| gin | 9 | 0 | 7 | 12 | 0 | 1 | 3 | 0 | 0 | 40 | 0 | 0 |
| express | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 9 | 0 | 0 |
| monolog | 5 | 18 | 8 | 12 | 1 | 0 | 0 | 0 | 0 | 29 | 0 | 0 |
| serde | 33 | 5 | 17 | 18 | 1 | 9 | 9 | 1 | 0 | 14 | 0 | 0 |
| gson | 17 | 3 | 14 | 34 | 8 | 4 | 0 | 6 | 2 | 51 | 0 | 0 |

javaweb's 11 s is almost all SQL: the inbound tallies (2.8 s), the file dependencies (2.0 s),
loading 122k symbols (1.1 s) and the measurements (1.1 s); the source-text check reads every
file once (57 MB) for the public candidates.

Top 5 of javaweb: `TemplateParser.java` 6930 lines (large_file) and its `TemplateParser` type
6884 lines (large_type); `SupplyWriter.java` 5026 lines and `SupplyWriter` 4933 lines;
`TagGenerator.java` 4378 lines.

Top 10 of zemble, read by hand:

| # | Finding | Verdict |
| --- | --- | --- |
| 1 | large_file `languages/catalog.py` 2914 lines | real; a declarative table of ~60 specs, so low value, but per-family files would read better |
| 2 | large_type `_JavaExtractor` 768 lines | real |
| 3 | long_callable `cli._build_parser` 131 lines | real, flat argparse wiring |
| 4 | long_callable `preflight.embed_status` 125 own lines | real |
| 5 | complex_callable `SqliteGraphProvider._walk` 46 | real |
| 6 | complex_callable `create.write_index` 31 | real |
| 7 | many_types_per_file `graph/facts.py` 21 types | real |
| 8 | complex_callable `Json.stringField` 16 (javac-facts) | real but marginal; p100 because Java is a population of one small module here |
| 9 | many_parameters `units._make_unit` 13 | real |
| 10 | many_parameters `ZembleIndex.__init__` 13 | real |

Before the rules above, a hand review (this top 10, plus samples of every machinery signal on
zemble, javaweb, serde and gin) found systematic false positives, each fixed by a general rule:
`long_callable` counted nested tool functions twice (`create_server`, `register_dupes_tool`:
own lines now); `unreferenced_symbol` flagged Python callables used as values (dispatch tables,
`key=`; 104 -> 5 findings on zemble: the source-text check); the machinery signals flagged
methods implementing an external or unseen contract (Python base classes, Rust trait impls on
foreign types, Rust trait members: the inheritance, owner and interface-member rules; serde
unreferenced 22 -> 0); Go `init` functions (`runtime_hook_patterns`); and JUnit classes in a
source set the graph does not call a test set (javaweb `artifactContractTest`: the
annotated-member rule for types).

### The duplication tier (2026-10-17)

In process, warm unit index, this machine (other work was running on it; walls vary +-15%):

| Run | Wall | Peak | Score | copying | vocabulary |
| --- | --- | --- | --- | --- | --- |
| javaweb whole tree, no tier | 12.3 s | 397 MB | 91.7 | not measured | not measured |
| javaweb whole tree, `--copying` | 103-114 s | 2.7 GB | 94.1 | 97.6 | 95.9 |
| javaweb `--paths zenit-cms` (349 files; tier on by default) | 50-57 s (135 s with a cold unit index) | 674 MB | 93.5 | 96.7 | 92.0 |
| javaweb `zenit/ --since HEAD~5` (130 changed files; CLI, graph refresh included) | 47.9 s (12.9 s with `--no-copying`) | | 92.5 | not scored | not scored |
| zemble whole tree, no tier | 0.44 s | 76 MB | 84.4 | not measured | not measured |
| zemble whole tree, `--copying` | 10.3-14.7 s | 222 MB | 91.9 | 98.7 | 100.0 |

Of a focused run's wall, the vocabulary channel's read of every file's sites is the floor
(11-14 s on javaweb); the clone kinds from the unit index take 1.4 s for one focus file and
17 s for 349. Parsing shaped bodies for the focus files only (rather than every file) took the
`zenit-cms` dedup run from 47 s to 34 s.

javaweb whole-tree findings: clone_mass 138 (of 4609 files), cloned_body 327 (of 17354
bodies), pass_through 39 (of 7525 public bodies with parameters), vocabulary_drift 103 (of
2528 classes and declaration sites). zemble: clone_mass 5, cloned_body 2, pass_through 0 (no
Java body qualifies), vocabulary_drift 0 (Java only, 5 declaration sites).

Read by hand: the top 10 duplication-tier findings of javaweb are all real (cross-project copies
`ServerCollectorOperations`, `HostBudgets`, `generate-place-timezones.py`; the `previousTotal`
body in 5 cruiser adapters; `quoteJs`/`quote` copied between cruiser and duiventil; the
`started_at` column written as a literal 15 times beside its constant). The one weak lead is
`CaptionStatus`: dedup chains 22 value sets into one family through pairwise overlaps while no
value is shared by all of them. All 7 of zemble's are real: `_evaluate_repo` and the `main`
windows copied between the `benchmarks/baselines/` scripts.

The first hand review found one systematic false positive, fixed by a general rule:
`pass_through` flagged 47 zero-parameter bodies (`return Set.of();`, `Map.of()`: hooks and
defaults that name a value) and 11 interface default methods and overridden or overriding
bodies (a contract role, not a middle man): 97 -> 39 on javaweb. The 39 left are named wrappers
over one static API (`AuthCookieSupport.issueSessionCookie` -> `SessionCookies.issue`,
`TypeResolver.getBoxedType` -> `BuiltInTypes.getBoxedType`).

## Known limits

- `lone_public_callable` counts one caller as a finding, as designed: in a Python codebase most
  module seams have one consumer (247 of 547 measured on zemble), so the machinery axis reads
  low there without any one finding being wrong. Its findings rank by percentile, so the
  never-called ones lead.
- A library's extension points (gson's `FieldNamingStrategy`, `ToNumberStrategy`) are
  single-implementation interfaces by design: they are implemented by the library's users.
- Dynamic dispatch by a computed name (`getattr(self, f"_keyword_{name}")` in `hwk.py`) is
  invisible to both the graph and the source-text check: those 4 `_keyword_*` methods stay
  reported on zemble.
- Python `Protocol`/`ABC`/`Enum` subclasses and every class with a supertype outside the graph
  have their methods excluded from the absence signals (fail closed), as do Go methods (any
  may satisfy an interface implicitly); abstract Python classes are not `abstract` types to
  `single_subclass_abstract`, which reads the `abstract` modifier.
- A Java functional interface implemented by lambdas has no IMPLEMENTS edge from them: it can
  read as a single-implementation interface.
- A name written anywhere withholds an absence finding, including in an unrelated identifier of
  the same spelling, a comment or an import: common names are under-reported, never over-reported.
- The file walker indexes a nested git checkout (a `.claude/worktrees/agent-*` worktree inside
  the tree) like any other directory, so both the graph and the duplication tier see a second
  copy of every file: whole-tree counts double and every body reads as cloned. `--since` never
  lists a nested checkout's files (git would add it as a gitlink). Not fixed here: the walker is
  indexing code.
- `pass_through` reads forwarding in Java only (the one dedup profile with shape hooks), and
  only bodies of at least 8 tokens: a void forwarder `Target.call(a);` is 7 and is not measured.
  `vocabulary_drift` declaration sites are Java only for the same reason.
- A `--since` run's duplication tier is not scored, and its vocabulary channel still reads every
  file: on javaweb it adds ~35 s to a delta run.
- A small language population (zemble's one Java module) makes a marginal finding p100.
