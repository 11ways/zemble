# zemble health

`zemble health [path]` (MCP `health`) is a ranked, reference-free report of code-shape
findings plus a score. It reads the symbol graph (`docs/graph.md`), for layering the
workspace's own `.zemble/home.toml` (`docs/home.md`), for the duplication tier the
`zemble dupes` channels (`docs/dedup.md`) through their per-root unit index, for the
consistency axis one plain read of each measured file's text, and for the history axis one
`git log` per checkout; it never builds a search index and never embeds anything. It is a
REPORT, never a gate: it exits 0 whenever it could answer, however many findings it has.

```
zemble health ./my-project                       # top 25 findings, the score and its axes
zemble health ./my-project --limit 100 --json    # the same payload as JSON
zemble health ./my-project --signals long_callable,complex_callable
zemble health ./my-project --paths src/core src/api
zemble health ./my-project/src/core              # a sub-directory: its own paths, its findings only
zemble health ./my-project --copying             # whole tree with the duplication tier (slow; see below)
zemble health ./my-project --since main          # the branch delta: findings touching changed lines
zemble health ./my-project --history-window 90   # the history axis over the last 90 days
zemble health ./my-project --no-history         # leave git out
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
| `style_outlier_repo` | consistency | population | a file's style vector lies past the cut from its language's files (value: distance in hundredths of the cut, > 100) | declared here: robust distance past the chi-square 0.99 quantile of the kept dimensions (see "The consistency axis") |
| `style_outlier_directory` | consistency | population | the same against its directory's files, where those are not its whole language | as above |
| `hotspot` | history | history | a file past `complex_callable` or `large_file` with at least 5 commits in the window, whose churn ranks in its checkout's top decile (value: thousandths of the checkout's changed files that changed strictly less often, > 900) | hotspots, change frequency x complexity (Tornhill, Your Code as a Crime Scene; CodeScene); the 5-commit floor is code-maat's `min-revs` default |
| `boundary_co_change` | history | history | two files in different modules whose commit sets overlap with a Jaccard > 300 per mille over at least 5 shared commits (one finding per pair) | temporal coupling (Tornhill); code-maat's `min-coupling` 30% and `min-shared-revs` 5 defaults, changesets over 30 files left out (`max-changeset-size`) |

The tool defaults quoted above are as the step's design states them; they were not re-verified
against the tools' current releases.

The measurements come from the graph (`docs/graph.md`): `CallableMetrics` per callable
(lines, cognitive complexity, nesting), `FileMetrics` per file, symbol spans and parameter
lists. `long_callable` uses a callable's OWN lines: the span of every named callable declared
directly inside it (a decorated tool inside its `register_*`, a closure) is subtracted, since
that one is measured on its own. Cognitive complexity follows Sonar and keeps nested
lambdas and functions in the enclosing callable.

## The history axis

`zemble.health.history` reads one window of git history (`--history-window DAYS`, default 365)
through `zemble.health.git`, the one module that runs git, in a subprocess only:

```
git -c core.quotePath=false log --since=<DAYS>.days.ago --no-merges --no-renames \
    --name-only --relative --format=<RS>%H -- .
```

A root inside a checkout is read from there (`--relative` keeps a sub-directory view's paths
relative to it). A root in no checkout that holds checkouts directly under it (a workspace of
sibling repositories, like javaweb with its 28) reads each, side by side, paths prefixed by the
directory; a deeper checkout is not looked for. A root with neither, a failing git, or a read past
`HISTORY_TIMEOUT_SECONDS` (20 s, every checkout together) leaves both signals out of the run with a
note, so the axis reads "not measured" and the run still exits 0. In the daemon the read runs in the
request's worker thread, so it holds that read slot 20 s at most and never the event loop.

Only current production files of the view take part: test files, deleted files and files that are
not measured never do, so test code is never flagged, as everywhere in health. Renames are not
followed: `--follow` is per file and so one process per file; without rename detection a rename
commit lists both paths, which credits the commit to the file under its current name exactly as
detection would (and detection cost 5x: 1.8 s against 0.34 s for javaweb's 28 logs in sequence),
while history before a rename stays under the old name either way.

- `hotspot` measures every production file of the scope the window touched. Its value is the
  churn rank, in thousandths of the changed production files of the same checkout that changed
  strictly less often (each checkout is its own history: one busy repository does not set the bar
  for the others), when the file is past `complex_callable` (its most complex callable) or
  `large_file`, and has at least 5 commits; otherwise 0, since it changed and is no hotspot. Files
  tied at the top rank below every one of them, so an evenly changed tree has no hotspot. The
  reason names the commits and what makes the file a hotspot (`complexity 23 in Daemon.handle,
  1322 lines`).
- `boundary_co_change` measures every pair of production files in different modules that changed
  together in at least 5 commits (`HomeConfig.module_of`: the declared modules of a `home.toml`,
  else the top-level directory). A changeset over 30 files counts toward churn but not toward
  co-change (a reformat or a sweep says nothing about which files belong together). Its value is
  the Jaccard of the two files' commit sets in thousandths; the reason reads `changed together in
  N of M commits with <other file>`, the key is the sorted pair, and the other file rides along as
  a place the finding also lives (so `--since` lists it when either file changed). Two files of
  the same name are not measured: one generated `sst-env.d.ts` per package, or one `cli.py` per
  package, are instances of one convention that a single sweep edits, and moving them into one
  module is not the fix.

## The consistency axis

No house style is declared: each file is compared with the other files of its own language
(`src/zemble/health/style.py`). The `population` tier says what that means for a reader: the
norm is the peers', so a finding can come or go when OTHER files change.

### The style vector

One vector per production file with at least 3 measured callables (`MIN_CALLABLES`; a smaller
file is not measured). Its dimensions are the members of `StyleDimension`, each with its
computation, transform and wording as facts on the member:

| Dimension | Computed as | Scale |
| --- | --- | --- |
| `callable_lines_median` | median own lines of the non-boilerplate callables | log |
| `callable_lines_p90` | nearest-rank 90th percentile of the same | log |
| `complexity_median` | median cognitive complexity of the same | log |
| `nesting_max` | deepest nesting of the same | log |
| `callable_density` | measured callables, boilerplate included, per 100 lines | log |
| `top_level_types` | named type declarations at file level | log |
| `comment_density` | lines a comment touches / lines | linear |
| `parameters_mean` | mean declared parameters of the non-boilerplate callables | log |
| `blank_line_ratio` | blank text lines / text lines | linear |
| `line_length_mean` | mean length of the non-blank lines, trailing whitespace stripped | log |
| `snake_case_names`, `camel_case_names`, `pascal_case_names` | share of the shaped method and function names spelled that way (`NameShape`) | linear |
| `public_share` | public share of the named types and the non-boilerplate methods and functions | linear |

Everything but the two text dimensions comes from the graph store (callable measurements,
file comment lines, symbols, visibility); blank lines and line length come from one plain read
of the file. Nothing is parsed again. `log` is `log1p`: lengths and counts are right-skewed.

Three rules keep the role of a file (a record, a registry) from passing for its style; each
came out of the hand review below:

- A callable that is one statement without control flow (`statements <= 1` and `nesting == 0`:
  an accessor, a delegation, a constant) is boilerplate. The callable-shape dimensions and
  `public_share` read the rest, and a file with fewer than 3 such bodies does not measure the
  callable-shape dimensions at all. Nesting rather than complexity decides, because a one-line
  predicate (`return a && b;`) scores a complexity of 1 without breaking the linear flow.
- Type names are left out of the naming shares (every language here spells them one way, so
  they only counted types again), and a one-word lower-case name (`run`) fits snake_case and
  camelCase alike, so it is evidence for neither.
- A dimension a file gives nothing to measure (no shaped name, too few bodies) is None and
  differs from any centre by nothing: no evidence is not a difference.

### Standardisation and the threshold

```
centre(d)    = median over the language's files of transform(x_d)
scale(d)     = 1.4826 * MAD(d), or 1.2533 * mean absolute deviation from the median when the MAD is 0
z_d          = (transform(x_d) - centre(d)) / scale(d)          a dimension with scale 0 is dropped
diff_d       = clip(z_d - c_d, -3, 3)                              c = 0 for the language, the directory's median z
distance     = sqrt(sum(diff_d^2)) / sqrt(chi2_0.99(k))          k = kept dimensions; reported * 100
```

- Per language: a language needs 6 measured files (`MIN_LANGUAGE_FILES`) and a dimension needs
  6 files that measure it; a dimension with zero spread (every file the same) is dropped for
  that language, and a language with fewer than 2 kept dimensions has no norm.
- Robust centre and scale (median, MAD with the normal-consistency factor 1.4826; the
  mean-absolute-deviation fallback of the modified z-score, Iglewicz and Hoaglin) so a few huge
  files do not define the norm. The diagonal scale only: no covariance is inverted.
- The clip at 3 scales (`DIFFERENCE_CAP`, Huber's clipping): past 3 a value is plainly unusual on
  that dimension and more says nothing new. Since 9 is under the cut of 2 or more dimensions,
  no single measurement makes an outlier: the size signals judge those, and reading different
  takes several dimensions at once.
- The cut: under a normal assumption the squared distance of k standardised dimensions is
  chi-square with k degrees of freedom, so 1% of files would pass its 0.99 quantile
  (`STYLE_CUT_QUANTILE`, Wilson-Hilferty approximation, within 1% of the tables from 3 degrees
  up). The value is that distance in hundredths of the cut, so the threshold is 100 whatever k
  is. Real code is heavier-tailed than normal: the measured shares past the cut are below.
- The directory norm: the file's parent directory, its sub-directories included; a directory
  with fewer than 3 measured files of the language (`MIN_DIRECTORY_FILES`) borrows its
  parent's, up to the root. Its centre is the median z vector of those files, in the LANGUAGE's
  scale, so both distances share units. A file whose directory norm would be its whole language
  is not measured by `style_outlier_directory` (it would repeat the repo signal).
- The reason names the norm and the up to three dimensions differing most (ranked by the
  unclipped difference; a second or third only when it differs by at least one scale), with the
  file's value and the norm's median: `against 154 python files: more commented (16% vs 3%);
  shorter callables (p90) (3 lines vs 26 lines); ...`.
- Scope: the norm is always every production file of the view, so `--paths` and `--since` list
  fewer files but judge them against the same peers. A sub-directory view's peers are that
  sub-directory's files (the view holds no others).

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

The history axis is on by default, because it costs under the 10 s it would have to stay under:
the git read is 0.05 s on zemble (342 commits) and 0.09-0.73 s on javaweb (28 checkouts, 4620
commits in 365 days, read side by side), and the whole history measurement 0.06 s and 1.0 s. Were
a window to pass 10 s, the rule would make it opt-in like the duplication tier; `--no-history`
(MCP `history=false`) turns it off anywhere.

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
layering 2, copying 1, vocabulary 1, consistency 1, history 1 (a forbidden edge breaks a rule the
workspace wrote down; the rest are general conventions or the code's own peers). A style finding
weighs 1 (its unit is hundredths of the cut, not lines), and so does a history finding (thousandths).
The history axis measures only files the window touched, so its `hotspot` fraction is the share of
the changed production files that are hotspots, and its `boundary_co_change` fraction the share of
the cross-module pairs that changed together at least 5 times that are coupled past the bar. An axis
with no measured unit (no `home.toml` with `[[forbidden]]` rules, the duplication tier not run, a
language below 6 measured files, no git checkout or `--no-history`, or `--signals` leaving it out) is
"not measured" and takes no part in the composite; it is never
100. The computation is deterministic: the same tree gives the same report (no wall time is
printed in it).

## Surfaces

- CLI `zemble health [path] [--limit N] [--signals a,b] [--paths P ...] [--since REF]
  [--copying | --no-copying] [--history | --no-history] [--history-window DAYS]
  [--baseline FILE] [--save-baseline FILE] [--json] [--no-daemon]`,
  through the warm daemon first, in process otherwise (graph built or refreshed as `outline`
  does). A path with no graph gets one built.
- MCP `health` (`repo`, `limit`, `signals`, `paths`, `since`, `copying`, `history`,
  `history_window`, `baseline`, `save_baseline`), returning the markdown; under
  `zemble nib-provider` it answers with one text part. Every surface builds one `HealthRequest`
  (`zemble.health.answers`), whose field names are the daemon's argument names.
- Daemon command `health`, reading its warm graph; no search index is loaded for it, the
  duplication tier included (it reads the unit index). The history axis runs git in the daemon's
  worker thread for the request, bounded by `HISTORY_TIMEOUT_SECONDS` (20 s). The read holds the
  whole graph's symbols and tallies in private memory (`read_estimate_mb`: 32 MiB plus 1.5 KiB per
  symbol and 64 B per edge, measured 209 MiB on javaweb and 321 MiB on the zenit workspace), so the
  daemon admits it against its memory ceiling first and otherwise answers `outgrown`: the CLI and the
  MCP tool then read the stored graph in their own process and say so (`docs/daemon.md`, fallback rules).
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
| javaweb | 674 | 669 | 490 | 1051 (1058 before graph format 10) | 243 | 87 | 0 | 135 | 12 | 3025 | 11 | 0 |
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

### The consistency axis (2026-10-17)

In process over an existing graph, this machine (walls vary +-15% with other load):

| Repo | Style measurement alone | Whole run without / with the axis | Peak | Score without / with | consistency | `style_outlier_repo` | `style_outlier_directory` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zemble | 0.04 s (159 files with a vector) | 0.43 / 0.45 s | 80 MB | 84.3 / 86.8 | 91.9 | 11 of 154 (7.1%) | 14 of 154 |
| javaweb (`--no-copying`) | 0.61 s (3222 files) | 10.7-11.3 / 11.4 s | 400 / 406 MB | 91.7 / 93.0 | 98.3 | 75 of 3208: Java 67 of 3160 (2.1%), JavaScript 8 of 48 | 31 of 3200 (34 on graph format 10) |

zemble's Python keeps 11 dimensions (the naming shares are constant: every shaped Python name
is snake_case); its 5 Java files are under the 6-file minimum. javaweb's Java keeps 10, its
JavaScript 12; its Python, Groovy, TypeScript, Bash and SCSS files are each under the minimum.

Five systematic false positives were found by hand on the way, each fixed by a general rule (Java
repo findings on javaweb at each step: 743 -> 489 -> 109 -> 107 -> 67, directory 364 -> 221 ->
60 -> 51 -> 28; zemble 16 -> 11):

1. Unclipped, one extreme dimension made an outlier alone (a 0%-public file at 6 scales): the
   clip at 3 scales.
2. Java's median callable was a 4-line getter, so every class with logic read different from
   the beans around it ("longer callables, more complex, fewer public"): the boilerplate rule.
3. `size() { return items.size(); }` scored a cognitive complexity of 1 (the metric counted the
   same-named call as recursion), so delegations escaped the boilerplate rule: nesting decides.
   The metric has since been fixed (graph format 10), and nesting still decides for predicates.
4. Accessors are public by role, so `public_share` still split data classes from logic classes:
   it reads the non-boilerplate declarations only.
5. In Python, `PascalCase` share only counted types again and one-word names read as "fewer
   snake_case names": type names are left out and one-word names are evidence for neither.

Top 10 findings read by hand. zemble:

| # | Finding | Verdict |
| --- | --- | --- |
| 1-2 | `types.py` (directory and repo): short lines, tiny callables, blank lines | real: a declarative module of dataclasses |
| 3, 6 | `dedup/languages/__init__.py`: tiny functions, no parameters, no nesting | real: a registry module |
| 4-5 | `graph/model.py`: more commented, tiny callables | real: a vocabulary module of enums and dataclasses with `#:` field docs |
| 7, 9 | `benchmarks/plot.py`: short lines, long callables | real: a different author's style (Semble's upstream benchmark scripts) |
| 8 | `languages/catalog.py`: 0.1 callables per 100 lines, few blank lines | real: a 2914-line declarative table |
| 10 | `benchmarks/metrics.py` against `benchmarks/`: many small functions | real: upstream code of another shape than its directory |

javaweb:

| # | Finding | Verdict |
| --- | --- | --- |
| 1 | directory `hawkeye/.../analysis/AnalysisContext.java`: short lines, no parameters, simple callables | role, not style: a scope-stack state holder among analysis passes (its methods are two statements, so past the boilerplate rule) |
| 2 | directory `orcono/mockup/lib/mock-data.js`: one-line callables among components | real: a mock-data table |
| 3 | repo `orcono/mockup/app/search/page.jsx`: PascalCase components, 52-line callables | real: the React mockup among plain JavaScript |
| 4, 8, 9 | repo `hawkeye/.../compiler/MethodCallTranspiler.java`, `PropertyRefHelper.java`, `TryAsyncTranspiler.java`: package-private, long callables, many parameters | real: the hawkeye compiler package's own style (package-private static helpers); the directory signal does not flag them, they agree with each other |
| 5 | directory `hawkeye/.../compiler/ReactiveIdRegistry.java`: public, short callables among package-private transpilers | role: the package's one public registry; it does read different, for a reason |
| 6 | repo `cruiser/.../seed/GeographySeeder.java`: 62-line callables, 71-char lines | real: a data seeder, table-like |
| 7 | repo `duiventil/.../config/SmtpConfig.java`: one-line bodies, dense | real: a builder written as one-liners with a field-comment table |
| 10 | directory `emberglyph/.../layout/SizeHints.java`: one-line fluent setters among layout logic | real: the one-liner style, with some role in it |

Generated and vendored files are legitimate findings: they read different by construction, and
the axis exists to surface such files (upstream Semble code shows up on zemble that way). The
two role findings above (a state holder, a lone public registry) are the residue the rules do
not remove; see Known limits.

### The history axis (2026-10-17)

In process, this machine: the git read is 0.05 s on zemble (1 checkout, 342 commits in 365 days)
and 0.09-0.73 s on javaweb (28 checkouts read side by side, 4620 commits); the whole history
measurement (read, per-file commit sets, co-change pairs) 0.06 s and 1.0 s, inside the walls'
own run-to-run noise. Composite score without / with the axis: zemble 86.6 / 88.8 (history 95.5:
`hotspot` 9 of 198 changed files), javaweb 93.0 / 93.5 (history 96.1: `hotspot` 173 of 4454).

Five systematic problems were found by hand on zemble, javaweb and opencode's `packages/` (a
multi-author TypeScript monorepo, 12585 commits a year), each fixed by a general rule:

1. One decile over all of javaweb's 28 checkouts let the busiest repository set the bar for every
   other (171 hotspots, 51 in hawkeye): the rank is per checkout (175; hawkeye 32).
2. A short history put a file changed 3 times in a small repository's top decile
   (`microcopy/tools/robbedoes/memory-profile.py`, its repository 16 commits a year): a file needs 5
   commits, code-maat's `min-revs` default (173).
3. Every one of opencode's 51 cross-module pairs past the bar was a generated `sst-env.d.ts`, one per
   package, regenerated by one sweep: two files of the same name are not measured (51 -> 0).
4. With those gone, a Jaccard bar of 0.5 (the design's example) found nothing on any tree, while
   the strongest pairs below it were real coupling: code-maat's `min-coupling` default 30% is the
   bar (zemble's `src/zemble` view 0 -> 3, opencode 0 -> 1).
5. Measuring only the complex or large files made the axis read "173 of 284 complex, active files
   are hotspots" (javaweb history 39.1): every changed file is measured, a non-hotspot at 0, so the
   fraction is the share of changed files that are hotspots (96.1).

Hand review, top findings:

| Tree | Finding | Verdict |
| --- | --- | --- |
| zemble | hotspots 1-9: `daemon/server.py` (37 commits, `Daemon.handle` 23, 1322 lines), `graph/store.py` (24, 2230 lines), `dedup/detect.py` (17, 1108 lines), `index/create.py` (17, `write_index` 31), `mcp.py` (17, `create_server` 23), `dedup/mcp.py` (14, `register_dupes_tool` 18), `dedup/cli.py` (12, `run_dupes` 18), `embedding/preflight.py` (12, `embed_status` 19), `graph/provider.py` (11, `_walk` 46) | 9 of 9 real; `register_dupes_tool` owes its complexity to the tool function nested in it, which Sonar's rule counts in the enclosing callable |
| zemble | `boundary_co_change`: none over the whole tree (generic config: all production code is under `src/`, one top-level module) | as designed |
| zemble `src/zemble` | `cache.py` with `index/create.py` (9 of 24), `index/types.py` (6 of 17), `index/index.py` (8 of 23) | 3 of 3 real: the index cache layer lives outside `index/` and changes with its internals |
| javaweb | hotspots 1-10: `DominoElement` (51, `registerReviver` 67), `HttpConduit` (70, `matchEndpoint` 32), `orcono/.../app.scss` (33, 1911 lines), `zenit/tools/lib/gradle.js` (25, `runGradleStreaming` 57), `HawkeyeCompile.groovy` (22, `lower` 27), `IRTranspiler` (210, 3354 lines), `orcono/.../Main.java` (51, 1123 lines), `ServerDominoElement` (43, `dispatchEvent` 22), `RenderContext` (139, `getValueFromPath` 41), `CouchbaseDatasource` (49, `upsert` 31) | 10 of 10 real; the two size-only ones are a stylesheet and a bootstrap class that keep growing |
| javaweb | `boundary_co_change`: none, by construction: every declared module is its own checkout, and a commit never spans two | not measurable there |
| opencode | `core/src/flag/flag.ts` with `opencode/src/effect/runtime-flags.ts` (15 of 48) | real: two flag vocabularies kept in step by hand |
| opencode | hotspot 2: `sdk/js/src/v2/gen/types.gen.ts` (364 commits, 13625 lines) | generated; a legitimate finding by the policy above (`large_file` reports it too) |

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
- The consistency axis measures structure as much as typography, so a file whose ROLE differs
  from its neighbours' (a state holder or a value type among logic classes, a public registry
  among package-private helpers) can still read different once its methods are two statements
  or more; the boilerplate rule removes only one-statement accessors. Correlated dimensions
  (median and p90 callable length) count separately on the diagonal metric.
- Real code is heavier-tailed than the normal assumption behind the cut: 7.1% of zemble's
  Python files and 2.1% of javaweb's Java files pass a cut that would pass 1% of normal data.
  A small population (javaweb's 48 JavaScript files of several kinds) passes more (17%).
- A sub-directory view compares its files with that sub-directory's files only, since the view
  holds no others; `--paths` and `--since` keep the whole view as the norm.
- History is per checkout: in a workspace of sibling repositories a cross-repository coupling is
  invisible (commits never span two histories), and a checkout nested deeper than the root's
  direct children is not read. History before a rename stays under the old path (`--follow` is
  per file), so a renamed file starts its churn again.
- `hotspot` reads size as well as complexity, so a data table or stylesheet that keeps growing
  (opencode's `i18n/en.ts`, javaweb's `DevSeed.java`) is a hotspot by size alone, and so is a
  generated file that is regenerated often.
- A pair below 5 shared commits is not measured however coupled it is, and files changed in
  sweeps over 30 files lose those commits for co-change (not for churn).
- Knowledge distribution (a file with one author, Tornhill's knowledge silos) is not a signal:
  every javaweb checkout but three and zemble have one author in the window, so it would flag
  every active file there and could not be checked by hand.
