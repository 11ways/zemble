# zemble health

`zemble health [path]` (MCP `health`) is a ranked, reference-free report of code-shape
findings plus a score. It reads the symbol graph (`docs/graph.md`), for layering the
workspace's own `.zemble/home.toml` (`docs/home.md`), for the duplication tier the
`zemble dupes` channels (`docs/dedup.md`) through their per-root unit index, for the
consistency axis one plain read of each measured file's text, and for the history axis one
`git log` per checkout; the documentation axis reads the comment facts the graph stores
(format 11). It never builds a search index and never embeds anything. It is a
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
graph's own `is_test`) is never measured. So is a generated or minified file (see "Generated and
minified files").

| Signal | Axis | Tier | Finding when | Threshold source |
| --- | --- | --- | --- | --- |
| `long_callable` | size | syntactic | own body lines > 60 | PMD ExcessiveMethodLength reports at 100 by default; 60 flags the tail before it gets there |
| `many_parameters` | size | syntactic | declared parameters > 5 | PMD ExcessiveParameterList reports at 10 by default; 5 is the common review convention |
| `deep_nesting` | size | syntactic | control-flow nesting > 3 | Sonar S134 default maximum 3 |
| `complex_callable` | size | syntactic | cognitive complexity > 15 | Sonar S3776 default threshold 15 |
| `large_type` | size | syntactic | type declaration span > 500 lines | PMD ExcessiveClassLength reports at 1000 by default; 500 flags the tail before it gets there |
| `large_file` | size | syntactic | file lines > 1000; a file holding exactly one named top-level type is not measured (`large_type` judges it: the type is the file), files with none or several are | Sonar S104 default maximum 1000 |
| `many_types_per_file` | size | syntactic | top-level types in one file > 5 | declared here; no common tool default |
| `single_implementation_interface` | machinery | graph | an interface, trait or protocol with exactly 1 production implementor in its own module (`module_of`); one whose single implementor lives in another module (an extension point by design), or that test code implements too (a seam for test doubles), is not measured, so a measured value of 1 is always a finding | speculative generality (Fowler, Refactoring) |
| `single_subclass_abstract` | machinery | graph | an `abstract` type with exactly 1 production subtype in its own module (one in another module, or a test subtype beside it: not measured, as above) | speculative generality (Fowler, Refactoring) |
| `deep_inheritance` | machinery | graph | a class (`class` kind) with more than 5 ancestors on its longest chain of EXTENDS edges at `unique_name` or better, counted from the topmost one the graph holds; interfaces are not measured | Sonar S110 (inheritance tree depth) default maximum 5 |
| `lone_public_callable` | machinery | graph | a public method or function with at least 1 production caller, all of them declared in its own outermost type (a file-level function's own type is its file); value: callers outside it, finding at 0 | declared here: a seam only its own type uses is a private helper |
| `unreferenced_symbol` | machinery | graph | a private or package-private method, function or type nothing references | declared here |
| `forbidden_crossing` | layering | graph | a file with edges into a module its own module may not depend on | the workspace's `[[forbidden]]` rules |
| `package_cycle` | coupling | graph | a directory of production files in a dependency cycle with other directories of its own module (EXACT edges only); value: directories in the cycle, finding above 1 | Acyclic Dependencies Principle (Martin); cyclic physical dependencies (Lakos); JDepend's package cycles |
| `feature_envy` | coupling | graph | a method whose getter calls into one other type outnumber its calls into its own type; value: those getter calls (0 when its own type is called as often), finding above 2 | Feature Envy (Lanza and Marinescu, Object-Oriented Metrics in Practice: ATFD > 2, LAA < 1/3), read off calls into one-statement getters |
| `clone_mass` | copying | duplication | file lines inside exact or renamed clone classes > 50 per mille (5%) | declared here: PMD CPD reports every copy past its token minimum but has no per-file share default |
| `cloned_body` | copying | duplication | a callable body in an exact or renamed clone class with a copy in another file | membership (Fowler, Refactoring: duplicated code) |
| `pass_through` | machinery | duplication | a public callable whose whole body hands its parameters, in order, to one other callable | existence: middle man (Fowler, Refactoring) |
| `vocabulary_drift` | vocabulary | duplication | a value or value set declared in more than one place (one per `vocabulary` class) | existence: a `zemble dupes --kind vocabulary` class |
| `style_outlier_repo` | consistency | population | a file's style vector lies past the cut from its language's files (value: distance in hundredths of the cut, > 100) | declared here: robust distance past the chi-square 0.99 quantile of the kept dimensions (see "The consistency axis") |
| `style_outlier_directory` | consistency | population | the same against its directory's files, where those are not its whole language | as above |
| `hotspot` | history | history | a file past `complex_callable` or `large_file` with at least 5 commits in the window, whose churn ranks in its checkout's top decile (value: thousandths of the checkout's changed files that changed strictly less often, > 900) | hotspots, change frequency x complexity (Tornhill, Your Code as a Crime Scene; CodeScene); the 5-commit floor is code-maat's `min-revs` default |
| `boundary_co_change` | history | history | two files in different modules whose commit sets overlap with a Jaccard > 300 per mille over at least 5 shared commits (one finding per pair) | temporal coupling (Tornhill); code-maat's `min-coupling` 30% and `min-shared-revs` 5 defaults, changesets over 30 files left out (`max-changeset-size`) |
| `undocumented_public` | documentation | syntactic | a public named type, or a public method or function, with no doc comment and no docstring (value: its doc lines, at most 0) | Sonar S1176 (public API should be documented); Checkstyle MissingJavadocType and MissingJavadocMethod at their default public scope |
| `long_docblock` | documentation | syntactic | a type's, callable's or field's doc comment plus docstring > 20 lines | declared here: the workspace convention is a one-sentence summary plus the gotchas; no common tool default |
| `commented_out_code` | documentation | syntactic | a file whose comment blocks that are nothing but code hold > 2 lines | Sonar S125 (sections of code should not be commented out), read by a line heuristic, not a re-parse; 2 lines tolerate one snippet |
| `stale_marker` | documentation | syntactic | a file whose comments hold an upper-case `TODO`, `FIXME`, `XXX` or `HACK` as a whole word (value: those lines, > 0) | Sonar S1135 (TODO tags) and S1134 (FIXME tags) |

The tool defaults quoted above are as the step's design states them; they were not re-verified
against the tools' current releases.

The measurements come from the graph (`docs/graph.md`): `CallableMetrics` per callable
(lines, cognitive complexity, nesting), `FileMetrics` per file, symbol spans and parameter
lists. `long_callable` uses a callable's OWN lines: the span of every named callable declared
directly inside it (a decorated tool inside its `register_*`, a closure) is subtracted, since
that one is measured on its own. Cognitive complexity follows Sonar and keeps nested
lambdas and functions in the enclosing callable.

## The machinery and coupling signals over callers and types

Every one reads graph edges in bulk, aggregated in SQL (`GraphProvider.caller_tallies`,
`call_tallies`, `file_dependencies`, `outgoing_edges`), never a query per symbol and never a row per
edge: on the zenit workspace (247k symbols, 2.7M edges) the caller aggregate is 232k rows and the
call aggregate 227k, each folded while it streams onto its callable with every far end reduced to its
home (the outermost type declaration holding it, `HealthFacts.home_of`; a file-level function's home
is its file). A sub-directory view spells a far end outside the folder `../`-relative to it, so it can
never equal an in-folder id.

- `lone_public_callable` measures a public method or function only once something in production
  calls it, and values it by the callers whose home is not its own: a nested type's callers are the
  outer type's own (a Java nested class reaches its outer type's private members). The reason names
  the callers and the type (`2 caller(s), all in Ledger`) and adds `tests call it too` when tests call
  it as well (package-private, or a test of the type itself, keeps those compiling). A public callable
  nothing calls is NOT measured: in a library its users outside the workspace call it, and without a
  declared library/application split that cannot be told apart from dead code (the earlier
  "at most one caller" rule flagged 5009 of 9018 public callables on the zenit workspace, most of
  them its API surface).
- `single_implementation_interface` and `single_subclass_abstract` read who the one subtype is from
  the hierarchy edges `collect` already loads; when it lives in another module (`module_of`: the
  declared `home.toml` modules, else the first path segment) the abstraction is an extension point
  and is left out of the population rather than measured, which keeps the signal's `== 1` honest: a
  measured 1 is always a finding. A subtype outside a sub-directory view cannot be placed, so such an
  abstraction is not measured there either. Nor is one that test code subtypes beside its one
  production subtype: that is a seam for test doubles, which inlining would remove (only where the
  graph holds that test code).
- `deep_inheritance` follows EXTENDS edges only (IMPLEMENTS is an interface, not an ancestor),
  iteratively and with a guard against a cycle in bad data; the reason names the topmost ancestor
  the graph holds (`up to ASTUnit`). A supertype outside the graph (a library class) is not counted.
- `package_cycle` nodes are (module, directory) pairs of the scope's production files; an edge is a
  `file_dependencies` row (IMPORTS, CALLS, REFERENCES_TYPE) resolved EXACT between two such files of
  the same module, so a by-name edge resolved into the wrong module or package cannot close a cycle.
  The floor is `exact` rather than `unique_name` because a by-name CALLS edge into a library-like
  name inflates cycles: on zenit the largest component grows from 59 to 86 directories with by-name
  calls, and the directories in cycles from 395 to 444. Components come from an iterative Tarjan.
  Each directory is one sample valued by its component's size; its reason lists up to 3 other
  directories of the cycle and names its lightest link into it (the fewest edges into one other
  directory of the cycle, and the file they come from, which also anchors the finding): the cheapest
  edge to cut. The key is `module:directory`.
- `feature_envy` approximates Lanza and Marinescu's ATFD (accesses to foreign data, directly or
  through accessors) with resolved calls (`HealthFacts.reads_data`): a call reads another type's data
  when it is written on a named receiver (`customer.name()`; not `Str.of(...)`, a type name as the
  extractor's own `receiver_type` marks it; not a bare call; not a chained receiver such as
  `builder.add(x).named()`, which acts on what a call just returned) and its callee is a getter: no
  parameter and a one-statement body without control flow (`CallableMetrics.is_boilerplate`, the rule
  the consistency axis uses for boilerplate). Its own type's count is every resolved call into its
  home or a supertype of it, bare calls included. The value is the reads of the most-read other type
  when they outnumber the own calls, else 0; the reason names that type (`3 getter calls into
  Customer, 0 calls into its own type`). Only methods with a measured body are measured. Each rule
  came out of the zenit workspace, measured in that order: every call on a receiver, bare calls
  skipped (the signal as first designed) gave 5119 findings of 73580 methods, led by migrations'
  `up` (610 calls into `TableBuilder`) and codecs; leaving out overrides and annotated methods, whose
  place a contract fixes, counting bare own calls and only one-statement callees: 1116; leaving out
  chained receivers (fluent builders): 1073; only parameterless getters, so a registration
  (`registry.register(x)`) is no read: 836 of 28211.

## The documentation axis

Four signals about comments, each read off the graph's format-11 facts (`docs/graph.md`), so
the axis costs no extra pass at report time.

**A declaration's doc** (`Symbol.doc_lines`, `metrics.doc_comment`) is the run of comments
directly above it: its preceding siblings, each ending on the line above the next (above the
wrapper the spec wraps it in, such as a decorator or an `export`), passing over annotation
siblings (Rust's `#[derive]` lines), plus a Python body's leading string
(`ComplexitySpec.docstring_kinds`). A javadoc and a `//` run count alike: the question is "is
there a doc", not "is it javadoc". A comment that trails code on its first line ends the run, a
blank line ends it, and a run whose every line reads as code is disabled code, not a doc. Both
extractors set it on types, callables and fields.

**`undocumented_public`** measures every named type, method and function that a reader of the
public API meets: the declaration and every type around it are PUBLIC (`HealthFacts.visibility`;
UNKNOWN is never measured, as Checkstyle reads the surrounding scope). Not measured, each for a
reason: constructors (the type says what they build), overrides and possible overrides
(`may_override`: the supertype documents the contract), runtime hooks (`__repr__`), `main`,
members of anonymous types, declarations inside a callable, and one-statement bodies
(`CallableMetrics.is_boilerplate`: a getter says what its name says). The finding spans the declaration's
first line only, where the doc goes, so a `--since` run lists it when the declaration is new or
re-declared, not on every edit inside it. The advice is the workspace convention: a
one-sentence docblock, then only the gotchas.

**`commented_out_code`** counts, per file, the lines of comment blocks that are nothing but
code (`metrics.disabled_code`). A block is a run of comments on adjacent lines that each start
their own line; a trailing comment annotates its line and never counts. A line reads as code
(`metrics.reads_as_code`) when, its comment markers stripped, it ends the way only code ends
(`;`, `{`, `}`, `),`), is a whole call (`a.b(c)`, no space before the parenthesis) or an
assignment of one token or call (for languages without semicolons), or opens a keyword block with
a colon and holds one of `(=[.`; it never does when it starts with a doc tag (`@return`), is a
JSON example, holds a URL or a `{@link}`, or holds four prose words in a row. A block is
skipped whole when it mixes prose with code (an explanation with its example), when it is written
with a doc opener (`/**`, `///`, `//!`, `/*!`), when it sits right below a block whose last line
ends in a colon ("do it like this:"), or when the extractor attributed it as a declaration's doc.
The signal is per file and each file weighs 1 in the score: weighed by its lines, a file under
the bar would weigh 0 and the fraction would say how concentrated the disabled code is, not how
widespread.

**`stale_marker`** counts the comment lines holding `TODO`, `FIXME`, `XXX` or `HACK` as an
upper-case whole word (lower case is prose: "a hack"). `AIDEV-NOTE:` is not a marker: it
documents a surprise, it does not defer work.

### Generated and minified files

`collect.unmeasured` takes a file out of every signal of every tier (its file measurements, its
symbols, its duplication units and clone memberships, its history) when a comment in its first
30 lines carries a generator's header (`@generated`, `DO NOT EDIT`, `generated by`,
`automatically generated`, `autogenerated`, `Code generated`, case-insensitive), when a line is
longer than 1000 bytes (`MINIFIED_LINE_BYTES`: minifiers put a program on a few lines, formatters
wrap hand-written code at 80 to 120 columns, and zemble's longest production line is 330 bytes),
or when its name ends in `.min.js`, `.min.css` or `.min.mjs`. The report carries one note, `N
generated or minified file(s) not measured: <up to 5 paths>`. The reason: a minified
`dc-runtime.js` in a javaweb docs mockup gave two findings nobody could act on.

## The history axis

`zemble.health.history` reads one window of git history (`--history-window DAYS`, default 365)
through `zemble.health.git`, the one module that runs git, in a subprocess only:

```
git -c core.quotePath=false log --since=<DAYS>.days.ago --no-merges --no-renames \
    --name-only --relative --format=<RS>%H -- .
```

A root inside a checkout is read from there (`--relative` keeps a sub-directory view's paths
relative to it), and so is every direct child directory holding a `.git` entry that is a checkout of
its own (`rev-parse --show-toplevel` is the child itself), side by side, paths prefixed by the
directory: a workspace of sibling repositories (javaweb with its 28), with or without a checkout at its
top (the zenit workspace's root is a checkout of changelogs holding its 40 module checkouts). A child
checkout's files never appear in the enclosing checkout's log (git records a gitlink, or nothing when
the child is ignored or untracked), so no commit is counted twice. A deeper checkout is not looked
for. A root with no checkout anywhere there, a failing git, or a read past
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
`lone_public_callable` also skips a member of an interface, trait or annotation type: it is the
contract, not a helper. `feature_envy` skips the callables of the first four rules and the members of
an anonymous class for a different reason, placement rather than reachability: a method a contract
or a framework places (an override, an annotated method, a migration's `up`) cannot move onto the
type it envies.

A remaining candidate with no inbound edge is finally checked against the source text,
because the graph has no edge for a callable used as a VALUE (a dispatch table, `key=f`,
`Foo::bar`): its simple name must not be written outside its own declaration in its own file
(private), its directory (package-private) or any file of the graph (anything wider; test
files count for `unreferenced_symbol` only). A mention in a comment or a string also withholds
the finding. A `lone_public_callable` candidate (all callers inside its own type) gets the same
check against every OTHER production file: a `Type::member` reference elsewhere would break if it
were made private, so a public name written in another file is not measured.

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
layering 2, coupling 1, copying 1, vocabulary 1, consistency 1, history 1, documentation 1 (a
forbidden edge breaks a rule the workspace wrote down; the rest are general conventions or the
code's own peers). A style finding weighs 1 (its unit is hundredths of the cut, not lines), and so
does a history finding (thousandths). The coupling axis's `package_cycle` fraction is the share of
the scope's production directories that sit in a cycle, and its `feature_envy` fraction the share of
the measured methods that envy another type. Of the documentation signals only `long_docblock`
weighs its lines: its fraction is the share of all doc lines that sit in overlong docblocks; the
others weigh 1 per declaration or file.
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
  symbol and 100 B per edge, measured 209 MiB on javaweb and 642 MiB on the zenit workspace with the
  caller and call aggregates, 599 MiB without them), so the
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

The per-signal counts below predate the precision rules of `lone_public_callable`,
`single_implementation_interface`, `single_subclass_abstract` and `large_file` and the coupling
axis; see "Precision rules and the coupling axis" for the current ones.

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

### Precision rules and the coupling axis (2026-10-17)

In process over the existing graphs, this machine under other load (load average 15-19, so walls
are rough). The zenit workspace (247k symbols, 2.69M edges; its root is a checkout holding 40 module
checkouts), past/measured before and after:

| Signal | Before | After |
| --- | --- | --- |
| `lone_public_callable` | 5009 / 9018 | 1129 / 6438 |
| `single_implementation_interface` | 327 / 1500 | 189 / 1362 (271 / 1444 with the module rule alone) |
| `single_subclass_abstract` | 17 / 260 | 11 / 254 |
| `large_file` | 170 / 8898 | 17 / 277 |
| `deep_inheritance` | - | 2 / 7169 |
| `package_cycle` | - | 395 / 939 directories (444 at `unique_name`) |
| `feature_envy` | - | 836 / 28211 methods |
| `hotspot` | not measured | 259 / 6556 |
| `boundary_co_change` | not measured | 0 / 0 (every module is its own checkout) |

Score 93.5 -> 93.1 (coupling 77.5, history 96.0). Wall 32-42 s before and 46-52 s after; peak RSS
758 MiB before and 850 MiB after; the caller and call aggregates read in 6.5 s and 3.1 s. On a copy of
zemble (generic config, no history): `lone_public_callable` 277 / 586 -> 72 / 561, `large_file`
6 / 200 -> 6 / 145, `package_cycle` 16 / 20 (the root package `src/zemble` imports every
sub-package for its CLI and MCP surfaces, and they import its `types`/`utils`), `feature_envy`
0 / 649, `deep_inheritance` 0 / 361.

Hand review on zenit:

| Finding | Verdict |
| --- | --- |
| `package_cycle` `zenit/common` (59 directories), lightest links of 2 edges: `Zenit.java` imports `api.ZenitRuntime`, and `api/ResponseCarrier.java` reads `Zenit.DRY` | real: the root package and its sub-packages depend on each other |
| `lone_public_callable` `ArcanaKnowledge.knowledgeRoleAllowed` (2 callers), `WorkProperties.canReadOption` (6 callers) | real: no file outside the type names them |
| `deep_inheritance` `LiteralDoubleUnit`, `LiteralIntegerUnit`: 6 ancestors up to `ASTUnit` | real by S110's count; an AST hierarchy, deliberate |
| `feature_envy` `BoardTemplate.Walk.attribute`: 26 token getter calls, 4 own | real: it rebuilds token ranges a `span()` on the token would give |
| `feature_envy` `ScreenCodec.read`, `AiModule.createDirector`, textum's `*Journeys` | not: a stream decoder, a composition root, and test code the graph does not mark as test (see Known limits) |
| `single_implementation_interface` `DelegationLookup` | not: a test double implements it, but the graph holds none of hohenheim's `src/browserTest/` files |

### The documentation axis (2026-10-17)

zemble (a clone of this branch, graph format 11, `--no-daemon`): documentation 99.7, composite
90.9. `undocumented_public` 7 of 1329 measured declarations, `long_docblock` 0 of 3761 documented
ones (the longest production doc is 20 lines, at the bar; 79 are over 10), `commented_out_code`
0 of 199 files, `stale_marker` 1 of 199 files, no file generated or minified. Hand review: the 7
are 7 public classes with no docstring (`benchmarks/data.py` `Target`, `RepoSpec`, `Task`;
`index/dense.py` `SelectableBasicBackend`; `index/files.py` `FileStatus`; `stats.py`
`BucketStats`, `SavingsSummary`), all real; the marker is `# TODO: make this configurable` in
`chunking/chunking.py`, real. The extraction cost of the format-11 facts: 1079 files (zemble's
`src` and `tests`, gson, serde, express) take 5.55 s of CPU against 5.15 s before (best of 3: 8%
more). A generated file now leaves every tier, so opencode's `types.gen.ts` hotspot above goes
away if the file carries a generator's header (not re-checked here).

## Known limits

- `lone_public_callable` cannot judge a public callable nothing in the workspace calls: a library's
  API and dead code look the same without a declared library/application split, so neither is
  measured (`unreferenced_symbol` still reports non-public dead code). Its source-text check
  withholds a candidate whose name is written in any other file, so common names (`get`, `apply`)
  are under-reported. A caller in test code does not count as outside: the reason says
  `tests call it too`, where package-private (Java) rather than private keeps them compiling.
- A library's extension points implemented in the SAME module only by the library's users
  (gson's `FieldNamingStrategy`, `ToNumberStrategy`) still read as single-implementation
  interfaces: only an implementor in another module, or a test double, marks the seam.
- `package_cycle` works on directories, not language packages: a Java package split across source
  sets (`src/common/java/x` and `src/server/java/x`) is two nodes, and a Python package's
  `__init__.py` is its directory. A cycle through a directory outside `--paths` is not seen (only
  scope files' edges are read). One large cycle yields one finding per directory, all at the same
  value; the reason's lightest link is the place to start (zenit's `zenit/common` cycle of 59).
- `feature_envy` reads getter-shaped calls (no parameter, one statement, no control flow), which also
  match the one-statement reads of a stream API: a wire codec reading a `BinaryReader` (`r.u16()`)
  is reported though moving it would be wrong, and so is a composition root reading a module's
  getters to wire services (quirkyquarters' `AiModule.createDirector`). It does not see field access
  (`other.x`), so a method reading public fields directly is never envious, and a sub-directory view
  never names a type outside it as the envied one. Test code the graph does not mark as test
  (textum's `src/browserTestCommon/`, which `TEST_PATH_SEGMENTS` does not list) is measured like
  production: on zenit it fills most of the top 15.
- `deep_inheritance` counts only ancestors the graph holds: a class whose chain continues into a
  library is measured from its topmost workspace ancestor, so its true depth is higher.
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
  invisible (commits never span two histories), so `boundary_co_change` measures nothing where every
  module is its own checkout (javaweb, zenit). The root and its direct children are read; a checkout
  nested deeper (under a plain directory, or inside a child checkout) is not. History before a rename
  stays under the old path (`--follow` is per file), so a renamed file starts its churn again.
- `hotspot` reads size as well as complexity, so a data table or stylesheet that keeps growing
  (opencode's `i18n/en.ts`, javaweb's `DevSeed.java`) is a hotspot by size alone, and so is a
  generated file that is regenerated often.
- A pair below 5 shared commits is not measured however coupled it is, and files changed in
  sweeps over 30 files lose those commits for co-change (not for churn).
- Knowledge distribution (a file with one author, Tornhill's knowledge silos) is not a signal:
  every javaweb checkout but three and zemble have one author in the window, so it would flag
  every active file there and could not be checked by hand.
- `commented_out_code` is a heuristic, not Sonar S125's re-parse. Measured by listing every line
  it counts: none in zemble's own `src`, `benchmarks` and `javac-facts`, the 3 planted lines in
  `tests/fixtures`, and 24 lines over 924 files of express, gin, gson, monolog and serde
  (`~/.cache/zemble-bench`), of which 16 are disabled code (67%); the other 8 are code-shaped
  notes that explain code (`// app.get(setting)` above the branch that handles that call, 4),
  sentences that end in code (`// Equivalent to Integer.parseInt(...);`, 2) and a JSON example
  written as an assignment (2). Over those five repositories no production file passes the bar
  of 2 lines; the two files that do are tests, both true. Before the block rule the same scan
  counted 1130 lines, nearly all examples inside doc comments. Recall is lower: a disabled block
  under a prose header (`/* legacy tests`), or one holding a line no rule reads as code (a string
  continuation `"/:id",`, a `...` placeholder), is not counted.
- `undocumented_public` treats every member of a type that may override (any type with a
  supertype the view cannot see into) as an override, so such members are never measured: the
  same fail-closed rule the absence signals use.
- Docstrings are read for Python and Starlark only (`docstring_kinds`); Julia's and Elixir's
  doc strings and attributes, written above or around the declaration as code, are not.
- A doc comment separated from its declaration by a blank line is no doc, as godoc reads it;
  javadoc would still attach it, so such a Java docblock is reported undocumented.
