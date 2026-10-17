# zemble health

`zemble health [path]` (MCP `health`) is a ranked, reference-free report of code-shape
findings plus a score. It reads only the symbol graph (`docs/graph.md`) and, for layering,
the workspace's own `.zemble/home.toml` (`docs/home.md`); it never builds a search index and
never embeds anything. It is a REPORT, never a gate: it exits 0 whenever it could answer,
however many findings it has.

```
zemble health ./my-project                       # top 25 findings, the score and its axes
zemble health ./my-project --limit 100 --json    # the same payload as JSON
zemble health ./my-project --signals long_callable,complex_callable
zemble health ./my-project --paths src/core src/api
zemble health ./my-project/src/core              # a sub-directory: its own paths, its findings only
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

The tool defaults quoted above are as the step's design states them; they were not re-verified
against the tools' current releases.

The measurements come from the graph (`docs/graph.md`): `CallableMetrics` per callable
(lines, cognitive complexity, nesting), `FileMetrics` per file, symbol spans and parameter
lists. `long_callable` uses a callable's OWN lines: the span of every named callable declared
directly inside it (a decorated tool inside its `register_*`, a closure) is subtracted, since
that one is measured on its own. Cognitive complexity follows Sonar and keeps nested
lambdas and functions in the enclosing callable.

COPYING, VOCABULARY, CONSISTENCY and HISTORY are later axes: they join `Axis` when a signal
measures them (duplication findings from `dupes` are the next step), so no axis exists that
could only ever read "not measured".

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
without its file (`com.example.Util.tangled(int,int)`) or, for a file-level unit, the path.
Never a line number, so a later baseline diff survives edits around a finding.

## The score

```
mass(unit)        = value for a lines signal (long_callable, large_type, large_file), else 1
fraction(signal)  = sum(mass of units past the threshold) / sum(mass of every measured unit)
axis score        = 100 * (1 - mean(fraction of the axis's measured signals))
composite         = sum(weight(axis) * axis score) / sum(weight(axis))   over measured axes
```

Weights (`AXIS_WEIGHTS` in `src/zemble/health/score.py`, the one table): size 1, machinery 1,
layering 2 (a forbidden edge breaks a rule the workspace wrote down; size and machinery are
general conventions). An axis with no measured unit (no `home.toml` with `[[forbidden]]`
rules, or `--signals` leaving it out) is "not measured" and takes no part in the composite; it
is never 100. The computation is deterministic: the same graph gives the same report.

## Surfaces

- CLI `zemble health [path] [--limit N] [--signals a,b] [--paths P ...] [--json] [--no-daemon]`,
  through the warm daemon first, in process otherwise (graph built or refreshed as `outline`
  does). A path with no graph gets one built.
- MCP `health` (`repo`, `limit`, `signals`, `paths`), returning the markdown; under
  `zemble nib-provider` it answers with one text part.
- Daemon command `health`, reading its warm graph; no index is loaded for it.
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
- A small language population (zemble's one Java module) makes a marginal finding p100.
