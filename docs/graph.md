# Symbol graph

`zemble graph` builds a symbol graph over a workspace in every language with a bundled
tree-sitter grammar and answers relationship questions about it: who calls this, who
implements this, what tests cover this. It is stored beside the search index and updated
incrementally. Java and Hawkeye templates go through hand-written extractors; every
other language goes through one grammar-driven extractor fed by a spec (see
[Languages](#languages)).

The graph is deliberately not a compiler. It is a tree-sitter extractor plus a
name resolver, and every answer it gives carries a grade saying how much it
actually knows. Use it to navigate; do not use it as proof.

## The model

Two tables of things, both language neutral (`zemble/graph/model.py`).

**Symbols** are declarations. Kinds: `PACKAGE`, `CLASS`, `INTERFACE`, `ENUM`,
`RECORD`, `ANNOTATION`, `STRUCT`, `TYPE` (an alias or typedef), `MODULE` (a file, or a
namespace with a body), `METHOD`, `CONSTRUCTOR`, `FUNCTION` (a callable that belongs to
no type), `FIELD`, `ENUM_CONSTANT`, `TEMPLATE`, `BLOCK`. A trait or protocol is an
`INTERFACE`; a Rust `impl` or a Swift `extension` adds members to the type it names.

A symbol's `id` is `<file-relative-path>#<qualified-name>` with a signature
disambiguator appended for callables, so the two `scale` overloads of
`com.example.core.Circle` are:

```
src/main/java/com/example/core/Circle.java#com.example.core.Circle.scale(double)
src/main/java/com/example/core/Circle.java#com.example.core.Circle.scale(double,int)
```

The disambiguator holds **erased** parameter types: `List<String>` becomes
`List`, `String...` becomes `String[]`. The human-readable `signature` field
keeps the generics.

Beyond the obvious fields (`file_path`, `start_line`, `end_line`,
`container_id`, `modifiers`, `annotations`) a symbol carries `annotation_args` -
the **string-literal** arguments of each of its annotations, keyed by annotation
simple name and then by element name, with the single unnamed argument keyed
`value` - and `is_test`, which is
true when any **directory** segment of its path is `test`, `tests`,
`browserTest`, `integrationTest` or `testFixtures`, matched case insensitively.
That covers both the Maven layout and the Gradle source sets used across the
javaweb workspace. It is a path fact, not a build fact: a class in
`src/main/java/.../testing/Contest.java` is not a test.

**Edges** are relationships: `EXTENDS`, `IMPLEMENTS`, `OVERRIDES`, `CALLS`,
`REFERENCES_TYPE`, `ANNOTATED_WITH`, `IMPORTS`, `TESTS`, `EXERCISES`.

An edge always records `dst_name` (the name as written in the source) and, when
resolution succeeded, `dst_id`. `TESTS` comes from naming (`FooTest`, `FooTests`,
`TestFoo`, `FooIT` all point at `Foo`); `EXERCISES` comes from use (a test file's
top-level type to every main-source symbol it touches).

## What the extractor sees

One file at a time, with no knowledge of any other file
(`zemble/graph/java.py`). It handles package declarations, all four import forms,
top-level, nested, local and anonymous types, interfaces with default and static
methods, enums including constant bodies, records (components become fields),
annotation types, constructors and overloads, generics, method invocations with
their receiver and argument count, `new Foo(...)`, `this(...)`, `super(...)`,
method references, and type references in every declaration position: extends,
implements, field, parameter, return, throws, local variable, cast, for-each,
try-with-resources, catch, instanceof pattern, type arguments and annotations.

Two shapes are worth naming:

- An **anonymous class** becomes a symbol named `<enclosing-type>$anon@<line>`
  whose container is the method that declares it, with an `EXTENDS` edge to its
  declared supertype (rewritten to `IMPLEMENTS` at resolution time when that
  supertype turns out to be an interface). Its own members belong to it, not to
  the enclosing method, so `implementations` can find it.
- A **lambda** gets no symbol at all. Calls inside it are attributed to the
  enclosing method, which is what you want when asking "who calls this".

Comments and string literals never produce references.

## The resolution ladder

Resolution is workspace wide and runs in two passes
(`zemble/graph/resolve.py`). Pass 1 indexes every file's declarations by
qualified name and by simple name. Pass 2 builds each file's scope and resolves
its edges. Supertype edges are resolved first so that call resolution can walk a
real chain.

Every edge records which rung it landed on:

| Resolution | Meaning |
| --- | --- |
| `EXACT` | The declaring type was pinned down through the file's scope, and exactly one member matched. |
| `UNIQUE_NAME` | Scope did not decide it, but exactly one symbol in the whole workspace carries that name. |
| `AMBIGUOUS` | Several did. `dst_id` is `None` and `candidate_count` says how many. |
| `UNRESOLVED` | Nothing did. The target is in the JDK or a third-party jar. |

A guess is never upgraded to `EXACT`.

An external tool that already knows the answer can replace this ladder for the
files it covers: see **[the graph facts overlay](graph-facts.md)**, which is the
documented file format any analyzer can write and `zemble graph facts status`
reports on. Every edge records a `source` saying which side produced it.

**Type names** climb: same file, then explicit import, then same package, then
wildcard imports, then the workspace by simple name. An explicit import that
names a type outside the workspace ends the climb at `UNRESOLVED` rather than
falling through to a same-named workspace type, because the import already said
which type was meant.

**Calls** climb: the receiver's type, then static imports, then the workspace by
name and arity. The receiver's type is known when the receiver is `this`, is
`super`, is absent, is written as a type name (`Helpers.twice(...)`), is a
`new` expression, is a cast, or is a variable whose declaration the extractor saw
in the same file: a parameter, a local, a field, a for-each binding, a
try-with-resources binding, a catch binding or an `instanceof` pattern. Once a
type is known, the search walks its resolved supertype chain and stops at the
nearest declaring type.

**Constructors** resolve by arity within the resolved type. A type that declares
no constructor resolves `new Foo()` to the type symbol itself, which is where the
implicit constructor lives.

**Overrides** are derived, not written: for each method, the supertype chain is
walked for a method with the same name and arity. Those edges are graded
`UNIQUE_NAME` because parameter types were never compared.

### Honest limits

These are real, not hypothetical:

- **No type inference.** A call on a chained expression (`a().b()`), on a `var`
  local, on a generic type variable, or on an element pulled out of a collection
  has no known receiver type and falls to the workspace-wide by-name rung.
- **Overloads are matched by name and arity only.** Two overloads with the same
  arity and different parameter types are `AMBIGUOUS`, and an override edge to
  the wrong one of a same-arity pair is possible.
- **Generics are erased.** `List<String>` and `List<Integer>` are the same
  receiver type; wildcards and bounds are ignored.
- **Inherited nested types are not in scope.** A subclass referring to a nested
  type it inherits resolves by the workspace rung, not by scope.
- **A varargs call whose argument count differs from the declared arity** does
  not match, and falls through to the by-name rung.
- **Two local classes with the same name in two overloads of one method** share a
  symbol id.
- **Files of a language without a bundled grammar are skipped**, counted per
  language, and named in the note a query prints when it has nothing to say.

## Languages

Every grammar `semble_grammars` ships except Java has a `LanguageSpec` in
`src/zemble/languages/catalog.py`: which node kinds declare a type, a namespace, a
callable, a field or a constant, where their name, body, parameters and supertypes live
(as small path expressions over the tree, `src/zemble/languages/paths.py`), which nodes
are calls and how their callee splits into receiver and name, and the conventions the
grammar does not spell out (`self` names, constructor names, whether a capitalised call
constructs, whether a leading underscore or a lowercase initial hides a declaration).
`src/zemble/graph/generic.py` reads any spec into the same symbols and edges the Java
extractor emits, so the resolver, the outline and the tools never know a second reader
exists. `tests/test_languages.py` fails the build when a spec names a node kind its
grammar lacks, or when a bundled code grammar has no spec.

What differs from the Java lane:

- **Every file is a `MODULE` symbol** holding its top-level declarations, qualified by the
  package header where the language has one (`app.core.Point` in Kotlin, `store.helper`
  in Go) and by the file's own dotted path otherwise (`src.zemble.graph.store.build_graph`).
  A namespace with a body (`namespace App.Core {`, `mod inner {`, `defmodule Store.Point`)
  names itself in full.
- **Resolution stays inside a language family.** `jvm` (Java, Kotlin, Scala, Groovy),
  `js` (JavaScript, TypeScript, TSX) and `c` (C, C++, Objective-C) resolve into each
  other; everything else only into itself. An unqualified call from inside a method reaches
  the file's own functions before the workspace, and an import that names a path the
  graph cannot see (`from pkg.mod import Base` for `src/pkg/mod.py`) matches by suffix.
  Only Java treats an import as authoritative.
- **Tests are recognised by file name too**: `test_x.py`, `x_test.go`, `x.test.ts`,
  `x_spec.rb` and their kin make a file a test source, its module a `TESTS` edge to the
  module it names, and every resolved reference out of it an `EXERCISES` edge.
- **Overrides are name and arity only**, walked through `EXTENDS`/`IMPLEMENTS` edges the
  same way, so `impl Trait for Type` and `class Circle(Shape)` both yield them.
- **No compiler facts**: the facts overlay is Java-only, so other languages keep their
  tree-sitter grades.

Adding a language is adding one spec to the catalog (parse a fixture and print the tree;
never guess node kinds) and one sample under `tests/fixtures/polyglot_samples/`.

## Hawkeye templates

A `.hwk` file is HTML carrying `{% ... %}` statements. The extractor
(`zemble/graph/hwk.py`) is **lexical, not a parser**: a component file wraps its
whole markup in one `{% tag PascalName { ... } %}` block, so the delimiters nest
and no grammar in the bundle can read them. It reads the facts the graph needs off
the text (`zemble/hwk.py`), and nothing else.

**Symbols.** One `TEMPLATE` per file, plus one `BLOCK` per `{% block "name" %}`.
A file that declares exactly one custom element **is** that element: its
qualified name is the tag (`pl-button`), so `zemble graph definition pl-button`
finds it. A file that declares none is named by its path. A file that declares
several - Hawkeye hoists tags globally, so `tabs.hwk` declares five - keeps the
path-named file symbol and gains one `TEMPLATE` per tag, and references land on
the element that owns those lines rather than on the file.

The tag itself is derived exactly as the compiler derives it
(`TypeUtils.toKebabCase`): `PlTabsTrigger` -> `pl-tabs-trigger`.

**Edges** from a template:

| Written | Edge | Resolves to |
| --- | --- | --- |
| `{% extend "zenitcms:shell" %}` | `EXTENDS` | the parent template |
| `{% render "zenitcms:nav-item" %}` | `IMPORTS` | the rendered partial |
| `<pl-button>` | `REFERENCES_TYPE` | the class or template declaring that tag |
| `String.presence(x)`, `t("add")` | `CALLS` | the `@HawkeyeFunction` method |

**The ladder, for each of those:**

*Template ids* are `namespace:path/below/templates`. The namespace is a Gradle
setting no single file can see, so it only narrows: a path that is unique in the
workspace is `UNIQUE_NAME`, and a path whose repository directory also agrees with
the written namespace (`zenit-cms` -> `zenitcms`, and a source set may append, as
in `plumage-browsertest`) is `EXACT`.

*Element tags* resolve to a `@HawkeyeCustomElement`-annotated class first and to
the declaring template second. A single hit is `EXACT`, because a tag is a global
registration key the compiler refuses to let two declarations share.

*Calls* resolve **only** against `@HawkeyeFunction` methods - nothing else in the
workspace is callable from a template, so a same-named plain Java method is never
a fallback. `namespace` plus `name` from the annotation matching what the template
wrote is `EXACT`; a name match alone is `UNIQUE_NAME`; several overloads sharing
one key are `AMBIGUOUS`. Arity is deliberately not compared: a template function's
Java method may take a leading `RenderContext` the call site never writes.

*Calls can be **exact facts** instead.* When a javac emitter has written facts about
the class a template was compiled into, those facts are mapped back onto the
template through Hawkeye's source maps and REPLACE the extracted call edges of that
template: the compiler already knew which method each call site reaches, and the
edge is then graded `EXACT` with `source = zemble-javac-facts` and the generated
member it was written about kept on the edge. A template's other edges - what it
extends, renders and references - stay the extractor's, because the generated class
knows nothing about them. The mapping, its freshness rule and its limits are in
`docs/graph-facts.md`.

### Honest limits

- **A tag's region ends where the next one begins.** Its closing `} %}` cannot be
  found lexically, so in a multi-element file the last declaration owns the rest
  of the file.
- **`Foo.bar(x)` is ambiguous in the language itself.** Hawkeye parses it as plain
  member access and only decides at transpile time whether `Foo` is a namespace or
  a local. The extractor records every one of them as a call; the ones that were
  member access simply find no `@HawkeyeFunction` and stay `UNRESOLVED`.
- **A registration written through a constant is invisible.** `annotation_args`
  keeps literals only, so `@HawkeyeCustomElement(tag = Microcopy.WRAPPER_TAG)` -
  two of the three such classes in the javaweb workspace - registers no tag here.
- **`{% render field.templateId %}`** names a template only at runtime, so it is
  recorded as a call, not as an include.
- **A tag's `extends` clause** (`tag PlTabsTrigger extends PlTabsMember`) is not
  an edge; only a template's `{% extend %}` is.
- **Style blocks are dropped** before scanning, so a `.hwk` never contributes SCSS
  identifiers - and never a `var(...)` read as a function call.
- **Blocks are structure, not behaviour**: a `BLOCK` symbol carries no edges of its
  own; what a block writes is attributed to the template.

Measured on the javaweb workspace: 619 templates yield 1,033 `TEMPLATE` and 124
`BLOCK` symbols and 7,806 edges - 4,294 element references of which 98.7 % are
`EXACT`, 3,291 calls of which 74 % land on one method and 21 % stay `UNRESOLVED`
(that last figure is mostly bare `name(...)` text that was never a template
function), and every one of the 106 `EXTENDS` and 115 `IMPORTS` edges resolved.
Extraction is cheap because it is only a scan: all 619 templates are read,
scanned and turned into symbols and edges in 0.33 s, single process.

## Storage and incremental updates

Sqlite in the index cache folder, created on demand: the graph is buildable with
no search index present (`zemble/graph/store.py`). The folder holds versions,
`graph-<n>.sqlite`, and a pointer, `graph.current`, naming the current version on
its first line and the one it replaced on its second. Tables:
`symbols`, `edges` (each carrying the `source` that produced it; see below), `files` (path,
mtime, size, package, imports), `decl_keys` (the Hawkeye registration keys a symbol
declares - element tag, template tag, template id, function name and namespaced
function key), `facts_symbols` (the `symbol` facts of every facts file, so mapping
one file can reach the others without parsing them), `facts_status` (one row per
facts file, see [the facts overlay](graph-facts.md)) and `meta` (format version,
root, covered languages, skipped languages). Format version 7: an ambiguous edge stores
how many symbols it could mean, not their ids, because nothing ever read the list and on
javaweb it was 1.6 GB of a 2.6 GB store (0.9 GB after the migration). Format version 8:
an edge spelled its source and destination symbol ids and its file as text, ~400 bytes a
row, and three indexes copied those strings again. Each id and file is now stored once in
`refs` and `edge_rows` holds its integer key; `edges` is a view with the old columns, so
every reader keeps its SQL and the plan reaches `edge_rows` through its integer indexes.
The zenit workspace graph went from 2.1 GB to 0.63 GB, and its adjacency and `dst_name`
queries got faster with it (fewer pages). Interned ids no edge uses any more are pruned
when a store is compacted. A column a graph
built by an older zemble lacks is added on the next open and `decl_keys` is filled
in one pass over the symbol table, so a version-4 graph is migrated rather than
rebuilt; a graph is derived data either way. A format-7 graph is converted on its next
writable open (20 s and 48 MiB for the 2.1 GB one) and compacted by that build.

### Durability

The store is written under `journal_mode=WAL` + `synchronous=FULL`: the log stays
on disk and every commit fsyncs, so a build that is killed mid-write leaves a file
sqlite recovers rather than a torn one. It was written under `synchronous=OFF` +
`journal_mode=MEMORY` until 2026-09-04, and three OOM kills tore the javaweb store:
1.28 GB of pages reachable from neither a btree nor the freelist, an `edges` tree
with out-of-order rowids, 1.26 M of its 1.4 M edges gone, and 4,751 "database disk
image is malformed" lines the daemon logged and ignored while it kept serving what
was left.

A build that reads the whole workspace anyway - a cold build, `--force`, or the
rebuild a malformed store triggers - writes a **new version** with the next free
number and publishes it by rewriting the pointer (a fsynced temp file renamed over
`graph.current`), so a kill leaves the previous version current and untouched, and
the new file is compact by construction. No database file is ever renamed over
another, which is what makes WAL safe here: a `-wal` left pointing at a replaced
inode would be a torn read. From 2026-09-04 to 2026-10-01 the store was one
`graph.sqlite` in rollback-journal mode, renamed into place, and that mode made
every reader wait on a refresh's transaction once it spilled past sqlite's page
cache: "database is locked" after 5 s while the daemon re-resolved under heavy
compiles.

Readers resolve the pointer and open that version read-only (`mode=ro`, 30 s busy
timeout); in WAL they read the last committed snapshot and never wait on the
writer. Each reader first takes a shared `flock` on the version's `-readers` sidecar
and keeps it until it closes; the lock is on a sidecar, never on the database,
because closing any descriptor of a sqlite file drops every POSIX lock sqlite holds
on it in that process. Only one process writes: every build holds an `flock` on
`graph.lock` (the kernel drops it when the holder dies, so a killed build leaves no
stale lock). `build_graph` waits for it; `refresh_graph`, the once-per-process
refresh `ensure_graph` runs, skips when it is held and reads the current version,
because the holder is already refreshing. The daemon's watcher calls `build_graph`,
so its change set lands after any other writer.

Under the lock, each build first retires every version but the current one that it
can lock **exclusively** - a killed build's half-written version, or a replaced one
no reader holds any more - deleting the database, its sidecars and the readers lock
last. A replaced version a reader still holds stays, named on the pointer's second
line, until the first writer after its release; it lives exactly as long as it is
read, rather than "one install". A reader whose version was retired between reading
the pointer and taking its lock sees the sidecar's inode change (or the database
gone) once the lock is granted, and reads the pointer again.

A single-file `graph.sqlite` left by an older zemble is moved into a version by the
first reader or incremental build that finds no pointer, without a rebuild (a
forced build ignores it). When no process has it (or a journal of it) open - judged
from `/proc/<pid>/fd`, never by opening it - it is **renamed** and sealed as the
first WAL version, so no second copy is left. While a process still running the
older code holds it, it is copied through sqlite's backup API instead, because that
process opens it by name in rollback-journal mode and turning the held inode into a
WAL database would give one database two journals; the original is then deleted by
the first writer that finds it unheld. Where `/proc` cannot answer, it counts as
held.

### Sub-directories

A graph request for a directory with no graph of its own is answered from the
nearest ancestor that has one, filtered to the sub-directory, the way search answers
from an ancestor index (`resolve_graph_root`, `SubtreeGraphProvider`). Precedence is
search's: a graph of exactly that path, then the nearest ancestor's, else a graph
of its own. `ensure_graph` refreshes the ancestor and builds nothing for the
sub-directory; a sub-directory the ancestor's walk skips (ignored, a nested repo it
excludes) is not covered and gets its own graph. The view speaks ids and paths
relative to the sub-directory and answers only with in-folder symbols, so its
answers equal what a graph of the sub-directory gives (pinned symbol by symbol by
`tests/test_graph_subtree.py`), with one difference by design: resolution and walks
run over the whole ancestor graph. A call into a sibling module resolves against its
real target instead of staying unresolved or landing by name on an in-folder
namesake, and a hierarchy chain that passes through a sibling module (`Leaf extends
Middle` in another module, `Middle extends Base`) reaches in-folder subtypes a
module-only graph cannot see. `graph facts status` on a sub-directory reports the
ancestor's graph, which owns the facts.

An incremental refresh writes **in place** in the current version: copying a
1.9 GB store per saved file would cost more than the refresh, and the WAL makes it
safe. What it cannot do is give freed pages back, so a refresh that leaves more
than a quarter of the store on the freelist is followed by a `VACUUM INTO` the next
version, published the same way (`_compact_if_drifted`; stores under 4096 pages
are left alone). `zemble graph compact` reclaims anything over 2% free.

A store sqlite calls malformed, or a pointer naming no version that exists, is
never repaired and never read past. `build_graph` catches it, says so at ERROR
naming the folder, and rebuilds from source into a new version that replaces the
malformed one outright, reporting `rebuilt_from_corruption` on the build
statistics; `graph_exists` is false for it (and logs), and the daemon's watcher
keys off `graph_present` - the pointer or the legacy file - so a torn store is the
one it keeps driving rather than the one it quietly stops refreshing.

A rebuild re-extracts only files whose mtime or size changed, then re-resolves
(a) those files, (b) their **dependents**: every file holding an edge whose
written destination name now points somewhere else, and (c) the files whose facts
coverage moved.

That second part is narrower than "every file mentioning a name declared in a
changed file", deliberately. A common method name such as `of` is written in
thousands of files, so the broad rule turns every save into a full re-resolve.
What actually invalidates a resolution is the name-to-symbol mapping changing,
so that is what is compared: name to declaring symbol ids, before against after,
both sides read straight off `symbols(file_path)` and `edges(dst_name)`. A rename,
a move between packages and a file rename all change it; re-saving a file does not.
On the javaweb workspace, touching `PageWindow.java` re-resolves one file, while
renaming `PageWindow.of` re-resolves about 2600.

### Nothing more than the targets need

A refresh of one file must not read the workspace. Four things used to, and none
of them does any more.

**The symbol tables the resolver runs on** live behind `SymbolLookup`
(`zemble/graph/lookup.py`). `SqliteLookup` answers each question with an indexed
query and caches what it touched: by id, by qualified name, by simple name, by
container, by file, by Hawkeye registration key, and the resolved supertypes of one
type. `MemoryLookup` builds every dictionary in one pass over the symbol list; builds
never use it, because on the zenit workspace it cost ~340 MiB and bought no speed over
the indexed lookup.

**The build itself is batched**, so its memory follows a batch, not the workspace.
Changed files are extracted `_BATCH_FILES` (100) at a time; their symbols go into
the store and their unresolved edges into a scratch database beside it
(`graph-scratch.building-<pid>`, attached for the build and deleted after). The
stored edges of every other file to re-resolve move there too. The supertype edges
of all targets are then resolved and indexed at once - a call chain can climb
through any of them - and everything else is resolved, derived and staged a batch at
a time, the lookup forgetting its caches between batches. The staged result is
copied into `edges` last. A full build of the zenit workspace (12,458 files, 2.06M
edges) peaks at ~190 MiB of private memory, where holding every extraction and the
whole symbol table took 2.2 GiB; it is ~10% slower. `tests/test_graph_incremental.py`
runs its journeys one file per batch and as one batch, against a full rebuild.

Reading the supertype map out of `edges` rather than out of a list handed to the
resolver is what makes it work: no re-resolved file's edges are in `edges` until
every batch is done, so what the table holds meanwhile is exactly the workspace
minus the edges about to be rewritten. The scratch database is attached before the
build's first write and the build commits when it is detached, so a refresh stays one
transaction; a failed one is rolled back.

**The facts files** are read only when something must be mapped, and then only the
ones that must. `plan_facts` decides that from `stat` alone: a facts file moved
when its own bytes moved or it vanished, and a template it mapped facts onto is
invalidated on its own when its modification time crosses the generated class it
was compiled from - a verdict the graph records per generated source and re-decides
with two stats. A facts file that did not move is asked only for the sources the
build is re-resolving, so a `.java` edit whose facts have just gone stale reads one
facts file, learns the sha no longer matches, and maps nothing at all.

Because a build maps only part of a facts file, what a facts file contributed is
accounted per SOURCE (`contributions` in its status row), never as one total: the
sources this build did not map keep the accounting they already had, so
`zemble graph build --stats` reports the whole truth however little was read.

**The `symbol` facts of the other facts files** live in `facts_symbols`. A ref
written in one facts file can be answered by a `symbol` fact in another, so mapping
needs all of them; parsing every facts file to answer a handful of refs is what an
incremental build must not do, and the table answers one ref at a time instead.

**The workspace walk** happens only when nobody named the change set. `build_graph`
takes `changed_paths` from the daemon's watcher, which already promises to name
every path that moved - facts files included, since the watcher matches them
explicitly - so the graph's own file record plus that change set replaces both the
source walk and the `**/build/zemble/*.jsonl` discovery walk. The two walks are
about 2 s of a javaweb refresh, and the CLI still pays them.

### Facts hierarchy is indexed before calls are resolved

A file whose facts own its supertypes contributes the tool's `EXTENDS` and
`IMPLEMENTS` edges to the hierarchy the call resolver walks, not the extractor's.
Resolving calls against the extractor's guesses and then storing the tool's edges
left a chain the stored graph did not have, which showed up as a cold build
grading a call `AMBIGUOUS` where a refreshed graph had it `EXACT`. Both now agree,
and the agreement is what the identity journeys check.

### It lands where a full rebuild would

Every narrowing above is a chance to leave a stale edge behind, and none of them is
visible in a count, so `tests/test_graph_incremental.py` applies a sequence of edits
- a rename, a move between packages, a deletion, a new caller, a template touch, a
template edit, a facts file appearing, changing and vanishing - and after each one
compares the whole `symbols`, `edges` and `decl_keys` tables against a from-scratch
build of the very same tree.

That found one real defect, which is fixed: a file the facts covered had its
extracted edges REPLACED in the table, so when its facts later vanished there was
nothing to re-resolve from but degraded copies of the fact edges. A target file
whose facts coverage moved is now re-extracted rather than reloaded.

The same check on the javaweb workspace - cold build, then seven edits applied
incrementally, then a fresh cold build of the resulting tree - reports zero rows
differing in either direction, for symbols and for edges.

### Measured

javaweb workspace (6.2k Java files, 1.6k templates, 35 facts files, 102k symbols,
~958k edges, 10 cores, warm page cache). "Change set" is the daemon's lane, where
the watcher names what moved; "walk" is the CLI's, where nothing does.

| Lane | Before | After |
| --- | --- | --- |
| Cold build | 64.9 s | 42.4 s |
| No-op rebuild (walk) | 38.2 s | 2.6 s |
| No-op rebuild (change set) | 37.4 s | 0.6 s |
| One `.java` edit (walk) | 37.6 s | 2.8 s |
| One `.java` edit (change set) | 38.0 s | 0.9 s |
| A method renamed in that file | 40.0 s | 2.4 s |
| One `.hwk` edit | 39.6 s | 0.6 s |
| One `.hwk` edit, template covered by facts | 40.1 s | 1.8 s |
| One facts file rewritten | 41.5 s | 5.5 s |
| A file deleted | 38.9 s | 0.6 s |

The before column has no cheap row because the old build had no cheap path: it read
every symbol, every hierarchy edge and every facts file on every build, and
re-resolved all 139 fact-mapped templates even when nothing had moved.

The daemon's `graph_ms` is exactly the change-set column: `_refresh_graph` hands the
watcher's paths straight to `build_graph`, so an edit is in the graph one 500 ms
debounce plus that number after it is saved.

The walk lane costs the source walk (0.6 s) plus the facts discovery walk (1.4 s),
the latter because `**/build/zemble/*.jsonl` can prune nothing - `**` keeps every
directory alive - so it descends the whole tree. A facts file rewrite is the
expensive lane by design: there is no smaller unit than the facts file it rewrote, and
every file it covers is re-extracted and re-resolved.

Mapping facts is bounded the same way the build is. Facts files are parsed, mapped and
released one at a time (a moved one twice: once for its `symbol` facts, which every
other file's refs may need, once to map it), refs map through a `SqliteLookup` that
forgets after every facts file, the fact edges wait in the scratch database
(`overlay_edges`) and are read back per batch, and skipped facts are counted by what
they are reported under rather than kept one record each. With every facts file of
the zenit workspace moved on archdev (163 files, 795 MB of JSONL, 903k fact edges,
8,274 files re-resolved) the refresh peaked at 3,388 MiB of private memory before and
338 MiB after, in about the same time (348 s before, 384 s after on a busy host); the
scratch database grew to ~2.2 GB on disk meanwhile.

Two constants were worth more than any of the structure. `_relative_to_workspace`
called `Path.resolve` - a realpath syscall - once per fact line, 1.4 M times per
build; memoising it took reading the workspace's facts files from 31 s to 1 s.
And `symbol_from_row` decodes four JSON columns per symbol, so it decodes them with
orjson: 2.4 s to 1.2 s for the whole table.

Queries answer off the indexes on `symbols(name)`, `symbols(qualified_name)`,
`edges(dst_id, kind)`, `edges(src_id, kind)` and `edges(dst_name, kind)`: a name
lookup is about 2 ms, a hierarchy or tests-of query is well under 1 ms, and the
worst case measured - `callers` of `Model.save`, 682 hits - is about 11 ms.

Extraction runs in a process pool using the `fork` start method where the
platform has it, because the alternatives re-import the host's `__main__` and so
break an embedded or piped interpreter. Any pool failure falls back to extracting
in the calling process.

## CLI

```
zemble graph build <path> [--stats] [--force] [--json]
zemble graph compact

zemble graph facts status <path> [--json] [--limit N]

zemble graph definition      <path> <symbol> [--json]
zemble graph callers         <path> <symbol> [--json]
zemble graph callees         <path> <symbol> [--json]
zemble graph references      <path> <symbol> [--json]
zemble graph implementations <path> <symbol> [--json]
zemble graph supertypes      <path> <symbol> [--json]
zemble graph overrides-of    <path> <symbol> [--json]
zemble graph overridden-by   <path> <symbol> [--json]
zemble graph tests-of        <path> <symbol> [--json]
zemble graph neighbors       <path> <symbol> [--hops N] [--kinds KIND ...] [--json]
```

`<symbol>` is a simple name (`PageWindow`), a qualified name
(`be.elevenways.zenit.common.data.PageWindow`) or `Type.member`
(`PageWindow.of`). A query builds the graph if none exists yet. `compact` brings every
graph in the cache to the current format and gives freed pages back, under each graph's
writer lock; a graph is otherwise only migrated when something next writes it.

Exit codes: `0` answered, `1` no such symbol, `2` the name is ambiguous, in which
case every candidate is listed on stderr so the next call can be qualified.

```
$ zemble graph callers ~/projects/javaweb PageWindow.of
PageWindow.of  [method]  zenit/src/common/java/.../PageWindow.java:32
callers: 8 result(s)
  zenit/src/common/java/.../StaticDataProvider.java:61  called from StaticDataProvider.load (line 61, exact match)
  zenit/src/server/java/.../RecordSourceHandlers.java:164  called from RecordSourceHandlers.handleQuery (line 164, exact match)
  ...
```

## MCP

Six tools are registered on the existing zemble MCP server:
`graph_definition`, `graph_callers`, `graph_implementations`, `graph_overrides`
(the method-level counterpart of `graph_implementations`: every override of one
`Type.member`), `graph_tests_of` and `graph_neighbors`. Each takes `symbol` and
an optional `repo` (defaulting to the server's start directory, announced in the
parameter description), builds the graph on first use and refreshes it once per
server process, and returns JSON. An ambiguous name comes back as an `error`
with a `candidates` list rather than as a failure. Every answer is capped at
`limit` results (default 50) and carries `total`, plus a `truncated` note when
the cap bit, so a cap is never silent. A wire command outside the query
vocabulary fails closed with an `error` instead of reaching the provider.

Neighbor and hierarchy walks enforce the cap during traversal, with one lookahead
hit. A stopped walk sets `total_exact: false`: `total` is then a lower bound, and
`truncated` explicitly says "at least". Subtree walks still cross sibling modules,
but bound visited symbols to `max(4096, (limit + 1) * 32)` so filtering cannot
materialize the whole ancestor graph before finding an in-folder answer.

## The provider seam

`GraphProvider` (`zemble/graph/provider.py`) is a `Protocol` holding
`definition`, `callers`, `callees`, `references`, `implementations`,
`supertypes`, `overrides_of`, `overridden_by`, `tests_of` and `neighbors`. It
mentions no sqlite and no tree-sitter types.

`SqliteGraphProvider` is the implementation shipped here. The point of the seam
is that a compiler-grade provider - `zenit-dev` handing over what javac already
knows - answers the same questions with `EXACT` where this one says
`UNIQUE_NAME`, and drops in without any consumer changing.

Every answer is a `Hit`: the symbol at the other end, the edge kind, the line,
the resolution and a one-line `reason` such as
`called from RecordSourceHandlers.handleQuery (line 164, exact match)`.
Hierarchy and neighbour walks also set `depth`.
