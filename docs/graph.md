# Java symbol graph

`zemble graph` builds a symbol graph over a Java workspace and answers
relationship questions about it: who calls this, who implements this, what tests
cover this. It is stored beside the search index and updated incrementally.

The graph is deliberately not a compiler. It is a tree-sitter extractor plus a
name resolver, and every answer it gives carries a grade saying how much it
actually knows. Use it to navigate; do not use it as proof.

## The model

Two tables of things, both language neutral (`zemble/graph/model.py`).

**Symbols** are declarations. Kinds: `PACKAGE`, `CLASS`, `INTERFACE`, `ENUM`,
`RECORD`, `ANNOTATION`, `METHOD`, `CONSTRUCTOR`, `FIELD`, `ENUM_CONSTANT`.

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
`container_id`, `modifiers`, `annotations`) a symbol carries `is_test`, which is
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
| `AMBIGUOUS` | Several did. `dst_id` is `None` and `candidates` lists them all. |
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
- **`.hwk` templates are not covered at all.** A call made from a Hawkeye
  template is invisible to the graph, so `callers` of a `@HawkeyeFunction` under
  reports. That is the next extractor, not a bug in this one.
- **Non-Java files are skipped**, counted per language, and named in the note a
  query prints when it has nothing to say.

## Storage and incremental updates

Sqlite at `<index cache folder>/graph.sqlite`, created on demand: the graph is
buildable with no search index present (`zemble/graph/store.py`). Tables:
`symbols`, `edges` (each carrying the `source` that produced it), `files` (path,
mtime, size, package, imports), `facts_status` (one row per facts file read, see
[the facts overlay](graph-facts.md)) and `meta` (format version, root, covered
language, skipped languages).

A rebuild re-extracts only files whose mtime or size changed, then re-resolves
(a) those files and (b) their **dependents**: every file holding an edge whose
written destination name now points somewhere else.

That last part is narrower than "every file mentioning a name declared in a
changed file", deliberately. A common method name such as `of` is written in
thousands of files, so the broad rule turns every save into a full re-resolve.
What actually invalidates a resolution is the name-to-symbol mapping changing,
so that is what is compared: name to declaring symbol ids, before against after.
A rename, a move between packages and a file rename all change it; re-saving a
file does not. On the javaweb workspace, touching `PageWindow.java` re-resolves
one file, while renaming `PageWindow.of` re-resolves about 2600.

Measured on the javaweb workspace (6.2k Java files, 10 cores): a full build takes
about 33 s and produces roughly 101k symbols and 923k edges; the zenit repository
alone takes 5.6 s for 19k symbols and 184k edges. Queries answer off the indexes
on `symbols(name)`, `symbols(qualified_name)`, `edges(dst_id, kind)`,
`edges(src_id, kind)` and `edges(dst_name, kind)`: a name lookup is about 2 ms, a
hierarchy or tests-of query is well under 1 ms, and the worst case measured -
`callers` of `Model.save`, 682 hits - is about 11 ms.

Extraction runs in a process pool using the `fork` start method where the
platform has it, because the alternatives re-import the host's `__main__` and so
break an embedded or piped interpreter. Any pool failure falls back to extracting
in the calling process.

## CLI

```
zemble graph build <path> [--stats] [--force] [--json]

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
(`PageWindow.of`). A query builds the graph if none exists yet.

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

Five tools are registered on the existing zemble MCP server:
`graph_definition`, `graph_callers`, `graph_implementations`, `graph_tests_of`
and `graph_neighbors`. Each takes `symbol` and `repo`, builds the graph on first
use and refreshes it once per server process, and returns JSON. An ambiguous
name comes back as an `error` with a `candidates` list rather than as a failure.

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
