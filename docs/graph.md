# Symbol graph

`zemble graph` builds a symbol graph over a Java and Hawkeye workspace and answers
relationship questions about it: who calls this, who implements this, what tests
cover this. It is stored beside the search index and updated incrementally.

The graph is deliberately not a compiler. It is a tree-sitter extractor plus a
name resolver, and every answer it gives carries a grade saying how much it
actually knows. Use it to navigate; do not use it as proof.

## The model

Two tables of things, both language neutral (`zemble/graph/model.py`).

**Symbols** are declarations. Kinds: `PACKAGE`, `CLASS`, `INTERFACE`, `ENUM`,
`RECORD`, `ANNOTATION`, `METHOD`, `CONSTRUCTOR`, `FIELD`, `ENUM_CONSTANT`,
`TEMPLATE`, `BLOCK`.

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
- **Non-Java, non-template files are skipped**, counted per language, and named
  in the note a query prints when it has nothing to say.

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

Sqlite at `<index cache folder>/graph.sqlite`, created on demand: the graph is
buildable with no search index present (`zemble/graph/store.py`). Tables:
`symbols`, `edges` (each carrying the `source` that produced it), `files` (path,
mtime, size, package, imports), `facts_status` (one row per facts file read, see
[the facts overlay](graph-facts.md)) and `meta` (format version, root, covered
languages, skipped languages). A column a graph built by an older zemble lacks is
added on the next open; a graph is derived data either way.

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
