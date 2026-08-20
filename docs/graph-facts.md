# Graph facts overlay

zemble's symbol graph is a tree-sitter extractor plus a name resolver
(`docs/graph.md`). It has no types, so an overload it cannot separate lands on
`AMBIGUOUS` and a call on a `var` local lands on `UNIQUE_NAME`. A tool that
already knows the answer - a javac plugin, a language server, any real front
end - can write those edges to a file, and zemble will prefer them.

This page is the contract. It is what an emitter is written against, so no field
name here changes without the emitters changing with it.

## The format

JSONL, UTF-8, one JSON object per line.

Line 1 is the header:

```json
{"zemble_facts": 1, "tool": "javac-facts", "tool_version": "0.1.0", "generated_at": "2026-08-20T09:00:00Z", "language": "java", "root": "/home/me/projects/app/module"}
```

- `zemble_facts` is the format version. A version this zemble does not read makes
  the whole file be refused; nothing in it is applied half way.
- `tool` is what every edge from this file will carry as its `source`, and what
  queries name in their reason.
- `root` is the directory the `path` fields are relative to. It may be absolute
  or relative to the facts file's own location, and it may be a **subdirectory**
  of the workspace zemble indexed: one Gradle module's compile task writing about
  its own module is the normal case. Paths that land outside the indexed
  workspace are skipped and counted.

Every following line is one fact:

| Line | Meaning |
| --- | --- |
| `{"t":"file","path":"rel/Foo.java","sha256":"<hex>"}` | The facts that follow for `path` were derived from that exact content. |
| `{"t":"symbol","ref":"<ref>","path":"rel/Foo.java","line":N,"kind":"class\|interface\|enum\|record\|annotation\|method\|constructor\|field"}` | Optional. Helps mapping; zemble still owns symbol ids. |
| `{"t":"call","from":"<ref>","to":"<ref>","path":"rel/Foo.java","line":N}` | A resolved call site. |
| `{"t":"override","from":"<ref>","to":"<ref>"}` | |
| `{"t":"extends","from":"<type ref>","to":"<type ref>"}` | |
| `{"t":"implements","from":"<type ref>","to":"<type ref>"}` | |
| `{"t":"annotation","ref":"<ref>","name":"<qualified annotation>","args":{...}}` | Optional; read and grouped, not yet turned into edges. |

A `file` line must precede the facts about that file. A fact that carries no
`path` of its own - `override`, `extends`, `implements`, `annotation` - belongs
to the **most recent** `file` line, which is why the ordering is part of the
contract and not a convenience. A fact before any `file` line is skipped and
counted. An unknown `t` is skipped and counted; it is never an error, so a newer
emitter may add fact kinds without breaking an older zemble.

## The ref grammar

A ref is an opaque string to everything except the per-language `RefMapper`. The
header's `language` picks the mapper, and the mapper is the only language-aware
piece of the overlay. Java is the flavour shipped:

- types `pkg.Outer.Inner`
- methods `pkg.Outer.Inner#name(erased.param.Type1,erased.param.Type2)` -
  fully qualified erased parameter types, arrays and varargs both as `Type[]`,
  no return type
- constructors `pkg.Outer.Inner#<init>(...)`
- fields `pkg.Outer.Inner#fieldName`

Another language defines its own flavour and registers its mapper in
`MAPPER_FACTORIES`; a facts file whose `language` has no mapper is reported and
ignored.

### Java flavour: the parts javac spells differently

These are the shapes the reference emitter (`zemble-javac-facts`) produces, and
what zemble does with each:

- **Anonymous and local types** use javac's flat names: `pkg.Top$1` for an
  anonymous class or an enum-constant body, `pkg.Top$1Local` for a local class,
  numbered per **outermost** class in source order. zemble names an anonymous
  class `<enclosing type>$anon@<line>` and keeps an enum constant's body on the
  constant itself, so the mapping is by order: the numbered declarations under
  `Top` are sorted by line and `$N` picks the Nth. A local class is matched by
  name first, by order only when the name is not unique. When neither lands, the
  ref is reported unmapped with the reason `anonymous flat name` - never guessed.
- **Initializers**: `pkg.Type#<clinit>()` (static initializer, including static
  field initializers) and `pkg.Type#<instance-init>()` (instance initializer and
  field initializers) map to the **enclosing type symbol**. zemble has no symbol
  for either, and field-initializer calls are not tracked per field on this side.
- **`extends java.lang.Object`** is never emitted. `java.lang.Enum`,
  `java.lang.Record` and `java.lang.annotation.Annotation` are, as external
  targets.
- An **interface** emits its super-interfaces as `extends`, a class emits its
  interfaces as `implements`. zemble's own extractor splits them the same way.
- One `override` per direct supertype branch.
- `new Anon(){}` emits a call to `pkg.Top$1#<init>(...)` and then a call from
  that constructor to the superclass constructor. A type that declares no
  constructor - every anonymous class - keeps its implicit one on the type
  symbol, which is where zemble's own resolver puts `new Foo()` too.
- Record components appear as `field` symbols on the record's line; implicit
  record members are not emitted. A call to an accessor - `rec.component()` -
  therefore names a method no source file declares, and maps onto the component
  **field**, which is the only symbol there is and where a reader would look.

### How a ref becomes a symbol

The type is resolved first, then the member is looked up among that type's own
declarations. Overloads are separated by arity and then by the **simple** names
of the erased parameter types, because zemble records parameter types as the
source wrote them (`List<String>` is stored as `List`). Matching is exact:

- The type is unknown to the workspace -> the target is **external**. It is kept
  as an edge with no `dst_id`, the full ref as `dst_name` and resolution
  `unresolved`, exactly like a JDK call the extractor could not resolve.
  `java.util.List#add(java.lang.Object)` is this case.
- The type is known but the member is not, or two members match -> the fact is
  **unmapped**: the edge is dropped, and the ref plus the reason is counted and
  listed by `zemble graph facts status`. A `symbol` fact for the same ref is
  consulted as a second rung (file plus line) before giving up.
- A source ref (`from`) that does not map drops the edge, because there is
  nothing to hang it on.

Generic parameters are the known sharp edge: javac erases `T` to its bound while
zemble stores the type variable as written, so a method whose overloads differ
only in a type variable can end up unmapped. It is reported, not guessed.

## Discovery

Facts files are found under the indexed workspace root by glob:

- `.zemble/facts/*.jsonl`
- `**/build/zemble/*.jsonl`

Both defaults apply when `<root>/.zemble/graph.toml` says nothing. To point
zemble somewhere else, list globs there:

```toml
[facts]
sources = ["**/build/zemble/facts.jsonl", "artifacts/*.jsonl"]
```

A configured list **replaces** the defaults.

The walk uses the file walker's ignored-directory list minus `build/` and
`.zemble/`, and it prunes any directory that cannot still match a glob, so
`**/build/zemble/*.jsonl` costs one `scandir` per build directory rather than a
walk of everything javac ever wrote. `.gitignore` is deliberately **not**
consulted here: generated facts are gitignored by construction, so honouring it
would make the documented convention undiscoverable.

## Freshness

Every `file` line names the sha256 of the content the facts were derived from.
Before applying anything, zemble hashes the file as it is on disk now:

- **fresh** - the hashes match. The file is covered, and its fact edges replace
  the extracted ones.
- **stale** - they differ, or the file is gone. Every fact for that file is
  ignored, and the extracted edges stay. The file is counted and named by
  `zemble graph facts status`.

Freshness is per source file, never per facts file: one edited file does not
invalidate the other 400 in the same emitter output.

## The replacement rule

For a source file with fresh facts, the `CALLS`, `OVERRIDES`, `EXTENDS` and
`IMPLEMENTS` edges originating in that file come from the facts and from nowhere
else. The extracted ones are dropped, and derived overrides are not derived for
that file. Files without fresh facts keep every extracted edge they had. The two
are never mixed for one file, because an answer that mixed them would be one
nobody could grade.

That cuts both ways, and an emitter has to know it: **a file you declare is a
file you own**. If you emit calls but no `implements`, that file has no
`implements` edges in the graph. Emit every one of the four kinds you know about
for the files you declare.

The kinds the overlay does not own - `REFERENCES_TYPE`, `ANNOTATED_WITH`,
`IMPORTS`, `TESTS`, `EXERCISES` - always stay with the extractor.

Two fresh facts files may declare the same source file: a Gradle module compiles
`common`, `server` and `browser` source sets with separate javac tasks, and a
file can be compiled by more than one. Their edge sets are **unioned**, and
identical edges are deduplicated. Nothing has to be coordinated between emitter
tasks.

Every edge carries a `source`: `tree-sitter` for zemble's own extractor,
otherwise the `tool` from the facts header. It is stored, it is on every `Hit`,
it is in the MCP and `--json` output, and it is what a query's reason names:

```
called from Consumer.measure (line 7, javac-facts)
```

## When facts are applied

At graph build time, inside the resolution pass - before overrides, tests and
"exercises" edges are derived, so those are derived from the edges the graph
actually keeps.

A build re-applies facts whenever a facts file appeared, changed or vanished:
every source file that facts file declares (now, or the last time it was read)
is re-resolved, even though not one byte of it changed. What each facts file
contributed last time is kept in the graph's `facts_status` table.

The daemon watches facts files as well as sources: its ignore rules make an
exception for anything matching a facts glob, which they would otherwise skip for
living under `build/` or `.zemble/`. A changed facts file therefore triggers the
same incremental graph refresh a changed `.java` file does. Any `zemble graph`
command re-checks them regardless, so nothing depends on the watcher having seen
the write.

## The status command

```
zemble graph facts status [path] [--json] [--limit N]
```

It builds the graph if needed, then reports:

- every facts file found, its tool, version, language and how long ago it was
  generated
- per file: how many source files it declares, how many are fresh, how many are
  stale and why, plus anything skipped (unknown fact kinds, facts with no `file`
  line before them, files outside the workspace)
- any facts file that was refused, and why
- coverage: fresh files, fact edges, unmapped refs, edges by `source`, and
  `CALLS` edges graded separately for covered files and for the rest - which is
  the number that says what the overlay bought
- the unmapped refs, most frequent first, with the reason each one was not mapped

```
$ zemble graph facts status ~/projects/app
Facts for ~/projects/app: 3 file(s) found
  module/build/zemble/common.jsonl  [javac-facts 0.1.0, java]  generated 4m ago
    118 file(s) declared, 117 fresh, 1 stale
      stale: module/src/common/java/com/example/Edited.java (content changed)
coverage: 117 of 118 declared file(s) fresh, 4210 fact edge(s), 3 unmapped ref(s)
  edges by source: javac-facts 4210, tree-sitter 18332
  calls in covered files: exact 3901, unresolved 309
  calls elsewhere: ambiguous 44, exact 812, unique_name 690, unresolved 120
unmapped refs (top 1):
  3x  com.example.Gen#pick(T)  [calls]  no pick in com.example.Gen
```

## How to write an emitter

1. Write the header first, with your own `tool` name and a `root` the `path`
   fields are relative to.
2. For each source file you have finished analysing, write its `file` line with
   the sha256 of the bytes **you compiled** - not of the file as it is when you
   flush, if you buffered.
3. Write the facts for that file before the next `file` line. Emit all four edge
   kinds you know about: you own the file once you declare it.
4. Use the ref grammar of your language exactly, with erased and fully qualified
   parameter types. If you cannot produce a ref for something, leave the fact out
   rather than writing an approximate ref: an unmapped ref costs the edge either
   way, and a wrong ref costs correctness.
5. Write to one of the discovered locations, one file per compile task is fine,
   and write it atomically (temp file plus rename) so a half-written file is
   never read.
6. Check your work with `zemble graph facts status <root>`: the unmapped list is
   the emitter's test suite.

The reference emitter for Java lives in `javac-facts/`.
