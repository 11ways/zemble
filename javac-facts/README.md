# zemble-javac-facts

A standalone javac plugin that writes down what the Java compiler already knows: every declared
symbol, every call with the callee javac actually selected, every override, every supertype edge,
and every annotation with constant arguments. The output is JSONL in zemble's graph-facts format,
which a reader can turn into a code graph without re-implementing Java name resolution or overload
resolution.

Nothing about it is project-specific: it only uses the documented `com.sun.source` API, needs no
`--add-exports`, and runs on any JDK 17 or newer with any build tool.

## Build

```bash
make build      # -> build/zemble-javac-facts.jar
make test       # compiles test/fixtures with the plugin and checks the facts
make clean
```

The jar is compiled with `--release 17`, so it runs on JDK 17+.

## Use

The jar must be on javac's *processor* path (that is where javac looks for `-Xplugin` service
providers), and the plugin is switched on with `-Xplugin:`.

Plain javac:

```bash
javac -processorpath zemble-javac-facts.jar \
      -Xplugin:"ZembleFacts out=build/zemble/facts.jsonl root=." \
      -d out $(find src -name '*.java')
```

Gradle:

```groovy
dependencies {
    annotationProcessor files("libs/zemble-javac-facts.jar")
}

tasks.withType(JavaCompile).configureEach {
    options.compilerArgs += ["-Xplugin:ZembleFacts out=${buildDir}/zemble/facts.jsonl root=${rootDir}"]
}
```

Maven:

```xml
<plugin>
  <artifactId>maven-compiler-plugin</artifactId>
  <configuration>
    <annotationProcessorPaths>
      <path>
        <groupId>io.zemble</groupId>
        <artifactId>zemble-javac-facts</artifactId>
        <version>0.1.0</version>
      </path>
    </annotationProcessorPaths>
    <compilerArgs>
      <arg>-Xplugin:ZembleFacts out=${project.build.directory}/zemble/facts.jsonl root=${maven.multiModuleProjectDirectory}</arg>
    </compilerArgs>
  </configuration>
</plugin>
```

If a build declares annotation processors explicitly, the plugin still needs to be on the processor
path but never runs as a processor; `-proc:none` is fine.

## Options

Options are space-separated `key=value` pairs inside the `-Xplugin:` value.

| Option | Default | Meaning |
| --- | --- | --- |
| `out` | `build/zemble/facts.jsonl` | Output file, resolved against the compiler's working directory. |
| `root` | the working directory | Emitted `path` values are relative to this, with forward slashes. |
| `append` | `false` | Keep whatever the file already holds instead of truncating it on the first write. |

With `append=false` the file is truncated once per JVM, on the first write, so a javac run that
calls the plugin over several rounds or many compilation units still produces one file with one
header. With `append=true` a header is written only when the file is empty, so several javac
invocations can fill one facts file (a multi-module build).

Any failure inside the plugin is caught: one line is printed to stderr as
`zemble-javac-facts: <msg>` and compilation continues. Whatever was written before the failure is
still valid JSONL.

## Output

Line 1 is the header:

```json
{"zemble_facts":1,"tool":"zemble-javac-facts","tool_version":"0.1.0","generated_at":"...","language":"java","root":"/abs/path"}
```

Then one fact per line. A `file` fact always precedes the facts about that file:

```json
{"t":"file","path":"demo/Demo.java","sha256":"b85cfa..."}
{"t":"symbol","ref":"demo.Demo#bar(int)","path":"demo/Demo.java","line":41,"kind":"method"}
{"t":"call","from":"demo.Demo#overloadCalls()","to":"demo.Demo#bar(java.lang.String)","path":"demo/Demo.java","line":72}
{"t":"override","from":"demo.Demo#greet(java.lang.String)","to":"demo.Greeter#greet(java.lang.String)"}
{"t":"extends","from":"demo.Demo$1","to":"demo.Base"}
{"t":"implements","from":"demo.Demo","to":"demo.Greeter"}
{"t":"annotation","ref":"demo.Demo","name":"demo.Marker","args":{"value":"demo","count":3,"level":"HIGH"}}
```

Ref grammar:

- Types: `pkg.Outer.Inner`, dotted source-style for nested types. Anonymous and local classes (and
  anything nested inside one) use javac's flat name instead: `pkg.Outer$1`, `pkg.Outer$1Local`.
- Methods: `pkg.Outer.Inner#name(erased.Param1,erased.Param2)` -- fully qualified erased parameter
  types, no return type, arrays and varargs as `Type[]`, type variables erased to their bound.
- Constructors: `pkg.Type#<init>(...)`.
- Fields: `pkg.Type#fieldName`.

Calls into the JDK and into libraries are emitted with the same grammar
(`java.util.List#add(java.lang.Object)`); the reader decides which targets are inside its
workspace.

### How some cases are resolved

- A call is attributed to the enclosing method or constructor. Code in a **lambda** belongs to the
  method containing the lambda; code in an **anonymous class** belongs to that anonymous class's own
  method (`pkg.Outer$1#run()`).
- Static field initialisers and `static {}` blocks are attributed to `pkg.Type#<clinit>()`; instance
  field initialisers and instance initialiser blocks to `pkg.Type#<instance-init>()`.
- A **method reference** (`Foo::bar`) is emitted as a `call` at its own position.
- `new Anon() {...}` emits the constructor javac selected, which for an anonymous class is the
  anonymous class's own constructor (`pkg.Outer$1#<init>(int)`), and the anonymous constructor then
  calls the superclass one.
- `extends` to `java.lang.Object` is not emitted; `java.lang.Enum`, `java.lang.Record` and
  `java.lang.annotation.Annotation` are, since they are the compiler's answer for those kinds.
- An interface listing super-interfaces emits them as `extends` (the source keyword); a class emits
  them as `implements`.
- `override` is emitted per supertype branch: the first overridden declaration found walking up each
  direct supertype.
- Annotation arguments are only emitted when constant: strings, numbers, booleans, chars (as
  strings), enum constants (as their simple name), and arrays of those. Class literals and nested
  annotations are skipped, as are defaults the source did not spell out.
- Implicit record members (accessors, `equals`, `hashCode`, `toString`) do not exist yet at the
  point the plugin runs, so only the record components (as fields) and explicitly declared members
  are emitted.

## Reader side

The consuming format, and what zemble does with these facts, is documented in zemble's
`docs/graph-facts.md`.
