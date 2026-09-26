import io.zemble.javacfacts.FactsPlugin;

import com.sun.source.util.JavacTask;

import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Journey test for the facts file across several compilations in ONE JVM, as a Gradle daemon runs them:
 * an incremental compile merges, a deleted source drops out, a full compile never appends.
 *
 * <p>Usage: {@code java -cp <plugin jar> test/MergeFacts.java <fixture dir> <scratch dir>}.
 */
public final class MergeFacts {

    private static final Pattern FILE_PATH = Pattern.compile("^\\{\"t\":\"file\",\"path\":\"([^\"]+)\"");

    private static final List<String> problems = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        Path fixtures = Path.of(args[0]).toAbsolutePath().normalize();
        Path scratch = Path.of(args[1]).toAbsolutePath().normalize();
        Path root = scratch.resolve("src");
        Path classes = scratch.resolve("classes");
        Path out = scratch.resolve("facts.jsonl");
        Files.createDirectories(root.resolve("demo"));
        Files.createDirectories(classes);
        try (Stream<Path> sources = Files.list(fixtures.resolve("demo"))) {
            for (Path source : sources.toList()) {
                Files.copy(source, root.resolve("demo").resolve(source.getFileName()));
            }
        }
        Path extra = root.resolve("demo/Extra.java");
        Files.writeString(extra, "package demo;\n\n/** Deleted between compilations. */\nclass Extra extends Base {\n"
                + "    Extra() {\n        super(1);\n    }\n}\n");

        // 1. A full compile records every source once, behind one header.
        compile(root, classes, out, all(root));
        List<String> first = paths(out);
        expect(first.size() == 8, "step 1: one file line per source, got " + first);

        // 2. An incremental compile of one source in the same JVM keeps every other block.
        String extraFacts = String.join("\n", linesAbout(out, "demo/Extra.java"));
        Files.delete(extra);
        compile(root, classes, out, List.of(root.resolve("demo/Base.java")));
        List<String> second = paths(out);
        expect(second.contains("demo/Base.java"), "step 2: the recompiled source is recorded");
        expect(second.stream().filter("demo/Base.java"::equals).count() == 1, "step 2: once, got " + second);
        expect(second.contains("demo/Demo.java") && second.contains("demo/Point.java"),
                "step 2: sources the compile did not see keep their facts, got " + second);
        expect(!extraFacts.isEmpty(), "step 2: the deleted source had facts before");
        expect(!second.contains("demo/Extra.java"), "step 2: a deleted source's facts are dropped, got " + second);
        expect(second.size() == 7, "step 2: nothing else lost or duplicated, got " + second);
        expect(linesAbout(out, "demo/Base.java").stream().anyMatch(line -> line.contains("\"t\":\"symbol\"")),
                "step 2: the recompiled block carries its facts");

        // 3. Another full compile in the same JVM rewrites instead of appending.
        compile(root, classes, out, all(root));
        List<String> third = paths(out);
        expect(third.size() == 7 && third.stream().distinct().count() == 7,
                "step 3: every source exactly once, got " + third);
        long headers = Files.readAllLines(out, StandardCharsets.UTF_8).stream()
                .filter(line -> line.contains("\"zemble_facts\"")).count();
        expect(headers == 1, "step 3: exactly one header, got " + headers);

        // 4. No temp file survives a finished compile.
        try (Stream<Path> left = Files.list(scratch)) {
            List<Path> temps = left.filter(path -> path.getFileName().toString().endsWith(".tmp")).toList();
            expect(temps.isEmpty(), "step 4: no temp file left, got " + temps);
        }

        if (!problems.isEmpty()) {
            problems.forEach(problem -> System.out.println("FAIL: " + problem));
            System.exit(1);
        }
        System.out.println("OK: merge journey, " + third.size() + " files");
    }

    private static void compile(Path root, Path classes, Path out, List<Path> sources) throws IOException {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        try (StandardJavaFileManager files = compiler.getStandardFileManager(null, null, StandardCharsets.UTF_8)) {
            Iterable<? extends JavaFileObject> units = files.getJavaFileObjectsFromPaths(sources);
            List<String> options = List.of("-proc:none", "-d", classes.toString(), "-cp", classes.toString());
            JavacTask task = (JavacTask) compiler.getTask(null, files, null, options, null, units);
            new FactsPlugin().init(task, "out=" + out, "root=" + root);
            if (!task.call()) {
                throw new IllegalStateException("compilation failed for " + sources);
            }
        }
    }

    private static List<Path> all(Path root) throws IOException {
        try (Stream<Path> sources = Files.list(root.resolve("demo"))) {
            return sources.filter(path -> path.toString().endsWith(".java")).sorted().toList();
        }
    }

    private static List<String> paths(Path out) throws IOException {
        List<String> paths = new ArrayList<>();
        for (String line : Files.readAllLines(out, StandardCharsets.UTF_8)) {
            Matcher matcher = FILE_PATH.matcher(line);
            if (matcher.find()) {
                paths.add(matcher.group(1));
            }
        }
        return paths;
    }

    private static List<String> linesAbout(Path out, String path) throws IOException {
        List<String> block = new ArrayList<>();
        boolean inside = false;
        for (String line : Files.readAllLines(out, StandardCharsets.UTF_8)) {
            Matcher matcher = FILE_PATH.matcher(line);
            if (matcher.find()) {
                inside = matcher.group(1).equals(path);
            } else if (inside) {
                block.add(line);
            }
        }
        return block;
    }

    private static void expect(boolean condition, String message) {
        if (!condition) {
            problems.add(message);
        }
    }
}
