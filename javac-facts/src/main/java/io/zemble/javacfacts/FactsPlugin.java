package io.zemble.javacfacts;

import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.util.JavacTask;
import com.sun.source.util.Plugin;
import com.sun.source.util.TaskEvent;
import com.sun.source.util.TaskListener;
import com.sun.source.util.TreePath;
import com.sun.source.util.Trees;

import javax.lang.model.element.TypeElement;
import javax.lang.model.util.Elements;
import javax.lang.model.util.Types;
import javax.tools.JavaFileObject;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/**
 * A javac plugin that emits compiler-resolved facts about the compiled sources as JSONL.
 *
 * <p>Enable with {@code -Xplugin:ZembleFacts out=<file> root=<dir> append=<bool>}; every failure is
 * swallowed after one stderr line so a broken plugin can never break a build.
 */
public final class FactsPlugin implements Plugin {

    static final String NAME = "ZembleFacts";
    static final String VERSION = "0.1.0";

    private static final String DEFAULT_OUT = "build/zemble/facts.jsonl";

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public void init(JavacTask task, String... args) {
        Path workingDir = Paths.get("").toAbsolutePath().normalize();

        Path out = workingDir.resolve(DEFAULT_OUT);
        Path root = workingDir;
        boolean append = false;

        for (String arg : args) {
            int split = arg.indexOf('=');
            if (split < 0) {
                continue;
            }
            String key = arg.substring(0, split).trim().toLowerCase(Locale.ROOT);
            String value = arg.substring(split + 1).trim();
            switch (key) {
                case "out" -> out = workingDir.resolve(value).normalize();
                case "root" -> root = workingDir.resolve(value).toAbsolutePath().normalize();
                case "append" -> append = Boolean.parseBoolean(value);
                default -> { }
            }
        }

        task.addTaskListener(new FactsListener(task, out, root, append));
    }

    /** Runs after ANALYZE so every tree the scanner touches is attributed. */
    private static final class FactsListener implements TaskListener {

        private final JavacTask task;
        private final FactWriter writer;
        private final Path root;
        private final Set<String> seenFiles = new HashSet<>();
        private final Set<String> seenClasses = new HashSet<>();

        private Trees trees;
        private Elements elements;
        private Types types;
        private Refs refs;
        private boolean reported;

        FactsListener(JavacTask task, Path out, Path root, boolean append) {
            this.task = task;
            this.writer = new FactWriter(out, append);
            this.root = root;
        }

        @Override
        public void finished(TaskEvent event) {
            if (event.getKind() != TaskEvent.Kind.ANALYZE) {
                return;
            }
            try {
                handle(event);
            } catch (Throwable throwable) {
                report(throwable);
            }
        }

        private void handle(TaskEvent event) throws IOException {
            CompilationUnitTree unit = event.getCompilationUnit();
            TypeElement typeElement = event.getTypeElement();
            if (unit == null || typeElement == null) {
                return;
            }

            if (trees == null) {
                trees = Trees.instance(task);
                elements = task.getElements();
                types = task.getTypes();
                refs = new Refs(elements, types);
            }

            JavaFileObject source = unit.getSourceFile();
            if (source == null) {
                return;
            }
            String relative = relativize(source);

            String classKey = relative + "|" + typeElement.getQualifiedName();
            if (!seenClasses.add(classKey)) {
                return;
            }

            StringBuilder out = new StringBuilder();
            if (seenFiles.add(relative)) {
                out.append('{');
                Json.field(out, "t", "file");
                out.append(',');
                Json.field(out, "path", relative);
                out.append(',');
                Json.field(out, "sha256", sha256(source));
                out.append("}\n");
            }

            TreePath path = trees.getPath(typeElement);
            if (path == null) {
                path = new TreePath(unit);
            }
            new FactScanner(trees, elements, types, refs, unit, relative, out)
                    .scan(path, null);

            writer.write(VERSION, root.toString(), out.toString());
        }

        private String relativize(JavaFileObject source) {
            String raw;
            try {
                raw = Paths.get(source.toUri()).toAbsolutePath().normalize().toString();
                Path absolute = Paths.get(raw);
                if (absolute.startsWith(root)) {
                    raw = root.relativize(absolute).toString();
                }
            } catch (RuntimeException exception) {
                raw = source.getName();
            }
            return raw.replace('\\', '/');
        }

        private static String sha256(JavaFileObject source) throws IOException {
            MessageDigest digest;
            try {
                digest = MessageDigest.getInstance("SHA-256");
            } catch (Exception exception) {
                throw new IOException(exception);
            }
            byte[] buffer = new byte[8192];
            try (InputStream stream = source.openInputStream()) {
                int read;
                while ((read = stream.read(buffer)) > 0) {
                    digest.update(buffer, 0, read);
                }
            } catch (UnsupportedOperationException exception) {
                // A file manager that only exposes character content (in-memory sources).
                CharSequence content = source.getCharContent(true);
                digest.update(content.toString().getBytes(StandardCharsets.UTF_8));
            }
            StringBuilder hex = new StringBuilder();
            for (byte value : digest.digest()) {
                hex.append(Character.forDigit((value >> 4) & 0xf, 16));
                hex.append(Character.forDigit(value & 0xf, 16));
            }
            return hex.toString();
        }

        private void report(Throwable throwable) {
            if (reported) {
                return;
            }
            reported = true;
            String message = throwable.getClass().getSimpleName()
                    + (throwable.getMessage() == null ? "" : ": " + throwable.getMessage());
            System.err.println("zemble-javac-facts: " + message);
        }
    }
}
