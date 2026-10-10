package io.zemble.javacfacts;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

/**
 * Writes one compilation's facts into the output file, merged with the facts earlier compilations
 * left there.
 *
 * <p>An incremental compile hands javac only the changed sources, so the file keeps every block
 * whose source this compilation did not analyze and that still exists under the root, and replaces
 * the rest. New facts stream into a sibling temp file that {@link #finish()} completes and moves
 * over the output, so a reader never sees a half-merged file. All state belongs to one instance,
 * i.e. one javac task: nothing leaks between compilations sharing a JVM (a Gradle daemon).
 * Two compilations writing the same output file CONCURRENTLY is unsupported; the later finish wins.
 *
 * <p>AIDEV-NOTE: the header records the root RELATIVE to the output's directory, so the file holds no
 * checkout path: a build cache restoring it into another checkout (same layout, other absolute path)
 * yields a file that resolves to, and merges in, that checkout.
 */
final class FactWriter {

    private static final String TOOL = "zemble-javac-facts";
    private static final String FILE_LINE_PREFIX = "{\"t\":\"file\",";

    private final Path output;
    private final Path root;
    private final String toolVersion;
    private final Set<String> analyzed = new HashSet<>();

    private Path temp;
    private Writer writer;
    private String lastPath;

    FactWriter(Path output, Path root, String toolVersion) {
        this.output = output.toAbsolutePath().normalize();
        this.root = root;
        this.toolVersion = toolVersion;
    }

    /**
     * Appends the facts of one analyzed type of {@code path}.
     *
     * <p>A {@code file} line opens the block whenever the previous block was about another file,
     * because pathless facts belong to the most recent {@code file} line.
     */
    synchronized void write(String path, String sha256, String lines) throws IOException {
        open();
        analyzed.add(path);
        if (!path.equals(lastPath)) {
            writer.write(fileLine(path, sha256));
            lastPath = path;
        }
        writer.write(lines);
    }

    /** Copies the still-valid blocks of the previous output behind the new facts and moves the result into place. */
    synchronized void finish() throws IOException {
        open();
        try {
            keepPrevious();
            writer.close();
            writer = null;
            try {
                Files.move(temp, output, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException exception) {
                Files.move(temp, output, StandardCopyOption.REPLACE_EXISTING);
            }
            temp = null;
        } finally {
            abandon();
        }
    }

    /** Drops this compilation's unfinished output, leaving the previous file untouched. */
    synchronized void abandon() {
        if (writer != null) {
            try {
                writer.close();
            } catch (IOException ignored) {
                // The temp file is deleted next; nothing else holds the stream.
            }
            writer = null;
        }
        if (temp != null) {
            try {
                Files.deleteIfExists(temp);
            } catch (IOException ignored) {
                // A leftover *.tmp file matches no facts discovery glob.
            }
            temp = null;
        }
    }

    private void open() throws IOException {
        if (writer != null) {
            return;
        }
        Path parent = output.getParent();
        Files.createDirectories(parent);
        // Not Files.createTempFile: its owner-only permissions would survive the move onto the output.
        temp = parent.resolve("." + output.getFileName() + "." + UUID.randomUUID() + ".tmp");
        writer = Files.newBufferedWriter(temp, StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW,
                StandardOpenOption.WRITE);
        writer.write(header());
    }

    /**
     * Copies every block of the previous output this compilation did not replace.
     *
     * <p>A previous file this writer cannot vouch for (another tool, format or root) is dropped whole:
     * its paths would be relative to something else.
     */
    private void keepPrevious() throws IOException {
        BufferedReader reader;
        try {
            reader = Files.newBufferedReader(output, StandardCharsets.UTF_8);
        } catch (NoSuchFileException absent) {
            return;
        }
        try (reader) {
            String header = reader.readLine();
            if (header == null || !compatible(header)) {
                return;
            }
            boolean keeping = false;
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith(FILE_LINE_PREFIX)) {
                    String path = Json.stringField(line, "path");
                    keeping = path != null && !analyzed.contains(path) && Files.isRegularFile(root.resolve(path));
                }
                if (keeping) {
                    writer.write(line);
                    writer.write('\n');
                }
            }
        }
    }

    /** Whether a previous header is ours and its root, absolute or relative to the output's directory, is this root. */
    private boolean compatible(String header) {
        if (!header.contains("\"zemble_facts\":1,") || !TOOL.equals(Json.stringField(header, "tool"))) {
            return false;
        }
        String recorded = Json.stringField(header, "root");
        if (recorded == null) {
            return false;
        }
        try {
            return canonical(output.getParent().resolve(recorded)).equals(canonical(root));
        } catch (InvalidPathException invalid) {
            return false;
        }
    }

    /**
     * The root as the header records it: relative to the output's directory, with forward slashes.
     *
     * <p>Relativized between real paths, because readers resolve it with symlinks followed. Only a root on
     * another filesystem root (a Windows drive) has no relative form and is recorded absolute.
     */
    private String recordedRoot() {
        Path from = canonical(output.getParent());
        Path to = canonical(root);
        if (from.getRoot() == null || !from.getRoot().equals(to.getRoot())) {
            return to.toString();
        }
        String relative = from.relativize(to).toString().replace('\\', '/');
        return relative.isEmpty() ? "." : relative;
    }

    /** The real path when it exists, else the normalized absolute one. */
    private static Path canonical(Path path) {
        try {
            return path.toRealPath();
        } catch (IOException missing) {
            return path.toAbsolutePath().normalize();
        }
    }

    private static String fileLine(String path, String sha256) {
        StringBuilder out = new StringBuilder();
        out.append('{');
        Json.field(out, "t", "file");
        out.append(',');
        Json.field(out, "path", path);
        out.append(',');
        Json.field(out, "sha256", sha256);
        out.append("}\n");
        return out.toString();
    }

    private String header() {
        StringBuilder out = new StringBuilder();
        out.append('{');
        Json.raw(out, "zemble_facts", "1");
        out.append(',');
        Json.field(out, "tool", TOOL);
        out.append(',');
        Json.field(out, "tool_version", toolVersion);
        out.append(',');
        Json.field(out, "generated_at", Instant.now().toString());
        out.append(',');
        Json.field(out, "language", "java");
        out.append(',');
        Json.field(out, "root", recordedRoot());
        out.append("}\n");
        return out.toString();
    }
}
