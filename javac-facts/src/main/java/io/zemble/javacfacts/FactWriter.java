package io.zemble.javacfacts;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.HashSet;
import java.util.Set;

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
        temp = Files.createTempFile(parent, "." + output.getFileName() + ".", ".tmp");
        writer = Files.newBufferedWriter(temp, StandardCharsets.UTF_8);
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

    private boolean compatible(String header) {
        return header.contains("\"zemble_facts\":1,")
                && TOOL.equals(Json.stringField(header, "tool"))
                && root.toString().equals(Json.stringField(header, "root"));
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
        Json.field(out, "root", root.toString());
        out.append("}\n");
        return out.toString();
    }
}
