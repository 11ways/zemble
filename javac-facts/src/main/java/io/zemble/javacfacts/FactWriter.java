package io.zemble.javacfacts;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.HashSet;
import java.util.Set;

/**
 * Appends JSONL facts to one output file.
 *
 * <p>Truncation and the header line are tracked per absolute output path for the lifetime of the
 * JVM, so a javac run that invokes the plugin over several rounds (or several compilation units)
 * still produces exactly one header.
 */
final class FactWriter {

    private static final Set<String> STARTED = new HashSet<>();

    private final Path output;
    private final boolean append;

    FactWriter(Path output, boolean append) {
        this.output = output.toAbsolutePath().normalize();
        this.append = append;
    }

    /** Writes one block of complete JSONL lines, preceded by the header on first use. */
    synchronized void write(String toolVersion, String root, String lines) throws IOException {
        String key = output.toString();
        boolean first;
        synchronized (STARTED) {
            first = STARTED.add(key);
        }

        Path parent = output.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        // Appending onto a file another JVM already headered must not add a second header.
        boolean needsHeader = first && !(append && Files.exists(output) && Files.size(output) > 0);

        StandardOpenOption mode = (first && !append)
                ? StandardOpenOption.TRUNCATE_EXISTING
                : StandardOpenOption.APPEND;

        try (OutputStream stream = Files.newOutputStream(output, StandardOpenOption.CREATE, StandardOpenOption.WRITE, mode);
             Writer writer = new OutputStreamWriter(stream, StandardCharsets.UTF_8)) {
            if (needsHeader) {
                writer.write(header(toolVersion, root));
            }
            writer.write(lines);
        }
    }

    private static String header(String toolVersion, String root) {
        StringBuilder out = new StringBuilder();
        out.append('{');
        Json.raw(out, "zemble_facts", "1");
        out.append(',');
        Json.field(out, "tool", "zemble-javac-facts");
        out.append(',');
        Json.field(out, "tool_version", toolVersion);
        out.append(',');
        Json.field(out, "generated_at", Instant.now().toString());
        out.append(',');
        Json.field(out, "language", "java");
        out.append(',');
        Json.field(out, "root", root);
        out.append("}\n");
        return out.toString();
    }
}
