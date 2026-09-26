package io.zemble.javacfacts;

import java.util.Map;

/** Minimal ASCII-only JSON writer; every non-ASCII character is escaped as \\uXXXX. */
final class Json {

    private Json() {
    }

    static void string(StringBuilder out, String value) {
        out.append('"');
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                case '\b' -> out.append("\\b");
                case '\f' -> out.append("\\f");
                default -> {
                    if (c < 0x20 || c > 0x7e) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        out.append('"');
    }

    /** Appends a key/value pair, where the value is written as a JSON string. */
    static void field(StringBuilder out, String key, String value) {
        string(out, key);
        out.append(':');
        string(out, value);
    }

    /** Appends a key/value pair, where the value is already valid JSON. */
    static void raw(StringBuilder out, String key, String rawValue) {
        string(out, key);
        out.append(':');
        out.append(rawValue);
    }

    /**
     * Writes an already-normalised value: String, Boolean, Number or a List of those.
     *
     * @throws IllegalArgumentException when the value is not one of those shapes
     */
    static void value(StringBuilder out, Object value) {
        if (value instanceof String s) {
            string(out, s);
        } else if (value instanceof Boolean || value instanceof Number) {
            out.append(value.toString());
        } else if (value instanceof Iterable<?> items) {
            out.append('[');
            boolean first = true;
            for (Object item : items) {
                if (!first) {
                    out.append(',');
                }
                first = false;
                value(out, item);
            }
            out.append(']');
        } else {
            throw new IllegalArgumentException("unsupported JSON value: " + value);
        }
    }

    /**
     * Reads the string value of a top-level key from one line this writer produced.
     *
     * <p>Not a general JSON parser: it finds the first {@code "key":"} occurrence, which is exact
     * for the header and {@code file} lines because their keys come before any free-form value.
     *
     * @return the decoded value, or null when the key is absent or its value is not a string
     */
    static String stringField(String line, String key) {
        StringBuilder needle = new StringBuilder();
        string(needle, key);
        needle.append(":\"");
        int start = line.indexOf(needle.toString());
        if (start < 0) {
            return null;
        }
        StringBuilder value = new StringBuilder();
        for (int i = start + needle.length(); i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                return value.toString();
            }
            if (c != '\\') {
                value.append(c);
                continue;
            }
            if (++i >= line.length()) {
                return null;
            }
            char escaped = line.charAt(i);
            switch (escaped) {
                case 'n' -> value.append('\n');
                case 'r' -> value.append('\r');
                case 't' -> value.append('\t');
                case 'b' -> value.append('\b');
                case 'f' -> value.append('\f');
                case 'u' -> {
                    if (i + 4 >= line.length()) {
                        return null;
                    }
                    try {
                        value.append((char) Integer.parseInt(line.substring(i + 1, i + 5), 16));
                    } catch (NumberFormatException exception) {
                        return null;
                    }
                    i += 4;
                }
                default -> value.append(escaped);
            }
        }
        return null;
    }

    static void object(StringBuilder out, Map<String, Object> values) {
        out.append('{');
        boolean first = true;
        for (Map.Entry<String, Object> entry : values.entrySet()) {
            if (!first) {
                out.append(',');
            }
            first = false;
            string(out, entry.getKey());
            out.append(':');
            value(out, entry.getValue());
        }
        out.append('}');
    }
}
