public final class Texts {
    public static String tidy(String value) {
        String trimmed = value.strip();
        return trimmed.isEmpty() ? null : trimmed.toLowerCase();
    }
}
