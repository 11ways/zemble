public final class Texts {
    /** @return the value as text, or null when it is absent or blank */
    public static String blankAsNull(Object value) {
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value);
        return text.isBlank() ? null : text;
    }

    /** @return the scoped copy key */
    public static Copy scoped(String key) {
        return Copy.of(key).withFilter("scope", "core").withFallback(key.strip()).trimmed();
    }
}
