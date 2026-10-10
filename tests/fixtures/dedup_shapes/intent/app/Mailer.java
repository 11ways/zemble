class Mailer {
    /** @return the value, or null when it is blank */
    private static String blankToNull(String value) {
        return value == null || value.isBlank() ? null : String.valueOf(value);
    }

    private static Copy scoped(String key) {
        return Copy.of(key).withFilter("scope", "mailer").withFallback(key.strip()).trimmed();
    }

    String blankAsText(String value) {
        return value == null || value.isBlank() ? null : String.valueOf(value);
    }
}
