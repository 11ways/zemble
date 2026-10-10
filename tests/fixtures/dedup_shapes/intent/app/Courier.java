class Courier {
    private static String nullIfBlank(String value) {
        return value == null || value.isBlank() ? null : String.valueOf(value);
    }

    private static int blankCount(String value) {
        return value == null || value.isBlank() ? 0 : String.valueOf(value).length();
    }

    /** @return the value, or null when it is blank */
    String blankOrNull(String value) {
        return value == null || value.isBlank() ? null : value.strip().isEmpty() ? null : String.valueOf(value);
    }
}
