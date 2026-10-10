public final class Labels {
    public static String clean(String text) {
        String stripped = text.strip();
        return stripped.isEmpty() ? null : stripped.toLowerCase();
    }

    public static String shout(String text) {
        String stripped = text.strip();
        return stripped.isEmpty() ? null : stripped.toUpperCase();
    }
}
