class Mail {
    private static String tidied(String raw) {
        String bare = raw.strip();
        return bare.isEmpty() ? null : bare.toLowerCase();
    }
}
