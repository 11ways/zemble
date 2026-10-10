class Lower {
    private static String upper(String input) {
        String bare = input.strip();
        return bare.isEmpty() ? null : bare.toUpperCase();
    }
}
