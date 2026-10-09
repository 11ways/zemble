public final class Slugs {
    public static String slugify(String input) {
        return SlugText.slugify(input);
    }

    public static String slugify(String input, String separator) {
        return SlugText.slugify(input, separator);
    }
}
