public final class SlugText {
    public static String slugify(String input) {
        return Normalizer.normalize(input, Normalizer.Form.NFD).replaceAll("[^a-z0-9]+", "-").toLowerCase();
    }

    public static String slugify(String input, String separator) {
        return Normalizer.normalize(input, Normalizer.Form.NFD).replaceAll("[^a-z0-9]+", separator).toLowerCase();
    }
}
