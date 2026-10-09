class Delta {
    static Copy copy(String key) {
        return Copy.of(key).withFilter("scope", "delta");
    }

    String title(Request request) {
        return Copy.of("delta_title").withFilter("scope", "delta").resolve(request.getLocales(), request.getMessageResolver());
    }

    static Map<String, int[]> counts(Conduit conduit) {
        Map<String, int[]> cached = conduit.getAttribute(COUNTS);
        if (cached == null) {
            cached = load();
            conduit.setAttribute(COUNTS, cached);
        }
        return cached;
    }

    Entry entry(Entry entry) {
        return entry.icon(Icon.of("delta-icon"));
    }
}
