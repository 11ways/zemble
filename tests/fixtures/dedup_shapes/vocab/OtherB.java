class OtherB {
    static Object cell(Row row, Field field) {
        return switch (field.label()) {
            case "alpha" -> row.alpha();
            case "beta" -> row.beta();
            case "gamma" -> row.gamma();
            default -> null;
        };
    }
}
