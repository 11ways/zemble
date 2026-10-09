class AppCells {
    static Object cell(Row row, Column column) {
        return switch (column.name()) {
            case "name" -> row.name();
            case "address" -> row.address();
            case "kind" -> row.kind();
            case "host" -> row.host();
            default -> null;
        };
    }
}
