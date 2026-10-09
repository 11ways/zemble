class ClientCells {
    static Object cell(Row row, Column column) {
        return switch (column.name()) {
            case "name" -> row.name();
            case "secret" -> row.secret();
            case "created" -> row.created();
            default -> null;
        };
    }
}
