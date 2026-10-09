class ProjectCells {
    static Object cell(Row row, Column column) {
        return switch (column.name()) {
            case "name" -> row.name();
            case "slug" -> row.slug();
            case "owner" -> row.owner();
            default -> null;
        };
    }
}
