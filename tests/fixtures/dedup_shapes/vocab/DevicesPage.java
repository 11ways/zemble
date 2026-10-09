class DevicesPage {
    String target(Panel panel) {
        return Routes.detail(panel, "instance-devices");
    }

    boolean failedRow(Row row) {
        return "failed".equals(row.get(STATUS));
    }

    String[] words(String text) {
        return text.split("\\s+");
    }

    boolean email(String text) {
        return text.matches("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$");
    }
}
