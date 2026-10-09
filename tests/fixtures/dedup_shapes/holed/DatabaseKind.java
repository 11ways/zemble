class DatabaseKind {
    public Runtime runtimeFor(String serverName) {
        return new DockerRuntime(new ServerService().clientFor(serverName), NetworkPolicy.forServer(serverName), Posture.PRIVATE, Egress.NONE);
    }

    public Icon getIcon() {
        return Icon.of("databasekind");
    }

    private Object pick(Row row) {
        return row.get(OWNER) == null ? this.databaseKindDefault : row.get(OWNER);
    }
}
