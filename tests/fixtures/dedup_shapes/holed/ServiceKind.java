class ServiceKind {
    public Runtime runtimeFor(String serverName) {
        return new DockerRuntime(new ServerService().clientFor(serverName), NetworkPolicy.forServer(serverName), Posture.PRIVATE, Egress.LIMITED);
    }

    public Icon getIcon() {
        return Icon.of("servicekind");
    }

    private Object pick(Row row) {
        return row.get(OWNER) == null ? this.serviceKindDefault : row.get(OWNER);
    }
}
