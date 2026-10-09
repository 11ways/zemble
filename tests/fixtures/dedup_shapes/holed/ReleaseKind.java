class ReleaseKind {
    public Runtime runtimeFor(String serverName) {
        return new DockerRuntime(new ServerService().clientFor(serverName), NetworkPolicy.forServer(serverName), Posture.PRIVATE, Egress.OPEN);
    }

    public Icon getIcon() {
        return Icon.of("releasekind");
    }

    private Object pick(Row row) {
        return row.get(OWNER) == null ? this.releaseKindDefault : row.get(OWNER);
    }
}
