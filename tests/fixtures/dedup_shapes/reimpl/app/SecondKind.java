class SecondKind implements Kind {
    @Override
    public Spec specFor(Map<String, Object> settings) {
        Map<String, String> env = new LinkedHashMap<>();
        settings.forEach((key, value) -> env.put(key.toUpperCase(), String.valueOf(value)));
        return Spec.builder().image(Images.resolve(settings)).env(env).footprint(Footprints.defaultFootprintMb()).build();
    }
}
