package demo;

/** An annotation with only constant arguments, so every one of them is emitted. */
public @interface Marker {

    String value();

    int count() default 1;

    boolean strict() default false;

    Level level() default Level.LOW;

    String[] tags() default {};
}
