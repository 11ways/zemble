package fixtures;

/** The same SHAPE with other field names: a member name is not a local. */
public class CtorB {

    private final String title;
    private final String caption;
    private final boolean air;
    private final int seconds;
    private final String shade;
    private final boolean fun;

    CtorB(String title, String caption, boolean air, int seconds, String shade, boolean fun) {
        this.title = title;
        this.caption = caption;
        this.air = air;
        this.seconds = seconds;
        this.shade = shade;
        this.fun = fun;
    }
}
