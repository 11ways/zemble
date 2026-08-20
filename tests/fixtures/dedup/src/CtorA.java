package fixtures;

/** A constructor that only assigns its parameters to same-named fields. */
public class CtorA {

    private final String key;
    private final String label;
    private final boolean sea;
    private final int minutes;
    private final String color;
    private final boolean joy;

    CtorA(String key, String label, boolean sea, int minutes, String color, boolean joy) {
        this.key = key;
        this.label = label;
        this.sea = sea;
        this.minutes = minutes;
        this.color = color;
        this.joy = joy;
    }
}
