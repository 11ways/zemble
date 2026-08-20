package demo;

/** A record implementing an interface, with a compact constructor. */
public record Point(int x, int y) implements Named {

    public Point {
        if (x < 0) {
            throw new IllegalArgumentException("negative");
        }
    }

    @Override
    public String name() {
        return "point";
    }
}
