package demo;

/** A superclass with an explicit constructor and an overridable method. */
public class Base {

    protected final int size;

    public Base(int size) {
        this.size = size;
    }

    public String describe() {
        return "base:" + size;
    }
}
