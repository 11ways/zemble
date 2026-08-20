package demo;

/** An interface with a default method and a static one. */
public interface Greeter {

    String greet(String name);

    default String greetLoudly(String name) {
        return greet(name).toUpperCase();
    }

    static Greeter plain() {
        return name -> "hello " + name;
    }
}
