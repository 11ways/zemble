package demo;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

import static java.util.Arrays.asList;

/** The workhorse fixture: overloads, varargs, generics, lambdas, nested and local classes. */
@Marker(value = "demo", count = 3, strict = true, level = Level.HIGH, tags = {"a", "b"})
public class Demo extends Base implements Greeter {

    static final String PREFIX = "d:";

    private final List<String> entries = new ArrayList<>();

    static {
        System.out.println(PREFIX);
    }

    public Demo() {
        this(1);
    }

    public Demo(int size) {
        super(size);
        entries.add(PREFIX);
    }

    @Override
    public String describe() {
        return super.describe() + "/demo";
    }

    @Override
    public String greet(String name) {
        return PREFIX + name;
    }

    public String bar(int value) {
        return "int:" + value;
    }

    public String bar(String value) {
        return "string:" + value;
    }

    public String bar(Object value) {
        return "object:" + value;
    }

    public int sum(int... values) {
        int total = 0;
        for (int value : values) {
            total += value;
        }
        return total;
    }

    public <T extends Number> double total(List<T> items) {
        double sum = 0;
        for (T item : items) {
            sum += item.doubleValue();
        }
        return sum;
    }

    /** Calls both overloads and the varargs method, so resolution is observable. */
    public String overloadCalls() {
        bar(1);
        bar("text");
        sum(1, 2, 3);
        return bar((Object) this);
    }

    /** A lambda body's calls belong to this method; the method reference is a call too. */
    public String lambdas() {
        Function<Demo, String> reference = Demo::describe;
        Supplier<String> supplier = () -> bar(7);
        entries.add(supplier.get());
        return reference.apply(this);
    }

    /** An anonymous subclass, an anonymous interface implementation and a local class. */
    public String inner() {
        Base anonymous = new Base(2) {
            @Override
            public String describe() {
                return "anon";
            }
        };

        Greeter greeter = new Greeter() {
            @Override
            public String greet(String name) {
                return "anon:" + name;
            }
        };

        class Local {
            String render() {
                return anonymous.describe() + greeter.greet("x");
            }
        }

        return new Local().render();
    }

    /** Static import plus a call straight into java.util. */
    public List<String> library() {
        List<String> values = new ArrayList<>(asList("a", "b"));
        values.add("c");
        return values;
    }

    /** A nested class with its own nested class. */
    public static class Nested {

        public static class Deeper {

            public String ping() {
                return "deep";
            }
        }

        public String call() {
            return new Deeper().ping();
        }
    }
}
