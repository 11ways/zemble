package fixtures;

/** The same run of statements, wrapped in different code. */
public class WindowB {

    public int prepare(String raw) {
        int seed = raw.length();
        int first = seed + 1;
        int second = first * 2;
        int third = second - 3;
        int fourth = third * third;
        int fifth = fourth + 5;
        int sixth = fifth % 7;
        int seventh = sixth + first;
        return seventh + raw.hashCode();
    }
}
