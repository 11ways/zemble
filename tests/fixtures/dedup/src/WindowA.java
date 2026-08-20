package fixtures;

/** A body whose middle run of statements is copied into WindowB. */
public class WindowA {

    public int prepare(int seed) {
        int first = seed + 1;
        int second = first * 2;
        int third = second - 3;
        int fourth = third * third;
        int fifth = fourth + 5;
        int sixth = fifth % 7;
        int seventh = sixth + first;
        return seventh;
    }
}
