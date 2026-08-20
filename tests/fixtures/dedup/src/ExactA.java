package fixtures;

/** Holds one body that is copied verbatim into ExactB. */
public class ExactA {

    private final int offset = 4;

    public int compute(int limit) {
        int total = 0;
        for (int index = 0; index < limit; index++) {
            int squared = index * index;
            if (squared % 3 == 0) {
                total += squared;
            }
        }
        return total + offset;
    }
}
