package fixtures;

/** The same body as ExactA, comments and formatting aside. */
public class ExactB {

    private final int offset = 9;

    // A different comment, and different whitespace, must not change the exact hash.
    public int recompute(int limit) {
        int total = 0;
        for (int index = 0; index < limit; index++) {
            int squared = index * index;
            if (squared % 3 == 0) {
                total  +=  squared;
            }
        }
        return total + offset;
    }
}
