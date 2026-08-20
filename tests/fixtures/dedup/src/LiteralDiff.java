package fixtures;

/** A near miss: identical but for one literal, which is a different decision. */
public class LiteralDiff {

    private final int offset = 1;

    public int gather(int bound) {
        int sum = 0;
        for (int step = 0; step < bound; step++) {
            int product = step * step;
            if (product % 5 == 0) {
                sum += product;
            }
        }
        return sum + offset;
    }
}
