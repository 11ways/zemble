package fixtures;

/** A near miss: identical but for a field name, which is not a local. */
public class FieldDiff {

    private final int shift = 1;

    public int gather(int bound) {
        int sum = 0;
        for (int step = 0; step < bound; step++) {
            int product = step * step;
            if (product % 3 == 0) {
                sum += product;
            }
        }
        return sum + shift;
    }
}
