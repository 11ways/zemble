package fixtures;

/** The alpha-renamed twin: same code, different local names. */
public class RenamedA {

    private final int offset = 1;

    public int gather(int bound) {
        int sum = 0;
        for (int step = 0; step < bound; step++) {
            int product = step * step;
            if (product % 3 == 0) {
                sum += product;
            }
        }
        return sum + offset;
    }
}
