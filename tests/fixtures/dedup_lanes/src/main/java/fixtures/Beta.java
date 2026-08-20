package fixtures;

/** Production copy two. */
public class Beta {

    public int compute(int limit) {
        int total = 0;
        for (int index = 0; index < limit; index++) {
            int squared = index * index;
            if (squared % 3 == 0) {
                total += squared;
            }
        }
        return total + 4;
    }
}
