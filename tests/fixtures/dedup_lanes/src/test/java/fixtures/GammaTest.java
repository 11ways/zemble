package fixtures;

/** Test half of a mixed class. */
public class GammaTest {

    public String describe(String label) {
        StringBuilder builder = new StringBuilder();
        for (int index = 0; index < label.length(); index++) {
            char letter = label.charAt(index);
            if (letter != 32) {
                builder.append(letter);
            }
        }
        return builder.toString().trim();
    }
}
