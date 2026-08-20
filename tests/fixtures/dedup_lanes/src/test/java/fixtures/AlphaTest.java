package fixtures;

/** Test copy one. */
public class AlphaTest {

    public void setUp() {
        Panel panel = new Panel("slug");
        for (int index = 0; index < 5; index++) {
            String name = "field" + index;
            if (name.length() > 2) {
                panel.add(name, index);
            }
        }
        panel.save();
    }
}
