package be.elevenways.hawkeye.generated.tags.demo;

/** Generated implementation of the `<demo-card>` element declared by `components/card.hwk`. */
public final class DemoCardImpl {

    private String heading = "";

    public String getHeading() {
        return this.heading;
    }

    public void render() {
        WidgetFunctions.render(this.heading);
    }
}
