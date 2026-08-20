package fixtures;

/** The same calls as LogicA and LogicB, but a control flow nothing shares. */
public class LogicC {

    public String summarize(Order order) {
        StringBuilder out = new StringBuilder();
        for (int index = 0; index < 3; index++) {
            out.append(order.title());
            while (order.isPaid()) {
                out.append(format(order.total()));
                break;
            }
        }
        out.append(fallback());
        return out.toString();
    }
}
