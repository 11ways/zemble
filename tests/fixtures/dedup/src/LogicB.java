package fixtures;

/** The other half: same calls and control flow, other local names, one statement more. */
public class LogicB {

    public String describe(Order order) {
        StringBuilder out = new StringBuilder();
        String heading = order.title();
        out.append(heading);
        if (order.isPaid()) {
            out.append(format(order.total()));
        } else {
            out.append(fallback());
        }
        return out.toString();
    }
}
