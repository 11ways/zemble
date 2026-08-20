package fixtures;

/** One half of the planted logic pair. */
public class LogicA {

    public String render(Order order) {
        StringBuilder builder = new StringBuilder();
        builder.append(order.title());
        if (order.isPaid()) {
            builder.append(format(order.total()));
        } else {
            builder.append(fallback());
        }
        return builder.toString();
    }
}
