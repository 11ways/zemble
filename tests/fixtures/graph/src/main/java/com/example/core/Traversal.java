package com.example.core;

import com.example.util.Helpers;
import static com.example.util.Helpers.*;

public class Traversal {

    public interface Doubler {

        double apply(double value);
    }

    public static class Session implements AutoCloseable {

        @Override
        public void close() {
        }

        public String name() {
            return "session";
        }
    }

    public double sumAreas(Shape[] shapes) {
        double total = 0;
        for (Shape shape : shapes) {
            total += shape.area();
        }
        return total;
    }

    public String narrow(Object value) {
        if (value instanceof Circle circle) {
            return circle.label();
        }
        Shape shape = (Shape) value;
        return shape.describe();
    }

    public double viaReference() {
        Doubler doubler = Helpers::twice;
        return doubler.apply(2.0);
    }

    public double staticWildcard() {
        return twice(4.0);
    }

    public String guarded() {
        try {
            return narrow(new Circle());
        } catch (IllegalStateException failure) {
            return failure.getMessage();
        }
    }

    public String withResource() {
        try (Session session = new Session()) {
            return session.name();
        }
    }
}
