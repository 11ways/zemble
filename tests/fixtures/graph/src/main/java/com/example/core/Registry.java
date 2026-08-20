package com.example.core;

import com.example.util.*;
import static com.example.util.Helpers.twice;

@Marker("registry")
public class Registry<T extends Shape> {

    public static class Entry {

        private final String name;

        Entry(String name) {
            this.name = name;
        }

        public String name() {
            return name;
        }
    }

    public Shape anonymousShape() {
        return new Shape() {

            @Override
            public double area() {
                return twice(2.0);
            }
        };
    }

    public String localHelper() {
        class Local {

            String go() {
                return new Entry("x").name();
            }
        }
        return new Local().go();
    }

    public double viaWildcard() {
        Helpers helpers = new Helpers();
        return helpers.instanceTwice(3.0);
    }

    public T identity(T shape) {
        return shape;
    }
}
