package com.example.app;

public class Consumer {

    // Deliberately unimportable: `Circle` is declared in com.example.core AND in
    // com.example.util, so the graph must report AMBIGUOUS instead of picking one.
    public double measure(Circle circle) {
        return circle.area();
    }
}
