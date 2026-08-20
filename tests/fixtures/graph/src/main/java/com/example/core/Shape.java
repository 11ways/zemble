package com.example.core;

public interface Shape {

    double area();

    default String describe() {
        return "shape " + area();
    }

    static Shape unit() {
        return new Circle(1.0);
    }
}
