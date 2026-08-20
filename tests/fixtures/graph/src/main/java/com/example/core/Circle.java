package com.example.core;

import com.example.util.Helpers;

public class Circle implements Shape {

    private final double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    public Circle() {
        this(1.0);
    }

    @Override
    public double area() {
        return Helpers.twice(radius);
    }

    public double scale(double factor) {
        return radius * factor;
    }

    public double scale(double factor, int times) {
        return radius * factor * times;
    }

    public String label() {
        return describe();
    }
}
