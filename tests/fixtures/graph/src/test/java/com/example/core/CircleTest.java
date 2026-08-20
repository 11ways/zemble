package com.example.core;

import com.example.util.Helpers;

public class CircleTest {

    public void areaJourney() {
        Circle circle = new Circle(2.0);
        double area = circle.area();
        double doubled = Helpers.twice(area);
        Point point = new Point(1, 2);
        int sum = point.sum();
    }
}
