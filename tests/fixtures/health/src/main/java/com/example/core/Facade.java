package com.example.core;

public class Facade {

    public static int compute(int left, int right) {
        return Calculator.add(left, right);
    }
}
