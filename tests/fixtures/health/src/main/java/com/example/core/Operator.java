package com.example.core;

import java.util.function.IntBinaryOperator;

public class Operator implements IntBinaryOperator {

    @Override
    public int applyAsInt(int left, int right) {
        return Calculator.add(left, right);
    }
}
