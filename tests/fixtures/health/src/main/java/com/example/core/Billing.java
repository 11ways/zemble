package com.example.core;

public class Billing {

    public static int total(int[] values) {
        int sum = 0;
        int largest = 0;
        for (int value : values) {
            sum += value;
            if (value > largest) {
                largest = value;
            }
        }
        int average = values.length == 0 ? 0 : sum / values.length;
        int spread = largest - average;
        return sum + spread;
    }
}
