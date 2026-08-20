package com.example.core;

public record Point(int x, int y) {

    public int sum() {
        return x + y;
    }
}
