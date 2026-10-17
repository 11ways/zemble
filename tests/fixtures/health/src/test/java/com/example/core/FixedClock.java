package com.example.core;

public class FixedClock implements Clock {

    @Override
    public long now() {
        return 42L;
    }
}
