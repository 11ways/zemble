package com.example.core;

public class SystemClock implements Clock {

    @Override
    public long now() {
        return System.currentTimeMillis();
    }
}
