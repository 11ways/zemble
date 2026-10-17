package com.example.core;

/** Implemented once in production and once by a test double: a seam, not speculative generality. */
public interface Clock {

    long now();
}
