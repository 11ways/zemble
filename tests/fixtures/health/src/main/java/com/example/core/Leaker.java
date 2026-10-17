package com.example.core;

import com.example.app.Launcher;

/** The core reaching into the application: a forbidden crossing. */
public class Leaker {

    public static String leak() {
        return Launcher.version();
    }
}
