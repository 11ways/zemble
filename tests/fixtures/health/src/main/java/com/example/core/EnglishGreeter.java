package com.example.core;

/** The only implementor of Greeter; its override is reached through the interface, never by name. */
public class EnglishGreeter implements Greeter {

    public String greet(String name) {
        return "Hello " + name;
    }
}
