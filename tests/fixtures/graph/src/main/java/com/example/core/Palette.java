package com.example.core;

public enum Palette {

    RED("r"),
    GREEN("g") {
        @Override
        public String tag() {
            return "GREEN";
        }
    };

    private final String code;

    Palette(String code) {
        this.code = code;
    }

    public String tag() {
        return code;
    }
}
