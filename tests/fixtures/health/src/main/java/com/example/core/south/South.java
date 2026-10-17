package com.example.core.south;

import com.example.core.north.North;

public class South {

    public static int down(int value) {
        return value - 1;
    }

    public static North origin() {
        return new North();
    }
}
