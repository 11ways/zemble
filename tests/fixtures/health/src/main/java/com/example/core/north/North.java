package com.example.core.north;

import com.example.core.south.South;

/** Depends on its sibling package, which depends back on it: a package cycle. */
public class North {

    public static int up(int value) {
        return South.down(value) + 1;
    }
}
