package com.example.docs;

/** Holds every documentation shape the health report reads. */
public class Documented {

    private int count;

    /**
     * Says hello.
     */
    public String hello(String name) {
        if (name == null) {
            return "nobody";
        }
        return "hello " + name;
    }

    // Adds, documented by a line comment.
    public int add(int left, int right) {
        int sum = left + right;
        return sum;
    }

    public int undocumented(int value) {
        int doubled = value * 2;
        return doubled + 1;
    }

    public int getCount() {
        return count;
    }

    public Documented() {
        count = 0;
    }

    private int hidden(int value) {
        int tripled = value * 3;
        return tripled + hidden(value - 1);
    }

    /**
     * Doubles a value, at length.
     *
     * One.
     * Two.
     * Three.
     * Four.
     * Five.
     * Six.
     * Seven.
     * Eight.
     * Nine.
     * Ten.
     * Eleven.
     * Twelve.
     * Thirteen.
     * Fourteen.
     * Fifteen.
     * Sixteen.
     * Seventeen.
     */
    public int verbose(int value) {
        int doubled = value * 2;
        return doubled;
    }

    /** Keeps the value; the old computation is still here. */
    public int disabled(int value) {
        // int old = value * 4;
        // old += 1;
        // return old;
        int kept = value + count;
        return kept; // TODO: drop the old computation
    }
}

class Hidden {

    public int inside(int value) {
        int moved = value + 1;
        return moved;
    }
}
