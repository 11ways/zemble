package com.example.core;

/** A public helper only its own type calls, once from a nested type: a lone public callable. */
public class Ledger {

    private int total;

    public int record(int amount) {
        total = normalize(amount) + total;
        return total;
    }

    public int normalize(int amount) {
        return Math.max(amount, 0);
    }

    public static class Audit {

        public int check(Ledger ledger) {
            return ledger.normalize(-1);
        }
    }
}
