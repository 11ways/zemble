package com.example.core;

/**
 * Asks its customer for everything it prints (feature envy); drives a ledger's logic and chains
 * calls on what a call returned, neither of which is envy.
 */
public class Invoice {

    private final Customer customer;

    public Invoice(Customer customer) {
        this.customer = customer;
    }

    public String label() {
        return customer.name() + ", " + customer.street() + ", " + customer.city();
    }

    public String moves() {
        return customer.moved("a").moved("b").moved("c").moved("d").city();
    }

    public int replay(Ledger ledger) {
        return ledger.record(1) + ledger.record(2) + ledger.record(3);
    }
}
