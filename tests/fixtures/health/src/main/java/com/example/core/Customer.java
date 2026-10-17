package com.example.core;

public class Customer {

    private final String name;
    private final String street;
    private final String city;

    public Customer(String name, String street, String city) {
        this.name = name;
        this.street = street;
        this.city = city;
    }

    public String name() {
        return name;
    }

    public String street() {
        return street;
    }

    public String city() {
        return city;
    }

    public Customer moved(String to) {
        return new Customer(name, street, to);
    }
}
