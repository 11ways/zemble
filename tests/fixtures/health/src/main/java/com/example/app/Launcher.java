package com.example.app;

import com.example.core.AbstractStore;
import com.example.core.Circle;
import com.example.core.EnglishGreeter;
import com.example.core.FileStore;
import com.example.core.Greeter;
import com.example.core.Leaker;
import com.example.core.Shape;
import com.example.core.Square;
import com.example.core.Util;

public class Launcher {

    public static void main(String[] args) {
        Greeter greeter = new EnglishGreeter();
        AbstractStore store = new FileStore();
        Shape first = new Circle(1.0);
        Shape second = new Square(2.0);
        System.out.println(Util.calledOnce(1) + Util.calledTwice(2) + Util.tangled(1, 2, 3, 4, 5, 6));
        System.out.println(Leaker.leak() + first.area() + second.area() + greeter.greet("you"));
        store.save();
        report();
    }

    static void report() {
        System.out.println(Util.calledTwice(3));
    }

    public static String version() {
        return "1.0";
    }
}
