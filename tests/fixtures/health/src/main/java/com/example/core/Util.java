package com.example.core;

public class Util {

    public static int neverCalled(int value) {
        return value + 1;
    }

    public static int calledOnce(int value) {
        return value * 2;
    }

    public static int calledTwice(int value) {
        return value * 3 + hiddenHelper(value);
    }

    @Deprecated
    public static void frameworkHook() {
        System.out.println("reached by reflection");
    }

    private static int hiddenHelper(int value) {
        return value - 1;
    }

    private static int deadHelper(int value) {
        return value - 2;
    }

    public static int tangled(int alpha, int beta, int gamma, int delta, int epsilon, int zeta) {
        int total = 0;
        for (int i = 0; i < alpha; i++) {
            if (i % 2 == 0) {
                for (int j = 0; j < beta; j++) {
                    if (j % 3 == 0) {
                        while (total < gamma) {
                            if (total % 5 == 0 && delta > 0) {
                                total += delta;
                            } else if (total % 7 == 0 || epsilon > 0) {
                                total += epsilon;
                            } else {
                                total += 1;
                            }
                        }
                    } else {
                        total -= 1;
                    }
                }
            } else if (i % 3 == 0) {
                total += zeta;
            } else {
                total += 2;
            }
        }
        if (total > 100) {
            total = 100;
        }
        if (total < -100) {
            total = -100;
        }
        int steps = 0;
        steps += 1;
        steps += 2;
        steps += 3;
        steps += 4;
        steps += 5;
        steps += 6;
        steps += 7;
        steps += 8;
        steps += 9;
        steps += 10;
        steps += 11;
        steps += 12;
        steps += 13;
        steps += 14;
        steps += 15;
        steps += 16;
        steps += 17;
        steps += 18;
        steps += 19;
        steps += 20;
        steps += 21;
        steps += 22;
        steps += 23;
        steps += 24;
        steps += 25;
        steps += 26;
        steps += 27;
        steps += 28;
        steps += 29;
        steps += 30;
        return total + steps;
    }
}
