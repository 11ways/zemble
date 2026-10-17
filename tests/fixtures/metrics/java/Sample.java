package metrics;

// A comment line.
class Sample {
    /* A block
       comment. */
    int seq(int a) {
        int b = a + 1;
        int c = b * 2;
        return c;
    }

    void nested(int[] xs, boolean a, boolean b) {
        for (int x : xs) {
            if (a) {
                if (b) {
                    work(x);
                }
            }
        }
    }

    int branches(int a) {
        if (a > 1) {
            return 1;
        } else if (a > 0) {
            return 2;
        } else {
            return 3;
        }
    }

    void booleans(boolean a, boolean b, boolean c) {
        if (a && b && c) {
            work(1);
        }
        if (a && b || c) {
            work(2);
        }
    }

    int ternary(boolean a) {
        return a ? 1 : 2;
    }

    void tryCatch(boolean a) {
        try {
            work(1);
        } catch (RuntimeException e) {
            if (a) {
                work(2);
            }
        }
    }

    int fact(int n) {
        if (n <= 1) {
            return 1;
        }
        return n * fact(n - 1);
    }

    void lambdaInLoop(int[] xs, boolean a) {
        for (int x : xs) {
            run(() -> {
                if (a) {
                    work(x);
                }
            });
        }
    }

    void labelled(int[][] grid) {
        outer:
        for (int[] row : grid) {
            for (int cell : row) {
                if (cell == 0) {
                    continue outer;
                }
            }
        }
    }

    void switcher(int a) {
        switch (a) {
            case 1:
                work(1);
                break;
            default:
                while (a > 0) {
                    a--;
                }
        }
    }

    void work(int x) {}

    void run(Runnable r) {}
}
