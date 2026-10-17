// A comment line.
int seq(int a) {
    int b = a + 1;
    int c = b * 2;
    return c;
}

void nested(int *xs, int n, bool a, bool b) {
    for (int i = 0; i < n; i++) {
        if (a) {
            if (b) {
                work(xs[i]);
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

void booleans(bool a, bool b, bool c) {
    if (a && b && c) {
        work(1);
    }
    if (a && b || c) {
        work(2);
    }
}

int ternary(bool a) {
    return a ? 1 : 2;
}

void tryCatch(bool a) {
    try {
        work(1);
    } catch (const std::exception &e) {
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

void lambdaInLoop(int *xs, int n, bool a) {
    for (int i = 0; i < n; i++) {
        run([&]() {
            if (a) {
                work(xs[i]);
            }
        });
    }
}

void work(int x) {}
