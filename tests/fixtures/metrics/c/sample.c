/* A comment line. */
int seq(int a) {
    int b = a + 1;
    int c = b * 2;
    return c;
}

void nested(int *xs, int n, int a, int b) {
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

void booleans(int a, int b, int c) {
    if (a && b && c) {
        work(1);
    }
    if (a && b || c) {
        work(2);
    }
}

int ternary(int a) {
    return a ? 1 : 2;
}

int fact(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * fact(n - 1);
}

void jumper(int a) {
    if (a) {
        goto done;
    }
    work(1);
done:
    work(2);
}

void work(int x) {}
