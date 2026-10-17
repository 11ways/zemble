// A comment line.
fun seq(a: Int): Int {
    val b = a + 1
    val c = b * 2
    return c
}

fun nested(xs: List<Int>, a: Boolean, b: Boolean) {
    for (x in xs) {
        if (a) {
            if (b) {
                work(x)
            }
        }
    }
}

fun branches(a: Int): Int {
    if (a > 1) {
        return 1
    } else if (a > 0) {
        return 2
    } else {
        return 3
    }
}

fun booleans(a: Boolean, b: Boolean, c: Boolean) {
    if (a && b && c) {
        work(1)
    }
    if (a && b || c) {
        work(2)
    }
}

fun tryCatch(a: Boolean) {
    try {
        work(1)
    } catch (e: RuntimeException) {
        if (a) {
            work(2)
        }
    }
}

fun fact(n: Int): Int {
    if (n <= 1) {
        return 1
    }
    return n * fact(n - 1)
}

fun lambdaInLoop(xs: List<Int>, a: Boolean) {
    for (x in xs) {
        run {
            if (a) {
                work(x)
            }
        }
    }
}

fun switcher(a: Int) {
    when (a) {
        1 -> work(1)
        else -> work(2)
    }
}

fun work(x: Int) {}

fun run(r: () -> Unit) {}
