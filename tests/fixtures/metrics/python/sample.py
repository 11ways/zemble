# A comment line.


def seq(a):
    b = a + 1
    c = b * 2
    return c


def nested(xs, a, b):
    for x in xs:
        if a:
            if b:
                work(x)


def branches(a):
    if a > 1:
        return 1
    elif a > 0:
        return 2
    else:
        return 3


def booleans(a, b, c):
    if a and b and c:
        work(1)
    if a and b or c:
        work(2)


def ternary(a):
    return 1 if a else 2


def try_catch(a):
    try:
        work(1)
    except RuntimeError:
        if a:
            work(2)


def fact(n):
    if n <= 1:
        return 1
    return n * fact(n - 1)


def lambda_in_loop(xs, a):
    for x in xs:
        run(lambda: work(x) if a else None)


def work(x):
    pass


def run(r):
    pass
