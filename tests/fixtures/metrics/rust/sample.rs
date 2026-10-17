// A comment line.
fn seq(a: i32) -> i32 {
    let b = a + 1;
    let c = b * 2;
    c
}

fn nested(xs: &[i32], a: bool, b: bool) {
    for x in xs {
        if a {
            if b {
                work(*x);
            }
        }
    }
}

fn branches(a: i32) -> i32 {
    if a > 1 {
        1
    } else if a > 0 {
        2
    } else {
        3
    }
}

fn booleans(a: bool, b: bool, c: bool) {
    if a && b && c {
        work(1);
    }
    if a && b || c {
        work(2);
    }
}

fn fact(n: u64) -> u64 {
    if n <= 1 {
        return 1;
    }
    n * fact(n - 1)
}

fn lambda_in_loop(xs: &[i32], a: bool) {
    for x in xs {
        run(|| {
            if a {
                work(*x);
            }
        });
    }
}

fn matcher(a: i32) {
    match a {
        1 => work(1),
        _ => work(2),
    }
}

fn work(x: i32) {}

fn run<F: Fn()>(r: F) {}
