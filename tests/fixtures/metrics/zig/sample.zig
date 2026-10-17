// A comment line.
fn seq(a: i32) i32 {
    const b = a + 1;
    const c = b * 2;
    return c;
}

fn nested(xs: []const i32, a: bool, b: bool) void {
    for (xs) |x| {
        if (a) {
            if (b) {
                work(x);
            }
        }
    }
}

fn branches(a: i32) i32 {
    if (a > 1) {
        return 1;
    } else if (a > 0) {
        return 2;
    } else {
        return 3;
    }
}

fn booleans(a: bool, b: bool, c: bool) void {
    if (a and b and c) {
        work(1);
    }
    if (a and b or c) {
        work(2);
    }
}

fn ternary(a: bool) i32 {
    return if (a) 1 else 2;
}

fn tryCatch(a: bool) void {
    risky() catch {
        if (a) {
            work(2);
        }
    };
}

fn fact(n: u64) u64 {
    if (n <= 1) {
        return 1;
    }
    return n * fact(n - 1);
}

fn switcher(a: i32) void {
    switch (a) {
        1 => work(1),
        else => {
            while (a > 0) {
                work(a);
            }
        },
    }
}

fn work(x: i32) void {
    _ = x;
}

fn risky() !void {}
