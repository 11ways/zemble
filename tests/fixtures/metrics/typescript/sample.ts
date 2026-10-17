// A comment line.
function seq(a: number): number {
  const b = a + 1;
  const c = b * 2;
  return c;
}

function nested(xs: number[], a: boolean, b: boolean): void {
  for (const x of xs) {
    if (a) {
      if (b) {
        work(x);
      }
    }
  }
}

function branches(a: number): number {
  if (a > 1) {
    return 1;
  } else if (a > 0) {
    return 2;
  } else {
    return 3;
  }
}

function booleans(a: boolean, b: boolean, c: boolean): void {
  if (a && b && c) {
    work(1);
  }
  if (a && b || c) {
    work(2);
  }
}

function ternary(a: boolean): number {
  return a ? 1 : 2;
}

function tryCatch(a: boolean): void {
  try {
    work(1);
  } catch (e) {
    if (a) {
      work(2);
    }
  }
}

function fact(n: number): number {
  if (n <= 1) {
    return 1;
  }
  return n * fact(n - 1);
}

function lambdaInLoop(xs: number[], a: boolean): void {
  for (const x of xs) {
    run(() => {
      if (a) {
        work(x);
      }
    });
  }
}

function work(x: number): void {}

function run(r: () => void): void {}
