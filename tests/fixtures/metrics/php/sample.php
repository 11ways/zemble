<?php
// A comment line.
function seq($a) {
    $b = $a + 1;
    $c = $b * 2;
    return $c;
}

function nested($xs, $a, $b) {
    foreach ($xs as $x) {
        if ($a) {
            if ($b) {
                work($x);
            }
        }
    }
}

function branches($a) {
    if ($a > 1) {
        return 1;
    } elseif ($a > 0) {
        return 2;
    } else {
        return 3;
    }
}

function booleans($a, $b, $c) {
    if ($a && $b && $c) {
        work(1);
    }
    if ($a && $b || $c) {
        work(2);
    }
}

function ternary($a) {
    return $a ? 1 : 2;
}

function tryCatch($a) {
    try {
        work(1);
    } catch (Exception $e) {
        if ($a) {
            work(2);
        }
    }
}

function fact($n) {
    if ($n <= 1) {
        return 1;
    }
    return $n * fact($n - 1);
}

function lambdaInLoop($xs, $a) {
    foreach ($xs as $x) {
        run(function () use ($a, $x) {
            if ($a) {
                work($x);
            }
        });
    }
}

function work($x) {}

function run($r) {}
