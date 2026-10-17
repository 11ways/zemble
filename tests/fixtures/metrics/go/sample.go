// Package metrics is a fixture.
package metrics

func seq(a int) int {
	b := a + 1
	c := b * 2
	return c
}

func nested(xs []int, a bool, b bool) {
	for _, x := range xs {
		if a {
			if b {
				work(x)
			}
		}
	}
}

func branches(a int) int {
	if a > 1 {
		return 1
	} else if a > 0 {
		return 2
	} else {
		return 3
	}
}

func booleans(a bool, b bool, c bool) {
	if a && b && c {
		work(1)
	}
	if a && b || c {
		work(2)
	}
}

func fact(n int) int {
	if n <= 1 {
		return 1
	}
	return n * fact(n-1)
}

func lambdaInLoop(xs []int, a bool) {
	for _, x := range xs {
		run(func() {
			if a {
				work(x)
			}
		})
	}
}

func switcher(a int) {
	switch a {
	case 1:
		work(1)
	default:
		work(2)
	}
}

func work(x int) {}

func run(r func()) {}
