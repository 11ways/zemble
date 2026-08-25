package app

type Shape interface {
	Area() float64
}

type Circle struct {
	Radius float64
}

func (c Circle) Area() float64 {
	return scale(3 * c.Radius * c.Radius)
}

func scale(value float64) float64 {
	return value * 2
}
