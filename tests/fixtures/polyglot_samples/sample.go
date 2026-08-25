package store

import ( "fmt"; "os" )

type Shape interface { Area() float64 }
type Point struct { X, Y int; Name string }
type Alias = Point
const Limit = 3
var count int

func (p *Point) Area() float64 { fmt.Println(p.X); return helper(p.X) }
func helper(n int) float64 { return float64(n) }
func New(x int) *Point { p := &Point{X: x}; p.Area(); return p }
