package app

import "testing"

func TestArea(t *testing.T) {
	if (Circle{Radius: 1}).Area() != 6 {
		t.Fail()
	}
}
