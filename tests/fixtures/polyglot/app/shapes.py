"""Shapes, in Python."""

from app.support import scale


class Shape:
    """A shape with an area."""

    def area(self):
        return 0

    def describe(self):
        return f"{self.name()} of {self.area()}"

    def name(self):
        return "shape"


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return scale(3 * self.radius * self.radius)

    def name(self):
        return "circle"


def largest(shapes):
    best = None
    for shape in shapes:
        if best is None or shape.area() > best.area():
            best = shape
    return best
