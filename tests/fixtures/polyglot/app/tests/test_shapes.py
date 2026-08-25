from app.shapes import Circle, largest


def test_largest():
    assert largest([Circle(1), Circle(2)]).radius == 2
