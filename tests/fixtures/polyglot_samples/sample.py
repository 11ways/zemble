from pkg.mod import Base
from pkg.mod import helper as h

CONST = 3


class Foo(Base, Mixin):
    """doc"""

    attr: int = 1

    def __init__(self, x):
        self.x = x

    @staticmethod
    def build(a, b=2, *args, **kw):
        return Foo(a)

    async def run(self):
        self.helper()
        h.other(1, 2)
        Foo.build(1)
        return await self.x


def top(n):
    if n:
        for i in range(n):
            print(i)
    return top(n - 1)
