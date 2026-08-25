load("@rules//:defs.bzl", "helper")
def area(x, y = 1):
    return helper(x) + y
def _impl(ctx):
    area(1)
point = rule(implementation = _impl)
