"""Fixture module: imports pkg.a, forming an intentional cycle with a.py."""

from pkg import a


def greet() -> str:
    return "hello from b, a has: " + str(dir(a))
