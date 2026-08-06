"""Fixture module: imports pkg.b, forming an intentional cycle with b.py."""

from pkg import b


def use_b() -> str:
    return b.greet()
