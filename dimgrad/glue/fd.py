"""Central finite differences, the independent check on generated gradients (RQ1)."""
from __future__ import annotations

import sys

EPS = sys.float_info.epsilon
STEP = EPS ** (1.0 / 3.0)  # the usual step for a central difference
TOLERANCE = 1e-6           # largest accepted relative error; fixed before any result was seen


def step_points(args: list[float], index: int):
    """(args with args[index] stepped up, stepped down, half-width of the step)."""
    x = args[index]
    h = STEP * (abs(x) if x != 0 else 1.0)
    hi, lo = list(args), list(args)
    hi[index], lo[index] = x + h, x - h
    return hi, lo, (hi[index] - lo[index]) / 2


def central_difference(f, args: list[float], index: int):
    """(derivative of f with respect to args[index], half-width of the step used)."""
    hi, lo, h = step_points(args, index)
    return (f(hi) - f(lo)) / (2 * h), h


def relative_error(ad: float, fd: float, value: float, h: float, tol: float = TOLERANCE) -> float:
    """|ad - fd| relative to the larger of the two.

    A finite difference carries round-off of about EPS*|f|/h. The denominator
    is never allowed below that level divided by `tol`, so a derivative too
    small for the finite difference to resolve is compared absolutely
    instead of failing on noise.
    """
    floor = EPS * abs(value) / h / tol
    return abs(ad - fd) / max(abs(ad), abs(fd), floor, sys.float_info.min)
