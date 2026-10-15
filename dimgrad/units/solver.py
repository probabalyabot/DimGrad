"""Unit equations solved by Gaussian elimination over the rationals.

Every equality the unit rules require is a linear equation over exponents.
`unify(a, b)` asserts a = b, i.e. a/b = 1. After substituting what is already
known, the residual either has no unknowns (then it must be dimensionless, or
the program has a unit error) or one unknown is eliminated in terms of the
rest. The substitution is kept fully reduced, so this is Gauss-Jordan
elimination performed one equation at a time.
"""
from __future__ import annotations

from fractions import Fraction

from .units import ONE, Unit


class Solver:
    def __init__(self):
        self.subst: dict[int, Unit] = {}
        self.kinds: dict[int, str] = {}   # "param" for `?`, "zero" for literal 0
        self.labels: dict[int, str] = {}  # how to print an unsolved variable

    def fresh(self, kind: str, label: str | None = None) -> Unit:
        var = len(self.kinds)
        self.kinds[var] = kind
        self.labels[var] = label or f"?{var}"
        return Unit(vars={var: 1})

    def resolve(self, unit: Unit) -> Unit:
        return unit.substitute(self.subst)

    def unify(self, a: Unit, b: Unit) -> bool:
        """Require a = b. Returns False when the two units cannot be equal."""
        residual = self.resolve(a / b)
        if not residual.vars:
            return residual.is_dimensionless
        # Prefer eliminating a zero-literal unknown, so `?` parameters stay as
        # the free unknowns for as long as possible.
        var = min(residual.vars, key=lambda v: (self.kinds[v] == "param", v))
        exp = residual.vars[var]
        rest = residual / Unit(vars={var: exp})
        solution = rest ** (Fraction(-1) / exp)
        for other in self.subst:
            self.subst[other] = self.subst[other].substitute({var: solution})
        self.subst[var] = solution
        return True

    def unsolved(self, kind: str | None = None) -> list[int]:
        return [v for v in self.kinds if v not in self.subst and (kind is None or self.kinds[v] == kind)]

    def default_to_unitless(self, var: int):
        self.unify(Unit(vars={var: 1}), ONE)
