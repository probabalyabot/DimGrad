"""Units as exponent vectors over base units, with exact rational exponents.

A unit is a map from base-unit names to fractions (Section 4.2 of the paper).
While a function is being checked a unit may also contain *unit variables*,
the unknowns that the solver eliminates: the unit of a literal 0 and the unit
of a parameter declared with `?`.
"""
from __future__ import annotations

from fractions import Fraction


def _combine(a: dict, b: dict, sign: int) -> dict:
    out = dict(a)
    for key, exp in b.items():
        out[key] = out.get(key, 0) + sign * exp
    return out


def _power_text(name: str, exp: Fraction) -> str:
    if exp == 1:
        return name
    if exp.denominator == 1:
        return f"{name}^{exp.numerator}"
    return f"{name}^({exp.numerator}/{exp.denominator})"


class Unit:
    """An immutable product of base units and unit variables raised to rational powers."""

    __slots__ = ("base", "vars", "_key")

    def __init__(self, base=None, vars=None):
        self.base = {k: Fraction(v) for k, v in (base or {}).items() if v != 0}
        self.vars = {k: Fraction(v) for k, v in (vars or {}).items() if v != 0}
        self._key = (frozenset(self.base.items()), frozenset(self.vars.items()))

    @staticmethod
    def of(name: str) -> "Unit":
        return Unit({name: 1})

    def __mul__(self, other: "Unit") -> "Unit":
        return Unit(_combine(self.base, other.base, 1), _combine(self.vars, other.vars, 1))

    def __truediv__(self, other: "Unit") -> "Unit":
        return Unit(_combine(self.base, other.base, -1), _combine(self.vars, other.vars, -1))

    def __pow__(self, power) -> "Unit":
        power = Fraction(power)
        return Unit(
            {k: v * power for k, v in self.base.items()},
            {k: v * power for k, v in self.vars.items()},
        )

    def __eq__(self, other) -> bool:
        return isinstance(other, Unit) and self._key == other._key

    def __hash__(self) -> int:
        return hash(self._key)

    @property
    def is_dimensionless(self) -> bool:
        return not self.base and not self.vars

    @property
    def is_concrete(self) -> bool:
        return not self.vars

    def substitute(self, subst: dict) -> "Unit":
        """Replace solved unit variables by their solutions."""
        if not self.vars:
            return self
        out = Unit(self.base)
        for var, exp in self.vars.items():
            out = out * (subst[var] ** exp if var in subst else Unit(vars={var: exp}))
        return out

    def _factors(self, order, var_names):
        rank = {name: i for i, name in enumerate(order)}
        items = sorted(self.base.items(), key=lambda kv: (rank.get(kv[0], len(rank)), kv[0]))
        for var, exp in sorted(self.vars.items()):
            label = (var_names or {}).get(var, f"?{var}")
            items.append((label, exp))
        return items

    def format(self, order=(), var_names=None) -> str:
        """Human-readable form, e.g. kg·m/s or m^2/(kg·s^2); `order` lists base units."""
        items = self._factors(order, var_names)
        num = [_power_text(n, e) for n, e in items if e > 0]
        den = [_power_text(n, -e) for n, e in items if e < 0]
        if not num and not den:
            return "1"
        text = "·".join(num) if num else "1"
        if len(den) == 1:
            text += "/" + den[0]
        elif den:
            text += "/(" + "·".join(den) + ")"
        return text

    def source(self, order=()) -> str:
        """The unit written in the source language's unit_expr syntax."""
        items = self._factors(order, None)
        num = [_power_text(n, e) for n, e in items if e > 0]
        den = [_power_text(n, -e) for n, e in items if e < 0]
        return "".join(["*".join(num) if num else "1"] + ["/" + d for d in den])

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return f"Unit({self.format()})"


ONE = Unit()
