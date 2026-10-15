from fractions import Fraction

from dimgrad.units.solver import Solver
from dimgrad.units.units import ONE, Unit

kg, m, s = Unit.of("kg"), Unit.of("m"), Unit.of("s")
J = kg * m ** 2 / s ** 2


def test_multiply_divide_power():
    assert m * m == m ** 2
    assert (m / s) * s == m
    assert J / (m / s) == kg * m / s
    assert m / m == ONE
    assert ONE.is_dimensionless


def test_fractional_exponents_are_exact():
    root = (m ** 2) ** Fraction(1, 2)
    assert root == m
    third = m ** Fraction(1, 3)
    assert third * third * third == m
    assert (s ** Fraction(-1, 2)) * (s ** Fraction(1, 2)) == ONE


def test_format():
    order = ["kg", "m", "s"]
    assert J.format(order) == "kg·m^2/s^2"
    assert (kg * m / s).format(order) == "kg·m/s"
    assert ONE.format(order) == "1"
    assert (ONE / s).format(order) == "1/s"
    assert (m ** 2 / (kg * s ** 2)).format(order) == "m^2/(kg·s^2)"
    assert (s ** Fraction(1, 2)).format(order) == "s^(1/2)"
    # Without an order, base units are listed alphabetically.
    assert (s * kg).format() == "kg·s"


def test_source_form_uses_language_syntax():
    order = ["kg", "m", "s"]
    assert J.source(order) == "kg*m^2/s^2"
    assert (ONE / s).source(order) == "1/s"
    assert (s ** Fraction(-1, 2)).source(order) == "1/s^(1/2)"


def test_solver_solves_for_an_unknown():
    solver = Solver()
    eta = solver.fresh("param", "?eta")
    # x - eta * (J/m) must be a length
    assert solver.unify(eta * J / m, m)
    assert solver.resolve(eta) == m ** 2 / J


def test_solver_finds_fractional_solutions():
    solver = Solver()
    u = solver.fresh("param")
    assert solver.unify(u * u, m)
    assert solver.resolve(u) == m ** Fraction(1, 2)


def test_solver_rejects_inconsistent_equations():
    solver = Solver()
    u = solver.fresh("zero")
    assert solver.unify(u, m)
    assert not solver.unify(u, s)
    assert not solver.unify(m, s)


def test_solver_keeps_substitution_reduced():
    solver = Solver()
    a, b = solver.fresh("zero"), solver.fresh("zero")
    assert solver.unify(a, b * m)
    assert solver.unify(b, s)
    assert solver.resolve(a) == m * s
