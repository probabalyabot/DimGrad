import os
from fractions import Fraction
from pathlib import Path

import pytest

from dimgrad import check_source
from dimgrad.frontend import ast_nodes as A
from dimgrad.glue.errors import DimGradError
from dimgrad.units.units import ONE, Unit

GOLDEN = Path(__file__).parent / "golden" / "errors"

kg, m, s, K = Unit.of("kg"), Unit.of("m"), Unit.of("s"), Unit.of("K")
J = kg * m ** 2 / s ** 2

HEADER = "unit kg, m, s, K;\nunit J = kg*m^2/s^2;\n"


def check(body):
    return check_source(HEADER + body)


def errors(body):
    with pytest.raises(DimGradError) as err:
        check(body)
    return [d.message for d in err.value.diagnostics]


# --- the sample of Section 4.1 ------------------------------------------------

def test_sample_report():
    result = check("""
fn energy(mass: kg, v: m/s) -> J {
    return 0.5 * mass * v * v;
}
grad energy wrt v;
""")
    assert result.report() == "fn energy(mass: kg, v: m/s) -> J\ngrad energy wrt v : kg·m/s"
    assert result.grads["energy_grad_v"].unit == kg * m / s


def test_every_expression_gets_a_unit():
    result = check("fn f(mass: kg, v: m/s) -> J { return 0.5 * mass * v * v; }")
    body = result.funcs["f"].node.body
    units = [e.unit for e in A.iter_exprs(body)]
    assert all(u is not None and u.is_concrete for u in units)
    assert units[0] == J                     # the whole product
    assert kg * m / s in units               # 0.5 * mass * v


# --- rules of Section 4.3 ---------------------------------------------------------

def test_literal_zero_takes_the_unit_of_its_context():
    result = check("""
fn distance(v: m/s, dt: s) -> m {
    let total = 0;
    for i in 0..10 {
        total = total + v * dt;
    }
    if (total > 0) {
        return total;
    } else {
        return 0;
    }
}
""")
    let = result.funcs["distance"].node.body.stmts[0]
    assert let.value.unit == m


def test_other_literals_are_unitless():
    assert errors("fn f(x: m) -> m { return x + 1; }") == ["cannot add m and 1"]
    check("fn f(x: m) -> m { return 2 * x + 0.5 * x; }")


def test_constants_carry_units():
    check("const g: m/s^2 = 9.81;\nfn weight(mass: kg) -> kg*m/s^2 { return mass * g; }")
    assert errors("const g: m/s^2 = 9.81;\nfn f(mass: kg) -> kg { return mass + g; }") == [
        "cannot add kg and m/s^2"]


def test_power_and_roots():
    result = check("""
fn f(a: m^2, v: m/s) -> m {
    let side = sqrt(a);
    let edge = cbrt(a * side);
    let inverse = v^-2;
    return side + edge;
}
""")
    stmts = result.funcs["f"].node.body.stmts
    assert stmts[0].value.unit == m
    assert stmts[1].value.unit == m
    assert stmts[2].value.unit == s ** 2 / m ** 2


def test_roots_give_fractional_exponents():
    result = check("fn f(t: s) -> s^(1/2) { return sqrt(t); }")
    assert result.funcs["f"].ret_unit == s ** Fraction(1, 2)


def test_unitless_builtins():
    check("fn f(x: m, scale: m) -> 1 { return sin(x / scale) + exp(-x / scale) + log(x / scale) + cos(0); }")
    assert errors("fn f(x: m) -> 1 { return exp(x); }") == [
        "the argument of 'exp' must be unitless but has unit m"]


def test_variables_are_block_scoped():
    check("""
fn f(x: m, t: s) -> m {
    if (x > 0) {
        let tmp = x;
    } else {
        let tmp = t;
    }
    return x;
}
""")


def test_function_may_be_called_before_its_definition():
    check("""
fn twice(x: m) -> m { return 2 * once(x); }
fn once(x: m) -> m { return x; }
""")


def test_alias_is_shown_in_messages():
    assert errors("fn f(e: J, x: m) -> J { return e + x; }") == [
        "cannot add J (= kg·m^2/s^2) and m"]


# --- derivative units and numerical-method updates (Section 5) ----------------

NEWTON = """
fn g(x: m, a: m^2) -> m^2 { return x^2 - a; }
grad g wrt x;
grad g wrt a;
"""


def test_gradient_unit_is_the_quotient():
    result = check(NEWTON)
    assert result.grads["g_grad_x"].unit == m
    assert result.grads["g_grad_a"].unit == ONE


def test_newton_step_is_accepted():
    check(NEWTON + "fn step(x: m, a: m^2) -> m { return x - g(x, a) / g_grad_x(x, a); }")


def test_newton_step_with_the_wrong_derivative_is_rejected():
    assert errors(NEWTON + "fn step(x: m, a: m^2) -> m { return x - g(x, a) / g_grad_a(x, a); }") == [
        "cannot subtract m^2 from m"]


DESCENT = """
fn f(x: m) -> J { return 0.5 * stiffness * x^2; }
const stiffness: J/m^2 = 50.0;
grad f wrt x;
"""


def test_learning_rate_unit_is_solved():
    result = check(DESCENT + "fn step(x: m, eta: ?) -> m { return x - eta * f_grad_x(x); }")
    step = result.funcs["step"]
    assert step.param_units[1] == m ** 2 / J      # [x]^2 / [f], the result in [25]
    assert "eta: s^2/kg (inferred)" in result.report()


def test_solved_unit_flows_to_callers():
    result = check(DESCENT + """
fn step(x: m, eta: ?) -> m { return x - eta * f_grad_x(x); }
fn run(x0: m, eta: ?) -> m {
    let x = x0;
    for i in 0..5 { x = step(x, eta); }
    return x;
}
""")
    assert result.funcs["run"].param_units[1] == m ** 2 / J


def test_wrong_learning_rate_unit_is_rejected():
    assert errors(DESCENT + "fn step(x: m, eta: m/J) -> m { return x - eta * f_grad_x(x); }") == [
        "cannot subtract 1 from m"]


EULER = "fn rate(T: K, r: 1/s) -> K/s { return -r * T; }\n"


def test_euler_step_size_is_a_time():
    result = check(EULER + "fn step(T: K, r: 1/s, h: ?) -> K { return T + h * rate(T, r); }")
    assert result.funcs["step"].param_units[2] == s


def test_euler_step_without_step_size_is_rejected():
    assert errors(EULER + "fn step(T: K, r: 1/s) -> K { return T + rate(T, r); }") == [
        "cannot add K and K/s"]


def test_placeholder_can_be_fractional():
    result = check("fn f(t: s, vol: ?) -> 1 { return exp(vol * sqrt(t)); }")
    assert result.funcs["f"].param_units[1] == s ** Fraction(-1, 2)


# --- golden error messages --------------------------------------------------

@pytest.mark.parametrize("path", sorted(GOLDEN.glob("*.dg")), ids=lambda p: p.stem)
def test_golden_error(path):
    with pytest.raises(DimGradError) as err:
        check_source(path.read_text(encoding="utf-8"))
    actual = err.value.format(path.name) + "\n"
    expected_path = path.with_suffix(".expected")
    if os.environ.get("UPDATE_GOLDEN"):
        expected_path.write_text(actual, encoding="utf-8", newline="\n")
    assert actual == expected_path.read_text(encoding="utf-8")
