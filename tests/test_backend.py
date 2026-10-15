"""Lowering, the interpreter, differentiation, the unit verifier and the optimiser."""
import dataclasses
import math

import pytest

from dimgrad import compile_source
from dimgrad.diff.verify import VerifyError, verify_function
from dimgrad.glue.fd import TOLERANCE, central_difference, relative_error
from dimgrad.middle import tac
from dimgrad.middle.interp import evaluate, normcdf
from dimgrad.units.units import Unit
from evaluation.common import benchmark_names, grid_points, load_grids, load_source

kg, m, s = Unit.of("kg"), Unit.of("m"), Unit.of("s")

ENERGY = """
unit kg, m, s;
unit J = kg*m^2/s^2;
fn energy(mass: kg, v: m/s) -> J {
    return 0.5 * mass * v * v;
}
grad energy wrt v;
"""


def instrs(func):
    return [tac.instr_text(i) for i in tac.walk(func.body) if isinstance(i, tac.Instr)]


def returned(func):
    """(value, derivative) operands of the function's final return."""
    last = func.body[-1]
    return last.value, last.tangent


def close(a, b, tol=1e-12):
    return a == b or abs(a - b) <= tol * max(abs(a), abs(b))


# --- the worked example of Section 6.2 -----------------------------------------

def test_lowering_of_the_sample():
    compiled = compile_source(ENERGY, optimize=False)
    energy = compiled.raw.functions["energy"]
    assert instrs(energy) == ["t1 = 0.5 * mass", "t2 = t1 * v", "t3 = t2 * v"]
    assert [energy.units[t] for t in ("t1", "t2", "t3")] == [kg, kg * m / s, kg * m ** 2 / s ** 2]


def test_tangent_code_before_optimisation():
    grad = compile_source(ENERGY, optimize=False).raw.functions["energy_grad_v"]
    # The paper writes dt1 = 0.0*mass + 0.5*0.0 on one line; in strict
    # three-address form each product gets its own temporary.
    assert instrs(grad) == [
        "t1 = 0.5 * mass", "t4 = 0.0 * mass", "t5 = 0.5 * 0.0", "d_t1 = t4 + t5",
        "t2 = t1 * v", "t6 = d_t1 * v", "t7 = t1 * 1.0", "d_t2 = t6 + t7",
        "t3 = t2 * v", "t8 = d_t2 * v", "t9 = t2 * 1.0", "d_t3 = t8 + t9",
    ]
    assert grad.units["d_t3"] == kg * m / s          # J / (m/s): a momentum
    assert grad.units["d_t1"] == kg * s / m          # kg / (m/s)
    assert grad.ret_unit == kg * m / s


def test_tangent_code_after_optimisation():
    compiled = compile_source(ENERGY)
    grad = compiled.program.functions["energy_grad_v"]
    # The paper stops at dt3 = t1*v + t2; common-subexpression elimination
    # also sees that t1*v is t2.
    assert instrs(grad) == ["t1 = 0.5 * mass", "t2 = t1 * v", "t3 = t2 * v", "d_t3 = t2 + t2"]
    assert compiled.stats["energy_grad_v"] == (12, 4)
    assert evaluate(compiled.program, "energy_grad_v", [2.0, 3.0]) == (6.0, 9.0)  # mass*v, energy


# --- each differentiation rule against its closed form ---------------------------

RULES = [
    ("x + a", lambda x, a: 1.0),
    ("a - x", lambda x, a: -1.0),
    ("-x", lambda x, a: -1.0),
    ("x * a", lambda x, a: a),
    ("x * x * x", lambda x, a: 3 * x * x),
    ("a / x", lambda x, a: -a / x ** 2),
    ("x / a", lambda x, a: 1 / a),
    ("x / (x + a)", lambda x, a: a / (x + a) ** 2),
    ("x^5", lambda x, a: 5 * x ** 4),
    ("x^2", lambda x, a: 2 * x),
    ("x^1", lambda x, a: 1.0),
    ("x^0", lambda x, a: 0.0),
    ("x^-2", lambda x, a: -2 / x ** 3),
    ("sqrt(x)", lambda x, a: 0.5 / math.sqrt(x)),
    ("cbrt(x)", lambda x, a: x ** (-2 / 3) / 3),
    ("sin(x)", lambda x, a: math.cos(x)),
    ("cos(x)", lambda x, a: -math.sin(x)),
    ("exp(x)", lambda x, a: math.exp(x)),
    ("log(x)", lambda x, a: 1 / x),
    ("normcdf(x)", lambda x, a: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)),
    ("exp(sin(x) * a) / sqrt(x)", lambda x, a: math.exp(math.sin(x) * a) * (a * math.cos(x) / math.sqrt(x)
                                                                             - 0.5 * x ** -1.5)),
    ("a", lambda x, a: 0.0),
    ("3", lambda x, a: 0.0),
]


@pytest.mark.parametrize("optimize", [False, True], ids=["raw", "optimised"])
@pytest.mark.parametrize("expr, derivative", RULES, ids=[r[0] for r in RULES])
def test_differentiation_rule(expr, derivative, optimize):
    source = f"fn f(x: 1, a: 1) -> 1 {{ return {expr}; }}\ngrad f wrt x;"
    program = compile_source(source, optimize=optimize).program
    for x, a in [(0.7, 1.3), (2.5, -0.4)]:
        deriv, value = evaluate(program, "f_grad_x", [x, a])
        assert close(deriv, derivative(x, a))
        assert close(value, evaluate(program, "f", [x, a]))


# --- control flow and calls ---------------------------------------------------

CONTROL = """
fn magnitude(x: 1) -> 1 {
    if (x < 0) {
        return -x;
    } else {
        return x;
    }
}

fn power_loop(x: 1) -> 1 {
    let p = 1;
    for i in 0..5 {
        p = p * x;
    }
    return p;
}

fn weighted(x: 1) -> 1 {
    let total = 0;
    for i in 1..4 {
        for j in 0..2 {
            total = total + i * magnitude(x) + j;
        }
    }
    return total * x;
}

fn sibling_scopes(x: 1) -> 1 {
    let y = x;
    if (x > 1) {
        let tmp = x * x;
        y = tmp;
    } else {
        let tmp = 3 * x;
        y = tmp;
    }
    return y;
}

grad magnitude wrt x;
grad power_loop wrt x;
grad weighted wrt x;
grad sibling_scopes wrt x;
"""


@pytest.mark.parametrize("optimize", [False, True], ids=["raw", "optimised"])
def test_control_flow(optimize):
    program = compile_source(CONTROL, optimize=optimize).program

    def run(name, x):
        return evaluate(program, name, [x])

    assert run("magnitude", -2.0) == 2.0
    assert run("magnitude_grad_x", -2.0) == (-1.0, 2.0)
    assert run("magnitude_grad_x", 2.0) == (1.0, 2.0)
    assert run("power_loop", 2.0) == 32.0
    assert run("power_loop_grad_x", 2.0) == (80.0, 32.0)          # 5 x^4
    # total = sum over i in 1..3, j in 0..1 of (i*|x| + j) = 12|x| + 3
    assert run("weighted", 2.0) == (12 * 2.0 + 3) * 2.0
    assert run("weighted_grad_x", 2.0)[0] == 24 * 2.0 + 3          # x > 0: f = 12x^2 + 3x
    assert run("weighted_grad_x", -2.0)[0] == -24 * -2.0 + 3       # x < 0: f = -12x^2 + 3x
    assert run("sibling_scopes_grad_x", 3.0) == (6.0, 9.0)
    assert run("sibling_scopes_grad_x", 0.5) == (3.0, 1.5)


def test_sibling_blocks_get_distinct_names():
    func = compile_source(CONTROL, optimize=False).raw.functions["sibling_scopes"]
    assert {"tmp", "tmp_2"} <= set(func.units)


def test_calls_are_kept_in_the_primal_and_inlined_in_the_gradient():
    raw = compile_source(CONTROL, optimize=False).raw
    ops = lambda name: {i.op for i in tac.walk(raw.functions[name].body) if isinstance(i, tac.Instr)}
    assert "call" in ops("weighted")
    assert "call" not in ops("weighted_grad_x")


def test_gradient_function_can_be_called():
    source = load_source("newton_root")
    program = compile_source(source).program
    # One Newton step for sqrt(9) from x = 3 stays at 3; six steps from 1 converge.
    assert evaluate(program, "newton_step", [3.0, 9.0]) == 3.0
    assert close(evaluate(program, "root", [1.0, 9.0]), 3.0)
    assert close(evaluate(program, "root", [5.0, 2.0]), math.sqrt(2.0))


# --- the unit verifier --------------------------------------------------------

def test_verifier_accepts_generated_code():
    compiled = compile_source(ENERGY)
    for program in (compiled.raw, compiled.program):
        for func in program.functions.values():
            verify_function(func, program)


def test_verifier_catches_a_wrong_tangent_unit():
    raw = compile_source(ENERGY, optimize=False).raw
    grad = raw.functions["energy_grad_v"]
    broken_units = dict(grad.units, d_t3=kg)
    body = [dataclasses.replace(i, unit=kg) if isinstance(i, tac.Instr) and i.dst == "d_t3" else i
            for i in grad.body]
    with pytest.raises(VerifyError):
        verify_function(dataclasses.replace(grad, units=broken_units, body=body), raw)


def test_verifier_catches_a_wrong_rule():
    raw = compile_source(ENERGY, optimize=False).raw
    grad = raw.functions["energy_grad_v"]
    # Replace d_t2 = t6 + t7 by d_t2 = t6 + t2: adds kg and kg*m/s.
    body = [dataclasses.replace(i, args=("t6", "t2")) if isinstance(i, tac.Instr) and i.dst == "d_t2" else i
            for i in grad.body]
    with pytest.raises(VerifyError):
        verify_function(dataclasses.replace(grad, body=body), raw)


# --- the optimiser ------------------------------------------------------------

def optimised(body, grad=""):
    source = f"fn f(x: 1, y: 1) -> 1 {{ {body} }}\n{grad}"
    return compile_source(source).program.functions


def test_constant_folding_and_identities():
    f = optimised("return 2 * 3 * x + 0 * y + (y - y) + x^1 * 1;")["f"]
    assert instrs(f) == ["t2 = 6.0 * x", "t9 = t2 + x"]


def test_common_subexpressions():
    f = optimised("return x * y + y * x;")["f"]
    assert instrs(f) == ["t1 = x * y", "t3 = t1 + t1"]


def test_dead_code_is_removed():
    f = optimised("let unused = sin(x) * y; let z = x; for i in 0..3 { let w = z * y; } return z;")["f"]
    assert instrs(f) == []
    assert returned(f) == ("x", None)


def test_constant_condition_selects_a_branch():
    f = optimised("let k = 2; if (k > 1) { return x; } else { return y; }")["f"]
    assert len(f.body) == 1 and returned(f) == ("x", None)


def test_reassigned_variable_is_not_treated_as_constant():
    f = optimised("let k = 1; for i in 0..3 { k = k * x; } return k * y;")["f"]
    program = compile_source("fn f(x: 1, y: 1) -> 1 { let k = 1; for i in 0..3 { k = k * x; } return k * y; }").program
    assert evaluate(program, "f", [2.0, 5.0]) == 40.0
    assert any(isinstance(item, tac.For) for item in f.body)


def test_zero_tangents_are_folded_through_loops():
    funcs = optimised("let acc = 0; for i in 0..5 { acc = acc + y * y; } return x * acc;", "grad f wrt x;")
    grad = funcs["f_grad_x"]
    # acc does not depend on x, so none of its tangent code survives: d f/d x = acc.
    assert not any(name.startswith("d_") for name in grad.units)
    value, derivative = returned(grad)
    assert derivative == "acc" and value == returned(funcs["f"])[0]
    assert tac.count_instrs(grad.body) == tac.count_instrs(funcs["f"].body)


def test_optimiser_reaches_a_fixed_point():
    from dimgrad.backend.optimizer import optimize_function
    for name in benchmark_names():
        program = compile_source(load_source(name)).program
        for func in program.functions.values():
            again = optimize_function(func)
            assert tac.count_instrs(again.body) == tac.count_instrs(func.body)


# --- every benchmark ----------------------------------------------------------

GRIDS = load_grids()
CASES = [(bench, func) for bench in benchmark_names() for func in GRIDS[bench]]


def test_every_benchmark_has_a_grid():
    assert set(GRIDS) == set(benchmark_names())
    assert len(benchmark_names()) >= 10


@pytest.mark.parametrize("bench, func", CASES, ids=[f"{b}:{f}" for b, f in CASES])
def test_optimised_code_agrees_with_unoptimised(bench, func):
    compiled = compile_source(load_source(bench))
    names = [func] + [g.name for g in compiled.check.grads.values() if g.func == func]
    params = compiled.program.functions[func].params
    for point in grid_points(GRIDS[bench][func], params):
        for name in names:
            raw = evaluate(compiled.raw, name, point)
            opt = evaluate(compiled.program, name, point)
            raw, opt = (raw, opt) if isinstance(raw, tuple) else ((raw,), (opt,))
            assert all(close(a, b, 1e-11) for a, b in zip(raw, opt)), (name, point, raw, opt)


@pytest.mark.parametrize("bench, func", CASES, ids=[f"{b}:{f}" for b, f in CASES])
def test_gradients_match_finite_differences(bench, func):
    compiled = compile_source(load_source(bench))
    program = compiled.program
    params = program.functions[func].params
    for grad in (g for g in compiled.check.grads.values() if g.func == func):
        for point in grid_points(GRIDS[bench][func], params):
            ad, value = evaluate(program, grad.name, point)
            fd, h = central_difference(lambda args: evaluate(program, func, args), point, grad.index)
            assert relative_error(ad, fd, value, h) < TOLERANCE, (grad.name, point, ad, fd)


def test_normcdf_matches_reference_values():
    assert close(normcdf(0.0), 0.5)
    assert close(normcdf(1.959963984540054), 0.975, 1e-9)
