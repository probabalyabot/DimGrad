"""The generated C, compiled with GCC, must agree with the interpreter."""
import pytest

from dimgrad import compile_source
from dimgrad.backend.cc import compile_c, find_gcc, run_queries
from dimgrad.backend.emit_c import emit_c
from dimgrad.middle.interp import evaluate
from evaluation.common import benchmark_names, grid_points, load_grids, load_source

pytestmark = pytest.mark.skipif(find_gcc() is None, reason="gcc is not installed")

GRIDS = load_grids()


def close(a, b, tol=1e-10):
    return a == b or abs(a - b) <= tol * max(abs(a), abs(b))


def queries_for(compiled, bench):
    queries = []
    for func, spec in GRIDS[bench].items():
        names = [func] + [g.name for g in compiled.check.grads.values() if g.func == func]
        for point in grid_points(spec, compiled.program.functions[func].params):
            queries += [(name, point) for name in names]
    return queries


@pytest.mark.parametrize("optimize", [True, False], ids=["optimised", "raw"])
@pytest.mark.parametrize("bench", benchmark_names())
def test_c_agrees_with_interpreter(bench, optimize, tmp_path):
    compiled = compile_source(load_source(bench), optimize=optimize)
    c_text = emit_c(compiled.program, f"{bench}.dg", harness=True)
    # Optimised output must be warning-free; unoptimised output keeps unused tangents.
    exe = compile_c(c_text, tmp_path, bench, werror=optimize)
    queries = queries_for(compiled, bench)
    results = run_queries(exe, queries)
    assert len(results) == len(queries)
    for (name, point), got in zip(queries, results):
        want = evaluate(compiled.program, name, point)
        want = list(want) if isinstance(want, tuple) else [want]
        assert len(got) == len(want)
        assert all(close(g, w) for g, w in zip(got, want)), (name, point, got, want)


def test_reserved_names_are_renamed(tmp_path):
    source = """
fn main(value: 1, double: 1) -> 1 {
    let log = value * double;
    let int = exp(log);
    return int + log;
}
grad main wrt value;
"""
    compiled = compile_source(source, optimize=False)
    exe = compile_c(emit_c(compiled.program, harness=True), tmp_path, "reserved", werror=False)
    (deriv, value), = run_queries(exe, [("main_grad_value", [0.5, 2.0])])
    want = evaluate(compiled.program, "main_grad_value", [0.5, 2.0])
    assert close(deriv, want[0]) and close(value, want[1])


def test_gradient_accepts_null_value_pointer(tmp_path):
    source = "fn f(x: 1) -> 1 { return x * x; }\ngrad f wrt x;\nfn g(x: 1) -> 1 { return f_grad_x(x); }"
    compiled = compile_source(source)
    c_text = emit_c(compiled.program, harness=True)
    assert "f_grad_x(x, NULL)" in c_text
    exe = compile_c(c_text, tmp_path, "null")
    assert run_queries(exe, [("g", [3.0])]) == [[6.0]]
