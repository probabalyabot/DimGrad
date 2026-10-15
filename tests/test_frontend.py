from fractions import Fraction

import pytest

from dimgrad import check_source
from dimgrad.frontend import ast_nodes as A
from dimgrad.frontend.lexer import tokenize
from dimgrad.frontend.parser import parse
from dimgrad.frontend.unparse import expr_str, unparse
from dimgrad.glue.errors import DimGradError
from evaluation.common import benchmark_names, load_source

SAMPLE = """
unit kg, m, s;
unit J = kg*m^2/s^2;

fn energy(mass: kg, v: m/s) -> J {
    return 0.5 * mass * v * v;
}

grad energy wrt v;
"""


def kinds(text):
    return [t.kind for t in tokenize(text)][:-1]


def test_lexer_range_is_not_a_float():
    assert kinds("0..10") == ["INT", "..", "INT"]


def test_lexer_numbers():
    tokens = tokenize("1.5e-3 2 3.0 4e2")
    assert [(t.kind, t.text) for t in tokens[:-1]] == [
        ("NUMBER", "1.5e-3"), ("INT", "2"), ("NUMBER", "3.0"), ("NUMBER", "4e2")]


def test_lexer_operators_keywords_and_comments():
    assert kinds("fn f(x: m) -> m // comment\n{ }") == [
        "fn", "NAME", "(", "NAME", ":", "NAME", ")", "->", "NAME", "{", "}"]
    assert kinds("a <= b != c") == ["NAME", "<=", "NAME", "!=", "NAME"]


def test_lexer_positions():
    tokens = tokenize("let x\n  = 1;")
    assert (tokens[2].kind, tokens[2].line, tokens[2].col) == ("=", 2, 3)


def test_lexer_rejects_unknown_character():
    with pytest.raises(DimGradError) as err:
        tokenize("let x = 1 % 2;")
    assert err.value.diagnostics[0].col == 11


def test_parse_sample_program():
    program = parse(SAMPLE)
    units, alias, fn, grad = program.decls
    assert [name for name, _, _ in units.names] == ["kg", "m", "s"]
    assert alias.name == "J"
    assert alias.expr.factors == [("kg", 1), ("m", 2), ("s", -2)]
    assert fn.name == "energy"
    assert [(p.name, p.unit.factors) for p in fn.params] == [("mass", [("kg", 1)]), ("v", [("m", 1), ("s", -1)])]
    assert (grad.func, grad.wrt) == ("energy", "v")
    ret = fn.body.stmts[0]
    assert isinstance(ret, A.Return)
    assert expr_str(ret.value) == "0.5 * mass * v * v"


def body_expr(text):
    program = parse(f"fn f(a: 1, b: 1, c: 1) -> 1 {{ return {text}; }}")
    return program.decls[0].body.stmts[0].value


def test_precedence_and_associativity():
    e = body_expr("a - b - c")
    assert isinstance(e.lhs, A.BinOp) and e.lhs.op == "-"       # (a - b) - c
    e = body_expr("a + b * c")
    assert isinstance(e.rhs, A.BinOp) and e.rhs.op == "*"
    e = body_expr("-a^2")
    assert isinstance(e, A.Neg) and isinstance(e.operand, A.Pow)  # -(a^2)
    e = body_expr("a^-2")
    assert isinstance(e, A.Pow) and e.exp == -2
    e = body_expr("(a + b) * c")
    assert isinstance(e.lhs, A.BinOp) and e.lhs.op == "+"


@pytest.mark.parametrize("text", [
    "a - (b - c)", "a / (b * c)", "(a + b) * c", "-(a + b)", "(-a)^2", "-a^2", "a - -b",
    "sqrt(a * b) + c", "a * (b + c) / (a - b)",
])
def test_expression_text_round_trips(text):
    assert expr_str(body_expr(text)) == text


def test_unit_expressions():
    program = parse("fn f(a: 1/s^(1/2), b: m^-2, c: ?) -> 1 { return 1; }")
    a, b, c = program.decls[0].params
    assert a.unit.factors == [("s", Fraction(-1, 2))]
    assert b.unit.factors == [("m", -2)]
    assert c.unit.placeholder


@pytest.mark.parametrize("source, message", [
    ("unit m", "expected ';' but found end of file"),
    ("fn f(x: m) -> m { let y = x return y; }", "expected ';' but found 'return'"),
    ("fn f(x: m) -> ? { return x; }", "'?' is only allowed as the unit of a parameter"),
    ("let x = 1;", "expected a declaration (unit, const, fn or grad) but found 'let'"),
    ("fn f(x: m) -> m { return x ^ y; }", "expected an integer exponent but found 'y'"),
    ("fn f(x: m) -> m { if (x) { return x; } }", "expected a comparison operator but found ')'"),
    ("fn f(x: m) -> m { for i in 0..n { } return x; }", "expected an integer loop bound but found 'n'"),
])
def test_syntax_errors(source, message):
    with pytest.raises(DimGradError) as err:
        parse(source)
    assert err.value.diagnostics[0].message == message


@pytest.mark.parametrize("name", benchmark_names())
def test_unparse_round_trips(name):
    source = load_source(name)
    text = unparse(parse(source))
    assert unparse(parse(text)) == text
    assert check_source(text).report() == check_source(source).report()
