"""Reference interpreter for three-address code.

Used as a testing oracle: it runs the same code the C emitter prints, so the
optimiser and the emitter can each be checked against it. Arithmetic follows
IEEE 754 the way the generated C does (division by zero gives inf or nan).
"""
from __future__ import annotations

import math

from . import tac

_RELOPS = {
    "<": lambda a, b: a < b, ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}


def _div(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        if a == 0 or a != a:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)


def _pow(a: float, n: int) -> float:
    try:
        return a ** n
    except ZeroDivisionError:
        return math.inf
    except OverflowError:
        return math.inf if a > 0 or n % 2 == 0 else -math.inf


def _guard(fn, on_error=math.nan):
    def wrapped(x: float) -> float:
        try:
            return fn(x)
        except ValueError:
            return on_error
        except OverflowError:
            return math.inf
    return wrapped


def _cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def normcdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _log(x: float) -> float:
    if x == 0:
        return -math.inf
    return math.log(x)


BUILTINS = {
    "sqrt": _guard(math.sqrt), "cbrt": _cbrt, "sin": _guard(math.sin), "cos": _guard(math.cos),
    "exp": _guard(math.exp), "log": _guard(_log), "normcdf": normcdf,
}

BINARY = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b, "div": _div,
}


def apply(op: str, args: list) -> float:
    """Evaluate one operation on constant operands (shared with the constant folder)."""
    if op == "copy":
        return args[0]
    if op == "neg":
        return -args[0]
    if op in BINARY:
        return BINARY[op](args[0], args[1])
    if op == "pow":
        return _pow(args[0], args[1])
    return BUILTINS[op](args[0])


def _run(program: tac.Program, body: list, env: dict):
    def value(operand):
        return operand if isinstance(operand, float) else env[operand]

    for item in body:
        if isinstance(item, tac.Instr):
            if item.op == "call":
                result = evaluate(program, item.args[0], [value(a) for a in item.args[1:]])
                env[item.dst] = result[0] if isinstance(result, tuple) else result
            elif item.op == "pow":
                env[item.dst] = _pow(value(item.args[0]), item.args[1])
            else:
                env[item.dst] = apply(item.op, [value(a) for a in item.args])
        elif isinstance(item, tac.If):
            branch = item.then if _RELOPS[item.relop](value(item.lhs), value(item.rhs)) else item.els
            result = _run(program, branch, env)
            if result is not None:
                return result
        elif isinstance(item, tac.For):
            for i in range(item.lo, item.hi):
                env[item.var] = float(i)
                result = _run(program, item.body, env)
                if result is not None:
                    return result
        elif isinstance(item, tac.Return):
            if item.tangent is None:
                return (value(item.value),)
            return (value(item.tangent), value(item.value))
    return None


def evaluate(program: tac.Program, name: str, args: list[float]):
    """Run a function. A gradient function returns (derivative, value)."""
    func = program.functions[name]
    if len(args) != len(func.params):
        raise ValueError(f"{name} expects {len(func.params)} arguments")
    env = {const: value for const, (_, value) in program.consts.items()}
    env.update(zip(func.params, (float(a) for a in args)))
    result = _run(program, func.body, env)
    return result[0] if len(result) == 1 else result
