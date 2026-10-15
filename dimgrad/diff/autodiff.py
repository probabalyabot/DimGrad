"""Forward-mode differentiation over three-address code (Section 5).

For `grad f wrt x`, every instruction `t = ...` gets a tangent instruction
`d_t = ...` placed right after it. The tangent of x is the constant 1 and the
tangents of the other inputs are 0. Each tangent is given the unit [t]/[x];
verify.py then re-derives every unit from the operands, which is the check
that both terms of each tangent rule agree.
"""
from __future__ import annotations

from ..glue.errors import Diagnostic, DimGradError
from ..middle import tac
from ..units.units import ONE

_INV_SQRT_2PI = 0.3989422804014327


def differentiate(func: tac.Function, wrt: str, program: tac.Program, name: str | None = None) -> tac.Function:
    """The gradient function of `func` (already inlined) with respect to parameter `wrt`."""
    x_unit = func.units[wrt]
    names = tac.NameGen(tac.all_names(func, program))
    units = dict(func.units)
    tangents: dict[str, str] = {}
    inactive = set(func.params) | set(program.consts)
    inactive |= {item.var for item in tac.walk(func.body) if isinstance(item, tac.For)}

    def unit_of(operand):
        if isinstance(operand, float):
            return ONE
        return units[operand] if operand in units else program.consts[operand][0]

    def tangent_name(operand: str) -> str:
        if operand not in tangents:
            tangents[operand] = names.named(f"d_{operand}")
            units[tangents[operand]] = units[operand] / x_unit
        return tangents[operand]

    def tan(operand):
        if isinstance(operand, float):
            return 0.0
        if operand == wrt:
            return 1.0
        if operand in inactive:
            return 0.0
        return tangent_name(operand)

    def rule(instr: tac.Instr, out: list):
        line = instr.line

        def tmp(op, args, unit):
            dst = names.temp()
            units[dst] = unit
            out.append(tac.Instr(dst, op, args, unit, line))
            return dst

        t, op, args = instr.dst, instr.op, instr.args
        dt = tangent_name(t)
        dt_unit = units[dt]

        def define(op, args):
            out.append(tac.Instr(dt, op, args, dt_unit, line))

        if op == "call":
            raise DimGradError(Diagnostic(line, 1, f"internal error: call to '{args[0]}' was not inlined"))
        a = args[0]
        da = tan(a)
        if op == "copy":
            define("copy", (da,))
        elif op == "neg":
            define("neg", (da,))
        elif op in ("add", "sub"):
            define(op, (da, tan(args[1])))
        elif op == "mul":
            b = args[1]
            p1 = tmp("mul", (da, b), dt_unit)
            p2 = tmp("mul", (a, tan(b)), dt_unit)
            define("add", (p1, p2))
        elif op == "div":
            # dt = (da*b - a*db) / b^2
            b = args[1]
            num_unit = unit_of(a) * unit_of(b) / x_unit
            p1 = tmp("mul", (da, b), num_unit)
            p2 = tmp("mul", (a, tan(b)), num_unit)
            p3 = tmp("sub", (p1, p2), num_unit)
            p4 = tmp("mul", (b, b), unit_of(b) ** 2)
            define("div", (p3, p4))
        elif op == "pow":
            # dt = n * a^(n-1) * da
            n = args[1]
            if n == 0:
                define("copy", (0.0,))
            else:
                p1 = tmp("pow", (a, n - 1), unit_of(a) ** (n - 1))
                p2 = tmp("mul", (float(n), p1), unit_of(a) ** (n - 1))
                define("mul", (p2, da))
        elif op == "sqrt":
            define("div", (da, tmp("mul", (2.0, t), units[t])))
        elif op == "cbrt":
            p1 = tmp("mul", (t, t), units[t] ** 2)
            define("div", (da, tmp("mul", (3.0, p1), units[t] ** 2)))
        elif op == "sin":
            define("mul", (tmp("cos", (a,), ONE), da))
        elif op == "cos":
            p1 = tmp("sin", (a,), ONE)
            define("mul", (tmp("neg", (p1,), ONE), da))
        elif op == "exp":
            define("mul", (t, da))
        elif op == "log":
            define("div", (da, a))
        elif op == "normcdf":
            # The derivative is the standard normal density exp(-a^2/2)/sqrt(2*pi).
            p1 = tmp("mul", (a, a), ONE)
            p2 = tmp("mul", (-0.5, p1), ONE)
            p3 = tmp("exp", (p2,), ONE)
            define("mul", (tmp("mul", (_INV_SQRT_2PI, p3), ONE), da))
        else:
            raise ValueError(f"no differentiation rule for '{op}'")

    def block(body: list) -> list:
        out = []
        for item in body:
            if isinstance(item, tac.Instr):
                out.append(item)
                rule(item, out)
            elif isinstance(item, tac.If):
                # The tangent follows the branch that runs (Section 5.3).
                out.append(tac.If(item.lhs, item.relop, item.rhs, block(item.then), block(item.els), item.line))
            elif isinstance(item, tac.For):
                out.append(tac.For(item.var, item.lo, item.hi, block(item.body), item.line))
            elif isinstance(item, tac.Return):
                out.append(tac.Return(item.value, tan(item.value), item.line))
        return out

    body = block(func.body)
    variables = set(func.vars) | {tangents[v] for v in func.vars if v in tangents}
    return tac.Function(name or f"{func.name}_grad_{wrt}", list(func.params), func.ret_unit / x_unit,
                        body, units, variables, kind="grad", of=func.name, wrt=wrt,
                        value_unit=func.ret_unit, tangents=tangents, line=func.line)
