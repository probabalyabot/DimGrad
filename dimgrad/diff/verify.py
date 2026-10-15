"""Re-derive the unit of every instruction from its operands and compare.

The unit checker works on the syntax tree. This pass checks the generated
code instead, with the same unit algebra: after differentiation it confirms
that both terms of every tangent rule agree and that each tangent has the
unit [t]/[x], and after each optimisation it confirms that the rewrite kept
every unit. A failure here is a compiler bug, never a user error.
"""
from __future__ import annotations

from fractions import Fraction

from ..middle import tac
from ..units.units import ONE

_ROOTS = {"sqrt": Fraction(1, 2), "cbrt": Fraction(1, 3)}


class VerifyError(AssertionError):
    pass


def verify_function(func: tac.Function, program: tac.Program):
    fmt = program.fmt

    def fail(message: str):
        raise VerifyError(f"{func.name}: {message}")

    def unit_of(operand):
        """Unit of an operand; None for the constant 0, which fits any unit."""
        if isinstance(operand, float):
            return None if operand == 0 else ONE
        if operand in func.units:
            return func.units[operand]
        if operand in program.consts:
            return program.consts[operand][0]
        fail(f"'{operand}' has no unit")

    def result_unit(instr: tac.Instr):
        u = [unit_of(a) for a in tac.operands(instr)]
        op = instr.op
        if op in ("copy", "neg"):
            return u[0]
        if op in ("add", "sub"):
            if u[0] is None:
                return u[1]
            if u[1] is not None and u[0] != u[1]:
                fail(f"'{tac.instr_text(instr)}' combines {fmt(u[0])} and {fmt(u[1])}")
            return u[0]
        if op == "mul":
            return None if None in u else u[0] * u[1]
        if op == "div":
            return None if None in u else u[0] / u[1]
        if op == "pow":
            return None if u[0] is None else u[0] ** instr.args[1]
        if op in _ROOTS:
            return None if u[0] is None else u[0] ** _ROOTS[op]
        if op == "call":
            callee = program.functions[instr.args[0]]
            for param, unit in zip(callee.params, u):
                if unit is not None and unit != callee.units[param]:
                    fail(f"'{tac.instr_text(instr)}' passes {fmt(unit)} for '{param}'")
            return callee.ret_unit
        if u[0] is not None and u[0] != ONE:
            fail(f"'{tac.instr_text(instr)}' applies {op} to {fmt(u[0])}")
        return ONE

    def check(body: list):
        for item in body:
            if isinstance(item, tac.Instr):
                if func.units.get(item.dst) != item.unit:
                    fail(f"'{item.dst}' is recorded with a different unit than its definition")
                unit = result_unit(item)
                if unit is not None and unit != item.unit:
                    fail(f"'{tac.instr_text(item)}' has unit {fmt(unit)} but is declared {fmt(item.unit)}")
            elif isinstance(item, tac.If):
                lhs, rhs = unit_of(item.lhs), unit_of(item.rhs)
                if lhs is not None and rhs is not None and lhs != rhs:
                    fail(f"condition compares {fmt(lhs)} and {fmt(rhs)}")
                check(item.then)
                check(item.els)
            elif isinstance(item, tac.For):
                check(item.body)
            elif isinstance(item, tac.Return):
                value_unit = func.value_unit if func.kind == "grad" else func.ret_unit
                unit = unit_of(item.value)
                if unit is not None and unit != value_unit:
                    fail(f"returns {fmt(unit)} but should return {fmt(value_unit)}")
                if func.kind == "grad":
                    unit = unit_of(item.tangent)
                    if unit is not None and unit != func.ret_unit:
                        fail(f"derivative has unit {fmt(unit)} but should have {fmt(func.ret_unit)}")

    check(func.body)
    if func.kind == "grad":
        x_unit = func.units[func.wrt]
        for name, tangent in func.tangents.items():
            if name in func.units and tangent in func.units:
                if func.units[tangent] != func.units[name] / x_unit:
                    fail(f"tangent '{tangent}' does not have the unit of '{name}' divided by the unit of '{func.wrt}'")
