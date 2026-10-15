"""Inline calls to user functions.

Differentiation works on one function at a time, so the copy that is about to
be differentiated has its calls expanded first. The language has no recursion
and `return` is always in tail position, so a call expands to: copy the
arguments into fresh names, run the callee's body, and turn each `return v`
into an assignment to the call's result.
"""
from __future__ import annotations

from ..glue.errors import Diagnostic, DimGradError
from ..middle import tac


def _rename_body(body: list, rename: dict, result: str, result_unit) -> list:
    def r(operand):
        return rename.get(operand, operand) if isinstance(operand, str) else operand

    out = []
    for item in body:
        if isinstance(item, tac.Instr):
            out.append(tac.Instr(rename[item.dst], item.op, tac.with_operands(item, r), item.unit, item.line))
        elif isinstance(item, tac.If):
            out.append(tac.If(r(item.lhs), item.relop, r(item.rhs),
                              _rename_body(item.then, rename, result, result_unit),
                              _rename_body(item.els, rename, result, result_unit), item.line))
        elif isinstance(item, tac.For):
            out.append(tac.For(rename[item.var], item.lo, item.hi,
                               _rename_body(item.body, rename, result, result_unit), item.line))
        elif isinstance(item, tac.Return):
            out.append(tac.Instr(result, "copy", (r(item.value),), result_unit, item.line))
    return out


def inline_calls(func: tac.Function, program: tac.Program) -> tac.Function:
    """A copy of `func` in which every call to a user function is expanded."""
    names = tac.NameGen(tac.all_names(func, program))
    units = dict(func.units)
    variables = set(func.vars)

    def expand(body: list) -> list:
        out = []
        for item in body:
            if isinstance(item, tac.Instr) and item.op == "call":
                callee = program.functions[item.args[0]]
                if callee.kind == "grad":
                    raise DimGradError(Diagnostic(item.line, 1,
                                                  f"cannot differentiate through gradient function '{callee.name}'"))
                callee = inline_calls(callee, program)
                rename = {name: names.named(f"{callee.name}_{name}") for name in callee.units}
                for name, new in rename.items():
                    units[new] = callee.units[name]
                variables.update(rename[v] for v in callee.vars)
                for param, arg in zip(callee.params, item.args[1:]):
                    out.append(tac.Instr(rename[param], "copy", (arg,), callee.units[param], item.line))
                out.extend(_rename_body(callee.body, rename, item.dst, item.unit))
                # The result may now be assigned once per returning branch.
                variables.add(item.dst)
            elif isinstance(item, tac.If):
                out.append(tac.If(item.lhs, item.relop, item.rhs, expand(item.then), expand(item.els), item.line))
            elif isinstance(item, tac.For):
                out.append(tac.For(item.var, item.lo, item.hi, expand(item.body), item.line))
            else:
                out.append(item)
        return out

    body = expand(func.body)
    return tac.Function(func.name, list(func.params), func.ret_unit, body, units, variables,
                        kind=func.kind, line=func.line)
