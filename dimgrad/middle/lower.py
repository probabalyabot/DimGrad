"""Lower the checked syntax tree to three-address code.

Invariant relied on by the differentiation pass: an operation always writes a
fresh temporary, and a named variable is only ever written by a `copy`. So an
instruction never overwrites one of its own operands.
"""
from __future__ import annotations

from ..frontend import ast_nodes as A
from ..units.checker import BUILTINS, CheckResult
from ..units.symtab import FuncInfo
from ..units.units import ONE
from . import tac

_BINARY_OPS = {"+": "add", "-": "sub", "*": "mul", "/": "div"}


class _FunctionLowerer:
    def __init__(self, info: FuncInfo, const_names):
        self.info = info
        self.names = tac.NameGen(set(info.param_names) | set(const_names))
        self.units = dict(zip(info.param_names, info.param_units))
        self.vars: set[str] = set()
        # Source name -> name used in the code. They differ only when two
        # sibling blocks declare variables with the same name.
        self.scopes: list[dict[str, str]] = [{p: p for p in info.param_names}]

    def resolve(self, name: str) -> str:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return name  # a global constant

    def declare(self, name: str, unit) -> str:
        unique = self.names.named(name)
        self.scopes[-1][name] = unique
        self.units[unique] = unit
        return unique

    def emit(self, out: list, op: str, args: tuple, unit, line: int) -> str:
        dst = self.names.temp()
        self.units[dst] = unit
        out.append(tac.Instr(dst, op, args, unit, line))
        return dst

    def expr(self, node: A.Expr, out: list):
        if isinstance(node, A.Num):
            return float(node.value)
        if isinstance(node, A.Name):
            return self.resolve(node.name)
        if isinstance(node, A.Neg):
            if isinstance(node.operand, A.Num):
                return -float(node.operand.value)
            return self.emit(out, "neg", (self.expr(node.operand, out),), node.unit, node.line)
        if isinstance(node, A.BinOp):
            lhs = self.expr(node.lhs, out)
            rhs = self.expr(node.rhs, out)
            return self.emit(out, _BINARY_OPS[node.op], (lhs, rhs), node.unit, node.line)
        if isinstance(node, A.Pow):
            return self.emit(out, "pow", (self.expr(node.base, out), node.exp), node.unit, node.line)
        if isinstance(node, A.Call):
            args = tuple(self.expr(arg, out) for arg in node.args)
            if node.name in BUILTINS:
                return self.emit(out, node.name, args, node.unit, node.line)
            return self.emit(out, "call", (node.name,) + args, node.unit, node.line)
        raise TypeError(f"not an expression: {node!r}")

    def block(self, block: A.Block, new_scope: bool = True) -> list:
        out: list = []
        if new_scope:
            self.scopes.append({})
        for stmt in block.stmts:
            if isinstance(stmt, A.Let):
                value = self.expr(stmt.value, out)
                name = self.declare(stmt.name, stmt.value.unit)
                self.vars.add(name)
                out.append(tac.Instr(name, "copy", (value,), stmt.value.unit, stmt.line))
            elif isinstance(stmt, A.Assign):
                value = self.expr(stmt.value, out)
                name = self.resolve(stmt.name)
                out.append(tac.Instr(name, "copy", (value,), self.units[name], stmt.line))
            elif isinstance(stmt, A.If):
                lhs = self.expr(stmt.cond.lhs, out)
                rhs = self.expr(stmt.cond.rhs, out)
                then = self.block(stmt.then)
                els = self.block(stmt.els) if stmt.els is not None else []
                out.append(tac.If(lhs, stmt.cond.op, rhs, then, els, stmt.line))
            elif isinstance(stmt, A.For):
                self.scopes.append({})
                var = self.declare(stmt.var, ONE)
                body = self.block(stmt.body, new_scope=False)
                self.scopes.pop()
                out.append(tac.For(var, stmt.lo, stmt.hi, body, stmt.line))
            elif isinstance(stmt, A.Return):
                out.append(tac.Return(self.expr(stmt.value, out), None, stmt.line))
        if new_scope:
            self.scopes.pop()
        return out

    def lower(self) -> tac.Function:
        body = self.block(self.info.node.body, new_scope=False)
        return tac.Function(self.info.name, list(self.info.param_names), self.info.ret_unit,
                            body, self.units, self.vars, line=self.info.node.line)


def lower_program(checked: CheckResult) -> tac.Program:
    consts = {name: (c.unit, c.value) for name, c in checked.consts.items()}
    functions = {name: _FunctionLowerer(info, consts).lower() for name, info in checked.funcs.items()}
    return tac.Program(consts, functions, checked.base_units, checked.aliases)
