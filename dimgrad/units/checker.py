"""Unit checker: computes and checks the unit of every expression (Section 4.3).

Each rule that requires two units to be equal hands the pair to the solver.
That one mechanism checks ordinary programs, gives a literal 0 the unit its
context requires, and solves the unit of a parameter declared with `?`.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..frontend import ast_nodes as A
from ..frontend.unparse import expr_str, unit_expr_str
from ..glue.errors import Diagnostic, DimGradError
from .solver import Solver
from .symtab import ConstInfo, FuncInfo, GradInfo, Scopes, VarInfo
from .units import ONE, Unit

ROOT_BUILTINS = {"sqrt": Fraction(1, 2), "cbrt": Fraction(1, 3)}
UNITLESS_BUILTINS = ("sin", "cos", "exp", "log", "normcdf")
BUILTINS = (*ROOT_BUILTINS, *UNITLESS_BUILTINS)

_KIND_WORDS = {"param": "parameter", "loop": "loop variable", "const": "constant"}


def _alias_of(unit: Unit, aliases: dict[str, Unit]):
    if unit.is_concrete and not unit.is_dimensionless:
        for name, value in aliases.items():
            if value == unit:
                return name
    return None


@dataclass
class CheckResult:
    """A checked program: every expression node carries its unit."""

    program: A.Program
    base_units: list[str]
    aliases: dict[str, Unit]
    consts: dict[str, ConstInfo]
    funcs: dict[str, FuncInfo]  # callees before callers
    grads: dict[str, GradInfo]

    def fmt(self, unit: Unit) -> str:
        return unit.format(self.base_units)

    def named(self, unit: Unit) -> str:
        """A declared name such as J when the unit has one, else its base form."""
        return _alias_of(unit, self.aliases) or self.fmt(unit)

    def describe(self, unit: Unit) -> str:
        alias = _alias_of(unit, self.aliases)
        return f"{alias} (= {self.fmt(unit)})" if alias else self.fmt(unit)

    def report(self) -> str:
        lines = []
        for decl in self.program.decls:
            if isinstance(decl, A.FnDecl):
                f = self.funcs[decl.name]
                params = []
                for i, param in enumerate(decl.params):
                    # Declared units are shown as written; solved ones in base units.
                    if i in f.inferred:
                        params.append(f"{param.name}: {self.named(f.param_units[i])} (inferred)")
                    else:
                        params.append(f"{param.name}: {unit_expr_str(param.unit)}")
                lines.append(f"fn {f.name}({', '.join(params)}) -> {unit_expr_str(decl.ret)}")
            elif isinstance(decl, A.GradDecl):
                g = self.grads[f"{decl.func}_grad_{decl.wrt}"]
                lines.append(f"grad {g.func} wrt {g.wrt} : {self.describe(g.unit)}")
        return "\n".join(lines)


class Checker:
    def __init__(self, program: A.Program):
        self.program = program
        self.diags: list[Diagnostic] = []
        self.solver = Solver()
        self.base_units: list[str] = []
        self.aliases: dict[str, Unit] = {}
        self.consts: dict[str, ConstInfo] = {}
        self.funcs: dict[str, FuncInfo] = {}
        self.grads: dict[str, GradInfo] = {}
        self.scopes = Scopes()
        self.current: FuncInfo | None = None

    # --- helpers ------------------------------------------------------------

    def error(self, node, message: str, *notes: str):
        self.diags.append(Diagnostic(node.line, node.col, message, list(notes)))

    def fmt(self, unit: Unit) -> str:
        return self.solver.resolve(unit).format(self.base_units, self.solver.labels)

    def describe(self, unit: Unit) -> str:
        alias = _alias_of(self.solver.resolve(unit), self.aliases)
        return f"{alias} (= {self.fmt(unit)})" if alias else self.fmt(unit)

    def unknown(self) -> Unit:
        """The unit of something already reported as an error: it matches anything."""
        return self.solver.fresh("zero")

    # --- driver -------------------------------------------------------------

    def run(self) -> CheckResult:
        self.collect()
        order = self.dependency_order()
        for name in order:
            self.check_function(self.funcs[name])
        for grad in self.grads.values():
            f = self.funcs[grad.func]
            grad.unit = f.ret_unit / f.param_units[grad.index]
            if f.uses_grad:
                self.error(grad.node,
                           f"cannot differentiate '{f.name}': it uses a gradient function, "
                           "and second derivatives are not supported")
        if self.diags:
            raise DimGradError(self.diags)
        return CheckResult(self.program, self.base_units, self.aliases, self.consts,
                           {name: self.funcs[name] for name in order}, self.grads)

    # --- global declarations ------------------------------------------------

    def unit_of(self, node: A.UnitExpr) -> Unit:
        unit = ONE
        for name, exp in node.factors:
            if name in self.aliases:
                unit = unit * self.aliases[name] ** exp
            elif name in self.base_units:
                unit = unit * Unit.of(name) ** exp
            else:
                self.error(node, f"unknown unit '{name}'")
        return unit

    def collect(self):
        for decl in self.program.decls:
            if isinstance(decl, A.UnitDecl):
                for name, line, col in decl.names:
                    if name in self.base_units or name in self.aliases:
                        self.diags.append(Diagnostic(line, col, f"unit '{name}' is already declared"))
                    else:
                        self.base_units.append(name)
            elif isinstance(decl, A.UnitAlias):
                if decl.name in self.base_units or decl.name in self.aliases:
                    self.error(decl, f"unit '{decl.name}' is already declared")
                else:
                    self.aliases[decl.name] = self.unit_of(decl.expr)
            elif isinstance(decl, A.ConstDecl):
                if decl.name in self.consts:
                    self.error(decl, f"'{decl.name}' is already defined (line {self.consts[decl.name].line})")
                else:
                    self.consts[decl.name] = ConstInfo(decl.name, self.unit_of(decl.unit), decl.value, decl.line)
            elif isinstance(decl, A.FnDecl):
                self.collect_function(decl)
        for decl in self.program.decls:
            if isinstance(decl, A.GradDecl):
                self.collect_grad(decl)

    def collect_function(self, decl: A.FnDecl):
        if decl.name in BUILTINS:
            self.error(decl, f"'{decl.name}' is a built-in function and cannot be redefined")
            return
        if decl.name in self.funcs:
            self.error(decl, f"function '{decl.name}' is already defined (line {self.funcs[decl.name].node.line})")
            return
        units, inferred = [], set()
        for i, param in enumerate(decl.params):
            if param.unit.placeholder:
                inferred.add(i)
                units.append(self.solver.fresh("param", f"?{param.name}"))
            else:
                units.append(self.unit_of(param.unit))
        self.funcs[decl.name] = FuncInfo(decl.name, decl, [p.name for p in decl.params],
                                         units, self.unit_of(decl.ret), inferred)

    def collect_grad(self, decl: A.GradDecl):
        f = self.funcs.get(decl.func)
        name = f"{decl.func}_grad_{decl.wrt}"
        if f is None:
            self.error(decl, f"cannot take the gradient of unknown function '{decl.func}'")
        elif decl.wrt not in f.param_names:
            self.error(decl, f"'{decl.wrt}' is not a parameter of '{decl.func}'")
        elif name in self.grads:
            self.error(decl, f"the gradient of '{decl.func}' with respect to '{decl.wrt}' is already declared")
        elif name in self.funcs:
            self.error(decl, f"the gradient function would be named '{name}', which is already a function")
        else:
            self.grads[name] = GradInfo(name, decl.func, decl.wrt, f.param_names.index(decl.wrt), decl)

    def dependency_order(self) -> list[str]:
        """Function names with callees first; recursion is reported and broken."""
        for f in self.funcs.values():
            for node in A.iter_exprs(f.node.body):
                if isinstance(node, A.Call):
                    if node.name in self.funcs:
                        f.calls.add(node.name)
                    elif node.name in self.grads:
                        f.calls.add(self.grads[node.name].func)
                        f.uses_grad = True
        order: list[str] = []
        state: dict[str, int] = {}

        def visit(name: str, path: list[str]):
            state[name] = 1
            for callee in sorted(self.funcs[name].calls):
                if state.get(callee) == 1:
                    cycle = path[path.index(callee):] + [callee]
                    self.error(self.funcs[name].node, f"recursion is not supported ({' -> '.join(cycle)})")
                elif callee not in state:
                    visit(callee, path + [callee])
            state[name] = 2
            order.append(name)

        for name in self.funcs:
            if name not in state:
                visit(name, [name])
        for name in order:
            f = self.funcs[name]
            f.uses_grad = f.uses_grad or any(self.funcs[c].uses_grad for c in f.calls)
        return order

    def signature(self, name: str):
        """(parameter names, parameter units, return unit) of a function or gradient."""
        if name in self.funcs:
            f = self.funcs[name]
            return f.param_names, f.param_units, f.ret_unit
        if name in self.grads:
            g = self.grads[name]
            f = self.funcs[g.func]
            return f.param_names, f.param_units, f.ret_unit / f.param_units[g.index]
        return None

    # --- functions ----------------------------------------------------------

    def declare(self, name: str, unit: Unit, kind: str, node):
        previous = self.scopes.lookup(name) or self.consts.get(name)
        if previous is not None:
            self.error(node, f"'{name}' is already defined (line {previous.line})")
            return
        self.scopes.declare(VarInfo(name, unit, kind, node.line))

    def check_function(self, f: FuncInfo):
        self.current = f
        self.scopes = Scopes()
        self.scopes.push()
        for param, unit in zip(f.node.params, f.param_units):
            self.declare(param.name, unit, "param", param)
        self.block(f.node.body, new_scope=False)
        if not self.returns(f.node.body, tail=True):
            self.error(f.node, f"function '{f.name}' does not return a value on every path")

        # A literal 0 that nothing constrained is simply unitless.
        for var in self.solver.unsolved("zero"):
            self.solver.default_to_unitless(var)
        for i in sorted(f.inferred):
            unit = self.solver.resolve(f.param_units[i])
            if not unit.is_concrete:
                param = f.node.params[i]
                self.error(param, f"cannot infer the unit of parameter '{param.name}': "
                                  f"nothing in '{f.name}' determines it")
                for var in unit.vars:
                    self.solver.default_to_unitless(var)
        f.param_units = [self.solver.resolve(u) for u in f.param_units]
        for node in A.iter_exprs(f.node.body):
            node.unit = self.solver.resolve(node.unit)

    def returns(self, block: A.Block, tail: bool) -> bool:
        """Check that `return` only appears in tail position; True if the block always returns."""
        always = False
        for i, stmt in enumerate(block.stmts):
            last = i == len(block.stmts) - 1
            if isinstance(stmt, A.Return):
                if tail and last:
                    always = True
                else:
                    self.error(stmt, "'return' must be the last statement of the function "
                                     "or of a final if/else branch")
            elif isinstance(stmt, A.If):
                then = self.returns(stmt.then, tail and last)
                els = self.returns(stmt.els, tail and last) if stmt.els is not None else False
                always = last and then and els
            elif isinstance(stmt, A.For):
                self.returns(stmt.body, False)
        return always

    # --- statements ---------------------------------------------------------

    def block(self, block: A.Block, new_scope: bool = True):
        if new_scope:
            self.scopes.push()
        for stmt in block.stmts:
            self.stmt(stmt)
        if new_scope:
            self.scopes.pop()

    def stmt(self, stmt):
        if isinstance(stmt, A.Let):
            self.declare(stmt.name, self.expr(stmt.value), "let", stmt)
        elif isinstance(stmt, A.Assign):
            unit = self.expr(stmt.value)
            info = self.scopes.lookup(stmt.name)
            kind = info.kind if info else ("const" if stmt.name in self.consts else None)
            if kind is None:
                self.error(stmt, f"assignment to undefined variable '{stmt.name}' (declare it with 'let')")
            elif kind != "let":
                self.error(stmt, f"cannot assign to {_KIND_WORDS[kind]} '{stmt.name}'")
            elif not self.solver.unify(info.unit, unit):
                self.error(stmt, f"cannot assign {self.describe(unit)} to '{stmt.name}', "
                                 f"which has unit {self.describe(info.unit)}",
                           f"in statement: {stmt.name} = {expr_str(stmt.value)}")
        elif isinstance(stmt, A.If):
            cond = stmt.cond
            lhs, rhs = self.expr(cond.lhs), self.expr(cond.rhs)
            if not self.solver.unify(lhs, rhs):
                self.error(cond, f"cannot compare {self.describe(lhs)} and {self.describe(rhs)}",
                           f"in condition: {expr_str(cond.lhs)} {cond.op} {expr_str(cond.rhs)}")
            self.block(stmt.then)
            if stmt.els is not None:
                self.block(stmt.els)
        elif isinstance(stmt, A.For):
            self.scopes.push()
            self.declare(stmt.var, ONE, "loop", stmt)
            self.block(stmt.body, new_scope=False)
            self.scopes.pop()
        elif isinstance(stmt, A.Return):
            unit = self.expr(stmt.value)
            f = self.current
            if not self.solver.unify(unit, f.ret_unit):
                self.error(stmt, f"function '{f.name}' returns {self.describe(f.ret_unit)} "
                                 f"but this value has unit {self.describe(unit)}",
                           f"in statement: return {expr_str(stmt.value)}")

    # --- expressions --------------------------------------------------------

    def expr(self, node: A.Expr) -> Unit:
        node.unit = self._expr(node)
        return node.unit

    def _expr(self, node: A.Expr) -> Unit:
        if isinstance(node, A.Num):
            # Kennedy's rule: 0 takes whatever unit its context requires.
            return self.solver.fresh("zero") if node.value == 0 else ONE
        if isinstance(node, A.Name):
            info = self.scopes.lookup(node.name) or self.consts.get(node.name)
            if info is None:
                self.error(node, f"undefined variable '{node.name}'")
                return self.unknown()
            return info.unit
        if isinstance(node, A.BinOp):
            lhs, rhs = self.expr(node.lhs), self.expr(node.rhs)
            if node.op == "*":
                return lhs * rhs
            if node.op == "/":
                return lhs / rhs
            if not self.solver.unify(lhs, rhs):
                if node.op == "+":
                    message = f"cannot add {self.describe(lhs)} and {self.describe(rhs)}"
                else:
                    message = f"cannot subtract {self.describe(rhs)} from {self.describe(lhs)}"
                self.error(node, message,
                           f"in expression: {expr_str(node)}",
                           f"left side:  {expr_str(node.lhs)} : {self.fmt(lhs)}",
                           f"right side: {expr_str(node.rhs)} : {self.fmt(rhs)}")
                return self.unknown()  # one message per mistake, not a cascade
            return lhs
        if isinstance(node, A.Neg):
            return self.expr(node.operand)
        if isinstance(node, A.Pow):
            return self.expr(node.base) ** node.exp
        if isinstance(node, A.Call):
            return self.call(node)
        raise TypeError(f"not an expression: {node!r}")

    def call(self, node: A.Call) -> Unit:
        arg_units = [self.expr(arg) for arg in node.args]
        name = node.name
        if name in BUILTINS:
            if len(arg_units) != 1:
                self.error(node, f"'{name}' expects 1 argument but got {len(arg_units)}")
                return self.unknown()
            if name in ROOT_BUILTINS:
                return arg_units[0] ** ROOT_BUILTINS[name]
            if not self.solver.unify(arg_units[0], ONE):
                self.error(node, f"the argument of '{name}' must be unitless but has unit "
                                 f"{self.describe(arg_units[0])}",
                           f"in expression: {expr_str(node)}")
            return ONE
        signature = self.signature(name)
        if signature is None:
            notes = []
            if "_grad_" in name:
                func, wrt = name.split("_grad_", 1)
                notes.append(f"to use a gradient, declare it first: grad {func} wrt {wrt};")
            self.error(node, f"unknown function '{name}'", *notes)
            return self.unknown()
        param_names, param_units, ret_unit = signature
        if len(arg_units) != len(param_names):
            plural = "" if len(param_names) == 1 else "s"
            self.error(node, f"'{name}' expects {len(param_names)} argument{plural} but got {len(arg_units)}")
            return ret_unit
        for i, (arg, unit) in enumerate(zip(node.args, arg_units)):
            if not self.solver.unify(unit, param_units[i]):
                self.error(arg, f"argument {i + 1} of '{name}' (parameter '{param_names[i]}') must have unit "
                                f"{self.describe(param_units[i])} but has unit {self.describe(unit)}",
                           f"in expression: {expr_str(node)}")
        return ret_unit


def check(program: A.Program) -> CheckResult:
    return Checker(program).run()
