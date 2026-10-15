"""RQ2 (detection): does the checker reject programs with injected unit errors?

Each mutation changes the unit of one expression in a correct benchmark. The
units computed by the checker for the original program are used to pick only
replacements whose unit differs, so no mutant preserves units by accident.
Five categories, as in Section 7.3 of the paper:

  1. adding or subtracting values with different units
  2. a wrong exponent in a power
  3. a missing or extra conversion constant
  4. swapped arguments with different units
  5. a wrong variable or unit in a derivative-based update

    python -m evaluation.mutate [--save-mutants]
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from typing import Callable

from dimgrad.frontend import ast_nodes as A
from dimgrad.frontend.parser import parse
from dimgrad.frontend.unparse import expr_str, unparse
from dimgrad.glue.errors import DimGradError
from dimgrad.units.checker import ROOT_BUILTINS, CheckResult, check
from dimgrad.units.units import Unit

from .common import RESULTS_DIR, benchmark_names, load_source, markdown_table, use_utf8_output

CATEGORIES = {
    1: "add/subtract different units",
    2: "wrong exponent",
    3: "missing or extra conversion constant",
    4: "swapped arguments",
    5: "wrong variable or unit in a derivative-based update",
}
PER_FUNCTION_CAP = 3  # for the mutation kinds that would otherwise apply to almost every operand
FACTOR = "mutant_factor"


@dataclass
class Slot:
    """A place in the tree that holds an expression, so it can be read and replaced."""
    owner: object
    attr: str
    index: int | None
    func: A.FnDecl

    def get(self):
        value = getattr(self.owner, self.attr)
        return value if self.index is None else value[self.index]

    def set(self, node):
        if self.index is None:
            setattr(self.owner, self.attr, node)
        else:
            getattr(self.owner, self.attr)[self.index] = node


def slots(program: A.Program) -> list[Slot]:
    """Every expression position, in a fixed order that survives a deep copy."""
    out: list[Slot] = []

    def expr(slot: Slot):
        out.append(slot)
        node = slot.get()
        if isinstance(node, A.BinOp):
            expr(Slot(node, "lhs", None, slot.func))
            expr(Slot(node, "rhs", None, slot.func))
        elif isinstance(node, A.Neg):
            expr(Slot(node, "operand", None, slot.func))
        elif isinstance(node, A.Pow):
            expr(Slot(node, "base", None, slot.func))
        elif isinstance(node, A.Call):
            for i in range(len(node.args)):
                expr(Slot(node, "args", i, slot.func))

    def block(node: A.Block, func: A.FnDecl):
        for stmt in node.stmts:
            if isinstance(stmt, (A.Let, A.Assign, A.Return)):
                expr(Slot(stmt, "value", None, func))
            elif isinstance(stmt, A.If):
                expr(Slot(stmt.cond, "lhs", None, func))
                expr(Slot(stmt.cond, "rhs", None, func))
                block(stmt.then, func)
                if stmt.els is not None:
                    block(stmt.els, func)
            elif isinstance(stmt, A.For):
                block(stmt.body, func)

    for decl in program.decls:
        if isinstance(decl, A.FnDecl):
            block(decl.body, decl)
    return out


@dataclass
class Mutation:
    category: int
    line: int
    function: str
    apply: Callable[[A.Program, list[Slot]], str]  # mutates a copy, returns a description


def _is_zero(node) -> bool:
    return isinstance(node, A.Num) and node.value == 0


def _name(name: str, like) -> A.Name:
    return A.Name(like.line, like.col, name)


def _replace(index: int, build) -> Callable:
    def apply(program: A.Program, sl: list[Slot]) -> str:
        before = sl[index].get()
        after = build(before, program)
        sl[index].set(after)
        return f"{expr_str(before)}  ->  {expr_str(after)}"
    return apply


def _unit_expr(unit, order, like) -> A.UnitExpr:
    rank = {name: i for i, name in enumerate(order)}
    factors = sorted(unit.base.items(), key=lambda kv: rank[kv[0]])
    return A.UnitExpr(like.line, like.col, factors)


def generate(base: A.Program, result: CheckResult) -> list[Mutation]:
    """All mutations of a checked program (whose expression nodes carry their units)."""
    out: list[Mutation] = []
    sl = slots(base)
    const_units = {c.name: c.unit for c in result.consts.values()}
    first, last = result.base_units[0], result.base_units[-1]
    counts: dict[tuple, int] = {}

    def capped(kind: str, func: str) -> bool:
        counts[kind, func] = counts.get((kind, func), 0) + 1
        return counts[kind, func] > PER_FUNCTION_CAP

    def with_factor_decl(program: A.Program):
        if not any(isinstance(d, A.ConstDecl) and d.name == FACTOR for d in program.decls):
            factors = [(first, 1)] + ([(last, -1)] if last != first else [])
            program.decls.append(A.ConstDecl(0, 0, FACTOR, A.UnitExpr(0, 0, factors), 1000.0, "1000.0"))

    for i, slot in enumerate(sl):
        node, fn = slot.get(), slot.func
        info = result.funcs[fn.name]
        scope = list(zip(info.param_names, info.param_units)) + list(const_units.items())

        # 1. add or subtract a value with a different unit
        if isinstance(node, A.BinOp) and node.op in "+-" and not _is_zero(node.lhs) and not _is_zero(node.rhs):
            other = next((n for n, u in scope if u != node.rhs.unit), None)
            if other is not None:
                out.append(Mutation(1, node.line, fn.name, _replace(
                    i, lambda n, p, other=other: A.BinOp(n.line, n.col, n.op, n.lhs, _name(other, n.rhs)))))
            else:
                factor = next((n for n, u in scope if not u.is_dimensionless), None)
                if factor is not None:
                    out.append(Mutation(1, node.line, fn.name, _replace(
                        i, lambda n, p, factor=factor: A.BinOp(
                            n.line, n.col, n.op, n.lhs, A.BinOp(n.line, n.col, "*", n.rhs, _name(factor, n.rhs))))))

        # 2. wrong exponent
        if isinstance(node, A.Pow) and not node.base.unit.is_dimensionless:
            out.append(Mutation(2, node.line, fn.name, _replace(
                i, lambda n, p: A.Pow(n.line, n.col, n.base, n.exp + (1 if n.exp > 0 else -1)))))
        if isinstance(node, A.Call) and node.name in ROOT_BUILTINS and not node.args[0].unit.is_dimensionless:
            swapped = "cbrt" if node.name == "sqrt" else "sqrt"
            out.append(Mutation(2, node.line, fn.name, _replace(
                i, lambda n, p, swapped=swapped: A.Call(n.line, n.col, swapped, n.args))))
        if (isinstance(node, A.BinOp) and node.op == "*" and isinstance(node.rhs, A.Name)
                and not node.rhs.unit.is_dimensionless and not capped("square", fn.name)):
            out.append(Mutation(2, node.line, fn.name, _replace(
                i, lambda n, p: A.BinOp(n.line, n.col, n.op, n.lhs, A.Pow(n.rhs.line, n.rhs.col, n.rhs, 2)))))

        # 3. missing or extra conversion constant
        if isinstance(node, A.BinOp) and node.op in "*/":
            def is_factor(operand):
                return (isinstance(operand, A.Name) and operand.name in const_units
                        and not const_units[operand.name].is_dimensionless)
            if is_factor(node.rhs):
                out.append(Mutation(3, node.line, fn.name, _replace(i, lambda n, p: n.lhs)))
            elif node.op == "*" and is_factor(node.lhs):
                out.append(Mutation(3, node.line, fn.name, _replace(i, lambda n, p: n.rhs)))
        if isinstance(node, A.Name) and node.name in info.param_names and not capped("factor", fn.name):
            def add_factor(n, program):
                with_factor_decl(program)
                return A.BinOp(n.line, n.col, "*", n, _name(FACTOR, n))
            out.append(Mutation(3, node.line, fn.name, _replace(i, add_factor)))

        # 4. swapped arguments, and 5a. derivative with respect to the wrong variable
        if isinstance(node, A.Call) and (node.name in result.funcs or node.name in result.grads):
            grad = result.grads.get(node.name)
            callee = result.funcs[grad.func if grad else node.name]
            units = callee.param_units
            pair = next(((a, b) for a in range(len(units)) for b in range(a + 1, len(units))
                         if units[a] != units[b] and not _is_zero(node.args[a]) and not _is_zero(node.args[b])),
                        None)
            if pair is not None:
                def swap(n, program, pair=pair):
                    args = list(n.args)
                    args[pair[0]], args[pair[1]] = args[pair[1]], args[pair[0]]
                    return A.Call(n.line, n.col, n.name, args)
                out.append(Mutation(4, node.line, fn.name, _replace(i, swap)))
            if grad is not None:
                for j, wrt in enumerate(callee.param_names):
                    if units[j] == units[grad.index]:
                        continue

                    def wrong_variable(n, program, wrt=wrt, func=grad.func):
                        declared = any(isinstance(d, A.GradDecl) and (d.func, d.wrt) == (func, wrt)
                                       for d in program.decls)
                        if not declared:
                            program.decls.append(A.GradDecl(n.line, n.col, func, wrt))
                        return A.Call(n.line, n.col, f"{func}_grad_{wrt}", n.args)
                    out.append(Mutation(5, node.line, fn.name, _replace(i, wrong_variable)))

    # 5b. a wrong unit on a parameter of a function that performs a derivative-based update
    for decl_index, decl in enumerate(base.decls):
        if not isinstance(decl, A.FnDecl) or not result.funcs[decl.name].uses_grad:
            continue
        info = result.funcs[decl.name]
        for j, param in enumerate(decl.params):
            before = info.param_units[j]
            wrong = before * Unit.of(first)

            def wrong_unit(program, sl, decl_index=decl_index, j=j, before=before, wrong=wrong):
                target = program.decls[decl_index].params[j]
                target.unit = _unit_expr(wrong, result.base_units, target)
                return f"parameter {target.name}: {result.fmt(before)}  ->  {result.fmt(wrong)}"
            out.append(Mutation(5, param.line, decl.name, wrong_unit))
    return out


def function_at(program: A.Program, line: int):
    """Name of the function whose definition contains `line`."""
    best = None
    for decl in program.decls:
        if isinstance(decl, A.FnDecl) and decl.line <= line and (best is None or decl.line > best.line):
            best = decl
    return best.name if best else None


def run_benchmark(bench: str, save_dir=None) -> list[dict]:
    source = load_source(bench)
    base = parse(source)
    result = check(base)
    rows = []
    for number, mutation in enumerate(generate(base, result), 1):
        mutant = copy.deepcopy(base)
        description = mutation.apply(mutant, slots(mutant))
        text = unparse(mutant)
        try:
            check(mutant)
            diagnostics = []
        except DimGradError as exc:
            diagnostics = exc.diagnostics
        # The mutant must be a real program: the verdict on its source text is the same.
        try:
            check(parse(text))
            reparsed_rejected = False
        except DimGradError:
            reparsed_rejected = True
        assert reparsed_rejected == bool(diagnostics), f"{bench} mutant {number}: verdict changed after unparsing"

        if save_dir is not None:
            header = f"// {bench}, mutant {number}: {CATEGORIES[mutation.category]}\n// {description}\n"
            (save_dir / f"{bench}_{number:03}.dg").write_text(header + text, encoding="utf-8")
        rows.append({
            "benchmark": bench, "mutant": number, "category": mutation.category,
            "function": mutation.function, "line": mutation.line, "mutation": description,
            "detected": bool(diagnostics),
            "same_line": any(d.line == mutation.line for d in diagnostics),
            "same_function": any(function_at(base, d.line) == mutation.function for d in diagnostics),
            "message": diagnostics[0].message if diagnostics else "",
        })
    return rows


def main(argv=None) -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save-mutants", action="store_true", help="write every mutant program to results/mutants/")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_dir = None
    if args.save_mutants:
        save_dir = RESULTS_DIR / "mutants"
        save_dir.mkdir(exist_ok=True)

    wrongly_rejected = []
    rows = []
    for bench in benchmark_names():
        try:
            rows += run_benchmark(bench, save_dir)
        except DimGradError:
            wrongly_rejected.append(bench)

    with open(RESULTS_DIR / "rq2_mutants.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def summarise(subset):
        n = len(subset)
        detected = [r for r in subset if r["detected"]]
        rate = f"{100 * len(detected) / n:.1f}%" if n else "-"
        line = f"{100 * sum(r['same_line'] for r in detected) / len(detected):.1f}%" if detected else "-"
        func = f"{100 * sum(r['same_function'] for r in detected) / len(detected):.1f}%" if detected else "-"
        return [n, len(detected), rate, line, func]

    table_rows = [[f"{c}. {label}"] + summarise([r for r in rows if r["category"] == c])
                  for c, label in CATEGORIES.items()]
    table_rows.append(["all"] + summarise(rows))
    table = markdown_table(["category", "mutants", "rejected", "detection rate",
                            "error on the mutated line", "error in the mutated function"], table_rows)

    missed = [r for r in rows if not r["detected"]]
    missed_table = markdown_table(
        ["benchmark", "category", "function", "line", "mutation"],
        [[r["benchmark"], r["category"], r["function"], r["line"], f"`{r['mutation']}`"] for r in missed])
    summary = (f"{len(rows)} unit-changing mutants from {len(benchmark_names())} programs; "
               f"{len(rows) - len(missed)} rejected, {len(missed)} accepted. "
               f"Original programs wrongly rejected: {len(wrongly_rejected)}.")

    text = f"# RQ2: detection of injected unit errors\n\n{summary}\n\n{table}\n"
    if missed:
        text += f"\n## Mutants the checker accepted\n\n{missed_table}\n"
    (RESULTS_DIR / "rq2_mutants.md").write_text(text, encoding="utf-8")
    print(table)
    print()
    print(summary)
    if missed:
        print()
        print("Accepted mutants:")
        print(missed_table)
    return 1 if wrongly_rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
