"""Emit C for every function and gradient. Units are erased; comments keep them readable."""
from __future__ import annotations

from collections import Counter

from ..middle import tac

# Names that cannot be used as-is for a variable or function in the generated C.
_RESERVED = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum",
    "extern", "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return",
    "short", "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void",
    "volatile", "while",
    "sqrt", "cbrt", "sin", "cos", "tan", "exp", "log", "pow", "erf", "erfc", "fabs", "floor", "ceil",
    "round", "gamma", "y0", "y1", "yn", "j0", "j1", "jn", "abs", "div", "time", "clock", "remove",
    "rename", "signal", "printf", "scanf", "strcmp", "main", "value", "NULL", "dg_normcdf",
}

_COMMENT_COLUMN = 44


def cname(name: str) -> str:
    return name + "_" if name in _RESERVED else name


def _operand(operand) -> str:
    if isinstance(operand, float):
        text = repr(operand)
        return f"({text})" if text.startswith("-") else text
    return cname(operand)


def _expression(instr: tac.Instr, program: tac.Program) -> str:
    a = [_operand(x) for x in tac.operands(instr)]
    op = instr.op
    if op == "copy":
        return a[0]
    if op == "neg":
        return f"-{a[0]}"
    if op in tac.BINARY_SYMBOLS:
        return f"{a[0]} {tac.BINARY_SYMBOLS[op]} {a[1]}"
    if op == "pow":
        n = instr.args[1]
        if n in (2, 3, 4):  # pow() is far slower than a few multiplications
            return " * ".join([a[0]] * n)
        if n == -1:
            return f"1.0 / {a[0]}"
        if n == -2:
            return f"1.0 / ({a[0]} * {a[0]})"
        return f"pow({a[0]}, {float(n)!r})"
    if op == "call":
        callee = program.functions[instr.args[0]]
        if callee.kind == "grad":
            a.append("NULL")  # only the derivative is wanted
        return f"{cname(callee.name)}({', '.join(a)})"
    if op == "normcdf":
        return f"dg_normcdf({a[0]})"
    return f"{op}({a[0]})"


def _prototype(func: tac.Function) -> str:
    params = [f"double {cname(p)}" for p in func.params]
    if func.kind == "grad":
        params.append("double *value")
    return f"double {cname(func.name)}({', '.join(params) or 'void'})"


def _describe(func: tac.Function, program: tac.Program) -> str:
    fmt = program.fmt
    if func.kind == "grad":
        return (f"d {func.of} / d {func.wrt} : {fmt(func.ret_unit)}"
                f"   (also stores {func.of} : {fmt(func.value_unit)} in *value)")
    params = ", ".join(f"{p}: {fmt(func.units[p])}" for p in func.params)
    return f"{func.name} : ({params}) -> {fmt(func.ret_unit)}"


def _emit_function(func: tac.Function, program: tac.Program, out: list[str]):
    fmt = program.fmt
    defs = Counter(item.dst for item in tac.walk(func.body) if isinstance(item, tac.Instr))
    top_level = {item.dst for item in func.body if isinstance(item, tac.Instr)}
    # A variable assigned in several places, or only inside a block, is declared up front.
    hoisted = sorted(n for n in defs if n in func.vars and not (defs[n] == 1 and n in top_level))

    def line(depth: int, code: str, unit=None):
        text = "    " * depth + code
        if unit is not None:
            text = f"{text:<{_COMMENT_COLUMN}}  /* {fmt(unit)} */"
        out.append(text)

    def block(body: list, depth: int):
        for item in body:
            if isinstance(item, tac.Instr):
                decl = "" if item.dst in hoisted else "double "
                line(depth, f"{decl}{cname(item.dst)} = {_expression(item, program)};", item.unit)
            elif isinstance(item, tac.If):
                line(depth, f"if ({_operand(item.lhs)} {item.relop} {_operand(item.rhs)}) {{")
                block(item.then, depth + 1)
                if item.els:
                    line(depth, "} else {")
                    block(item.els, depth + 1)
                line(depth, "}")
            elif isinstance(item, tac.For):
                var = cname(item.var)
                line(depth, f"for (int {var} = {item.lo}; {var} < {item.hi}; {var}++) {{")
                block(item.body, depth + 1)
                line(depth, "}")
            elif isinstance(item, tac.Return):
                if func.kind == "grad":
                    line(depth, f"if (value) *value = {_operand(item.value)};")
                    line(depth, f"return {_operand(item.tangent)};")
                else:
                    line(depth, f"return {_operand(item.value)};")

    out.append(f"/* {_describe(func, program)} */")
    out.append(_prototype(func))
    out.append("{")
    if hoisted:
        line(1, f"double {', '.join(cname(n) for n in hoisted)};")
    block(func.body, 1)
    out.append("}")


def _emit_harness(program: tac.Program, out: list[str]):
    """A main() that reads `name arg...` lines from stdin and prints the results."""
    arity = max((len(f.params) for f in program.functions.values()), default=0)
    out += [
        "#include <stdio.h>",
        "#include <string.h>",
        "",
        "int main(void)",
        "{",
        "    char name[128];",
        f"    double a[{max(arity, 1)}];",
        "    while (scanf(\"%127s\", name) == 1) {",
    ]
    keyword = "if"
    for func in program.functions.values():
        n = len(func.params)
        args = [f"a[{i}]" for i in range(n)]
        out.append(f"        {keyword} (strcmp(name, \"{func.name}\") == 0) {{")
        if n:
            reads = ", ".join(f"&a[{i}]" for i in range(n))
            out.append(f"            if (scanf(\"{' '.join(['%lf'] * n)}\", {reads}) != {n}) return 2;")
        if func.kind == "grad":
            out.append("            double value;")
            out.append(f"            double deriv = {cname(func.name)}({', '.join(args + ['&value'])});")
            out.append("            printf(\"%.17g %.17g\\n\", deriv, value);")
        else:
            out.append(f"            printf(\"%.17g\\n\", {cname(func.name)}({', '.join(args)}));")
        keyword = "} else if"
    if program.functions:
        out += ["        } else {",
                "            fprintf(stderr, \"unknown function %s\\n\", name);",
                "            return 2;",
                "        }"]
    out += ["    }", "    return 0;", "}"]


def emit_c(program: tac.Program, source_name: str = "", harness: bool = False) -> str:
    origin = f" from {source_name}" if source_name else ""
    out = [
        f"/* Generated by DimGrad{origin}.",
        " * Units are checked at compile time and erased here; the comments record them. */",
        "#include <math.h>",
        "#include <stddef.h>",
        "",
    ]
    for name, (unit, value) in program.consts.items():
        out.append(f"{f'static const double {cname(name)} = {value!r};':<{_COMMENT_COLUMN}}  /* {program.fmt(unit)} */")
    if program.consts:
        out.append("")
    uses_normcdf = any(isinstance(item, tac.Instr) and item.op == "normcdf"
                       for f in program.functions.values() for item in tac.walk(f.body))
    if uses_normcdf:
        out += ["static double dg_normcdf(double x)",
                "{",
                "    return 0.5 * erfc(-x * 0.70710678118654752440);",
                "}",
                ""]
    for func in program.functions.values():
        out.append(_prototype(func) + ";")
    for func in program.functions.values():
        out.append("")
        _emit_function(func, program, out)
    if harness:
        out.append("")
        _emit_harness(program, out)
    return "\n".join(out) + "\n"
