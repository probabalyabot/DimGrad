"""Turn syntax trees back into source text (error messages and mutant programs)."""
from __future__ import annotations

from . import ast_nodes as A

_PREC = {"+": 1, "-": 1, "*": 2, "/": 2}


def _prec(node) -> int:
    if isinstance(node, A.BinOp):
        return _PREC[node.op]
    if isinstance(node, A.Neg):
        return 3
    if isinstance(node, A.Pow):
        return 4
    return 5


def _wrap(node, needs_parens: bool) -> str:
    text = expr_str(node)
    return f"({text})" if needs_parens else text


def expr_str(node) -> str:
    if isinstance(node, A.Num):
        return node.text
    if isinstance(node, A.Name):
        return node.name
    if isinstance(node, A.BinOp):
        prec = _PREC[node.op]
        lhs = _wrap(node.lhs, _prec(node.lhs) < prec)
        rhs = _wrap(node.rhs, _prec(node.rhs) <= prec)
        return f"{lhs} {node.op} {rhs}"
    if isinstance(node, A.Neg):
        return "-" + _wrap(node.operand, _prec(node.operand) < 3)
    if isinstance(node, A.Pow):
        return f"{_wrap(node.base, _prec(node.base) < 5)}^{node.exp}"
    if isinstance(node, A.Call):
        return f"{node.name}({', '.join(expr_str(a) for a in node.args)})"
    raise TypeError(f"not an expression: {node!r}")


def _exponent(exp) -> str:
    if exp == 1:
        return ""
    if exp.denominator == 1:
        return f"^{exp.numerator}"
    return f"^({exp.numerator}/{exp.denominator})"


def unit_expr_str(node: A.UnitExpr) -> str:
    if node.placeholder:
        return "?"
    text = ""
    for name, exp in node.factors:
        if exp > 0:
            text += ("*" if text else "") + name + _exponent(exp)
        else:
            text = (text or "1") + "/" + name + _exponent(-exp)
    return text or "1"


def _block(block: A.Block, depth: int, out: list[str]):
    pad = "    " * depth
    for stmt in block.stmts:
        if isinstance(stmt, A.Let):
            out.append(f"{pad}let {stmt.name} = {expr_str(stmt.value)};")
        elif isinstance(stmt, A.Assign):
            out.append(f"{pad}{stmt.name} = {expr_str(stmt.value)};")
        elif isinstance(stmt, A.Return):
            out.append(f"{pad}return {expr_str(stmt.value)};")
        elif isinstance(stmt, A.For):
            out.append(f"{pad}for {stmt.var} in {stmt.lo}..{stmt.hi} {{")
            _block(stmt.body, depth + 1, out)
            out.append(f"{pad}}}")
        elif isinstance(stmt, A.If):
            c = stmt.cond
            out.append(f"{pad}if ({expr_str(c.lhs)} {c.op} {expr_str(c.rhs)}) {{")
            _block(stmt.then, depth + 1, out)
            if stmt.els is not None:
                out.append(f"{pad}}} else {{")
                _block(stmt.els, depth + 1, out)
            out.append(f"{pad}}}")
        else:
            raise TypeError(f"not a statement: {stmt!r}")


def unparse(program: A.Program) -> str:
    out: list[str] = []
    for decl in program.decls:
        if isinstance(decl, A.UnitDecl):
            out.append(f"unit {', '.join(name for name, _, _ in decl.names)};")
        elif isinstance(decl, A.UnitAlias):
            out.append(f"unit {decl.name} = {unit_expr_str(decl.expr)};")
        elif isinstance(decl, A.ConstDecl):
            out.append(f"const {decl.name}: {unit_expr_str(decl.unit)} = {decl.text};")
        elif isinstance(decl, A.GradDecl):
            out.append(f"grad {decl.func} wrt {decl.wrt};")
        elif isinstance(decl, A.FnDecl):
            params = ", ".join(f"{p.name}: {unit_expr_str(p.unit)}" for p in decl.params)
            out.append("")
            out.append(f"fn {decl.name}({params}) -> {unit_expr_str(decl.ret)} {{")
            _block(decl.body, 1, out)
            out.append("}")
        else:
            raise TypeError(f"not a declaration: {decl!r}")
    return "\n".join(out) + "\n"
